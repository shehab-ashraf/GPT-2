import argparse
import os
import subprocess
import sys
import time

import torch

from src.model.gpt import GPT, GPTConfig
from src.model.muon import Muon
from src.data.dataloader import DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# -----------------------------------------------------------------------------
# utilities

def print_system_info() -> None:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")
    print()
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout.rstrip())
    except FileNotFoundError:
        print("nvidia-smi not found, running on CPU?")
    print()


def gpu_memory_gb() -> float:
    return torch.cuda.memory_reserved() / 1e9


def estimate_mfu(
    num_params_no_embed: int,
    avg_tps: float,
    seq_len: int,
    num_layers: int,
    model_dim: int,
    peak_flops: float = 312e12,  # A100 SXM4 bf16 peak
) -> float:
    flops_per_token = 6 * num_params_no_embed + 12 * num_layers * seq_len * model_dim
    return flops_per_token * avg_tps / peak_flops


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = sum(
        p.grad.detach().float().norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return total ** 0.5


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# -----------------------------------------------------------------------------
# evaluation

VAL_TOKENS = 10_485_760


@torch.no_grad()
def evaluate(model, val_loader, device, batch_size: int, seq_len: int) -> float:
    val_steps = VAL_TOKENS // (batch_size * seq_len)
    model.eval()
    val_loader.reset()

    losses = []
    for _ in range(val_steps):
        batch = val_loader.next_batch()
        if batch is None:
            break
        x, y, cu_seqlens, pos, max_seqlen = batch
        x = x.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        y = y.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        cu_seqlens = cu_seqlens.to(device, non_blocking=True)
        pos = pos.to(device, dtype=torch.long, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, cu_seqlens, pos, max_seqlen)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# -----------------------------------------------------------------------------
# training loop

def train(args) -> float:
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    grad_accum_steps = args.total_batch_size // (args.micro_batch * args.seq_len)
    assert args.total_batch_size % (args.micro_batch * args.seq_len) == 0, (
        f"total_batch_size {args.total_batch_size} must be divisible by "
        f"micro_batch × seq_len = {args.micro_batch * args.seq_len}"
    )

    print_system_info()
    print(f"  micro batch : {args.micro_batch} × {args.seq_len} = "
          f"{args.micro_batch * args.seq_len:,} tokens")
    print(f"  total batch : {args.total_batch_size:,} tokens  "
          f"(grad_accum={grad_accum_steps})")
    print(f"  total steps : {args.total_steps}")
    print(f"  schedule    : {args.warmup_steps} warmup → hold → linear warmdown")
    print(f"  optimizer   : Muon lr={args.muon_lr} wd={args.muon_wd}"
          f"  |  AdamW lr={args.adam_lr} wd={args.adam_wd}")

    config = GPTConfig(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        model_dim=args.model_dim,
        max_seq_len=args.seq_len,
    )
    print(f"\n{config}")

    vram_before = gpu_memory_gb()
    model = GPT(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_no_embed = sum(
        p.numel() for name, p in model.named_parameters()
        if "wte" not in name and "lm_head" not in name
    )
    print(f"  params    : {num_params:,}  ({num_params/1e6:.1f}M total, "
          f"{num_params_no_embed/1e6:.1f}M non-embedding)")
    print(f"  model VRAM: {gpu_memory_gb() - vram_before:.2f} GB")

    if args.compile:
        print("  compiling : torch.compile(dynamic=True) ...")
        model = torch.compile(model, dynamic=True)

    muon_optimizer = Muon(
        model.blocks.parameters(),
        lr=args.muon_lr,
        weight_decay=args.muon_wd,
    )
    adam_optimizer = torch.optim.AdamW(
        model.wte.parameters(),
        lr=args.adam_lr,
        betas=(0.9, 0.95),
        weight_decay=args.adam_wd,
        fused=True,
    )

    def lr_schedule(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        elif step < args.total_steps - args.cooldown_steps:
            return 1.0
        else:
            decay = (args.total_steps - step) / args.cooldown_steps
            return decay * (1 - 0.15) + 0.15

    optimizers = [muon_optimizer, adam_optimizer]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule) for opt in optimizers]

    train_loader = DataLoader(args.train_data_dir, args.micro_batch, args.seq_len)
    val_loader = DataLoader(args.val_data_dir, args.micro_batch, args.seq_len)

    num_shards = len(train_loader.shards)
    num_tokens = sum(
        int(torch.from_file(s, False, 256, dtype=torch.int32)[2])
        for s in train_loader.shards
    )
    print(f"  train data : {num_tokens:,} tokens across {num_shards} shards")

    run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("  wandb not installed")
        else:
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"nanogpt-{args.total_steps}s",
                config={**vars(args), "num_params": num_params, "grad_accum": grad_accum_steps},
            )
            print(f"  wandb : {run.url}")

    step_width = len(str(args.total_steps))

    model.train()
    tokens_seen = 0
    best_val_loss = float("inf")
    throughput_window = []  
    train_start_time = None

    for step in range(args.total_steps):
        step_start = time.time()
        muon_optimizer.zero_grad(set_to_none=True)
        adam_optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.zeros(1, device=device)
        for _ in range(grad_accum_steps):
            x, y, cu_seqlens, pos, _ = train_loader.next_batch()
            x = x.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
            y = y.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
            cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            pos = pos.to(device, dtype=torch.long, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, cu_seqlens, pos, args.seq_len)

            loss_accum += loss.detach()
            loss.div_(grad_accum_steps).backward()

        grad_norm = compute_grad_norm(model)
        muon_optimizer.step()
        adam_optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

        torch.cuda.synchronize()
        train_loss = loss_accum.item() / grad_accum_steps
        step_time = time.time() - step_start
        tokens_seen += args.total_batch_size
        tokens_per_sec = args.total_batch_size / step_time

        if step == 0:
            train_start_time = time.time()
        else:
            throughput_window = (throughput_window + [tokens_per_sec])[-20:]

        current_mfu = (
            estimate_mfu(
                num_params_no_embed,
                sum(throughput_window) / len(throughput_window),
                args.seq_len, config.num_layers, config.model_dim,
            )
            if throughput_window else 0.0
        )

        if step % args.log_interval == 0:
            print(
                f"  [step {step:>{step_width}}/{args.total_steps}]"
                f"  loss={train_loss:.4f}"
                f"  | {tokens_per_sec:>9,.0f} tok/s"
                f"  | step_ms={step_time*1000:.0f}"
                f"  | MFU={current_mfu:.1%}"
                f"  | mem={gpu_memory_gb():.1f}GB"
                f"  | gnorm={grad_norm:.3f}"
            )

        if run is not None:
            run.log({
                "train/loss": train_loss,
                "train/gnorm": grad_norm,
                "train/lr_muon": muon_optimizer.param_groups[0]["lr"],
                "train/lr_adam": adam_optimizer.param_groups[0]["lr"],
                "perf/tok_per_sec": tokens_per_sec,
                "perf/mfu": current_mfu,
                "perf/step_ms": step_time * 1000,
                "perf/mem_reserved_gb": gpu_memory_gb(),
                "tokens_seen": tokens_seen,
            }, step=step)

        if step > 0 and step % args.val_interval == 0:
            val_loss = evaluate(model, val_loader, device,
                                batch_size=args.micro_batch, seq_len=args.seq_len)
            best_val_loss = min(best_val_loss, val_loss)
            avg_step_ms = (
                args.total_batch_size / (sum(throughput_window) / len(throughput_window)) * 1000
                if throughput_window else 0.0
            )
            print(
                f"  [val  {step:>{step_width}}/{args.total_steps}]"
                f"  val_loss={val_loss:.4f}"
                f"  | best={best_val_loss:.4f}"
                f"  | avg_ms={avg_step_ms:.0f}"
            )
            if run is not None:
                run.log({"val/loss": val_loss, "val/best": best_val_loss}, step=step)

            if val_loss <= args.target_loss:
                print(f"\n  TARGET REACHED  val_loss={val_loss:.4f}"
                      f"  time={format_time(time.time() - train_start_time)}")
                break

            model.train()

    total_time = time.time() - train_start_time
    avg_tps = tokens_seen / total_time
    avg_mfu = estimate_mfu(num_params_no_embed, avg_tps,
                           args.seq_len, config.num_layers, config.model_dim)
    print()
    print(f"  done")
    print(f"  best val_loss : {best_val_loss:.4f}")
    print(f"  time          : {format_time(total_time)} ({total_time/60:.1f} min)")
    print(f"  tokens seen   : {tokens_seen:,}")
    print(f"  avg tok/s     : {avg_tps:,.0f}")
    print(f"  avg MFU       : {avg_mfu:.1%}")

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "final.pt")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": raw_model.state_dict(),
        "config": config.__dict__,
        "val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
        "total_steps": args.total_steps,
    }, checkpoint_path)
    print(f"\n  saved: {checkpoint_path}")

    if run is not None:
        run.summary.update({
            "final_val_loss": best_val_loss,
            "total_time_min": total_time / 60,
            "avg_mfu": avg_mfu,
        })
        run.finish()

    return best_val_loss


# -----------------------------------------------------------------------------
# cli

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NanoGPT(124M) training on fineweb10B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("model")
    g.add_argument("--vocab_size", type=int, default=50304)
    g.add_argument("--num_layers", type=int, default=12)
    g.add_argument("--num_heads", type=int, default=12)
    g.add_argument("--model_dim", type=int, default=768)

    g = p.add_argument_group("data")
    g.add_argument("--train_data_dir", default="cache/fineweb-10B/fineweb_train_*.bin",
                   help="glob pattern for train shards")
    g.add_argument("--val_data_dir", default="cache/fineweb-10B/fineweb_val_*.bin",
                   help="glob pattern for val shard")
    g.add_argument("--save_dir", default="cache/checkpoints")

    g = p.add_argument_group("batch")
    g.add_argument("--micro_batch", type=int, default=32)
    g.add_argument("--seq_len", type=int, default=2048)
    g.add_argument("--total_batch_size", type=int, default=524288,
                   help="tokens/step = micro_batch × seq_len × grad_accum")

    g = p.add_argument_group("training")
    g.add_argument("--total_steps", type=int, default=1500)
    g.add_argument("--warmup_steps", type=int, default=100)
    g.add_argument("--cooldown_steps", type=int, default=900)
    g.add_argument("--target_loss", type=float, default=3.28)
    g.add_argument("--compile", action="store_true",
                   help="torch.compile(dynamic=True)")

    g = p.add_argument_group("optimizer")
    g.add_argument("--muon_lr", type=float, default=0.02)
    g.add_argument("--adam_lr", type=float, default=0.006)
    g.add_argument("--muon_wd", type=float, default=0.01)
    g.add_argument("--adam_wd", type=float, default=0.1)

    g = p.add_argument_group("logging")
    g.add_argument("--log_interval", type=int, default=10)
    g.add_argument("--val_interval", type=int, default=50)
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb_project", default="nanoGPT")
    g.add_argument("--wandb_name", default=None)

    return p


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()