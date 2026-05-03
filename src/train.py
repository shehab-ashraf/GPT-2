"""
train.py — NanoGPT training script.

Target: val_loss ≤ 3.28 on FineWeb-10B in ~70 min on a single A100 80GB.

Usage:
    bash scripts/run.sh
    python -m src.train --help
"""

import os
import sys
import time
import subprocess
import argparse

import torch

from src.model.gpt  import GPT, GPTConfig
from src.model.muon import Muon
from src.data.dataloader import DataLoader

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ---------------------------------------------------------------------------
# system info

def print_system_info() -> None:
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")
    print()
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout.rstrip())
    except FileNotFoundError:
        print("nvidia-smi not found — running on CPU?")
    print()


# ---------------------------------------------------------------------------
# metrics

def mem_gb() -> float:
    return torch.cuda.memory_reserved() / 1e9


def compute_mfu(
    n_params_no_emb: int,
    tok_per_sec:     float,
    seq_len:         int,
    num_layers:      int,
    model_dim:       int,
    peak_flops:      float = 312e12,  # A100 SXM4 bf16 peak
) -> float:
    # PaLM Appendix B: 6N (fwd+bwd matmuls) + 4·L·T·d (attention)
    flops_per_tok = 6 * n_params_no_emb + 4 * num_layers * seq_len * model_dim
    return flops_per_tok * tok_per_sec / peak_flops


def grad_norm(model: torch.nn.Module) -> float:
    total = sum(
        p.grad.detach().float().norm() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
    return total ** 0.5


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# evaluation
#
# VAL_TOKENS matches the speedrun target: val_loss ≤ 3.28 is defined as
# P(first 10,485,760 val tokens) ≥ exp(-3.28 × 10,485,760).

VAL_TOKENS = 10_485_760


@torch.no_grad()
def evaluate(model, val_loader, device, B: int, T: int) -> float:
    val_steps = VAL_TOKENS // (B * T)
    model.eval()
    val_loader.reset()

    losses = []
    for _ in range(val_steps):
        batch = val_loader.next_batch()
        if batch is None:
            break
        x, y, cu_seqlens, pos, max_seqlen = batch
        x          = x.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        y          = y.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        cu_seqlens = cu_seqlens.to(device, non_blocking=True)
        pos        = pos.to(device, dtype=torch.long, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, cu_seqlens, pos, max_seqlen)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ---------------------------------------------------------------------------
# training

def train(args) -> float:
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    grad_accum = args.total_batch_size // (args.micro_batch * args.seq_len)
    assert args.total_batch_size % (args.micro_batch * args.seq_len) == 0, (
        f"total_batch_size {args.total_batch_size} must be divisible by "
        f"micro_batch × seq_len = {args.micro_batch * args.seq_len}"
    )

    # startup
    print_system_info()
    print("\n")
    print("Training Configuration")
    print("---------------------------------------------------------")
    print(f"Micro batch size : {args.micro_batch} x {args.seq_len} = {args.micro_batch * args.seq_len:,} tokens")
    print(f"Total batch size : {args.total_batch_size:,} tokens (Grad accum: {grad_accum})")
    print(f"Total steps      : {args.total_steps}")
    print(f"Warmup steps     : {args.warmup_steps}")
    print(f"Cooldown steps   : {args.cooldown_steps}")
    print(f"Muon Optimizer   : lr={args.muon_lr}, weight_decay={args.muon_wd}")
    print(f"AdamW Optimizer  : lr={args.adam_lr}, weight_decay={args.adam_wd}")
    print("\n")

    # model
    config = GPTConfig(
        vocab_size=50304,
        num_layers=12,
        num_heads=12,
        model_dim=768,
        max_seq_len=args.seq_len,
    )
    vram_before = mem_gb()
    model = GPT(config).to(device)

    n_params        = sum(p.numel() for p in model.parameters())
    n_params_no_emb = sum(p.numel() for n, p in model.named_parameters()
                          if "wte" not in n and "lm_head" not in n)

    print("Model Configuration")
    print("---------------------------------------------------------")
    print(f"Vocab size       : {config.vocab_size}")
    print(f"Num layers       : {config.num_layers}")
    print(f"Num heads        : {config.num_heads}")
    print(f"Model dim        : {config.model_dim}")
    print(f"Max seq len      : {config.max_seq_len}")
    print(f"Total params     : {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Non-embed params : {n_params_no_emb:,} ({n_params_no_emb/1e6:.1f}M)")
    print(f"Model VRAM       : {mem_gb() - vram_before:.2f} GB")
    print("\n")

    if args.compile:
        print("Compiling model ...")
        model = torch.compile(model, dynamic=True)

    # optimizer: Muon for transformer blocks, AdamW for embeddings
    muon = Muon(model.blocks.parameters(), lr=args.muon_lr, weight_decay=args.muon_wd)
    adam = torch.optim.AdamW(
        model.wte.parameters(), lr=args.adam_lr,
        betas=(0.9, 0.95), weight_decay=args.adam_wd,
        fused=True,
    )

    # trapezoidal LR: warmup → hold at peak → linear warmdown to 15%
    def get_lr(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        elif step < args.total_steps - args.cooldown_steps:
            return 1.0
        else:
            decay = (args.total_steps - step) / args.cooldown_steps
            return decay * (1 - 0.15) + 0.15

    optimizers = [muon, adam]
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # data
    train_loader = DataLoader(args.train_data, args.micro_batch, args.seq_len)
    val_loader   = DataLoader(args.val_data,   args.micro_batch, args.seq_len)

    n_shards = len(train_loader.shards)
    n_tokens = sum(int(torch.from_file(s, False, 256, dtype=torch.int32)[2])
                   for s in train_loader.shards)
    print(f"Training data    : {n_tokens:,} tokens across {n_shards} shards")
    print("\n")

    # W&B
    run = None
    if args.wandb:
        if not _WANDB:
            print("wandb not installed.")
        else:
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"nanogpt-{args.seq_len}t-{args.total_steps}s",
                config={**vars(args), "n_params": n_params, "grad_accum": grad_accum},
                tags=["nanogpt", "gpt2-124m", "muon", "fineweb"],
            )
            print(f"wandb       : {run.url}")

    W = len(str(args.total_steps))

    # training loop
    model.train()
    tokens_seen = 0
    best_val    = float("inf")
    tok_window  = []   # rolling 20-step window (excludes compile step)
    t_train     = None

    for step in range(args.total_steps):
        t_step = time.time()

        muon.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)

        # gradient accumulation — loss.detach() avoids CPU sync inside the loop
        loss_accum = torch.zeros(1, device=device)
        for _ in range(grad_accum):
            x, y, cu_seqlens, pos, _ = train_loader.next_batch()
            x          = x.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
            y          = y.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
            cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            pos        = pos.to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, cu_seqlens, pos, args.seq_len)
            loss_accum += loss.detach()
            loss.div_(grad_accum).backward()

        gnorm = grad_norm(model)
        muon.step()
        adam.step()
        for s in schedulers:
            s.step()
        torch.cuda.synchronize()

        train_loss   = loss_accum.item() / grad_accum
        dt           = time.time() - t_step
        tokens_seen += args.total_batch_size
        tps          = args.total_batch_size / dt

        if step == 0:
            t_train = time.time()  # exclude compile time from ETA
        else:
            tok_window = (tok_window + [tps])[-20:]

        cur_mfu = (compute_mfu(n_params_no_emb, sum(tok_window) / len(tok_window),
                               args.seq_len, config.num_layers, config.model_dim)
                   if tok_window else 0.0)

        if step % args.log_interval == 0:
            print(
                f"  [step {step:>{W}}/{args.total_steps}]"
                f"  loss={train_loss:.4f}"
                f"  |  {tps:>9,.0f} tok/s"
                f"  |  step_ms={dt*1000:.0f}"
                f"  |  MFU={cur_mfu:.1%}"
                f"  |  mem={mem_gb():.1f}GB"
                f"  |  gnorm={gnorm:.3f}"
            )

        if run is not None:
            run.log({
                "train/loss":           train_loss,
                "train/gnorm":          gnorm,
                "train/lr_muon":        muon.param_groups[0]["lr"],
                "train/lr_adam":        adam.param_groups[0]["lr"],
                "perf/tok_per_sec":     tps,
                "perf/mfu":             cur_mfu,
                "perf/step_ms":         dt * 1000,
                "perf/mem_reserved_gb": mem_gb(),
                "tokens_seen":          tokens_seen,
            }, step=step)

        # validation
        if step > 0 and step % args.val_interval == 0:
            val_loss = evaluate(model, val_loader, device,
                                B=args.micro_batch, T=args.seq_len)
            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            avg_ms   = (args.total_batch_size / (sum(tok_window) / len(tok_window)) * 1000
                        if tok_window else 0.0)
            print(
                f"  [val  {step:>{W}}/{args.total_steps}]"
                f"  val_loss={val_loss:.4f}"
                f"  |  best={best_val:.4f}"
                f"  |  avg_ms={avg_ms:.0f}"
            )

            if is_best:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, "nanogpt-best.pt")
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model":        raw.state_dict(),
                    "config":       config.__dict__,
                    "val_loss":     best_val,
                    "tokens_seen":  tokens_seen,
                    "total_steps":  args.total_steps,
                }, ckpt_path)
                print(f"  [save] new best model saved to {ckpt_path}")

            if run is not None:
                run.log({"val/loss": val_loss, "val/best": best_val}, step=step)

            if val_loss <= args.target_loss:
                print(f"\n  TARGET REACHED  val_loss={val_loss:.4f}"
                      f"  time={fmt_time(time.time() - t_train)}")
                break

            model.train()

    # final summary
    total_time = time.time() - t_train
    avg_tps    = tokens_seen / total_time
    avg_mfu    = compute_mfu(n_params_no_emb, avg_tps,
                             args.seq_len, config.num_layers, config.model_dim)

    print()
    print(f"  done")
    print(f"  best val_loss : {best_val:.4f}")
    print(f"  time          : {fmt_time(total_time)}  ({total_time/60:.1f} min)")
    print(f"  tokens seen   : {tokens_seen:,}")
    print(f"  avg tok/s     : {avg_tps:,.0f}")
    print(f"  avg MFU       : {avg_mfu:.1%}")

    # checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "nanogpt-final.pt")
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model":        raw.state_dict(),
        "config":       config.__dict__,
        "val_loss":     best_val,
        "tokens_seen":  tokens_seen,
        "total_steps":  args.total_steps,
    }, ckpt_path)
    print(f"\n  saved final model  →  {ckpt_path}")

    if run is not None:
        run.summary.update({
            "final_val_loss": best_val,
            "total_time_min": total_time / 60,
            "avg_mfu":        avg_mfu,
        })
        run.finish()

    return best_val


# ---------------------------------------------------------------------------
# cli

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NanoGPT — GPT-2 (124M) training on FineWeb-10B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("data")
    g.add_argument("--train_data", default="cache/fineweb-10B/fineweb_train_*.bin",
                   help="glob pattern for train shards")
    g.add_argument("--val_data",   default="cache/fineweb-10B/fineweb_val_*.bin",
                   help="glob pattern for val shard")
    g.add_argument("--save_dir",   default="checkpoints")

    g = p.add_argument_group("batch")
    g.add_argument("--micro_batch",      type=int, default=32)
    g.add_argument("--seq_len",          type=int, default=2048)
    g.add_argument("--total_batch_size", type=int, default=524288,
                   help="tokens/step = micro_batch × seq_len × grad_accum")

    g = p.add_argument_group("training")
    g.add_argument("--total_steps",    type=int,   default=1500)
    g.add_argument("--warmup_steps",   type=int,   default=100)
    g.add_argument("--cooldown_steps", type=int,   default=900)
    g.add_argument("--target_loss",    type=float, default=3.28)
    g.add_argument("--compile",        action="store_true",
                   help="torch.compile(dynamic=True) — recommended")

    g = p.add_argument_group("optimizer")
    g.add_argument("--muon_lr", type=float, default=0.02)
    g.add_argument("--adam_lr", type=float, default=0.006)
    g.add_argument("--muon_wd", type=float, default=0.01)
    g.add_argument("--adam_wd", type=float, default=0.1)

    g = p.add_argument_group("logging")
    g.add_argument("--log_interval",  type=int, default=10)
    g.add_argument("--val_interval",  type=int, default=50)
    g.add_argument("--wandb",         action="store_true")
    g.add_argument("--wandb_project", default="nanoGPT")
    g.add_argument("--wandb_name",    default=None)

    return p


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()