"""Train the 124M NanoGPT model on packed FineWeb-10B shards."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
from dataclasses import asdict

import torch
import torch._dynamo as dynamo
import torch.distributed as dist

from src.checkpoint_manager import load_checkpoint, save_checkpoint
from src.data.loader import FineWebLoader
from src.model import GPT, GPTConfig
from src.optim import Muon
from src.utils import (
    A100_PEAK_BF16,
    DummyWandb,
    compute_cleanup,
    compute_init,
    fmt_time,
    get_mfu,
    get_wandb,
    grad_norm,
    mem_gb,
    print0,
    print_system_info,
    seed,
)

# -----------------------------------------------------------------------------
# 1. CLI

VAL_TOKENS = 10_485_760

parser = argparse.ArgumentParser(description="Pretrain GPT-2 124M on FineWeb-10B")

# Logging & Data
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables)")
parser.add_argument("--wandb-project", default="nanogpt")
parser.add_argument("--train-data", default="cache/fineweb-10B/fineweb_train_*.bin")
parser.add_argument("--val-data", default="cache/fineweb-10B/fineweb_val_*.bin")
parser.add_argument("--save-dir", default="checkpoints")
parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")

# Runtime
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--total-batch-size", type=int, default=524288, help="global tokens per step")
parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--compile-mode",
    default="auto",
    choices=["auto", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
)
parser.add_argument("--bucket-cap-mb", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)

# Training horizon
parser.add_argument("--num-iterations", type=int, default=3000)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--cooldown-steps", type=int, default=900)
parser.add_argument("--final-lr-frac", type=float, default=0.0)

# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.0036, help="AdamW lr")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon lr")
parser.add_argument("--muon-momentum", type=float, default=0.95)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.95)
parser.add_argument("--adam-wd", type=float, default=0.1)

# Eval and checkpoints
parser.add_argument("--eval-tokens", type=int, default=VAL_TOKENS)
parser.add_argument("--eval-every", type=int, default=250, help="-1 disables eval")
parser.add_argument("--log-interval", type=int, default=1)
parser.add_argument("--save-every", type=int, default=-1, help="-1 saves best/final only")

args = parser.parse_args()
user_config = vars(args).copy()
COMPUTE_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

# -----------------------------------------------------------------------------
# 2. Distributed setup & logging

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0

if args.compile:
    dynamo.config.recompile_limit = 64

seed(args.seed)
torch.set_float32_matmul_precision("high")
print_system_info(ddp_world_size)

wandb_run = get_wandb(args.run != "dummy", master_process)
if not isinstance(wandb_run, DummyWandb):
    wandb_run = wandb_run.init(
        project=args.wandb_project,
        name=args.run,
        config=user_config,
        tags=["nanogpt", "gpt2-124m", "muon", "fineweb"],
    )
    print0(f"wandb: {wandb_run.url}")

# -----------------------------------------------------------------------------
# 3. Model

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens == 0, (
    "total_batch_size must be a multiple of world_tokens_per_fwdbwd"
)
grad_accum_steps = args.total_batch_size // world_tokens

model_config = GPTConfig(
    vocab_size=50304,
    num_layers=12,
    num_heads=12,
    model_dim=768,
    max_seq_len=args.max_seq_len,
)

vram_before = mem_gb()
orig_model = GPT(model_config).to(device)
num_params = sum(p.numel() for p in orig_model.parameters())
num_params_no_emb = sum(
    p.numel() for n, p in orig_model.named_parameters() if "wte" not in n and "lm_head" not in n
)

print0(
    f"model  : {model_config.num_layers}L/{model_config.num_heads}H d={model_config.model_dim} | "
    f"vocab {model_config.vocab_size} | seq {model_config.max_seq_len} | "
    f"{num_params / 1e6:.1f}M params ({num_params_no_emb / 1e6:.1f}M non-embed) | "
    f"{mem_gb() - vram_before:.2f} GB"
)
print0(
    f"batch  : {args.device_batch_size}x{args.max_seq_len} = {tokens_per_fwdbwd:,} tok/rank | "
    f"{args.total_batch_size:,} tok total (accum {grad_accum_steps}) | "
    f"world {ddp_world_size}{' (DDP)' if ddp else ''}"
)
print0(
    f"optim  : muon lr={args.matrix_lr} wd={args.weight_decay} | "
    f"adamw lr={args.embedding_lr} wd={args.adam_wd} | "
    f"steps {args.num_iterations} (warmup {args.warmup_steps}, cooldown {args.cooldown_steps})"
)

if not isinstance(wandb_run, DummyWandb):
    wandb_run.config.update(
        {"n_params": num_params, "grad_accum": grad_accum_steps, "world_size": ddp_world_size},
        allow_val_change=True,
    )

# Resume if requested
checkpoint = None
if args.resume:
    checkpoint = load_checkpoint(args.save_dir, "last", device)
    if checkpoint is not None:
        if checkpoint.get("model_config") != asdict(model_config):
            raise ValueError("Checkpoint model config mismatch. Use the same --max-seq-len.")
        orig_model.load_state_dict(checkpoint.pop("model"), strict=True)
        print0(f"resuming from step {int(checkpoint['step'])}")
    else:
        print0("no checkpoint found, starting from scratch")

# -----------------------------------------------------------------------------
# 4. Compile & DDP

model = orig_model
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp_local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        broadcast_buffers=False,
        bucket_cap_mb=args.bucket_cap_mb,
    )

if args.compile:
    print0(f"compiling model (mode={args.compile_mode}) ...")
    mode = args.compile_mode if args.compile_mode != "auto" else "default"
    model = torch.compile(model, mode=mode)

# -----------------------------------------------------------------------------
# 5. Optimizer & LR schedule

muon = Muon(
    orig_model.blocks.parameters(),
    lr=args.matrix_lr,
    momentum=args.muon_momentum,
    weight_decay=args.weight_decay,
)
adam = torch.optim.AdamW(
    orig_model.wte.parameters(),
    lr=args.embedding_lr,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_wd,
    fused=True,
)
optimizers = [muon, adam]


def lr_mult(step: int) -> float:
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    if step < args.num_iterations - args.cooldown_steps:
        return 1.0
    progress = (args.num_iterations - step) / args.cooldown_steps
    return progress * (1 - args.final_lr_frac) + args.final_lr_frac


schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_mult) for opt in optimizers]

if checkpoint is not None:
    for opt, state in zip(optimizers, checkpoint.get("optimizers", [])):
        opt.load_state_dict(state)
    for sch, state in zip(schedulers, checkpoint.get("schedulers", [])):
        sch.load_state_dict(state)

# -----------------------------------------------------------------------------
# 6. Data loaders

train_loader = FineWebLoader(
    args.train_data,
    args.device_batch_size,
    args.max_seq_len,
    rank=ddp_rank,
    world_size=ddp_world_size,
)
val_loader = FineWebLoader(
    args.val_data,
    args.device_batch_size,
    args.max_seq_len,
    rank=ddp_rank,
    world_size=ddp_world_size,
)

if checkpoint is not None and "train_loader" in checkpoint:
    train_loader.load_state_dict(checkpoint["train_loader"])

if master_process:
    n_tokens = sum(int(torch.from_file(p, False, 256, dtype=torch.int32)[2]) for p in train_loader.shards)
    print0(f"data   : {n_tokens:,} tokens across {len(train_loader.shards)} shards\n")

# -----------------------------------------------------------------------------
# 7. Evaluation


@torch.no_grad()
def evaluate():
    eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
    model.eval()
    val_loader.reset()

    losses = []
    for _ in range(eval_steps):
        batch = val_loader.next_batch()
        if batch is None:
            break
        x, y, cu_seqlens, pos, max_seqlen = batch

        x = x.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        y = y.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        cu_seqlens = cu_seqlens.to(device, non_blocking=True)
        pos = pos.to(device, dtype=torch.long, non_blocking=True)

        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            _, loss = model(x, y, cu_seqlens, pos, max_seqlen)
        losses.append(loss.detach())

    model.train()
    if not losses:
        return float("inf")

    val_loss = torch.stack(losses).mean()
    if ddp:
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    return float(val_loss)


# -----------------------------------------------------------------------------
# 8. Training state

start_step = int(checkpoint["step"]) + 1 if checkpoint else 0
tokens_seen = int(checkpoint.get("tokens_seen", 0)) if checkpoint else 0
best_val = float(checkpoint.get("best_val") or "inf") if checkpoint else float("inf")
best_step = checkpoint.get("best_step") if checkpoint else None
last_val_loss = checkpoint.get("validation_loss") if checkpoint else None

model.train()
t_train = time.time() if args.resume else None
t_step = time.time()
tok_window = []
W = len(str(args.num_iterations))

# -----------------------------------------------------------------------------
# 9. Checkpoint helper


def save_ckpt(step: int, val_loss: float | None, is_best: bool):
    if not master_process:
        return
    state = {
        "model": orig_model.state_dict(),
        "optimizers": [o.state_dict() for o in optimizers],
        "schedulers": [s.state_dict() for s in schedulers],
        "version": 1,
        "step": step,
        "model_config": asdict(model_config),
        "user_config": user_config,
        "tokens_seen": tokens_seen,
        "best_val": best_val if best_val != float("inf") else None,
        "best_step": best_step,
        "validation_loss": val_loss,
        "is_best": is_best,
        "world_size": ddp_world_size,
        "train_loader": train_loader.state_dict(),
    }
    save_checkpoint(args.save_dir, "last", state)
    if is_best:
        save_checkpoint(args.save_dir, "best", state)
    print0(f"  [save] {'best' if is_best else 'last'} checkpoint at step {step}")


# -----------------------------------------------------------------------------
# 10. Main loop

for step in range(start_step, args.num_iterations):
    # ---- forward / backward / accumulate ----
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss_accum = torch.zeros(1, device=device)

    for _ in range(grad_accum_steps):
        x, y, cu_seqlens, pos, _ = train_loader.next_batch()
        x = x.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        y = y.to(device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        cu_seqlens = cu_seqlens.to(device, non_blocking=True)
        pos = pos.to(device, dtype=torch.long, non_blocking=True)

        with torch.autocast("cuda", dtype=COMPUTE_DTYPE):
            _, loss = model(x, y, cu_seqlens, pos, args.max_seq_len)

        loss_accum += loss.detach()
        (loss / grad_accum_steps).backward()

    # ---- optimizer step ----
    for opt in optimizers:
        opt.step()
    for sch in schedulers:
        sch.step()
    tokens_seen += args.total_batch_size

    # ---- skip timing on step 0 to pay compile cost ----
    if step == 0 and not args.resume:
        if ddp:
            dist.barrier()
        torch.cuda.synchronize()
        t_train = time.time()
        t_step = time.time()
        continue

    # ---- logging ----
    if args.log_interval > 0 and step % args.log_interval == 0:
        torch.cuda.synchronize()
        train_loss = loss_accum.item() / grad_accum_steps
        gnorm = grad_norm(orig_model)
        dt = time.time() - t_step
        tps = args.total_batch_size * args.log_interval / dt
        tok_window = (tok_window + [tps])[-20:]

        cur_mfu = get_mfu(
            num_params_no_emb,
            sum(tok_window) / len(tok_window),
            args.max_seq_len,
            model_config.num_layers,
            model_config.model_dim,
        )
        pct = 100 * step / args.num_iterations
        eta = (args.num_iterations - step) * dt / args.log_interval

        print0(
            f"step {step:>{W}}/{args.num_iterations} ({pct:.1f}%) | "
            f"loss={train_loss:.4f} | "
            f"dt={dt * 1000 / args.log_interval:.0f}ms | "
            f"tok/s={tps:,.0f} | "
            f"mfu={cur_mfu:.1%} | "
            f"gnorm={gnorm:.3f} | "
            f"mem={mem_gb():.1f}GB | "
            f"eta={fmt_time(eta)}"
        )

        if not isinstance(wandb_run, DummyWandb):
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "train/gnorm": gnorm,
                    "train/lr_muon": optimizers[0].param_groups[0]["lr"],
                    "train/lr_adam": optimizers[1].param_groups[0]["lr"],
                    "perf/tok_per_sec": tps,
                    "perf/mfu": cur_mfu,
                    "perf/step_ms": dt * 1000 / args.log_interval,
                    "perf/mem_reserved_gb": mem_gb(),
                    "tokens_seen": tokens_seen,
                    "world_size": ddp_world_size,
                },
                step=step,
            )
        t_step = time.time()

    # ---- evaluation ----
    saved_at_step = False
    if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
        val_loss = evaluate()
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        last_val_loss = val_loss
        if is_best:
            best_step = step

        avg_ms = (
            (args.total_batch_size / (sum(tok_window) / len(tok_window)) * 1000)
            if tok_window
            else 0.0
        )
        print0(
            f"val  {step:>{W}}/{args.num_iterations} | "
            f"loss={val_loss:.4f} | best={best_val:.4f} | avg_ms={avg_ms:.0f}"
        )

        if is_best:
            save_ckpt(step, val_loss, True)
            saved_at_step = True

        if ddp:
            dist.barrier()
        if not isinstance(wandb_run, DummyWandb):
            wandb_run.log({"val/loss": val_loss, "val/best": best_val}, step=step)
        model.train()

    # ---- periodic checkpoint ----
    if args.save_every > 0 and step > 0 and step % args.save_every == 0 and not saved_at_step:
        save_ckpt(step, last_val_loss, False)
        if ddp:
            dist.barrier()

# -----------------------------------------------------------------------------
# 11. Finalize

if t_train is None:
    t_train = time.time()
total_time = time.time() - t_train
avg_tps = tokens_seen / total_time
avg_mfu = get_mfu(
    num_params_no_emb, avg_tps, args.max_seq_len, model_config.num_layers, model_config.model_dim
)

print0(f"\ndone")
print0(f"  best val_loss : {best_val:.4f}")
print0(f"  time          : {fmt_time(total_time)} ({total_time / 60:.1f} min)")
print0(f"  tokens seen   : {tokens_seen:,}")
print0(f"  avg tok/s     : {avg_tps:,.0f}")
print0(f"  avg MFU       : {avg_mfu:.1%} (peak {A100_PEAK_BF16 / 1e12:.0f} TFLOPS)")

save_ckpt(step, last_val_loss, step == best_step)
print0(f"saved final checkpoint at step {step} in {args.save_dir}")

if not isinstance(wandb_run, DummyWandb):
    wandb_run.summary.update(
        {
            "final_val_loss": best_val,
            "total_time_min": total_time / 60,
            "avg_mfu": avg_mfu,
            "world_size": ddp_world_size,
        }
    )
    wandb_run.finish()

compute_cleanup(ddp)
