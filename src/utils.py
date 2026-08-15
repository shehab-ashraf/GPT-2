import os
import sys

import torch

# -----------------------------------------------------------------------------
# Constants

# NVIDIA A100 SXM4 peak bf16 dense FLOPS
A100_PEAK_BF16 = 312e12

# -----------------------------------------------------------------------------
# Printing & seeding

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def seed(s=42):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# -----------------------------------------------------------------------------
# Distributed setup


def compute_init():
    if "RANK" not in os.environ:
        return False, 0, 0, 1, torch.device("cuda")

    import torch.distributed as dist

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.empty(1, device=f"cuda:{local_rank}", requires_grad=True).backward()

    return True, rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def compute_cleanup(is_ddp):
    if is_ddp:
        import torch.distributed as dist

        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# W&B stand-in


class DummyWandb:

    class Summary:
        def update(self, d):
            pass

    def __init__(self):
        self.url = "dummy"
        self.summary = self.Summary()

    def log(self, d, step=None):
        pass

    def finish(self):
        pass


def get_wandb(use_wandb, master):
    if not (use_wandb and master):
        return DummyWandb()
    try:
        import wandb

        return wandb
    except ImportError:
        return DummyWandb()


# -----------------------------------------------------------------------------
# System info


def print_system_info(world_size=1):
    gpu = ""
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        prefix = f"{world_size}x " if world_size > 1 else ""
        gpu = f" | {prefix}{p.name} ({p.total_memory / 1e9:.1f}GB, CC {p.major}.{p.minor})"
    print0(
        f"python {sys.version.split()[0]} | pytorch {torch.__version__} | cuda {torch.version.cuda}{gpu}"
    )


# -----------------------------------------------------------------------------
# Metrics


def get_mfu(
    n_params_no_emb, tok_per_sec, seq_len, num_layers, model_dim, peak_flops=A100_PEAK_BF16
):
    flops_per_tok = 6 * n_params_no_emb + 4 * num_layers * seq_len * model_dim
    return flops_per_tok * tok_per_sec / peak_flops


def grad_norm(model):
    sq = sum((p.grad.detach().float() ** 2).sum() for p in model.parameters() if p.grad is not None)
    return float(sq) ** 0.5


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def mem_gb(kind="reserved"):
    fn = torch.cuda.memory_reserved if kind == "reserved" else torch.cuda.memory_allocated
    return fn() / 1e9
