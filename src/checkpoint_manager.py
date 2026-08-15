import os
import tempfile
from pathlib import Path

import torch


def save_checkpoint(dir, name, state):
    path = Path(dir) / f"ckpt_{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=dir, prefix=".", suffix=".tmp")
    os.close(fd)

    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_checkpoint(dir, name, device):
    path = Path(dir) / f"ckpt_{name}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=True)
