from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import torch

EOT = 50256


# -----------------------------------------------------------------------------
# load shard

def _load_shard(path: str) -> torch.Tensor:
    filepath = Path(path)
    header = torch.from_file(str(filepath), shared=False, size=256, dtype=torch.int32)
    assert header[0] == 20240520, f"bad magic in {path}"
    assert header[1] == 1, f"unsupported version in {path}"

    num_tokens = int(header[2])
    tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
    with filepath.open("rb", buffering=0) as f:
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    return tokens


# -----------------------------------------------------------------------------
# dataloader

class DataLoader:
    def __init__(self, file_glob: str, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shards = sorted(glob(file_glob))
        assert self.shards, f"no shards found for glob: {file_glob!r}"
        self.reset()

    def reset(self):
        self._shard_idx = 0
        self._tokens = _load_shard(self.shards[0])
        self._eot_pos = (self._tokens == EOT).nonzero(as_tuple=True)[0]
        self._doc_idx = 0

    def next_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        needed = self.batch_size * self.seq_len + 1

        starts, ends, collected = [], [], 0
        while collected < needed:
            doc = self._next_doc()
            if doc is None:
                return None
            start, end = doc
            end = min(end, start + needed - collected)
            starts.append(start)
            ends.append(end)
            collected += end - start

        T = self.batch_size * self.seq_len
        buf = torch.cat([self._tokens[s:e] for s, e in zip(starts, ends)])
        x = buf[:T].clone()
        y = buf[1:T + 1].clone()

        lengths = torch.tensor([e - s for s, e in zip(starts, ends)], dtype=torch.int32)
        lengths[-1] -= 1

        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)

        position_ids = (
            torch.arange(T, dtype=torch.int32)
            - torch.repeat_interleave(cu_seqlens[:-1], lengths)
        )

        return x, y, cu_seqlens, position_ids, int(lengths.max())

    def _next_doc(self) -> Optional[Tuple[int, int]]:
        if self._doc_idx >= len(self._eot_pos):
            self._shard_idx += 1
            if self._shard_idx >= len(self.shards):
                return None
            self._tokens = _load_shard(self.shards[self._shard_idx])
            self._eot_pos = (self._tokens == EOT).nonzero(as_tuple=True)[0]
            self._doc_idx = 0

        start = int(self._eot_pos[self._doc_idx])
        end = (
            int(self._eot_pos[self._doc_idx + 1])
            if self._doc_idx + 1 < len(self._eot_pos)
            else len(self._tokens)
        )
        end = min(end, start + self.seq_len)
        self._doc_idx += 1
        return start, end