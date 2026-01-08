import numpy as np
import torch
import os

def load_tokens_bin(filename, dtype=np.int32, offset_bytes=0):
    npt = np.memmap(
        filename,
        dtype=dtype,
        mode="r",
        offset=offset_bytes
    )
    return torch.from_numpy(npt).long()

class TokenDataLoader:
    
    def __init__(self, data_root, B, T):
        self.B = B
        self.T = T

        shards = [
            os.path.join(data_root, f)
            for f in sorted(os.listdir(data_root))
        ]
        assert len(shards) > 0, f"No shards found"

        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens_bin(self.shards[self.current_shard])
        self.current_position = 0

    def _load_next_shard(self):
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens_bin(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1

        if self.current_position + needed > len(self.tokens):
            self._load_next_shard()

        buf = self.tokens[self.current_position : self.current_position + needed]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T
        return x, y


