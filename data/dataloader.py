import numpy as np
import torch
import os
from pathlib import Path

def load_tokens(filename):
    file_path = Path(filename)
    header = torch.from_file(str(file_path), False, 256, dtype=torch.int32)
    num_tokens = int(header[2])
    
    tokens = torch.empty(num_tokens, dtype=torch.int16, pin_memory=True)
    
    with file_path.open("rb", buffering=0) as f:
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    
    return tokens

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
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def _load_next_shard(self):
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
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


