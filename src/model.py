from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from src.kernels import Rotary, flash_attn_func, flash_attn_kvcache, norm


# ---------------------------------------------------------------------------
# config


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    model_dim: int = 768
    max_seq_len: int = 2048
    logit_softcap: float = 30.0


# ---------------------------------------------------------------------------
# causal self-attention


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, rotary: Rotary):
        super().__init__()
        assert config.model_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads

        self.c_attn = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.c_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.rotary = rotary
        self.k_cache = None
        self.v_cache = None

    def forward(
        self, x: torch.Tensor, cu_seqlens=None, pos=None, max_seqlen=None, cache_seqlens=None
    ) -> torch.Tensor:
        B, T, C = x.shape
        N, H = self.num_heads, self.head_dim

        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, N, H)
        k = k.view(B, T, N, H)
        v = v.view(B, T, N, H)

        q = F.rms_norm(q, (H,))
        k = F.rms_norm(k, (H,))

        q = self.rotary(q, pos)
        k = self.rotary(k, pos)

        if cache_seqlens is not None:
            y = flash_attn_kvcache(q, self.k_cache, self.v_cache, k, v, cache_seqlens)
        else:
            y = flash_attn_func(q, k, v, cu_seqlens, max_seqlen)

        return self.c_proj(y)


# ---------------------------------------------------------------------------
# feed-forward network


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.model_dim, 4 * config.model_dim, bias=False)
        self.c_proj = nn.Linear(4 * config.model_dim, config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


# ---------------------------------------------------------------------------
# transformer block


class Block(nn.Module):
    def __init__(self, config: GPTConfig, rotary: Rotary):
        super().__init__()
        self.attn = CausalSelfAttention(config, rotary)
        self.mlp = MLP(config)
        self.attn_scale = 1.0 / math.sqrt(2.0 * config.num_layers)

    def forward(self, x, cu_seqlens=None, pos=None, max_seqlen=None, cache_seqlens=None):
        x_norm = norm(x)
        x = x + self.attn_scale * self.attn(x_norm, cu_seqlens, pos, max_seqlen, cache_seqlens)
        x_norm = norm(x)
        x = x + self.mlp(x_norm)
        return x


# ---------------------------------------------------------------------------
# GPT model


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.num_layers % 2 == 0, "num_layers must be even (U-Net skip connections)"
        self.config = config
        self.half_n = config.num_layers // 2

        self.wte = nn.Embedding(config.vocab_size, config.model_dim)
        self.rotary = Rotary(config.model_dim // config.num_heads, max_seq_len=config.max_seq_len)
        self.blocks = nn.ModuleList([Block(config, self.rotary) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, std=residual_std)

    def allocate_cache(self, batch_size: int, device=None, dtype=torch.bfloat16) -> None:
        if device is None:
            device = next(self.parameters()).device
        N = self.config.num_heads
        H = self.config.model_dim // N
        S = self.config.max_seq_len
        for block in self.blocks:
            block.attn.k_cache = torch.zeros(batch_size, S, N, H, device=device, dtype=dtype)
            block.attn.v_cache = torch.zeros(batch_size, S, N, H, device=device, dtype=dtype)

    def forward(
        self, idx, targets=None, cu_seqlens=None, pos=None, max_seqlen=None, cache_seqlens=None
    ):
        B, T = idx.shape

        if pos is None:
            pos = torch.arange(T, device=idx.device)

        x = self.wte(idx)

        skips = []
        for i, block in enumerate(self.blocks):
            if i < self.half_n:
                skips.append(x)
            else:
                x = x + skips.pop()
            x = block(x, cu_seqlens, pos, max_seqlen, cache_seqlens)

        x = norm(x)
        logits = self.lm_head(x)
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return None, loss
        return logits, None
