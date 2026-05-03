from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


# ---------------------------------------------------------------------------
# config

@dataclass
class GPTConfig:
    vocab_size:    int   = 50304
    num_layers:    int   = 12
    num_heads:     int   = 12
    model_dim:     int   = 768
    max_seq_len:   int   = 2048
    logit_softcap: float = 30.0


# ---------------------------------------------------------------------------
# rotary position embeddings

class Rotary(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t        = torch.arange(max_seq_len).float()
        freqs    = torch.outer(t, inv_freq)
        # bfloat16: avoids dtype casts in the training hot path under autocast
        self.register_buffer("cos", freqs.cos().bfloat16(), persistent=False)
        self.register_buffer("sin", freqs.sin().bfloat16(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # pos: (T,) — always 1D, works for both training (packed) and inference
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        c = self.cos[pos].unsqueeze(0).unsqueeze(2)  # (1, T, 1, H//2)
        s = self.sin[pos].unsqueeze(0).unsqueeze(2)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ---------------------------------------------------------------------------
# normalization

def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# causal self-attention

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.model_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim  = config.model_dim // config.num_heads

        self.c_attn = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.c_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.rotary = Rotary(self.head_dim, max_seq_len=config.max_seq_len)

    def forward(self, x: torch.Tensor, cu_seqlens=None, pos=None, max_seqlen=None) -> torch.Tensor:
        B, T, C = x.shape
        N, H    = self.num_heads, self.head_dim

        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, N, H)
        k = k.view(B, T, N, H)
        v = v.view(B, T, N, H)

        q = F.rms_norm(q, (H,))
        k = F.rms_norm(k, (H,))

        q = self.rotary(q, pos)
        k = self.rotary(k, pos)

        if cu_seqlens is not None:
            # training: flash_attn varlen enforces per-document causal masking
            q = q.reshape(B * T, N, H)
            k = k.reshape(B * T, N, H)
            v = v.reshape(B * T, N, H)
            y = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=0.0, causal=True,
            )
            return self.c_proj(y.view(B, T, C))

        # inference: standard SDPA (works on any backend, no flash_attn needed)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# feed-forward network

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.model_dim, 4 * config.model_dim, bias=False)
        self.c_proj = nn.Linear(4 * config.model_dim, config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


# ---------------------------------------------------------------------------
# transformer block

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn       = CausalSelfAttention(config)
        self.mlp        = MLP(config)
        self.attn_scale = 1.0 / math.sqrt(2.0 * config.num_layers)

    def forward(self, x, cu_seqlens=None, pos=None, max_seqlen=None):
        x = x + self.attn_scale * self.attn(norm(x), cu_seqlens, pos, max_seqlen)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# GPT model

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.num_layers % 2 == 0, "num_layers must be even (U-Net skip connections)"
        self.config = config
        self.half_n = config.num_layers // 2

        self.wte     = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, std=residual_std)

    def forward(self, idx, targets=None, cu_seqlens=None, pos=None, max_seqlen=None):
        B, T = idx.shape

        # inference: auto-generate positions when not provided by the dataloader
        if pos is None:
            pos = torch.arange(T, device=idx.device)

        x = self.wte(idx)

        # U-Net skip connections: first half saves, second half adds back
        skips = []
        for i, block in enumerate(self.blocks):
            if i < self.half_n:
                skips.append(x)
            else:
                x = x + skips.pop()
            x = block(x, cu_seqlens, pos, max_seqlen)

        x      = norm(x)
        logits = self.lm_head(x)
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return None, loss
        return logits, None