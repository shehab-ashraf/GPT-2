import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn

# -----------------------------------------------------------------------------
# RoPE


class Rotary(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        c = self.cos[pos].unsqueeze(0).unsqueeze(2)
        s = self.sin[pos].unsqueeze(0).unsqueeze(2)
        out = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return out.to(orig_dtype)


# -----------------------------------------------------------------------------
# RMSNorm


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# flash attention, varlen (training: packed documents, per-doc causal masking)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:

    B, T, N, H = q.shape
    q = q.reshape(B * T, N, H)
    k = k.reshape(B * T, N, H)
    v = v.reshape(B * T, N, H)
    y = flash_attn.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    return y.view(B, T, N * H)


# -----------------------------------------------------------------------------
# flash attention with KV cache


def flash_attn_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens,
) -> torch.Tensor:

    B, _, N, H = q.shape
    y = flash_attn.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        softmax_scale=H**-0.5,
        causal=True,
    )
    return y.reshape(B, -1, N * H)
