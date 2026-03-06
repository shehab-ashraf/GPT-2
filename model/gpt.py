from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    embd_size: int = 768
    num_heads: int = 12


# -----------------------------------------------------------------------------
# Rotary Embeddings
class Rotary(nn.Module):

    def __init__(self, head_dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# -----------------------------------------------------------------------------
# Normalization
def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# Attention
class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.embd_size % config.num_heads == 0
        self.n_head = config.num_heads
        self.n_embed = config.embd_size
        self.head_size = config.embd_size // config.num_heads
        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size)
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


# -----------------------------------------------------------------------------
# MLP
class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size)
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()          
        x = self.c_proj(x)
        return x


# -----------------------------------------------------------------------------
# Transformer Block
class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / (2 * config.num_layers) ** 0.5

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn_scale * self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# Full GPT Model
class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embd_size),
            h=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
        ))

        self.lm_head = nn.Linear(config.embd_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.rotary = Rotary(
            head_dim=config.embd_size // config.num_heads,
            max_seq_len=config.context_length
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor, return_logits: bool = False):

        B, T = idx.size()
        assert T <= self.config.context_length

        x = self.transformer.wte(idx)
        cos, sin = self.rotary(T, x.device)

        for block in self.transformer.h:
            x = block(x, cos, sin)

        x = norm(x)
        logits = self.lm_head(x)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_logits:
            return logits, loss
        return None, loss