from dataclasses import dataclass 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class GPTConfig:
    """Configuration for GPT model architecture."""
    context_length: int = 1024  # Maximum sequence length
    vocab_size: int = 50257     # Size of vocabulary
    num_layers: int = 12        # Number of transformer blocks
    embd_size: int = 768        # Embedding dimension
    num_heads: int = 12         # Number of attention heads

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        # Calculate the inverse frequencies for the rotary embeddings
        # This creates a sequence of frequencies [theta_0, theta_1, ...]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Register as a buffer so it's part of the state_dict but not a parameter
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        # x is assumed to be (Batch, Seq_Len, Dim)
        seq_len = x.shape[1]
        
        # Cache the sine and cosine values if sequence length changes
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            # Create position indices [0, 1, ..., seq_len-1]
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # Compute outer product to get frequencies for each position
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            # Compute cos and sin values
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
            
        # Return shapes aligned for broadcasting with (Batch, Heads, Time, Head_Dim)
        # We need (1, 1, Time, Head_Dim/2) to broadcast correctly
        return self.cos_cached[None, None, :, :], self.sin_cached[None, None, :, :]

def apply_rotary_emb(x, cos, sin):
    # Apply the rotary embeddings to the queries and keys
    # x: (Batch, Heads, Time, Head_Dim)
    # cos, sin: (1, 1, Time, Head_Dim/2)
    
    assert x.ndim == 4 # Ensure we are working with multihead attention format
    d = x.shape[3]//2  # Split the dimension into two halves
    
    x1 = x[..., :d]
    x2 = x[..., d:]
    
    # Apply rotation matrix:
    # [x1, x2] * [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
    # Note: user provided implementation has signs:
    # y1 = x1 * cos + x2 * sin
    # y2 = x1 * (-sin) + x2 * cos
    # This corresponds to a specific rotation direction.
    
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    
    return torch.cat([y1, y2], 3)

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer for autoregressive modeling."""

    def __init__(self, config):
        super().__init__()
        # Ensure embedding dimension is evenly divisible across all attention heads
        assert config.embd_size % config.num_heads == 0, f"embedding dim should be divisible by number of heads"
        # Key, query, value projections for all heads in a single batch operation
        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size)
        # Output projection to bring multi-head results back to embedding dimension
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)
        self.n_head = config.num_heads
        self.n_embed = config.embd_size
        
        # Rotary Embedding setup
        # Head size is embedding dimension divided by number of heads
        self.head_size = config.embd_size // config.num_heads
        self.rotary = Rotary(self.head_size)
    
    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension
        
        # Compute query, key, value for all heads in one go
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        # Reshape and transpose to separate heads: (B, T, C) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Apply Rotary Positional Embeddings
        # Calculate cos and sin for the current sequence length
        cos, sin = self.rotary(x) 
        # Apply RoPE to queries and keys
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Flash Attention: Fused, memory-efficient attention
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )
        
        # Concatenate all heads back together: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """Feed-forward network with GELU activation (4x expansion factor)."""

    def __init__(self, config):
        super().__init__()
        # Expand to 4x embedding dimension
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size)
        self.gelu = nn.GELU()
        # Project back to embedding dimension
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size)
    
    def forward(self, x):
        x = self.c_fc(x)      # Expand
        x = self.gelu(x)       # Non-linear activation
        x = self.c_proj(x)     # Project back
        return x

class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP, with residual connections."""
    
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / (2 * config.num_layers) ** 0.5
    
    def forward(self, x):
        # Attention with residual connection (pre-norm style)
        x = x + self.attn_scale * self.attn(norm(x))
        # MLP with residual connection (pre-norm style)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    """GPT Language Model: token + position embeddings -> transformer blocks -> language modeling head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_size),    # Token embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),  # Transformer blocks
        ))
        # Language modeling head: maps embeddings back to vocabulary logits
        self.lm_head = nn.Linear(config.embd_size, config.vocab_size, bias=False)

        # weight sharing scheme (reduces 768*50257=~40M params, fewer params, more efficient)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets, return_logits=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, block size is only {self.config.context_length}"
        # forward the token embeddings
        x = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_logits:
            return logits, loss
        return None, loss
