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
        
        # Causal mask: lower triangular matrix to prevent attending to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length))
                                     .view(1, 1, config.context_length, config.context_length))
        
    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension
        
        # Compute query, key, value for all heads in one go
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        # Reshape and transpose to separate heads: (B, T, C) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask to prevent attending to future positions
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
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
        self.ln_1 = nn.LayerNorm(config.embd_size)  # Pre-normalization for attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embd_size)  # Pre-normalization for MLP
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Attention with residual connection (pre-norm style)
        x = x + self.attn(self.ln_1(x))
        # MLP with residual connection (pre-norm style)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """GPT Language Model: token + position embeddings -> transformer blocks -> language modeling head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_size),    # Token embeddings
            wpe = nn.Embedding(config.context_length, config.embd_size),  # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),  # Transformer blocks
            ln_f = nn.LayerNorm(config.embd_size)  # Final layer norm before output
        ))
        # Language modeling head: maps embeddings back to vocabulary logits
        self.lm_head = nn.Linear(config.embd_size, config.vocab_size, bias=False)

