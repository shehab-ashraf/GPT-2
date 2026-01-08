from numpy.lib._arraysetops_impl import isin
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
        self.attn_scale = 1 / (2 * config.num_layer) ** 0.5
    
    def forward(self, x):
        # Attention with residual connection (pre-norm style)
        x = x + self.attn_scale * self.attn(self.ln_1(x))
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

        # weight sharing scheme (reduces 768*50267=~40M params, fewer params, more efficient)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, block size is only {self.config.context_length}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
