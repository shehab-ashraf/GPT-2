import os
import time
import torch
import torch.nn as nn
from model.gpt import GPT, GPTConfig
from data.dataloader import TokenDataLoader

# Hyperparameters
total_batch_size = 524288 # 2**19 tokens per optimizer step
batch_size = 32 # micro-batch size
sequence_length = 1024
max_steps = 38 # 20M tokens / 524288 approx 38 steps
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Calculate gradient accumulation steps
assert total_batch_size % (batch_size * sequence_length) == 0, "total_batch_size must be divisible by batch_size * sequence_length"
grad_accum_steps = total_batch_size // (batch_size * sequence_length)
print(f"Total batch size: {total_batch_size}")
print(f"Tokens per micro-batch: {batch_size * sequence_length}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Data Loader
data_root = "data/fineweb20M"
if not os.path.exists(data_root):
    print(f"Warning: {data_root} not found.")

train_loader = TokenDataLoader(data_root, B=batch_size, T=sequence_length)

torch.set_float32_matmul_precision('high')

# Model
config = GPTConfig(context_length=sequence_length)
model = GPT(config)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training Loop
model.train()
step = 0

# Start timing total training
total_start_time = time.time()

while step < max_steps:
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        # Get batch
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        # Scale loss to account for accumulation
        _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        
        # Backward pass
        loss.backward()
    
    optimizer.step()
    
    # Wait for GPU to finish all queued operations for accurate timing
    if device == "cuda": torch.cuda.synchronize()
        
    t1 = time.time()
    dt = (t1 - t0) * 1000 # milliseconds
    tokens_per_sec = total_batch_size / (t1 - t0)
    
    # Log: step:0 train_loss:10.8299 train_time:0ms token/s:
    print(f"step:{step}/{max_steps} train_loss:{loss_accum.item():.4f} train_time:{dt:.0f}ms token/s:{tokens_per_sec:.2f}")
    
    step += 1

# End timing total training
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Total tokens processed: {max_steps * total_batch_size:,}")
print(f"Average tokens/sec: {(max_steps * total_batch_size) / total_time:.2f}")
