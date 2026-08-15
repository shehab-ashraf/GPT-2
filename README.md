# nanogpt

I trained a 124M-parameter large language model from scratch on FineWeb-10B.

I built a 124M-parameter GPT-2 from scratch in PyTorch and incrementally 
replaced every 2019-era component with its modern
version: FlashAttention varlen with packed documents and per-document causal
masking, RoPE, RMSNorm, QK-norm, squared ReLU,
U-Net skip connections, Gemma-style logit soft-capping, and a
Muon + AdamW optimizer split.

3,000 steps on 2× A100, ~70 minutes of training, and it hits **70% MFU** at
375,000 tok/s. It reaches 40.50 perplexity on WikiText-2 vs. OpenAI's 25.2
with ~40B tokens. Not close yet, but it's learning fast on far less data.

- Blog (deep dive): <https://shehab-ashraf.github.io/posts/nanogpt/>
- Weights & Biases: <https://wandb.ai/ashrafshehab-/nanoGPT>
- Pretrained weights: <https://huggingface.co/ashrafs1/nanogpt-3000>

## feel the magic

First, navigate to the folder where you keep your projects and clone this
repository:

```bash
git clone https://github.com/shehab-ashraf/nanogpt.git && cd nanogpt
```

Then, let's run an inference to see it in action:

```bash
uv sync --extra eval
uv run python -m src.infer --start "What is the answer to life, the universe, and everything?"
```

And it streams back:

```
prompt: What is the answer to life, the universe, and everything?
generated_text: The answer is yes. What can I do for my life, my family or even a little piece of me with all that I have lost in this time? How could I have changed that and made it work again on my own? How could I have made some changes to my life over the years so that it has been taken into account when it comes back from this loss? How could I make myself more productive? In addition, what would be my priority if I had no other words to say to me now? Would I choose to follow up any of these things and give up? What would happen if someone else was involved in my
```


## how good is it

I benchmarked it head-to-head against the official OpenAI GPT-2 on WikiText-2,
sliding-window perplexity:

| model | tokens trained | WikiText-2 PPL |
| --- | ---: | ---: |
| OpenAI GPT-2 (124M) | ~40 B | 25.2 |
| **nanoGPT (3,000 steps)** | ~1.57 B | 40.50 |

The architecture upgrades accelerate learning, but it needs more data to
fully close the gap.

## what's inside

The 2019 GPT-2 internals were rebuilt with modern components:

- **Muon + AdamW**: [Muon](https://github.com/KellerJordan/modded-nanogpt) (Newton-Schulz orthogonalized momentum) for all 2-D matrix weights; AdamW for the embeddings and 1-D params. Best of both worlds.
- **FlashAttention-2 (varlen)**: packed documents, per-document causal masking via `cu_seqlens`, **zero padding waste**. Flash-attn only, run on CUDA.
- **RoPE**: replacing absolute positional embeddings.
- **RMSNorm** + **QK-norm**: replacing LayerNorm and raw attention logits.
- **Squared ReLU**: the MLP activation.
- **U-Net skip connections**: first half of the blocks saves activations, second half adds them back.
- **Logit soft-capping**: Gemma-style `30 * tanh(logits / 30)` for stability.
- **Vocab 50,304**: padded to a multiple of 128 for tensor-core efficiency.
- **bf16 compute, TF32 matmul, `torch.compile`**: forwards/backwards run under bf16 `autocast`, `set_float32_matmul_precision("high")` lets fp32 matmuls ride the tensor cores as TF32, and the whole graph is fused by `torch.compile`.

## models

Pretrained checkpoints on the Hub, in plain `model.safetensors` + `config.json` form:

| run | steps | tokens | WikiText-2 PPL | download |
| --- | ---: | ---: | ---: | --- |
| nanogpt-3000 | 3,000 | ~1.57 B | 40.50 | [ashrafs1/nanogpt-3000s](https://huggingface.co/ashrafs1/nanogpt-3000s) |


## train it yourself

One launcher handles `uv sync`, FineWeb-10B shard download, and single- or two-GPU DDP training. Designed for A100 40GB (peaks at 32.9 GB).

```bash 
NPROC=2 TOTAL_STEPS=3000 WANDB_RUN=nanogpt-3000 bash runs/speedrun.sh 
```

Environment controls:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `NPROC` | `1` | Number of A100 GPUs |
| `TOTAL_STEPS` | `3000` | Optimizer steps |
| `MICRO_BATCH` | `32` | Sequences per GPU per micro-batch |
| `RUN_NAME` | `nanogpt-3000` | Output directory under `out/` |
| `RESUME` | `0` | `1` resumes from `ckpt_last.pt` |
| `WANDB_RUN` | unset | Enables W&B logging when set |
| `DOWNLOAD_SHARDS` | `50` | FineWeb train shards to keep locally |

Extra `src.train` flags pass through at the end:

```bash
NPROC=2 TOTAL_STEPS=6000 bash runs/speedrun.sh --eval-every 250
```
  
### the recipe

- **Batch**: 524,288 global tokens / step. Micro-batch 32 × seq 2048 × grad-accum, DDP-all-reduced. 3,000 steps → ~1.57 B tokens seen.
- **Schedule**: trapezoidal LR. Linear warmup → constant → linear cooldown.
- **Optimizers**: Muon LR 0.02 (weight-decay 0.01, momentum 0.95); AdamW LR 0.0036 (weight-decay 0.1, β1 0.9, β2 0.95).
- **Eval**: every 250 steps on 10.49 M tokens.



## ack

Architecture and training recipe heavily inspired by
[Keller Jordan](https://github.com/KellerJordan/modded-nanogpt)'s modded-nanogpt
speedrun and [Tyler Romero](https://github.com/tyler-romero)'s contributions,
built on top of Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[llm.c](https://github.com/karpathy/llm.c). Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) was a great
learning resource.
