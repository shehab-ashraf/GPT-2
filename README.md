# nanoGPT

Training a 124M parameter large language model.

This isn't the 2019 GPT-2. I kept the classic 124M parameter footprint, but rebuilt the internals using modern LLMs.

Read the full technical deep dive on my blog: [nanoGPT](https://shehab-ashraf.github.io/posts/nanogpt/) | Training logs: [WandB](https://wandb.ai/ashrafshehab-/nanoGPT)

## Quickstart

Let's get it running on your machine for quick CPU inference. First, clone and install:

```bash
git clone https://github.com/shehab-ashraf/GPT-2.git
cd GPT-2
pip install -r requirements.txt
```

Then generate text:

```bash
python -m src.inference.sample --start "What is the answer to life, the universe, and everything?"
```

```text
prompt: What is the answer to life, the universe, and everything?
generated_text: The answer is yes. What can I do for my life, my family or even a little piece of me with all that I have lost in this time? How could I have changed that and made it work again on my own? How could I have made some changes to my life over the years so that it has been taken into account when it comes back from this loss? How could I make myself more productive? In addition, what would be my priority if I had no other words to say to me now? Would I choose to follow up any of these things and give up? What would happen if someone else was involved in my
```

## Performance

How good is it actually? I benchmarked it head-to-head against the official OpenAI GPT-2 on the WikiText-2 dataset:

*   **nanoGPT (1,500 steps)**: `56.47` perplexity 
*   **OpenAI GPT-2 (124M)**: `27.22` perplexity

The official GPT-2 was trained on roughly 40 billion tokens. This checkpoint only read 786 million tokens (over 50x less data). The architecture upgrades accelerates learning, but the model still needs to read more data to close the gap. 

**Training Stats** (Single A100 GPU):
*   **Time**: 69 minutes
*   **Tokens Processed**: 786 Million
*   **Throughput**: 191,000 tokens/sec
*   **MFU**: 35.8%

## The Architecture

I kept the core 124M parameters but swapped in modern architecture choices to make training blazing fast and stable. Here's what I changed and why:

| Component | Original GPT-2 | What I did |
|-----------|----------------|------------|
| Position encoding | Learned absolute | **RoPE**: Encodes relative positions directly into attention |
| Normalization | LayerNorm | **RMSNorm**: Drops the mean-centering, fewer ops, same quality |
| Activation | GELU | **Squared ReLU**: `max(0, x)²`, dead simple and faster |
| Vocab size | 50,257 | **50,304**: Padded to keep Tensor Cores happy |
| Skip connections | Sequential | **U-Net**: First 6 layers save residuals, layers 7-12 add them back |
| Attention | Standard | **QK-norm + FlashAttention varlen**: Stable training + document packing |
| Logit capping | None | **Soft-cap (Gemma-style)**: Prevents logits from exploding |
| Optimizer | AdamW | **Muon** (transformer blocks) + AdamW (embeddings) |


### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Micro batch size | 64 |
| Sequence length | 2,048 |
| Total batch size | 524,288 tokens/step |
| Total steps | 1,500 |
| LR schedule | Trapezoidal (100 warmup → hold → 900 cooldown) |
| Muon LR / weight decay | 0.02 / 0.01 |
| AdamW LR / weight decay | 0.006 / 0.1 |
| Precision | bfloat16 |

## Train It Yourself

I use **Lightning AI** for training because it gives me free access to A100 GPUs 

```sh
bash scripts/setup.sh
```

Download the 10B subset of FineWeb and train:

```sh
python -m src.data.download --shards 50
python -m src.train
```

## Acknowledgements

The architecture and training recipe are heavily inspired by [Keller Jordan](https://github.com/KellerJordan/modded-nanogpt)'s modded-nanogpt speedrun and [Tyler Romero](https://github.com/tyler-romero)'s contributions. And of course, the whole thing stands on the shoulders of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [minGPT](https://github.com/karpathy/minGPT), which showed that a GPT implementation doesn't need to be 10,000 lines.
