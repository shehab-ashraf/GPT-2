# nanogpt

Training a 124M parameter large language model.

This is a complete train + inference for a modernized GPT-2 architecture, with a focus on maximum speed and training stability. I kept the classic 124M parameter footprint from the 2019 GPT-2, but completely rebuilt the internals using modern LLM architecture choices (RoPE, RMSNorm, FlashAttention, Muon, U-Net skips).

Read the full technical deep dive on my [blog](https://shehab-ashraf.github.io/posts/nanogpt/) or check out the raw training logs on [WandB](https://wandb.ai/ashrafshehab-/nanoGPT).

## quickstart

First, navigate to the folder where you keep your projects and clone this repository:

```bash
git clone https://github.com/shehab-ashraf/GPT-2.git
cd GPT-2
pip install -r requirements.txt
```

Then, let's run a quick CPU inference to see it in action:

```bash
python -m src.inference.sample --start "What is the answer to life, the universe, and everything?"
```

You'll see the text stream a sample:

> prompt: What is the answer to life, the universe, and everything?
> generated_text: The answer is yes. What can I do for my life, my family or even a little piece of me with all that I have lost in this time? How could I have changed that and made it work again on my own? How could I have made some changes to my life over the years so that it has been taken into account when it comes back from this loss? How could I make myself more productive? In addition, what would be my priority if I had no other words to say to me now? Would I choose to follow up any of these things and give up? What would happen if someone else was involved in my

## performance

How good is it actually? I benchmarked it head-to-head against the official OpenAI GPT-2 on the WikiText-2 dataset using a sliding-window evaluation:

* nanoGPT (3,000 steps): `42.15` perplexity 
* OpenAI GPT-2 (124M): `25.17` perplexity

The official GPT-2 was trained on roughly 40 billion tokens. This 3,000-step checkpoint only read ~1.5 billion tokens (over 25x less data). The architecture upgrades massively accelerate learning efficiency, but it just needs more data to fully close the gap.

**Training Stats** (Single A100 GPU):
* Hardware: 1x NVIDIA A100
* Time: ~2 hours
* Throughput: ~200,000 tokens/sec
* MFU: ~37.8%
* Best Validation Loss: `3.3110`

## model

The model follows the GPT-2 124M parameter footprint but incorporates several modern improvements for better speed and stability:

* **RoPE** relative position encodings
* **RMSNorm** instead of LayerNorm
* **Squared ReLU** activation
* **Vocab 50,304** (Tensor Core optimized)
* **U-Net skip connections**
* **QK-norm + FlashAttention**
* **Logit soft-capping** (Gemma-style)
* **Muon + AdamW** optimizers

Training: micro batch 64, seq 2048, total batch 524k tokens/step, 3000 steps, trapezoidal LR schedule. Muon LR 0.02, AdamW LR 0.006, bf16/TF32 precision.

## train it yourself

I use Lightning AI for training because it gives me access to A100 GPUs. First set up the environment:

```bash
bash scripts/setup.sh
```

Download the 10B subset of FineWeb and start training:

```bash
python -m src.data.download --shards 50
python -m src.train
```

## ack

Architecture and training recipe heavily inspired by [Keller Jordan](https://github.com/KellerJordan/modded-nanogpt)'s modded-nanogpt speedrun and [Tyler Romero](https://github.com/tyler-romero)'s contributions. Built on top of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) was a great learning resource.
