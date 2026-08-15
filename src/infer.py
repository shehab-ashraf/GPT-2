"""Inference for nanoGPT-124M: streaming text generation with a KV cache."""

import argparse
import json
import time
import torch
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# configuration

HF_REPO = "ashrafs1/nanogpt-3000"
GEN_TOKENS = 128
TEMPERATURE = 0.7
TOP_K = 50
REPETITION_PENALTY = 1.3
enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# model loading


def load_model(repo_id: str, device: str = "cuda") -> GPT:
    conf = json.load(open(hf_hub_download(repo_id, "config.json")))
    model = GPT(GPTConfig(**conf))
    state = load_file(hf_hub_download(repo_id, "model.safetensors"), device=device)
    model.load_state_dict(state)
    model.to(device).to(torch.bfloat16).eval()
    model.rotary.cos = model.rotary.cos.float()
    model.rotary.sin = model.rotary.sin.float()
    model.allocate_cache(batch_size=1, device=device, dtype=torch.bfloat16)
    return model


# -----------------------------------------------------------------------------
# sampling


def sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[-1], -float("inf"))
    probs = F.softmax(logits.float() / temperature, dim=-1)
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div_(noise).argmax(dim=-1)


# -----------------------------------------------------------------------------
# generation


@torch.inference_mode()
def generate(
    model: GPT,
    prompt: str,
    max_new_tokens: int,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    rep_pen: float = REPETITION_PENALTY,
):
    device = next(model.parameters()).device
    cfg = model.config

    ids = enc.encode(prompt) if prompt else [enc.eot_token]
    if len(ids) > cfg.max_seq_len:
        ids = ids[-cfg.max_seq_len :]
    T = len(ids)
    max_new_tokens = min(max_new_tokens, cfg.max_seq_len - T)

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(idx, cache_seqlens=0)[0][:, -1, :]
    pos = T

    seen = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
    seen[ids] = True

    for _ in range(max_new_tokens):
        if rep_pen != 1.0:
            vals = logits[0][seen]
            logits[0][seen] = torch.where(vals > 0, vals / rep_pen, vals * rep_pen)

        next_id = sample(logits[0], temperature, top_k)
        seen[next_id] = True
        yield enc.decode([next_id.item()])

        if pos >= cfg.max_seq_len - 1:
            break

        p = torch.tensor([pos], dtype=torch.long, device=device)
        cs = torch.tensor([pos], dtype=torch.int32, device=device)
        logits = model(next_id.view(1, 1), pos=p, cache_seqlens=cs)[0][:, -1, :]
        pos += 1


# -----------------------------------------------------------------------------
# main


def main():
    parser = argparse.ArgumentParser(description="nanoGPT-124M text generation")
    parser.add_argument("--start", type=str, default="", help="prompt to condition on")
    parser.add_argument(
        "--max_new", type=int, default=GEN_TOKENS, help="number of new tokens to generate"
    )
    parser.add_argument(
        "--repo", type=str, default=HF_REPO, help="huggingface repo id of the trained model"
    )
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

    model = load_model(args.repo)

    print(f"prompt: {args.start}")
    print("generated_text: ", end="", flush=True)
    t0 = time.time()
    n = 0
    for chunk in generate(model, args.start, args.max_new):
        print(chunk, end="", flush=True)
        n += 1
    dt = time.time() - t0
    print(f"\n\n[{n} tokens in {dt:.2f}s, {n / dt:.1f} tok/s]")


if __name__ == "__main__":
    main()
