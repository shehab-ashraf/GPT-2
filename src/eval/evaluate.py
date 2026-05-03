"""
Perplexity evaluation on Wikitext-2 using sliding-window approach.
Reference: https://huggingface.co/docs/transformers/en/perplexity
"""

import sys, argparse, json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# Model loading
def load_model(name, device):

    print(f"Loading {name}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        max_len = getattr(
            model.config, "n_positions",
            getattr(model.config, "max_position_embeddings", 1024),
        )
        return model, tokenizer, max_len
    except Exception:
        pass

    from src.model.gpt import GPT, GPTConfig

    conf = json.load(open(hf_hub_download(name, "config.json")))
    model = GPT(GPTConfig(**conf))
    state = {
        k.replace("_orig_mod.", ""): v
        for k, v in load_file(hf_hub_download(name, "model.safetensors")).items()
    }
    model.load_state_dict(state)
    return model.to(device), tiktoken.get_encoding("gpt2"), conf["max_seq_len"]


# Tokenization
def tokenize(tokenizer, text):

    if hasattr(tokenizer, "encode_ordinary"):
        return torch.tensor(tokenizer.encode_ordinary(text)).unsqueeze(0)
    return tokenizer(text, return_tensors="pt").input_ids


# Sliding-window perplexity (see HF docs linked above)
def compute_perplexity(model, ids, max_len, device):

    stride = max_len // 2
    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, ids.size(1), stride)):
        end = min(begin + max_len, ids.size(1))
        chunk = ids[:, begin:end].to(device)

        with torch.no_grad():
            out = model(chunk)
            logits = out[0] if isinstance(out, tuple) else out.logits

        loss = F.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="none")
        nlls.append(loss[-(end - prev_end):])

        prev_end = end
        if end == ids.size(1):
            break

    return torch.exp(torch.cat(nlls).mean()).item()


# Main
def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 perplexity evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, max_len = load_model(args.model, device)
    if args.compile:
        model = torch.compile(model)
    model.eval()

    print("Loading wikitext-2-raw-v1...")
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenize(tokenizer, text)

    print(f"Evaluating (stride={max_len // 2}, max_len={max_len})...")
    ppl = compute_perplexity(model, ids, max_len, device)
    print(f"Wikitext PPL: {ppl:.4f}")

if __name__ == "__main__":
    main()
