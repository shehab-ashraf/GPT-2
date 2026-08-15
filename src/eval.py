import argparse
import json

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# -----------------------------------------------------------------------------
# model loading


def load_model(name, device):

    print(f"Loading {name}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        max_len = getattr(
            model.config,
            "n_positions",
            getattr(model.config, "max_position_embeddings", 1024),
        )
        return model, tokenizer, max_len
    except Exception:
        pass

    from src.model import GPT, GPTConfig

    conf = json.load(open(hf_hub_download(name, "config.json")))
    model = GPT(GPTConfig(**conf))
    state = {
        k.replace("_orig_mod.", ""): v  # strip the torch.compile prefix, if any
        for k, v in load_file(hf_hub_download(name, "model.safetensors")).items()
    }
    model.load_state_dict(state)
    return model.to(device), tiktoken.get_encoding("gpt2"), conf["max_seq_len"]


# -----------------------------------------------------------------------------
# tokenization


def tokenize(tokenizer, text):
    if hasattr(tokenizer, "encode_ordinary"):
        return torch.tensor(tokenizer.encode_ordinary(text)).unsqueeze(0)
    return tokenizer(text, return_tensors="pt").input_ids


# -----------------------------------------------------------------------------
# sliding-window perplexity


def compute_perplexity(model, ids, max_len, device):
    stride = max_len // 2
    nlls = []
    prev_end = 0
    is_ours = type(model).__name__ == "GPT"

    for begin in tqdm(range(0, ids.size(1), stride)):
        end = min(begin + max_len, ids.size(1))
        chunk = ids[:, begin:end].to(device)

        dtype = torch.bfloat16 if (is_ours and device.startswith("cuda")) else torch.float32
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=is_ours),
        ):
            if is_ours:  # flash-attention path: single sequence, no padding
                T = chunk.size(1)
                cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=device)
                pos = torch.arange(T, device=device)
                out = model(chunk, cu_seqlens=cu_seqlens, pos=pos, max_seqlen=T)
            else:
                out = model(chunk)
            logits = out[0] if isinstance(out, tuple) else out.logits

        loss = F.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="none")
        nlls.append(loss[-(end - prev_end) :])

        prev_end = end
        if end == ids.size(1):
            break

    return torch.exp(torch.cat(nlls).mean()).item()


# -----------------------------------------------------------------------------
# main


def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 perplexity evaluation")
    parser.add_argument(
        "--model", type=str, required=True, help="a HF model id, or one of our checkpoint repos"
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model before evaluating"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, max_len = load_model(args.model, device)
    if args.compile:
        model = torch.compile(model)
    model.eval()

    print("Loading wikitext-2-raw-v1 (test split)...")
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    ids = tokenize(tokenizer, text)

    print(f"Evaluating (stride={max_len // 2}, max_len={max_len})...")
    ppl = compute_perplexity(model, ids, max_len, device)
    print(f"Wikitext PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()
