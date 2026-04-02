import argparse
import time

import torch
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from src.model.gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# configuration

HF_REPO = 'ashrafs1/nanoGPT'
GEN_TOKENS         = 128
TEMPERATURE        = 0.85
TOP_K              = 50
TOP_P              = 0.95
REPETITION_PENALTY = 1.15

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")


# -----------------------------------------------------------------------------
# inference

@torch.inference_mode()
def generate_text(model, prompt, max_new=GEN_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, rep_pen=REPETITION_PENALTY):
    prompt_ids = enc.encode(prompt) if prompt else [enc.eot_token]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    seen_tokens = set(prompt_ids)
    
    for _ in range(max_new):
        logits, _ = model(ids)
        logits    = logits[:, -1, :].clone()
        
        logits = logits / max(temperature, 1e-5)
        
        if rep_pen != 1.0:
            for tid in seen_tokens:
                val = logits[0, tid]
                logits[0, tid] = val / rep_pen if val > 0 else val * rep_pen
                
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[0, sorted_idx[0][remove[0]]] = -float('Inf')
        
        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        
        token = next_id.item()
        seen_tokens.add(token)
        ids = torch.cat([ids, next_id], dim=1)
        
        yield enc.decode([token])


# -----------------------------------------------------------------------------
# cli

def parse_args():
    parser = argparse.ArgumentParser(description="nanoGPT sampling")
    parser.add_argument("--start", default="", help="Custom starting prompt")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--max_new", type=int, default=GEN_TOKENS)
    return parser.parse_args()

def main():
    args = parse_args()

    import warnings
    warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')

    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename='model.safetensors')
    state_dict = load_safetensors(ckpt_path, device='cpu')

    model = GPT(GPTConfig())
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    prompt = args.start 
    
    print(f"prompt: {prompt}")
    print("generated_text: ", end="", flush=True)
    
    t0 = time.time()
    for token_str in generate_text(
        model=model,
        prompt=prompt,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=TOP_P,
        rep_pen=REPETITION_PENALTY
    ):
        print(token_str, end="", flush=True)
    
    dt = time.time() - t0
    print(f"\n\n[Tokens: {args.max_new} | Time: {dt:.2f}s | Speed: {args.max_new / dt:.1f} tok/s]\n")

if __name__ == "__main__":
    main()