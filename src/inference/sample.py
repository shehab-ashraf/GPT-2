import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import argparse
import time
import torch
import torch.nn.functional as F
import tiktoken
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.model.gpt import GPT, GPTConfig

HF_REPO = 'ashrafs1/nanogpt-3000s'
GEN_TOKENS, TEMP, TOP_K, TOP_P, REP_PEN = 128, 0.85, 50, 0.95, 1.15
enc = tiktoken.get_encoding("gpt2")

@torch.inference_mode()
def generate_text(model, prompt, max_new=GEN_TOKENS, temp=TEMP, top_k=TOP_K, top_p=TOP_P, rep_pen=REP_PEN):
    ids = torch.tensor([enc.encode(prompt) if prompt else [enc.eot_token]], dtype=torch.long)
    seen_tokens = set(ids[0].tolist())
    
    for _ in range(max_new):
        # truncate to context length, forward pass, grab last logit
        logits = model(ids[:, -model.config.max_seq_len:])[0][:, -1, :].clone() / max(temp, 1e-5)
        
        if rep_pen != 1.0:
            for t in seen_tokens:
                val = logits[0, t]
                logits[0, t] = val / rep_pen if val > 0 else val * rep_pen
                
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            logits[0, sorted_idx[0][remove[0]]] = -float('Inf')
        
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        token = next_id.item()
        seen_tokens.add(token)
        ids = torch.cat([ids, next_id], dim=1)
        yield enc.decode([token])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="", help="Prompt")
    parser.add_argument("--max_new", type=int, default=GEN_TOKENS)
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings('ignore', message='.*unauthenticated requests.*')

    ckpt = hf_hub_download(repo_id=HF_REPO, filename='model.safetensors')
    model = GPT(GPTConfig())
    model.load_state_dict(load_file(ckpt, device='cpu'))
    model.eval()
    
    print(f"prompt: {args.start}\ngenerated_text: ", end="", flush=True)
    
    t0 = time.time()
    for t in generate_text(model, args.start, args.max_new):
        print(t, end="", flush=True)
    
    dt = time.time() - t0
    print(f"\n\n[Tokens: {args.max_new} | Time: {dt:.2f}s | Speed: {args.max_new / dt:.1f} tok/s]")

if __name__ == "__main__":
    main()