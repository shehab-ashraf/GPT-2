"""
FineWeb Preprocessing Script (10B Sample)

This script downloads, tokenizes, and shards the FineWeb 10B dataset.
It stores the results in .bin files, which includes a 1024-byte header.

Usage Examples:

1. Dev Mode (20M tokens):
   python prepare.py --total_tokens 20000000 --shard_size 2000000
   -> Output will be in 'fineweb20M'

2. Full Training:
   python prepare.py --shard_size 100000000
   -> Output will be in 'fineweb10B'
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

MAGIC = 20240520  
VERSION = 1       
HEADER_SIZE = 256 

def write_shard(path, tokens: np.ndarray):
    """Writes a list of tokens to a binary file with a header."""
    assert tokens.dtype == np.uint16
    
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens) 

    with open(path, "wb") as f:
        f.write(header.tobytes())  
        f.write(tokens.tobytes())  

def main():
    parser = argparse.ArgumentParser(description="Clean and simple FineWeb preprocessing")
    parser.add_argument("--shard_size", type=int, default=100_000_000, 
                        help="How many tokens per shard")
    parser.add_argument("--total_tokens", type=int, default=None,
                        help="How many tokens to process in total (leave empty for full dataset)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (defaults to fineweb20M or fineweb10B)")
    args = parser.parse_args()

    remote_name = "sample-10BT"
    
    if args.out_dir:
        out_dir = args.out_dir
    elif args.total_tokens == 20_000_000:
        out_dir = "data/fineweb20M"
    else:
        out_dir = "data/fineweb10B"
        
    os.makedirs(out_dir, exist_ok=True)

    print(f"--- FineWeb Preprocessing (10B Sample) ---")
    print(f"Output Directory : {out_dir}")
    print(f"Total Tokens     : {args.total_tokens or 'ENTIRE DATASET'}")
    print(f"Tokens per Shard : {args.shard_size:,}")


    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=remote_name,
        split="train",
        streaming=True,
    )

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] 

    def tokenize(text: str) -> np.ndarray:
        """Tokenizes a string and adds an <|endoftext|> prefix."""
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(text))
        return np.array(tokens, dtype=np.uint16)

    progress = tqdm(
        total=args.total_tokens,
        unit="tokens",
        desc="Processing tokens",
        disable=args.total_tokens is None,
    )

    shard_idx = 0
    shard = np.empty(args.shard_size, dtype=np.uint16)
    shard_pos = 0
    total_processed = 0
    
    total_limit = args.total_tokens if args.total_tokens is not None else float('inf')

    for doc in dataset:
        if total_processed >= total_limit:
            break
            
        tokens = tokenize(doc["text"])
        pos = 0
        
        while pos < len(tokens) and total_processed < total_limit:
            space_left = args.shard_size - shard_pos
            
            if space_left == 0:
                shard_path = os.path.join(out_dir, f"fineweb_{shard_idx:06d}.bin")
                write_shard(shard_path, shard)
                shard_idx += 1
                shard_pos = 0
                space_left = args.shard_size
            
            take = min(space_left, len(tokens) - pos, total_limit - total_processed)
            
            shard[shard_pos:shard_pos + take] = tokens[pos:pos + take]
            
            shard_pos += take
            pos += take
            total_processed += take
            progress.update(take)

    if shard_pos > 0:
        shard_path = os.path.join(out_dir, f"fineweb_{shard_idx:06d}.bin")
        write_shard(shard_path, shard[:shard_pos])

    progress.close()

if __name__ == "__main__":
    main()
