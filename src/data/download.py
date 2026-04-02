import argparse
import os

from huggingface_hub import hf_hub_download


# -----------------------------------------------------------------------------
# configuration

REPO_ID   = "kjj0/fineweb10B-gpt2"
NUM_TOTAL = 103  


# -----------------------------------------------------------------------------
# download

def download_shard(filename: str, output_dir: str) -> None:
    """Download a sigle shard file if it doesn't already exist."""
    local_path = os.path.join(output_dir, filename)
    if os.path.exists(local_path):
        print(f"  already exists: {filename}")
        return

    print(f"  downloading {filename} ...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=output_dir,
    )


# -----------------------------------------------------------------------------
# cli

def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-10B pre-tokenized shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shards", type=int, default=NUM_TOTAL,
        help="number of training shards to download (each ~100M tokens)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "cache", "fineweb-10B"),
        help="directory to save shard files",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    download_shard("fineweb_val_%06d.bin" % 0, args.output_dir)

    for i in range(1, args.shards + 1):
        download_shard("fineweb_train_%06d.bin" % i, args.output_dir)


if __name__ == "__main__":
    main()
