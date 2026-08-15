"""FineWeb-10B dataloader with BOS-aligned document packing."""

import threading
from glob import glob
from pathlib import Path
from typing import Optional

import torch

# -----------------------------------------------------------------------------
# Constants

BOS_TOKEN_ID = 50256
HEADER_ITEMS = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1


# -----------------------------------------------------------------------------
# helpers

def _round_up(value: int, multiple: int = 128) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _max_doc_capacity(total_tokens: int) -> int:
    known = {
        16384: 64,
        32768: 96,
        49152: 128,
        65536: 192,
        98304: 256,
    }
    if total_tokens in known:
        return known[total_tokens]
    return max(8, _round_up(total_tokens // 300))


def _load_shard(path: str | Path) -> torch.Tensor:
    path = Path(path)

    header = torch.from_file(str(path), shared=False, size=HEADER_ITEMS, dtype=torch.int32)
    assert int(header[0]) == HEADER_MAGIC, f"bad magic in {path}"
    assert int(header[1]) == HEADER_VERSION, f"unsupported version in {path}"
    num_tokens = int(header[2])

    tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
    with path.open("rb", buffering=0) as f:
        f.seek(HEADER_ITEMS * 4)
        bytes_read = f.readinto(tokens.numpy())
    assert bytes_read == 2 * num_tokens, f"short read in {path}"

    return tokens


def _find_bos(tokens: torch.Tensor) -> torch.Tensor:
    positions = (tokens == BOS_TOKEN_ID).nonzero(as_tuple=True)[0]
    sentinel = torch.tensor([tokens.numel()], dtype=torch.int64)
    return torch.cat([positions, sentinel])


def _pack_documents(
    bounds: torch.Tensor,
    num_docs: int,
    start_doc: int,
    target_tokens: int,
    max_seq_len: int,
) -> Optional[tuple[list[int], list[int], int]]:
    target = target_tokens + 1
    starts, ends = [], []
    filled = 0
    cursor = start_doc

    while filled < target:
        if cursor >= num_docs:
            return None

        start = int(bounds[cursor])
        end = int(bounds[cursor + 1])
        end = min(end, start + max_seq_len, start + target - filled)

        starts.append(start)
        ends.append(end)
        filled += end - start
        cursor += 1

    assert filled == target
    return starts, ends, cursor


# -----------------------------------------------------------------------------
# DataLoader

class FineWebLoader:

    def __init__(
        self,
        file_glob: str,
        device_batch_size: int,
        max_seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.tokens_per_rank = device_batch_size * max_seq_len

        self.max_docs = _max_doc_capacity(self.tokens_per_rank)
        self.max_cu_len = self.max_docs + 1

        self._shards = sorted(glob(file_glob))
        assert self._shards, f"no shards match: {file_glob!r}"

        # Current shard
        self._shard_idx = 0
        self._cursor = 0
        self._current_tokens: Optional[torch.Tensor] = None
        self._current_bounds: Optional[torch.Tensor] = None
        self._current_num_docs = 0

        # Next shard
        self._next_tokens: Optional[torch.Tensor] = None
        self._next_bounds: Optional[torch.Tensor] = None
        self._next_error: Optional[Exception] = None
        self._next_ready = threading.Event()
        self._next_thread: Optional[threading.Thread] = None

        self.reset()

    @property
    def shards(self) -> list[str]:
        return self._shards

    def reset(self) -> None:
        self._drain_worker()
        self._shard_idx = 0
        self._cursor = 0
        self._set_shard(0)
        self._prefetch_shard(1)

    def next_batch(self):
        if self._current_tokens is None:
            return None

        try:
            starts_all, ends_all = self._pack_for_all_ranks()
        except StopIteration:
            if not self._advance_shard():
                return None
            try:
                starts_all, ends_all = self._pack_for_all_ranks()
            except StopIteration:
                return None

        return self._build_batch(starts_all[self.rank], ends_all[self.rank])


    def state_dict(self) -> dict[str, int]:
        if self._current_tokens is None:
            return {"shard_idx": len(self._shards), "cursor": 0}
        return {"shard_idx": self._shard_idx, "cursor": self._cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        shard_idx = int(state["shard_idx"])
        cursor = int(state["cursor"])

        if not (0 <= shard_idx < len(self._shards)):
            raise ValueError(f"invalid shard index: {shard_idx}")
        if cursor < 0:
            raise ValueError(f"invalid cursor: {cursor}")

        self._drain_worker()

        self._shard_idx = shard_idx
        self._set_shard(shard_idx)

        if cursor > self._current_num_docs:
            raise ValueError(f"cursor {cursor} exceeds {self._current_num_docs} docs in shard")

        self._cursor = cursor
        self._prefetch_shard(self._shard_idx + 1)


    def _set_shard(self, idx: int) -> None:
        self._current_tokens = _load_shard(self._shards[idx])
        self._current_bounds = _find_bos(self._current_tokens)
        self._current_num_docs = len(self._current_bounds) - 1

    def _prefetch_shard(self, idx: int) -> None:
        if idx >= len(self._shards):
            self._next_thread = None
            return

        self._next_ready.clear()
        self._next_error = None
        self._next_thread = threading.Thread(
            target=self._worker_load,
            args=(self._shards[idx],),
            daemon=True,
        )
        self._next_thread.start()

    def _worker_load(self, path: str) -> None:
        try:
            self._next_tokens = _load_shard(path)
            self._next_bounds = _find_bos(self._next_tokens)
        except Exception as exc:
            self._next_error = exc
        finally:
            self._next_ready.set()

    def _drain_worker(self) -> None:
        if self._next_thread is not None:
            self._next_ready.wait()
            self._next_thread = None

    def _advance_shard(self) -> bool:
        if self._shard_idx + 1 >= len(self._shards):
            self._current_tokens = None
            self._current_bounds = None
            return False

        self._next_ready.wait()
        if self._next_error is not None:
            raise self._next_error

        self._current_tokens = self._next_tokens
        self._current_bounds = self._next_bounds
        self._current_num_docs = len(self._current_bounds) - 1
        self._shard_idx += 1
        self._cursor = 0

        self._prefetch_shard(self._shard_idx + 1)
        return True


    def _pack_for_all_ranks(self):
        cursor = self._cursor
        starts_all, ends_all = [], []

        for _ in range(self.world_size):
            result = _pack_documents(
                self._current_bounds,
                self._current_num_docs,
                cursor,
                self.tokens_per_rank,
                self.max_seq_len,
            )
            if result is None:
                raise StopIteration("shard exhausted")
            starts, ends, cursor = result
            starts_all.append(starts)
            ends_all.append(ends)

        self._cursor = cursor
        return starts_all, ends_all

    def _build_batch(self, doc_starts: list[int], doc_ends: list[int]):
        T = self.tokens_per_rank

        flat = torch.cat([self._current_tokens[s:e] for s, e in zip(doc_starts, doc_ends)])
        x = flat[:T].to(torch.int32)
        y = flat[1 : T + 1].to(torch.int64)

        lengths = torch.tensor([e - s for s, e in zip(doc_starts, doc_ends)], dtype=torch.int32)
        lengths[-1] -= 1

        cumsum = torch.zeros(len(lengths) + 1, dtype=torch.int32)
        cumsum[1:] = lengths.cumsum(0)

        cu_seqlens = torch.full((self.max_cu_len,), T, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens[1 : len(lengths) + 1] = cumsum[1:]

        pos = torch.arange(T, dtype=torch.int32)
        pos -= torch.repeat_interleave(cumsum[:-1], lengths)
        pos = pos.to(torch.long)

        max_seqlen = int(lengths.max().item())
        return x, y, cu_seqlens, pos, max_seqlen
