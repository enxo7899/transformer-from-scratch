"""Corpus loading, split by document, and batching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bpe import BPETokenizer

DOC_SEPARATOR = "<<<DOC>>>"


def load_documents(path: str | Path) -> list[str]:
    return [d.strip("\n") for d in Path(path).read_text(encoding="ascii").split(f"\n{DOC_SEPARATOR}\n") if d.strip()]


def split_documents(docs: list[str], val_fraction: float = 0.08, seed: int = 0) -> tuple[list[str], list[str]]:
    """Shuffle documents with a fixed seed and hold out whole documents until val_fraction of characters is reached."""
    order = np.random.default_rng(seed).permutation(len(docs))
    total = sum(len(d) for d in docs)
    val_idx, chars = set(), 0
    for i in order:
        if chars >= val_fraction * total:
            break
        val_idx.add(int(i))
        chars += len(docs[i])
    train = [d for i, d in enumerate(docs) if i not in val_idx]
    val = [d for i, d in enumerate(docs) if i in val_idx]
    return train, val


def tokenize_documents(tok: BPETokenizer, docs: list[str]) -> np.ndarray:
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(d))
        ids.append(tok.eot_id)
    return np.asarray(ids, dtype=np.int64)


@dataclass
class Windows:
    x: np.ndarray  # (N, T) inputs
    y: np.ndarray  # (N, T) next-token targets


def fixed_windows(tokens: np.ndarray, context: int) -> Windows:
    """Non-overlapping windows covering the token stream, used for evaluation."""
    n = (len(tokens) - 1) // context
    x = tokens[: n * context].reshape(n, context)
    y = tokens[1: n * context + 1].reshape(n, context)
    return Windows(x, y)


def random_batch(tokens: np.ndarray, batch: int, context: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    starts = rng.integers(0, len(tokens) - context - 1, size=batch)
    x = np.stack([tokens[s: s + context] for s in starts])
    y = np.stack([tokens[s + 1: s + 1 + context] for s in starts])
    return x, y
