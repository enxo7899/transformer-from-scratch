"""Unigram and bigram baselines over the same BPE tokens, scored on the same targets as the transformer."""
from __future__ import annotations

import math

import numpy as np

from .data import Windows


class Unigram:
    def __init__(self, tokens: np.ndarray, vocab: int, k: float = 1.0):
        counts = np.bincount(tokens, minlength=vocab).astype(np.float64) + k
        self.p = counts / counts.sum()

    def nll(self, w: Windows) -> float:
        return float(-np.log(self.p[w.y]).mean())


class Bigram:
    """p(b | a) = (count(a, b) + alpha * p_unigram(b)) / (count(a) + alpha): smoothed towards the unigram."""

    def __init__(self, tokens: np.ndarray, vocab: int, alpha: float = 1.0):
        self.vocab = vocab
        self.uni = Unigram(tokens, vocab).p
        self.counts = np.zeros((vocab, vocab), dtype=np.float64)
        np.add.at(self.counts, (tokens[:-1], tokens[1:]), 1.0)
        self.alpha = alpha

    def probs(self, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        c = self.counts[prev, nxt]
        total = self.counts[prev].sum(axis=-1)
        return (c + self.alpha * self.uni[nxt]) / (total + self.alpha)

    def nll(self, w: Windows) -> float:
        return float(-np.log(self.probs(w.x.reshape(-1), w.y.reshape(-1))).mean())


def fit_bigram(train_tokens: np.ndarray, vocab: int, grid=(0.1, 0.3, 1.0, 3.0, 10.0)) -> Bigram:
    """Choose alpha on the last 10% of the training tokens, then refit on all training tokens. Validation is never used."""
    cut = int(len(train_tokens) * 0.9)
    from .data import fixed_windows
    held = fixed_windows(train_tokens[cut:], 128)
    best = min(grid, key=lambda a: Bigram(train_tokens[:cut], vocab, a).nll(held))
    return Bigram(train_tokens, vocab, best)


def perplexity(nll: float) -> float:
    return math.exp(nll)
