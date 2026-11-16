"""Byte-pair encoding trained from scratch.

The tokenizer works on UTF-8 bytes, so any string round-trips. Text is first cut
into pre-tokens (words, numbers, punctuation runs, whitespace runs) and merges
never cross a pre-token boundary. The JavaScript port in web/bpe.js implements
the same pre-tokenizer and merge procedure; tests/fixtures/parity.json keeps
the two in step.

Token ids: 0..255 are raw bytes, 256.. are merges in training order, and the
last id is the end-of-text marker.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PRETOKEN_PATTERN = r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9 \t\n\r\f\v]+|[ \t\n\r\f\v]+(?![^ \t\n\r\f\v])|[ \t\n\r\f\v]+"
_PRETOKEN_RE = re.compile(PRETOKEN_PATTERN)
END_OF_TEXT = "<|endoftext|>"


class BPEError(ValueError):
    """Raised for invalid tokenizer input, such as an unknown token id."""


def pretokenize(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.eot_id = len(self.vocab)
        self.vocab_size = self.eot_id + 1
        self._cache: dict[str, tuple[int, ...]] = {}

    # ---- training ---------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 512) -> "BPETokenizer":
        """Learn vocab_size - 257 merges (257 = 256 bytes + end-of-text)."""
        if vocab_size < 257:
            raise BPEError("vocab_size must be at least 257")
        word_counts = Counter(pretokenize(text))
        words = [(list(w.encode("utf-8")), n) for w, n in word_counts.items()]
        merges: list[tuple[int, int]] = []
        while len(merges) < vocab_size - 257:
            pairs: Counter[tuple[int, int]] = Counter()
            for seq, n in words:
                for pair in zip(seq, seq[1:]):
                    pairs[pair] += n
            if not pairs:
                break
            best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = 256 + len(merges)
            merges.append(best)
            words = [(_merge(seq, best, new_id), n) for seq, n in words]
        return cls(merges)

    # ---- encode / decode ---------------------------------------------
    def _encode_pretoken(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        seq = list(piece.encode("utf-8"))
        while len(seq) > 1:
            best_rank, best_pair = None, None
            for pair in zip(seq, seq[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            seq = _merge(seq, best_pair, 256 + best_rank)
        out = tuple(seq)
        self._cache[piece] = out
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in pretokenize(text):
            ids.extend(self._encode_pretoken(piece))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        out = bytearray()
        for i in ids:
            if i == self.eot_id:
                out += END_OF_TEXT.encode()
            elif 0 <= i < self.eot_id:
                out += self.vocab[i]
            else:
                raise BPEError(f"unknown token id {i}")
        return bytes(out)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ---- persistence ----------------------------------------------------
    def to_json(self) -> dict:
        return {"type": "byte-level-bpe", "pretoken_pattern": PRETOKEN_PATTERN, "vocab_size": self.vocab_size,
                "eot_id": self.eot_id, "merges": [list(p) for p in self.merges]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), separators=(",", ":")), encoding="ascii")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="ascii"))
        return cls([tuple(p) for p in data["merges"]])


def _merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out
