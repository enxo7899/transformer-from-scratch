"""Write the parity fixture the browser and the node tests compare against.

python -m tiny_gpt.export --model-dir model
Logits are computed from the float32 weights on disk, in float64, exactly as the JavaScript forward pass sees them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bpe import BPETokenizer
from .data import load_documents, split_documents
from .infer import generate
from .weights import load_weights

PARITY_PROMPT = "def read_file(path):\n    with open(path) as f:\n        return f.read()"
TOKENIZER_STRINGS = [
    "Hello, world!", "  two  spaces and\ttabs\n\nnewlines  ", "x = [i for i in range(10) if i % 2 == 0]",
    "The quick brown fox jumps over the lazy dog.", "numbers 3.14159 and 2026-11-16", "caf\u00e9 \u2192 \u65e5\u672c\u8a9e",
    "import os, sys; print(os.path.join('a', 'b'))", "", " ", "A" * 40, "Return the value of the *args and **kwargs.",
]


def heldout_passage(tok: BPETokenizer, corpus: Path, n_tokens: int = 256) -> dict:
    """The first n_tokens of the first validation document long enough, used by the page's context-length experiment."""
    _, val = split_documents(load_documents(corpus))
    for doc in val:
        ids = tok.encode(doc)
        if len(ids) >= n_tokens:
            return {"ids": ids[:n_tokens], "text": tok.decode(ids[:n_tokens])}
    raise ValueError("no validation document is long enough")


def build_fixture(model_dir: Path, corpus: Path) -> dict:
    tok = BPETokenizer.load(model_dir / "tokenizer.json")
    model = load_weights(model_dir / "weights.bin", dtype=np.float64)
    ids = tok.encode(PARITY_PROMPT)
    logits, _ = model.forward(np.asarray(ids)[None])
    att = model.attention_maps()
    greedy = generate(model, ids, 24, temperature=0)
    return {
        "prompt": PARITY_PROMPT,
        "ids": ids,
        "logits": [[round(float(v), 6) for v in row] for row in logits[0]],
        "attention_layer0_head0": [[round(float(v), 7) for v in row] for row in att[0][0, 0]],
        "attention_last_layer_head3": [[round(float(v), 7) for v in row] for row in att[-1][0, min(3, att[-1].shape[1] - 1)]],
        "greedy_continuation": greedy,
        "heldout": heldout_passage(tok, corpus),
        "tokenizer": [{"text": s, "ids": tok.encode(s)} for s in TOKENIZER_STRINGS],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--corpus", default="data/corpus.txt")
    args = ap.parse_args()
    out = Path(args.model_dir) / "parity.json"
    out.write_text(json.dumps(build_fixture(Path(args.model_dir), Path(args.corpus)), separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
