"""Seeded training is reproducible, and the shipped model files agree with each other."""
import json
from pathlib import Path

import numpy as np
import pytest

from tiny_gpt.bpe import BPETokenizer
from tiny_gpt.export import build_fixture
from tiny_gpt.train import parser, train
from tiny_gpt.weights import load_weights

ROOT = Path(__file__).resolve().parents[1]


def tiny_args(tmp_path, seed):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    rng = np.random.default_rng(0)
    docs = [" ".join(rng.choice(words, 400)) for _ in range(30)]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n<<<DOC>>>\n".join(docs) + "\n")
    out = tmp_path / f"run{seed}"
    out.mkdir()
    return parser().parse_args(["--corpus", str(corpus), "--model-dir", str(out), "--results", str(out / "r.json"), "--seed", str(seed),
                                "--vocab", "300", "--context", "16", "--layers", "1", "--heads", "2", "--dim", "16", "--steps", "12",
                                "--batch", "4", "--warmup", "3", "--eval-every", "6", "--eval-windows", "4"])


def test_same_seed_gives_identical_weights_and_a_different_seed_does_not(tmp_path):
    train(tiny_args(tmp_path, 1))
    (tmp_path / "again").mkdir()
    train(tiny_args(tmp_path / "again", 1))
    a = (tmp_path / "run1" / "weights.bin").read_bytes()
    b = (tmp_path / "again" / "run1" / "weights.bin").read_bytes()
    assert a == b
    train(tiny_args(tmp_path, 2))
    assert (tmp_path / "run2" / "weights.bin").read_bytes() != a


def test_training_run_writes_results_with_baselines(tmp_path):
    r = train(tiny_args(tmp_path, 3))
    assert set(r["results"]) == {"transformer", "bigram", "unigram"}
    assert r["model"]["parameters"] > 0 and len(r["curve"]) == 2
    assert json.loads((tmp_path / "run3" / "r.json").read_text())["training"]["seed"] == 3


@pytest.mark.skipif(not (ROOT / "model" / "parity.json").exists(), reason="train the model first")
def test_shipped_parity_fixture_matches_the_shipped_weights():
    shipped = json.loads((ROOT / "model" / "parity.json").read_text())
    fresh = build_fixture(ROOT / "model", ROOT / "data" / "corpus.txt")
    assert fresh["ids"] == shipped["ids"] and fresh["greedy_continuation"] == shipped["greedy_continuation"]
    assert np.allclose(fresh["logits"], shipped["logits"], atol=1e-6)


@pytest.mark.skipif(not (ROOT / "model" / "weights.bin").exists(), reason="train the model first")
def test_shipped_weights_and_tokenizer_belong_together():
    model = load_weights(ROOT / "model" / "weights.bin")
    tok = BPETokenizer.load(ROOT / "model" / "tokenizer.json")
    assert model.config.vocab_size == tok.vocab_size
    assert (ROOT / "model" / "weights.bin").stat().st_size < 5_000_000
