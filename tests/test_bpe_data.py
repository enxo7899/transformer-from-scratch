"""Tokenizer, corpus split and baselines."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tiny_gpt.baselines import Bigram, Unigram
from tiny_gpt.bpe import BPEError, BPETokenizer, pretokenize
from tiny_gpt.data import fixed_windows, load_documents, split_documents

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "The quick brown fox jumps over the lazy dog. " * 30 + "def add(a, b):\n    return a + b  # 12 + 30\n" * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(SAMPLE, 300)


def test_bpe_round_trips_plain_text_code_and_whitespace(tok):
    for text in [SAMPLE, "", " ", "   leading and trailing   ", "a\n\n\nb\t c", "x=1;y=2"]:
        assert tok.decode(tok.encode(text)) == text


def test_bpe_round_trips_non_ascii_through_raw_bytes(tok):
    text = "café → 日本語 \U0001f600"
    assert tok.decode(tok.encode(text)) == text


def test_bpe_round_trip_holds_for_random_strings(tok):
    rng = np.random.default_rng(0)
    alphabet = list("abcde fgh\n\t.,;:()0123456789THE") + ["é", "中"]
    for _ in range(200):
        text = "".join(rng.choice(alphabet, size=int(rng.integers(0, 40))))
        assert tok.decode(tok.encode(text)) == text


def test_bpe_compresses_text_it_was_trained_on(tok):
    assert len(tok.encode(SAMPLE)) < 0.5 * len(SAMPLE.encode())


def test_bpe_training_is_deterministic():
    assert BPETokenizer.train(SAMPLE, 300).merges == BPETokenizer.train(SAMPLE, 300).merges


def test_bpe_vocab_layout_and_unknown_ids(tok):
    assert tok.vocab_size == 300 and tok.eot_id == 299
    assert tok.decode([tok.eot_id]) == "<|endoftext|>"
    with pytest.raises(BPEError):
        tok.decode([300])


def test_bpe_merges_never_cross_a_word_boundary(tok):
    for piece in tok.vocab[256:]:
        text = piece.decode("utf-8", errors="replace")
        assert len(pretokenize(text)) == 1 or text.strip() == ""


def test_bpe_saved_file_loads_to_the_same_tokenizer(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    assert BPETokenizer.load(tmp_path / "t.json").encode(SAMPLE) == tok.encode(SAMPLE)


def test_split_by_document_keeps_every_document_on_one_side():
    docs = [f"document number {i} " * (5 + i) for i in range(40)]
    train, val = split_documents(docs, 0.2, seed=1)
    assert set(train).isdisjoint(val) and sorted(train + val) == sorted(docs)
    assert sum(map(len, val)) >= 0.2 * sum(map(len, docs))


def test_shipped_corpus_is_ascii_and_split_has_no_shared_documents():
    docs = load_documents(ROOT / "data" / "corpus.txt")
    train, val = split_documents(docs)
    assert len(docs) > 300 and set(train).isdisjoint(val) and len(val) > 10


def test_unigram_and_bigram_are_proper_distributions():
    tokens = np.random.default_rng(0).integers(0, 6, 500)
    uni, bi = Unigram(tokens, 6), Bigram(tokens, 6, alpha=1.0)
    assert uni.p.sum() == pytest.approx(1.0)
    for a in range(6):
        assert bi.probs(np.full(6, a), np.arange(6)).sum() == pytest.approx(1.0)


def test_bigram_beats_unigram_on_text_with_real_structure():
    tokens = np.tile(np.arange(8), 200)
    w = fixed_windows(tokens, 16)
    assert Bigram(tokens, 8).nll(w) < 0.1 < Unigram(tokens, 8).nll(w)


def test_results_file_is_consistent_when_present():
    path = ROOT / "docs" / "results.json"
    if not path.exists():
        pytest.skip("run the training script first")
    r = json.loads(path.read_text())
    res = r["results"]
    assert res["transformer"]["val_loss"] < res["bigram"]["val_loss"] < res["unigram"]["val_loss"]
    assert res["transformer"]["perplexity"] == pytest.approx(math.exp(res["transformer"]["val_loss"]), rel=1e-3)
    assert r["model"]["weights_bytes"] < 5_000_000
