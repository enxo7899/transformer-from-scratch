"""Model behaviour: causality, context limit, KV cache, overfitting, weights file."""
import numpy as np
import pytest

from tiny_gpt.infer import KVCache, generate, sample_logits, step
from tiny_gpt.model import GPT, ContextOverflowError, GPTConfig
from tiny_gpt.optim import AdamW, clip_grad_norm, cosine_lr
from tiny_gpt.weights import WeightFileError, load_weights, save_weights

from test_gradients import tiny_model


def test_attention_weights_above_the_diagonal_are_exactly_zero():
    model = tiny_model()
    model.forward(np.random.default_rng(1).integers(0, 11, (2, 6)))
    for att in model.attention_maps():
        assert np.all(np.triu(att, k=1) == 0)
        assert np.allclose(att.sum(axis=-1), 1.0)


def test_changing_a_future_token_never_changes_earlier_logits():
    model = tiny_model()
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 11, (1, 6))
    base, _ = model.forward(ids)
    base = base.copy()
    for pos in range(1, 6):
        changed = ids.copy()
        changed[0, pos] = (changed[0, pos] + 1) % 11
        out, _ = model.forward(changed)
        assert np.array_equal(out[0, :pos], base[0, :pos])
        assert not np.allclose(out[0, pos], base[0, pos])


def test_sequence_longer_than_context_is_refused():
    model = tiny_model()
    with pytest.raises(ContextOverflowError):
        model.forward(np.zeros((1, 7), dtype=np.int64))


def test_kv_cache_step_matches_full_forward_pass():
    model = tiny_model()
    ids = np.random.default_rng(4).integers(0, 11, 6)
    full, _ = model.forward(ids[None])
    cache = KVCache(model)
    for t, tok in enumerate(ids):
        assert np.allclose(step(model, int(tok), cache), full[0, t], atol=1e-10)


def test_kv_cache_refuses_to_go_past_the_context():
    model = tiny_model()
    cache = KVCache(model)
    for _ in range(6):
        step(model, 1, cache)
    with pytest.raises(ContextOverflowError):
        step(model, 1, cache)


def test_generation_stops_at_the_context_limit():
    out = generate(tiny_model(), [1, 2, 3], max_new=50, temperature=1.0, seed=0)
    assert len(out) == 3


def test_temperature_zero_is_greedy_and_seeded_sampling_repeats():
    model = tiny_model()
    assert generate(model, [1], 5, temperature=0) == generate(model, [1], 5, temperature=0, seed=99)
    assert generate(model, [1], 5, temperature=1.5, seed=4) == generate(model, [1], 5, temperature=1.5, seed=4)


def test_top_k_never_samples_outside_the_k_best_tokens():
    logits = np.array([5.0, 4.0, 3.0, -2.0, -3.0, -4.0])
    rng = np.random.default_rng(0)
    picks = {sample_logits(logits, 5.0, 2, rng) for _ in range(300)}
    assert picks <= {0, 1}


def test_loss_drops_when_overfitting_one_tiny_batch():
    model = GPT(GPTConfig(vocab_size=16, context=8, n_layers=2, n_heads=2, dim=16), seed=0)
    opt = AdamW(model.named_parameters(), lr=1e-2)
    ids = np.random.default_rng(0).integers(0, 16, (4, 9))
    x, y = ids[:, :-1], ids[:, 1:]
    losses = []
    for _ in range(60):
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        opt.step()
        losses.append(loss)
    assert losses[0] > np.log(16) - 0.3
    assert losses[-1] < 0.3 * losses[0]


def test_saved_weights_reload_bit_identical(tmp_path):
    model = GPT(GPTConfig(vocab_size=20, context=8, n_layers=2, n_heads=2, dim=16), seed=5)
    save_weights(model, tmp_path / "w.bin")
    loaded = load_weights(tmp_path / "w.bin")
    assert loaded.config == model.config
    for (name, a, _), (_, b, _) in zip(model.named_parameters(), loaded.named_parameters()):
        assert a.tobytes() == b.tobytes(), name
    ids = np.arange(8)[None]
    assert np.array_equal(model.forward(ids)[0], loaded.forward(ids)[0])


def test_weight_file_rejects_garbage_and_truncation(tmp_path):
    (tmp_path / "bad.bin").write_bytes(b"nope")
    with pytest.raises(WeightFileError):
        load_weights(tmp_path / "bad.bin")
    model = GPT(GPTConfig(vocab_size=20, context=8, n_layers=1, n_heads=2, dim=16))
    save_weights(model, tmp_path / "w.bin")
    raw = (tmp_path / "w.bin").read_bytes()
    (tmp_path / "cut.bin").write_bytes(raw[:-100])
    with pytest.raises(WeightFileError):
        load_weights(tmp_path / "cut.bin")


def test_adamw_matches_a_hand_computed_first_step():
    p, g = np.array([[1.0, -2.0]]), np.array([[0.5, 0.25]])
    AdamW([("w", p, g)], lr=0.1, weight_decay=0.1).step()
    expected = np.array([[1.0, -2.0]]) * (1 - 0.1 * 0.1) - 0.1 * np.sign(g) * (np.abs(g) / (np.abs(g) + 1e-8))
    assert np.allclose(p, expected)


def test_adamw_does_not_decay_vectors():
    p, g = np.array([1.0, 1.0]), np.zeros(2)
    AdamW([("b", p, g)], lr=0.1, weight_decay=0.5).step()
    assert np.array_equal(p, [1.0, 1.0])


def test_cosine_schedule_warms_up_then_decays_to_the_floor():
    lrs = [cosine_lr(s, 100, 1.0, 10) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[-1] == pytest.approx(0.1, abs=0.01)


def test_gradient_clipping_caps_the_global_norm_only_when_needed():
    g1, g2 = np.full(4, 3.0), np.full(4, 4.0)
    norm = clip_grad_norm([g1, g2], 1.0)
    assert norm == pytest.approx(10.0)
    assert np.sqrt((g1 ** 2).sum() + (g2 ** 2).sum()) == pytest.approx(1.0, abs=1e-5)
    small = np.full(4, 0.1)
    clip_grad_norm([small], 1.0)
    assert np.array_equal(small, np.full(4, 0.1))
