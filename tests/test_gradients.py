"""Every layer's hand-written backward pass against central finite differences, in float64."""
import numpy as np
import pytest

from tiny_gpt.layers import GELU, MLP, Block, CausalSelfAttention, Embedding, LayerNorm, Linear, cross_entropy
from tiny_gpt.model import GPT, GPTConfig

EPS = 1e-6
TOL = 1e-6


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Norm-wise relative error, robust to entries whose true gradient is zero (such as the key bias)."""
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12))


def numeric_grad(f, arr: np.ndarray) -> np.ndarray:
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + EPS
        hi = f()
        arr[i] = old - EPS
        lo = f()
        arr[i] = old
        g[i] = (hi - lo) / (2 * EPS)
    return g


def check_layer(layer, x: np.ndarray, rng) -> None:
    """Compare analytic and numeric gradients for the layer input and every parameter."""
    r = rng.standard_normal(layer.forward(x).shape)
    loss = lambda: float((layer.forward(x) * r).sum())
    layer.zero_grad()
    layer.forward(x)
    dx = layer.backward(r)
    assert rel_err(dx, numeric_grad(loss, x)) < TOL, "input gradient"
    for name, p, g in layer.named_parameters():
        assert rel_err(g, numeric_grad(loss, p)) < TOL, f"parameter gradient {name}"


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_linear_gradient_matches_finite_differences(rng):
    check_layer(Linear(5, 4, rng, std=0.5, dtype=np.float64), rng.standard_normal((2, 3, 5)), rng)


def test_layernorm_gradient_matches_finite_differences(rng):
    ln = LayerNorm(6, dtype=np.float64)
    ln.gamma[:] = rng.standard_normal(6)
    ln.beta[:] = rng.standard_normal(6)
    check_layer(ln, rng.standard_normal((2, 3, 6)), rng)


def test_gelu_gradient_matches_finite_differences(rng):
    check_layer(GELU(), rng.standard_normal((3, 4)) * 2, rng)


def test_mlp_gradient_matches_finite_differences(rng):
    check_layer(MLP(4, rng, proj_std=0.3, dtype=np.float64), rng.standard_normal((2, 3, 4)), rng)


def test_causal_attention_gradient_matches_finite_differences(rng):
    attn = CausalSelfAttention(8, 2, rng, proj_std=0.4, dtype=np.float64)
    for _, p, _ in attn.named_parameters():
        p += rng.standard_normal(p.shape) * 0.3
    check_layer(attn, rng.standard_normal((2, 5, 8)), rng)


def test_transformer_block_gradient_matches_finite_differences(rng):
    block = Block(8, 2, 2, rng, np.float64)
    for _, p, _ in block.named_parameters():
        p += rng.standard_normal(p.shape) * 0.1
    check_layer(block, rng.standard_normal((2, 4, 8)), rng)


def test_embedding_gradient_matches_finite_differences(rng):
    emb = Embedding(7, 5, 4, rng, np.float64)
    ids = rng.integers(0, 7, (3, 5))
    r = rng.standard_normal((3, 5, 4))
    loss = lambda: float((emb.forward(ids) * r).sum())
    emb.zero_grad()
    emb.forward(ids)
    emb.backward(r)
    for name, p, g in emb.named_parameters():
        assert rel_err(g, numeric_grad(loss, p)) < TOL, name


def test_cross_entropy_gradient_matches_finite_differences(rng):
    logits = rng.standard_normal((2, 3, 6))
    targets = rng.integers(0, 6, (2, 3))
    _, d = cross_entropy(logits, targets)
    assert rel_err(d, numeric_grad(lambda: cross_entropy(logits, targets)[0], logits)) < TOL


def test_cross_entropy_of_uniform_logits_is_log_vocab():
    loss, _ = cross_entropy(np.zeros((1, 4, 10)), np.array([[1, 2, 3, 4]]))
    assert loss == pytest.approx(np.log(10))


def tiny_model(seed: int = 3) -> GPT:
    model = GPT(GPTConfig(vocab_size=11, context=6, n_layers=2, n_heads=2, dim=8), seed=seed, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _, p, _ in model.named_parameters():
        p += rng.standard_normal(p.shape) * 0.1  # move away from the near-zero init so every path matters
    return model


def test_full_model_gradient_checks_every_parameter_including_tied_head():
    model = tiny_model()
    rng = np.random.default_rng(5)
    ids, targets = rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))
    model.zero_grad()
    model.forward(ids, targets)
    model.backward()
    for name, p, g in model.named_parameters():
        num = numeric_grad(lambda: model.forward(ids, targets)[1], p)
        assert rel_err(g, num) < 1e-5, name


def test_gradients_accumulate_until_zeroed():
    model = tiny_model()
    ids = np.array([[1, 2, 3, 4, 5, 6]])
    model.zero_grad()
    model.forward(ids, ids)
    model.backward()
    once = model.children["emb"].grads["wte"].copy()
    model.forward(ids, ids)
    model.backward()
    assert np.allclose(model.children["emb"].grads["wte"], 2 * once)
    model.zero_grad()
    assert not model.children["emb"].grads["wte"].any()
