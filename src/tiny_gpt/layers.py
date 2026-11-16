"""Layers with hand-written forward and backward passes.

Every layer caches what its backward pass needs during forward(). backward(dy)
adds parameter gradients into the `grads` dict (call zero_grad first) and
returns the gradient with respect to its input. Nothing here uses autograd.
"""
from __future__ import annotations

import math
from typing import Iterator

import numpy as np

Array = np.ndarray
NEG_INF = -1e30
_GELU_C = math.sqrt(2.0 / math.pi)


class Layer:
    """Base class: owns named parameters and matching gradient buffers."""

    def __init__(self) -> None:
        self.params: dict[str, Array] = {}
        self.grads: dict[str, Array] = {}
        self.children: dict[str, "Layer"] = {}

    def _add(self, name: str, value: Array) -> Array:
        self.params[name] = value
        self.grads[name] = np.zeros_like(value)
        return value

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Array, Array]]:
        for name, p in self.params.items():
            yield prefix + name, p, self.grads[name]
        for cname, child in self.children.items():
            yield from child.named_parameters(f"{prefix}{cname}.")

    def zero_grad(self) -> None:
        for g in self.grads.values():
            g.fill(0.0)
        for child in self.children.values():
            child.zero_grad()


class Linear(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, std: float = 0.02, dtype=np.float32):
        super().__init__()
        self.w = self._add("w", (rng.standard_normal((n_in, n_out)) * std).astype(dtype))
        self.b = self._add("b", np.zeros(n_out, dtype=dtype))
        self._x: Array | None = None

    def forward(self, x: Array) -> Array:
        self._x = x
        return x @ self.w + self.b

    def backward(self, dy: Array) -> Array:
        x2 = self._x.reshape(-1, self._x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.grads["w"] += x2.T @ dy2
        self.grads["b"] += dy2.sum(axis=0)
        return dy @ self.w.T


class LayerNorm(Layer):
    def __init__(self, dim: int, dtype=np.float32, eps: float = 1e-5):
        super().__init__()
        self.gamma = self._add("gamma", np.ones(dim, dtype=dtype))
        self.beta = self._add("beta", np.zeros(dim, dtype=dtype))
        self.eps = eps
        self._cache: tuple[Array, Array] | None = None

    def forward(self, x: Array) -> Array:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        rstd = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + self.eps)
        xhat = xc * rstd
        self._cache = (xhat, rstd)
        return xhat * self.gamma + self.beta

    def backward(self, dy: Array) -> Array:
        xhat, rstd = self._cache
        self.grads["gamma"] += (dy * xhat).reshape(-1, xhat.shape[-1]).sum(axis=0)
        self.grads["beta"] += dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        dxhat = dy * self.gamma
        return rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))


class GELU(Layer):
    """tanh approximation of GELU."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: tuple[Array, Array] | None = None

    def forward(self, x: Array) -> Array:
        t = np.tanh(_GELU_C * (x + 0.044715 * x ** 3))
        self._cache = (x, t)
        return 0.5 * x * (1.0 + t)

    def backward(self, dy: Array) -> Array:
        x, t = self._cache
        dt = (1.0 - t * t) * _GELU_C * (1.0 + 3 * 0.044715 * x * x)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * dt)


class MLP(Layer):
    def __init__(self, dim: int, rng: np.random.Generator, proj_std: float, dtype=np.float32):
        super().__init__()
        self.children = {"fc": Linear(dim, 4 * dim, rng, dtype=dtype), "proj": Linear(4 * dim, dim, rng, std=proj_std, dtype=dtype)}
        self.act = GELU()

    def forward(self, x: Array) -> Array:
        return self.children["proj"].forward(self.act.forward(self.children["fc"].forward(x)))

    def backward(self, dy: Array) -> Array:
        return self.children["fc"].backward(self.act.backward(self.children["proj"].backward(dy)))


def softmax(x: Array) -> Array:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class CausalSelfAttention(Layer):
    """Multi-head self-attention where position t may only look at positions <= t."""

    def __init__(self, dim: int, n_heads: int, rng: np.random.Generator, proj_std: float, dtype=np.float32):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
        self.children = {"qkv": Linear(dim, 3 * dim, rng, dtype=dtype), "proj": Linear(dim, dim, rng, std=proj_std, dtype=dtype)}
        self.att: Array | None = None  # (B, H, T, T) attention weights of the last forward pass
        self._cache: tuple[Array, Array, Array] | None = None

    def _split(self, a: Array) -> Array:
        b, t, _ = a.shape
        return a.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Array) -> Array:
        b, t, d = x.shape
        q, k, v = (self._split(a) for a in np.split(self.children["qkv"].forward(x), 3, axis=-1))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))
        causal = np.tril(np.ones((t, t), dtype=bool))
        scores = np.where(causal, scores, np.asarray(NEG_INF, dtype=scores.dtype))
        att = softmax(scores)
        self.att, self._cache = att, (q, k, v)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.children["proj"].forward(out)

    def backward(self, dy: Array) -> Array:
        q, k, v = self._cache
        att = self.att
        b, t, d = dy.shape
        d_out = self._split(self.children["proj"].backward(dy))
        d_att = d_out @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ d_out
        d_scores = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True)) * (1.0 / math.sqrt(self.head_dim))
        dq = d_scores @ k
        dk = d_scores.transpose(0, 1, 3, 2) @ q
        merge = lambda a: a.transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.children["qkv"].backward(np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1))


class Block(Layer):
    """Pre-LayerNorm transformer block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, dim: int, n_heads: int, n_layers: int, rng: np.random.Generator, dtype=np.float32):
        super().__init__()
        proj_std = 0.02 / math.sqrt(2 * n_layers)
        self.children = {
            "ln1": LayerNorm(dim, dtype), "attn": CausalSelfAttention(dim, n_heads, rng, proj_std, dtype),
            "ln2": LayerNorm(dim, dtype), "mlp": MLP(dim, rng, proj_std, dtype),
        }

    def forward(self, x: Array) -> Array:
        c = self.children
        x = x + c["attn"].forward(c["ln1"].forward(x))
        return x + c["mlp"].forward(c["ln2"].forward(x))

    def backward(self, dy: Array) -> Array:
        c = self.children
        dy = dy + c["ln2"].backward(c["mlp"].backward(dy))
        return dy + c["ln1"].backward(c["attn"].backward(dy))


class Embedding(Layer):
    """Token embedding plus learned absolute position embedding."""

    def __init__(self, vocab: int, context: int, dim: int, rng: np.random.Generator, dtype=np.float32):
        super().__init__()
        self.wte = self._add("wte", (rng.standard_normal((vocab, dim)) * 0.02).astype(dtype))
        self.wpe = self._add("wpe", (rng.standard_normal((context, dim)) * 0.02).astype(dtype))
        self._ids: Array | None = None

    def forward(self, ids: Array) -> Array:
        self._ids = ids
        return self.wte[ids] + self.wpe[: ids.shape[1]]

    def backward(self, dx: Array) -> None:
        np.add.at(self.grads["wte"], self._ids, dx)
        self.grads["wpe"][: dx.shape[1]] += dx.sum(axis=0)


def cross_entropy(logits: Array, targets: Array) -> tuple[float, Array]:
    """Mean cross-entropy over all positions and its gradient with respect to logits."""
    flat = logits.reshape(-1, logits.shape[-1])
    tgt = targets.reshape(-1)
    shifted = flat - flat.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=-1))
    n = tgt.shape[0]
    loss = float((logsum - shifted[np.arange(n), tgt]).mean())
    d = np.exp(shifted - logsum[:, None])
    d[np.arange(n), tgt] -= 1.0
    return loss, (d / n).reshape(logits.shape)
