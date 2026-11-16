"""Incremental decoding with a key/value cache, and token sampling."""
from __future__ import annotations

import numpy as np

from .layers import Array, softmax
from .model import GPT, ContextOverflowError


class KVCache:
    def __init__(self, model: GPT):
        c = model.config
        dtype = model.children["emb"].wte.dtype
        self.keys = [np.zeros((c.n_heads, c.context, c.dim // c.n_heads), dtype) for _ in range(c.n_layers)]
        self.values = [np.zeros_like(k) for k in self.keys]
        self.length = 0


def _ln(x: Array, layer) -> Array:
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + layer.eps) * layer.gamma + layer.beta


def step(model: GPT, token: int, cache: KVCache) -> Array:
    """Feed one token at position cache.length and return logits for the next token (V,)."""
    c = model.config
    pos = cache.length
    if pos >= c.context:
        raise ContextOverflowError(f"position {pos} is beyond the trained context of {c.context}")
    emb = model.children["emb"]
    x = emb.wte[token] + emb.wpe[pos]
    hd = c.dim // c.n_heads
    for i, block in enumerate(model.blocks):
        ch = block.children
        h = _ln(x, ch["ln1"])
        qkv = h @ ch["attn"].children["qkv"].w + ch["attn"].children["qkv"].b
        q, k, v = (a.reshape(c.n_heads, hd) for a in np.split(qkv, 3))
        cache.keys[i][:, pos], cache.values[i][:, pos] = k, v
        scores = np.einsum("hd,htd->ht", q, cache.keys[i][:, : pos + 1]) / np.sqrt(hd)
        out = np.einsum("ht,htd->hd", softmax(scores), cache.values[i][:, : pos + 1]).reshape(-1)
        x = x + out @ ch["attn"].children["proj"].w + ch["attn"].children["proj"].b
        h = _ln(x, ch["ln2"])
        mlp = ch["mlp"]
        a = h @ mlp.children["fc"].w + mlp.children["fc"].b
        a = mlp.act.forward(a)
        x = x + a @ mlp.children["proj"].w + mlp.children["proj"].b
    cache.length += 1
    return _ln(x, model.children["ln_f"]) @ emb.wte.T


def sample_logits(logits: Array, temperature: float, top_k: int, rng: np.random.Generator) -> int:
    """Greedy when temperature is 0; otherwise sample from the top_k most likely tokens (top_k 0 keeps all)."""
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    if 0 < top_k < z.size:
        cut = np.partition(z, -top_k)[-top_k]
        z = np.where(z >= cut, z, -np.inf)
    p = softmax(z)
    return int(rng.choice(z.size, p=p))


def generate(model: GPT, prompt_ids: list[int], max_new: int, temperature: float = 1.0, top_k: int = 0, seed: int = 0) -> list[int]:
    """Generate up to max_new tokens. Stops (without error) when the context is full."""
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    rng = np.random.default_rng(seed)
    cache = KVCache(model)
    logits = None
    for t in prompt_ids:
        logits = step(model, t, cache)
    out: list[int] = []
    for _ in range(max_new):
        if cache.length >= model.config.context:
            break
        nxt = sample_logits(logits, temperature, top_k, rng)
        out.append(nxt)
        logits = step(model, nxt, cache)
    return out
