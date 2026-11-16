"""The decoder-only transformer: embeddings, N pre-LN blocks, final LayerNorm, tied head."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .layers import Array, Block, Embedding, Layer, LayerNorm, cross_entropy


class ContextOverflowError(ValueError):
    """Raised when a sequence is longer than the context the model was trained for."""


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 512
    context: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


class GPT(Layer):
    def __init__(self, config: GPTConfig, seed: int = 0, dtype=np.float32):
        super().__init__()
        self.config = config
        rng = np.random.default_rng(seed)
        c = config
        self.children["emb"] = Embedding(c.vocab_size, c.context, c.dim, rng, dtype)
        for i in range(c.n_layers):
            self.children[f"block{i}"] = Block(c.dim, c.n_heads, c.n_layers, rng, dtype)
        self.children["ln_f"] = LayerNorm(c.dim, dtype)
        self._xf: Array | None = None
        self._dlogits: Array | None = None

    @property
    def blocks(self) -> list[Block]:
        return [self.children[f"block{i}"] for i in range(self.config.n_layers)]

    def n_parameters(self) -> int:
        return sum(p.size for _, p, _ in self.named_parameters())

    def forward(self, ids: Array, targets: Array | None = None) -> tuple[Array, float | None]:
        """Return logits (B, T, V) and, when targets are given, the mean loss."""
        if ids.shape[1] > self.config.context:
            raise ContextOverflowError(f"sequence of {ids.shape[1]} tokens exceeds the trained context of {self.config.context}")
        x = self.children["emb"].forward(ids)
        for block in self.blocks:
            x = block.forward(x)
        self._xf = self.children["ln_f"].forward(x)
        logits = self._xf @ self.children["emb"].wte.T  # tied output head
        if targets is None:
            return logits, None
        loss, self._dlogits = cross_entropy(logits, targets)
        return logits, loss

    def backward(self) -> None:
        """Backpropagate the loss from the last forward(ids, targets) call."""
        if self._dlogits is None:
            raise RuntimeError("call forward with targets before backward")
        emb = self.children["emb"]
        d_logits = self._dlogits
        emb.grads["wte"] += d_logits.reshape(-1, d_logits.shape[-1]).T @ self._xf.reshape(-1, self._xf.shape[-1])
        dx = self.children["ln_f"].backward(d_logits @ emb.wte)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        emb.backward(dx)

    def attention_maps(self) -> list[Array]:
        """Attention weights (B, H, T, T) per layer from the last forward pass."""
        return [b.children["attn"].att for b in self.blocks]

    def state(self) -> dict[str, Array]:
        return {name: p for name, p, _ in self.named_parameters()}

    def load_state(self, state: dict[str, Array]) -> None:
        own = self.state()
        if own.keys() != state.keys():
            raise KeyError("parameter names do not match the model")
        for name, p in own.items():
            if p.shape != state[name].shape:
                raise ValueError(f"shape mismatch for {name}: {p.shape} vs {state[name].shape}")
            p[...] = state[name]
