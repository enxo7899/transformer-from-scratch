# Design notes

## The pieces

```
 data/corpus.txt ──► split by document ──► BPE (train docs only) ──► token ids
                                                                       │
        ┌──────────────────────────────────────────────────────────────┘
        ▼
 tiny_gpt (numpy, float32)                                    tests (float64)
 ┌───────────────────────────────────────────┐                finite-difference
 │ Embedding (tokens + learned positions)    │   backward()   check of every
 │ Block x N:  x + Attn(LN(x)); x + MLP(LN(x))│ ◄──────────── layer's gradient
 │ LayerNorm ─► tied head (x · Eᵀ) ─► loss   │
 └──────────┬────────────────────────────────┘
            │ AdamW + cosine LR + clipping
            ▼
   model/weights.bin  model/tokenizer.json  docs/results.json  model/parity.json
            │                    │                 │                   │
            ▼                    ▼                 ▼                   ▼
   web/weights.js          web/bpe.js        charts on the page   node --test and
   web/model.js (same maths, KV cache)                            the page's own
   web/generate.js, sampling.js                                   parity check
```

## Decisions and trade-offs

**Hand-written backward pass.** Each layer in `layers.py` caches what it needs in `forward` and implements `backward` explicitly. There is no autograd and no graph. The price is that every new layer needs its own derivation; the check is `tests/test_gradients.py`, which compares analytic gradients with central finite differences in float64 for the input and every parameter of every layer, then for the whole model including the tied head (where the embedding matrix gets gradient from two places).

**Float32 to train, float64 to check.** Layers take their dtype from their parameters. Training runs in float32 (about twice as fast on CPU); the gradient tests build the same classes in float64 so finite differences are meaningful. The browser keeps the float32 weights and does its arithmetic in float64, so it matches Python's float64 evaluation of the same weights to well under 1e-4.

**Pre-LayerNorm, GELU, tied head.** Pre-LN trains stably without long warmup at this size. Tying the output matrix to the token embedding saves 65k parameters, a large share of a model this small. Residual output projections are initialised with a 1/sqrt(2N) scale.

**Learned absolute positions, so a hard context limit.** The model has an embedding for positions 0 to 127 and nothing beyond. Both the Python and JavaScript forward passes raise `ContextOverflowError` instead of silently doing something else. The demo then shows four policies (refuse, stop, slide, unguarded) and scores the same held-out passage under each. Rotary or relative positions would soften the limit but not remove it.

**Byte-level BPE, 512 tokens.** Bytes make every string encodable. Merges never cross a pre-token boundary (word, number, punctuation run, whitespace run), which keeps training fast (it works on a table of unique words) and makes the JavaScript port short. The pre-token regex uses only ASCII classes so Python's `re` and JavaScript's `RegExp` agree. Parity is enforced by a fixture of strings and ids exported from Python.

**Weight file.** `TGPT`, a length-prefixed JSON header (config plus tensor names, shapes, offsets) and raw little-endian float32. No pickle, no dependency, 3.5 MB.

**Baselines on the same targets.** Unigram and bigram models are fitted on the same training tokens and scored on exactly the same validation windows and targets as the transformer. The bigram's smoothing strength is chosen on the last 10% of the training tokens, never on validation.

**The page runs the shipped code.** `index.html` imports `web/*.js`, the same modules the node tests import. Nothing on the page is a mock-up, and every number is read from `docs/results.json` or computed live.

## What I would do next for production

- Train on a far larger and more varied corpus with a held-out set drawn from a different source, so validation measures generalisation and not just unseen documents of the same kind.
- Rotary position embeddings and a longer context; a fused attention kernel or a compiled backend instead of numpy for the training loop.
- Mixed precision and gradient accumulation; checkpointing and resumable runs.
- A real inference path in the browser using typed-array matmul in a worker or WebAssembly; the current one is clear, not fast.
- Evaluate beyond perplexity (for example sampled-text review and task probes) before drawing conclusions about quality.
