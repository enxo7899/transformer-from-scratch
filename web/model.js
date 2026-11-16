// The transformer forward pass in plain JavaScript: same maths as src/tiny_gpt (pre-LN blocks,
// causal multi-head attention, tanh-GELU MLP, tied output head). Weights are float32, activations float64.

export class ContextOverflowError extends Error {
  constructor(message) {
    super(message);
    this.name = "ContextOverflowError";
  }
}

const GELU_C = Math.sqrt(2 / Math.PI);

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

/** out (rows x nOut) = x (rows x nIn) @ w (nIn x nOut) + b */
function linear(x, rows, nIn, w, b, nOut) {
  const out = new Float64Array(rows * nOut);
  for (let r = 0; r < rows; r++) {
    const o = r * nOut;
    for (let j = 0; j < nOut; j++) out[o + j] = b[j];
    for (let i = 0; i < nIn; i++) {
      const xi = x[r * nIn + i];
      const wi = i * nOut;
      for (let j = 0; j < nOut; j++) out[o + j] += xi * w[wi + j];
    }
  }
  return out;
}

function layerNorm(x, rows, dim, gamma, beta) {
  const out = new Float64Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    let mu = 0;
    for (let i = 0; i < dim; i++) mu += x[r * dim + i];
    mu /= dim;
    let v = 0;
    for (let i = 0; i < dim; i++) v += (x[r * dim + i] - mu) ** 2;
    const rstd = 1 / Math.sqrt(v / dim + 1e-5);
    for (let i = 0; i < dim; i++) out[r * dim + i] = (x[r * dim + i] - mu) * rstd * gamma[i] + beta[i];
  }
  return out;
}

/** In-place softmax over the first n entries of a. */
export function softmaxInPlace(a, n = a.length) {
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (a[i] > max) max = a[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    a[i] = Math.exp(a[i] - max);
    sum += a[i];
  }
  for (let i = 0; i < n; i++) a[i] /= sum;
  return a;
}

export class KVCache {
  /** @param {GPT} model @param {boolean} unbounded allow positions past the trained context (used only to demonstrate the failure) */
  constructor(model, unbounded = false) {
    const { n_layers: layers, dim } = model.config;
    this.length = 0;
    this.unbounded = unbounded;
    this.capacity = unbounded ? 2048 : model.config.context;
    this.keys = Array.from({ length: layers }, () => new Float64Array(this.capacity * dim));
    this.values = Array.from({ length: layers }, () => new Float64Array(this.capacity * dim));
  }
}

export class GPT {
  /** @param {{config: object, tensors: Map<string, Float32Array>}} weights from parseWeights */
  constructor({ config, tensors }) {
    this.config = config;
    this.t = tensors;
  }

  get(name) {
    const w = this.t.get(name);
    if (!w) throw new Error(`missing tensor ${name}`);
    return w;
  }

  /** Output logits for every position. With attention:true also returns attention[layer][head] as T*T row-major weights. */
  forward(ids, { attention = false } = {}) {
    const { context, n_layers: L, n_heads: H, dim: D, vocab_size: V } = this.config;
    const T = ids.length;
    if (T > context) throw new ContextOverflowError(`sequence of ${T} tokens exceeds the trained context of ${context}`);
    const hd = D / H;
    const wte = this.get("emb.wte");
    const wpe = this.get("emb.wpe");
    let x = new Float64Array(T * D);
    for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) x[t * D + i] = wte[ids[t] * D + i] + wpe[t * D + i];
    const attn = [];
    for (let l = 0; l < L; l++) {
      const p = (n) => this.get(`block${l}.${n}`);
      const h = layerNorm(x, T, D, p("ln1.gamma"), p("ln1.beta"));
      const qkv = linear(h, T, D, p("attn.qkv.w"), p("attn.qkv.b"), 3 * D);
      const merged = new Float64Array(T * D);
      const maps = [];
      for (let head = 0; head < H; head++) {
        const att = new Float64Array(T * T);
        const row = new Float64Array(T);
        for (let t = 0; t < T; t++) {
          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let i = 0; i < hd; i++) dot += qkv[t * 3 * D + head * hd + i] * qkv[s * 3 * D + D + head * hd + i];
            row[s] = dot / Math.sqrt(hd);
          }
          softmaxInPlace(row, t + 1);
          for (let s = 0; s <= t; s++) {
            att[t * T + s] = row[s];
            for (let i = 0; i < hd; i++) merged[t * D + head * hd + i] += row[s] * qkv[s * 3 * D + 2 * D + head * hd + i];
          }
        }
        maps.push(att);
      }
      attn.push(maps);
      const proj = linear(merged, T, D, p("attn.proj.w"), p("attn.proj.b"), D);
      for (let i = 0; i < x.length; i++) x[i] += proj[i];
      const h2 = layerNorm(x, T, D, p("ln2.gamma"), p("ln2.beta"));
      const a = linear(h2, T, D, p("mlp.fc.w"), p("mlp.fc.b"), 4 * D);
      for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]);
      const m = linear(a, T, 4 * D, p("mlp.proj.w"), p("mlp.proj.b"), D);
      for (let i = 0; i < x.length; i++) x[i] += m[i];
    }
    const xf = layerNorm(x, T, D, this.get("ln_f.gamma"), this.get("ln_f.beta"));
    const logits = new Float64Array(T * V);
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < V; v++) {
        let s = 0;
        for (let i = 0; i < D; i++) s += xf[t * D + i] * wte[v * D + i];
        logits[t * V + v] = s;
      }
    }
    return attention ? { logits, attention: attn } : { logits };
  }

  /** Feed one token at position cache.length; returns logits for the next token. Pass out = [] to receive per-layer, per-head attention rows. */
  step(token, cache, out = null) {
    const { context, n_layers: L, n_heads: H, dim: D, vocab_size: V } = this.config;
    const pos = cache.length;
    if (pos >= context && !cache.unbounded) throw new ContextOverflowError(`position ${pos} is beyond the trained context of ${context}`);
    if (pos >= cache.capacity) throw new ContextOverflowError("cache is full");
    const hd = D / H;
    const wte = this.get("emb.wte");
    const wpe = this.get("emb.wpe");
    const posRow = Math.min(pos, context - 1); // only reachable in unbounded mode: reuse the last learned position
    let x = new Float64Array(D);
    for (let i = 0; i < D; i++) x[i] = wte[token * D + i] + wpe[posRow * D + i];
    for (let l = 0; l < L; l++) {
      const p = (n) => this.get(`block${l}.${n}`);
      const h = layerNorm(x, 1, D, p("ln1.gamma"), p("ln1.beta"));
      const qkv = linear(h, 1, D, p("attn.qkv.w"), p("attn.qkv.b"), 3 * D);
      cache.keys[l].set(qkv.subarray(D, 2 * D), pos * D);
      cache.values[l].set(qkv.subarray(2 * D, 3 * D), pos * D);
      const merged = new Float64Array(D);
      const rows = [];
      for (let head = 0; head < H; head++) {
        const row = new Float64Array(pos + 1);
        for (let s = 0; s <= pos; s++) {
          let dot = 0;
          for (let i = 0; i < hd; i++) dot += qkv[head * hd + i] * cache.keys[l][s * D + head * hd + i];
          row[s] = dot / Math.sqrt(hd);
        }
        softmaxInPlace(row);
        for (let s = 0; s <= pos; s++) for (let i = 0; i < hd; i++) merged[head * hd + i] += row[s] * cache.values[l][s * D + head * hd + i];
        rows.push(row);
      }
      if (out) out.push(rows);
      const proj = linear(merged, 1, D, p("attn.proj.w"), p("attn.proj.b"), D);
      for (let i = 0; i < D; i++) x[i] += proj[i];
      const h2 = layerNorm(x, 1, D, p("ln2.gamma"), p("ln2.beta"));
      const a = linear(h2, 1, D, p("mlp.fc.w"), p("mlp.fc.b"), 4 * D);
      for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]);
      const m = linear(a, 1, 4 * D, p("mlp.proj.w"), p("mlp.proj.b"), D);
      for (let i = 0; i < D; i++) x[i] += m[i];
    }
    const xf = layerNorm(x, 1, D, this.get("ln_f.gamma"), this.get("ln_f.beta"));
    const logits = new Float64Array(V);
    for (let v = 0; v < V; v++) {
      let s = 0;
      for (let i = 0; i < D; i++) s += xf[i] * wte[v * D + i];
      logits[v] = s;
    }
    cache.length += 1;
    return logits;
  }
}
