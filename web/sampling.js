// Seeded sampling with temperature and top-k.
import { softmaxInPlace } from "./model.js";

/** mulberry32: a small seeded PRNG so a run can be repeated exactly. @returns {() => number} uniform in [0, 1) */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Plain softmax probabilities (temperature 1) of a logits vector. */
export function probabilities(logits) {
  return softmaxInPlace(Float64Array.from(logits));
}

/** The k-th largest value, or -Infinity when k is 0 or covers the whole vocabulary. */
function kthLargest(values, k) {
  if (k <= 0 || k >= values.length) return -Infinity;
  return Float64Array.from(values).sort()[values.length - k];
}

/** Distribution actually sampled from after temperature and top-k. Temperature 0 puts all mass on the argmax. */
export function samplingDistribution(logits, temperature, topK) {
  const n = logits.length;
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    const p = new Float64Array(n);
    p[best] = 1;
    return p;
  }
  const z = Float64Array.from(logits, (v) => v / temperature);
  const cut = kthLargest(z, topK);
  for (let i = 0; i < n; i++) if (z[i] < cut) z[i] = -Infinity;
  return softmaxInPlace(z);
}

/** @param {Float64Array} logits @param {() => number} rng @returns {number} token id */
export function sampleToken(logits, temperature, topK, rng) {
  const p = samplingDistribution(logits, temperature, topK);
  if (temperature <= 0) return p.indexOf(1);
  let u = rng();
  for (let i = 0; i < p.length; i++) {
    u -= p[i];
    if (u < 0) return i;
  }
  return p.length - 1;
}
