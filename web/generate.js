// Text generation on top of the KV-cache forward pass, with explicit handling of the context limit.
import { ContextOverflowError, KVCache } from "./model.js";
import { mulberry32, probabilities, sampleToken } from "./sampling.js";

/**
 * What to do when the sequence reaches the trained context length:
 *  refuse     throw ContextOverflowError (nothing past the limit is ever computed)
 *  stop       stop generating quietly and report that the context is full
 *  slide      drop the oldest half of the window, re-read the rest, and keep going
 *  unguarded  the broken baseline: keep the cache growing and reuse the last position embedding
 */
export const OVERFLOW_POLICIES = ["refuse", "stop", "slide", "unguarded"];

/** Most likely n tokens of a probability vector as [id, p] pairs. */
export function topTokens(probs, n = 12) {
  return Array.from(probs, (p, id) => [id, p]).sort((a, b) => b[1] - a[1]).slice(0, n);
}

/** Probability the model gave each token given the tokens before it (position 0 has no context, so null). */
export function scoreSequence(model, ids) {
  const { context, vocab_size: V } = model.config;
  const start = Math.max(0, ids.length - context);
  const window = ids.slice(start);
  const { logits, attention } = model.forward(window, { attention: true });
  const records = window.map((id, t) => {
    if (t === 0) return { id, prob: null, probs: null };
    const p = probabilities(logits.subarray((t - 1) * V, t * V));
    return { id, prob: p[id], probs: p };
  });
  return { start, records, attention };
}

/**
 * @param {import("./model.js").GPT} model
 * @param {{promptIds: number[], maxNew: number, temperature: number, topK: number, seed: number, policy: string}} opts
 * @returns {{generated: {id: number, prob: number, probs: Float64Array}[], events: {type: string, at: number}[], stopped: string | null}}
 */
export function generate(model, { promptIds, maxNew, temperature, topK, seed, policy }) {
  if (!OVERFLOW_POLICIES.includes(policy)) throw new Error(`unknown overflow policy ${policy}`);
  if (promptIds.length === 0) throw new Error("prompt must contain at least one token");
  const { context } = model.config;
  const rng = mulberry32(seed);
  const events = [];
  let window = promptIds.slice();
  if (window.length > context) {
    if (policy === "refuse") throw new ContextOverflowError(`prompt of ${window.length} tokens exceeds the trained context of ${context}`);
    if (policy === "stop") {
      return { generated: [], events: [{ type: "prompt-too-long", at: 0 }], stopped: "context" };
    }
    if (policy === "slide") {
      window = window.slice(window.length - (context - 1));
      events.push({ type: "prompt-truncated", at: 0 });
    }
  }
  const unbounded = policy === "unguarded";
  let cache = new KVCache(model, unbounded);
  let logits = null;
  for (const id of window) logits = model.step(id, cache);
  const generated = [];
  for (let n = 0; n < maxNew; n++) {
    if (cache.length >= context && !unbounded) {
      if (policy === "refuse") {
        const err = new ContextOverflowError(`context of ${context} tokens is full after ${generated.length} generated tokens`);
        err.partial = { generated, events };
        throw err;
      }
      if (policy === "stop") return { generated, events, stopped: "context" };
      window = window.slice(window.length - context / 2);
      cache = new KVCache(model, false);
      for (const id of window) logits = model.step(id, cache);
      events.push({ type: "slide", at: generated.length });
    }
    const id = sampleToken(logits, temperature, topK, rng);
    const p = probabilities(logits);
    generated.push({ id, prob: p[id], probs: p });
    window.push(id);
    logits = model.step(id, cache);
  }
  return { generated, events, stopped: null };
}

/**
 * Teacher-forced loss of a known text under an overflow policy: nll[i] is the cost (in nats) of token i+1 given the
 * tokens the policy lets the model see. Returns how many tokens were scored and, for "refuse", the error that stopped it.
 */
export function scoreWithPolicy(model, ids, policy) {
  if (!OVERFLOW_POLICIES.includes(policy)) throw new Error(`unknown overflow policy ${policy}`);
  const { context } = model.config;
  const unbounded = policy === "unguarded";
  let cache = new KVCache(model, unbounded);
  let window = [];
  const nll = [];
  let logits = null;
  for (let i = 0; i < ids.length; i++) {
    if (logits) nll.push(-Math.log(probabilities(logits)[ids[i]]));
    if (cache.length >= context && !unbounded) {
      if (policy === "refuse") return { nll, error: new ContextOverflowError(`context of ${context} tokens is full at token ${i}`) };
      if (policy === "stop") return { nll, error: null, stoppedAt: i };
      window = window.slice(window.length - context / 2);
      cache = new KVCache(model, false);
      for (const id of window) logits = model.step(id, cache);
    }
    window.push(ids[i]);
    logits = model.step(ids[i], cache);
  }
  return { nll, error: null };
}
