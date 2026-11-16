import { test } from "node:test";
import assert from "node:assert/strict";
import { KVCache, ContextOverflowError } from "../web/model.js";
import { generate, scoreSequence, scoreWithPolicy } from "../web/generate.js";
import { mulberry32, probabilities, sampleToken, samplingDistribution } from "../web/sampling.js";
import { parseWeights, WeightFileError } from "../web/weights.js";
import { loadAll, maxAbsDiff } from "./helpers.js";

const { model, tok, fixture, weights } = loadAll();
const V = model.config.vocab_size;
const ctx = model.config.context;

test("JS logits match the Python logits at every position to 1e-4", () => {
  const { logits } = model.forward(fixture.ids);
  let worst = 0;
  fixture.logits.forEach((row, t) => (worst = Math.max(worst, maxAbsDiff(logits.subarray(t * V, (t + 1) * V), row))));
  assert.ok(worst < 1e-4, `max difference ${worst}`);
});

test("JS attention weights match the Python attention weights", () => {
  const { attention } = model.forward(fixture.ids, { attention: true });
  const T = fixture.ids.length;
  const last = attention.length - 1;
  const h = Math.min(3, model.config.n_heads - 1);
  for (let t = 0; t < T; t++) {
    assert.ok(maxAbsDiff(attention[0][0].subarray(t * T, (t + 1) * T), fixture.attention_layer0_head0[t]) < 1e-4);
    assert.ok(maxAbsDiff(attention[last][h].subarray(t * T, (t + 1) * T), fixture.attention_last_layer_head3[t]) < 1e-4);
  }
});

test("greedy generation in JS produces the same tokens as Python", () => {
  const out = generate(model, { promptIds: fixture.ids, maxNew: fixture.greedy_continuation.length, temperature: 0, topK: 0, seed: 1, policy: "refuse" });
  assert.deepEqual(out.generated.map((g) => g.id), fixture.greedy_continuation);
});

test("the KV cache gives the same logits as the full forward pass", () => {
  const { logits } = model.forward(fixture.ids);
  const cache = new KVCache(model);
  fixture.ids.forEach((id, t) => {
    const step = model.step(id, cache);
    assert.ok(maxAbsDiff(step, logits.subarray(t * V, (t + 1) * V)) < 1e-9);
  });
});

test("attention rows sum to one and nothing attends to the future", () => {
  const T = fixture.ids.length;
  for (const layer of model.forward(fixture.ids, { attention: true }).attention) {
    for (const att of layer) {
      for (let t = 0; t < T; t++) {
        let sum = 0;
        for (let s = 0; s < T; s++) {
          if (s > t) assert.equal(att[t * T + s], 0);
          sum += att[t * T + s];
        }
        assert.ok(Math.abs(sum - 1) < 1e-9);
      }
    }
  }
});

test("changing a later token never changes earlier logits", () => {
  const a = model.forward(fixture.ids).logits;
  const changed = fixture.ids.slice();
  const cut = Math.floor(changed.length / 2);
  changed[cut] = (changed[cut] + 7) % 256;
  const b = model.forward(changed).logits;
  assert.equal(maxAbsDiff(a.subarray(0, cut * V), b.subarray(0, cut * V)), 0);
  assert.ok(maxAbsDiff(a.subarray(cut * V, (cut + 1) * V), b.subarray(cut * V, (cut + 1) * V)) > 0);
});

test("a sequence longer than the trained context is refused", () => {
  assert.throws(() => model.forward(new Array(ctx + 1).fill(1)), ContextOverflowError);
  const cache = new KVCache(model);
  for (let i = 0; i < ctx; i++) model.step(1, cache);
  assert.throws(() => model.step(1, cache), ContextOverflowError);
});

const base = { temperature: 0, topK: 0, seed: 3 };

test("refuse policy throws at the context limit and keeps the tokens made so far", () => {
  const promptIds = tok.encode("The ");
  assert.throws(() => generate(model, { ...base, promptIds, maxNew: ctx + 10, policy: "refuse" }), (err) => {
    assert.ok(err instanceof ContextOverflowError);
    assert.equal(err.partial.generated.length, ctx - promptIds.length);
    return true;
  });
});

test("stop policy ends exactly at the context limit without an error", () => {
  const promptIds = tok.encode("The ");
  const out = generate(model, { ...base, promptIds, maxNew: ctx + 10, policy: "stop" });
  assert.equal(out.generated.length, ctx - promptIds.length);
  assert.equal(out.stopped, "context");
});

test("slide policy keeps generating past the context and records where it slid", () => {
  const out = generate(model, { ...base, promptIds: tok.encode("The "), maxNew: ctx + 30, policy: "slide" });
  assert.equal(out.generated.length, ctx + 30);
  assert.ok(out.events.some((e) => e.type === "slide"));
});

test("an over-long prompt is refused, rejected or truncated depending on the policy", () => {
  const promptIds = new Array(ctx + 20).fill(65);
  assert.throws(() => generate(model, { ...base, promptIds, maxNew: 3, policy: "refuse" }), ContextOverflowError);
  assert.equal(generate(model, { ...base, promptIds, maxNew: 3, policy: "stop" }).generated.length, 0);
  const slid = generate(model, { ...base, promptIds, maxNew: 3, policy: "slide" });
  assert.equal(slid.generated.length, 3);
  assert.equal(slid.events[0].type, "prompt-truncated");
});

test("the unguarded policy runs past the context only because the guard was switched off", () => {
  const out = generate(model, { ...base, promptIds: tok.encode("The "), maxNew: ctx + 20, policy: "unguarded" });
  assert.equal(out.generated.length, ctx + 20);
});

test("the same seed gives the same text and a different seed usually does not", () => {
  const run = (seed) => generate(model, { promptIds: tok.encode("The "), maxNew: 30, temperature: 1, topK: 0, seed, policy: "stop" }).generated.map((g) => g.id).join();
  assert.equal(run(5), run(5));
  assert.notEqual(run(5), run(6));
});

test("temperature 0 always picks the most likely token", () => {
  const { logits } = model.forward(fixture.ids);
  const last = logits.subarray((fixture.ids.length - 1) * V);
  const best = last.indexOf(Math.max(...last));
  for (let s = 0; s < 5; s++) assert.equal(sampleToken(last, 0, 0, mulberry32(s)), best);
});

test("top-k never samples outside the k most likely tokens", () => {
  const logits = Float64Array.from([5, 4, 3, -2, -3, -4]);
  const rng = mulberry32(1);
  for (let i = 0; i < 300; i++) assert.ok(sampleToken(logits, 5, 2, rng) < 2);
  const p = samplingDistribution(logits, 1, 3);
  assert.equal(p.filter((x) => x > 0).length, 3);
  assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-12);
});

test("higher temperature spreads probability mass out", () => {
  const logits = Float64Array.from({ length: 20 }, (_, i) => i * 0.3);
  const peak = (t) => Math.max(...samplingDistribution(logits, t, 0));
  assert.ok(peak(0.5) > peak(1) && peak(1) > peak(2));
});

test("scoring a sequence gives each token a probability and leaves the first one unscored", () => {
  const { records } = scoreSequence(model, fixture.ids);
  assert.equal(records[0].prob, null);
  assert.ok(records.slice(1).every((r) => r.prob > 0 && r.prob <= 1));
  assert.ok(Math.abs(probabilities(model.forward(fixture.ids).logits.subarray(0, V)).reduce((a, b) => a + b) - 1) < 1e-12);
});

test("a damaged weight file is rejected", () => {
  assert.throws(() => parseWeights(new ArrayBuffer(16)), WeightFileError);
  assert.ok(weights.tensors.get("emb.wte").length === V * model.config.dim);
});

test("scoring a held-out passage gives identical losses under every policy until the context fills", () => {
  const ids = fixture.heldout.ids.slice(0, ctx + 40);
  const results = ["refuse", "stop", "slide", "unguarded"].map((p) => scoreWithPolicy(model, ids, p));
  const head = results[2].nll.slice(0, ctx - 1);
  for (const r of results) assert.deepEqual(r.nll.slice(0, ctx - 1), head);
  assert.ok(results[0].error instanceof ContextOverflowError);
  assert.equal(results[2].nll.length, ids.length - 1);
  assert.equal(results[3].nll.length, ids.length - 1);
});
