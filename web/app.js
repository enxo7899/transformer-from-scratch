// Page wiring: loads the model, runs the real forward pass from web/model.js, and draws the panels.
import { BPETokenizer } from "./bpe.js";
import { parseWeights } from "./weights.js";
import { GPT, ContextOverflowError } from "./model.js";
import { generate, scoreSequence, scoreWithPolicy, OVERFLOW_POLICIES } from "./generate.js";
import { renderChips, markChips } from "./chips.js";
import { AttentionView } from "./attention.js";
import { drawCurve, drawResultsTable, drawDistribution } from "./charts.js";
import { h, pct } from "./dom.js";

const EXAMPLES = [
  ["Python docs", 'The "for" statement is used to iterate over the elements of a'],
  ["A function", "def read_file(path):\n    with open(path) as f:\n        "],
  ["A definition", "A decorator is a function that"],
  ["An error", "Raises a ValueError if"],
];
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const runs = new Set((params.get("run") ?? "").split(",").filter(Boolean));
const wants = (name) => runs.has(name) || runs.has("all");

const state = { tok: null, model: null, fixture: null, results: null, tokens: [], records: [], start: 0, attention: null, layer: 0, head: 0, hover: null, pinned: null };
let attView;

function controls() {
  return { temperature: Number($("temp").value), topK: Number($("topk").value), seed: Number($("seed").value) || 0 };
}

function syncLabels() {
  $("tempv").textContent = Number($("temp").value) === 0 ? "0 (greedy)" : Number($("temp").value).toFixed(2);
  $("topkv").textContent = Number($("topk").value) === 0 ? "off" : $("topk").value;
  $("maxnewv").textContent = $("maxnew").value;
  $("breakNv").textContent = $("breakN").value;
}

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = `status ${kind}`;
}

const tokText = (id) => state.tok.decode([id]);
const idle = () => new Promise((r) => setTimeout(r, 20));

async function load() {
  const [wbuf, tokSpec, fixture, results] = await Promise.all([
    fetch("model/weights.bin").then((r) => r.arrayBuffer()),
    fetch("model/tokenizer.json").then((r) => r.json()),
    fetch("model/parity.json").then((r) => r.json()),
    fetch("docs/results.json").then((r) => r.json()),
  ]);
  state.model = new GPT(parseWeights(wbuf));
  state.tok = new BPETokenizer(tokSpec);
  state.fixture = fixture;
  state.results = results;
}

function showFacts() {
  const r = state.results, m = r.model, t = r.training;
  $("facts").replaceChildren(
    h("b", {}, m.parameters.toLocaleString("en")), " parameters, ", h("b", {}, String(m.n_layers)), " layers, ", h("b", {}, String(m.n_heads)),
    " heads, d=", h("b", {}, String(m.dim)), ", context ", h("b", {}, String(m.context)), ", ", String(m.vocab_size), "-token BPE. Trained in ",
    h("b", {}, `${(t.seconds / 60).toFixed(1)} min`), ` on ${t.hardware.cores} CPU cores, numpy only. Weights ${(m.weights_bytes / 1e6).toFixed(1)} MB.`);
}

function chipTokens() {
  return state.tokens;
}

function applyFocus() {
  const f = state.hover ?? state.pinned;
  const root = $("chipsOut");
  markChips(root, state.hover, state.pinned);
  const q = f == null ? null : f - state.start;
  attView.setFocus(q != null && q >= 0 ? q : null);
  $("attRead").textContent = attView.readout(q != null && q >= 0 ? q : null);
  const rec = f == null ? null : state.records[f];
  const { temperature, topK } = controls();
  if (f == null) $("dist").replaceChildren(h("div", { class: "empty" }, "Hover or click a token to see the model's top guesses at that position."));
  else drawDistribution($("dist"), tokText, rec, temperature, topK);
}

function setHover(i) {
  state.hover = i;
  applyFocus();
}

function pin(i) {
  state.pinned = state.pinned === i ? null : i;
  applyFocus();
}

function drawAttention() {
  if (!state.attention) return;
  const T = state.tokens.length - state.start;
  const labels = state.tokens.slice(state.start).map((t) => t.text);
  attView.set({ att: state.attention[state.layer][state.head], T, labels }, null);
  applyFocus();
}

function buildSegs() {
  const { n_layers: L, n_heads: H } = state.model.config;
  const make = (root, count, key, label) => {
    root.querySelectorAll("button").forEach((b) => b.remove());
    for (let i = 0; i < count; i++) {
      const b = h("button", { class: "small", "aria-pressed": String(state[key] === i), onclick: () => {
        state[key] = i;
        root.querySelectorAll("button").forEach((x, j) => x.setAttribute("aria-pressed", String(j === i)));
        drawAttention();
      } }, `${label}${i + 1}`);
      root.append(b);
    }
  };
  make($("layerSeg"), L, "layer", "L");
  make($("headSeg"), H, "head", "H");
}

function setSeg(key, value) {
  state[key] = value;
  const root = $(key === "layer" ? "layerSeg" : "headSeg");
  root.querySelectorAll("button").forEach((x, j) => x.setAttribute("aria-pressed", String(j === value)));
}

async function runGenerate() {
  const text = $("prompt").value;
  const status = $("genStatus");
  const promptIds = state.tok.encode(text);
  if (!promptIds.length) {
    setStatus(status, "Type a prompt first: the model needs at least one token to continue from.", "err");
    return;
  }
  setStatus(status, "Generating...");
  await idle();
  const t0 = performance.now();
  const { temperature, topK, seed } = controls();
  const out = generate(state.model, { promptIds, maxNew: Number($("maxnew").value), temperature, topK, seed, policy: "stop" });
  const ids = [...promptIds, ...out.generated.map((g) => g.id)];
  const scored = scoreSequence(state.model, ids);
  state.start = scored.start;
  state.records = Array.from({ length: ids.length }, (_, i) => (i >= scored.start ? scored.records[i - scored.start] : { id: ids[i], prob: null, probs: null }));
  state.tokens = ids.map((id, i) => ({ id, text: tokText(id), prob: state.records[i].prob, prompt: i < promptIds.length }));
  state.attention = scored.attention;
  state.hover = null;
  state.pinned = null;
  renderChips($("chipsOut"), chipTokens(), { onHover: setHover, onPin: pin });
  drawAttention();
  const ms = Math.round(performance.now() - t0);
  const mean = out.generated.length ? out.generated.reduce((a, g) => a + g.prob, 0) / out.generated.length : 0;
  let msg = `${out.generated.length} new tokens in ${ms} ms. Mean probability the model gave its own picks: ${pct(mean)}.`;
  if (out.stopped) msg += ` Stopped because the ${state.model.config.context}-token context is full.`;
  setStatus(status, msg, out.stopped ? "err" : "ok");
}

async function runBreak() {
  const status = $("breakStatus");
  const promptIds = state.tok.encode($("prompt").value);
  if (!promptIds.length) return setStatus(status, "Type a prompt first.", "err");
  const policy = $("policy").value;
  const { temperature, topK, seed } = controls();
  const ctx = state.model.config.context;
  setStatus(status, "Running...");
  await idle();
  let generated, events = [], error = null, stopped = null;
  try {
    ({ generated, events, stopped } = generate(state.model, { promptIds, maxNew: Number($("breakN").value), temperature, topK, seed, policy }));
  } catch (err) {
    if (!(err instanceof ContextOverflowError)) throw err;
    error = err;
    generated = err.partial?.generated ?? [];
  }
  const toks = [...promptIds.map((id) => ({ id, text: tokText(id), prob: null, prompt: true })), ...generated.map((g) => ({ id: g.id, text: tokText(g.id), prob: g.prob, prompt: false }))];
  renderChips($("breakOut"), toks, { onHover() {}, onPin() {} }, toks.length > ctx ? ctx : -1);
  const first = generated.slice(0, Math.max(0, ctx - promptIds.length));
  const later = generated.slice(Math.max(0, ctx - promptIds.length));
  const avg = (a) => (a.length ? pct(a.reduce((x, g) => x + g.prob, 0) / a.length) : "none");
  if (error) setStatus(status, `${error.name}: ${error.message}. The ${generated.length} tokens made before the limit are kept; nothing was computed past it.`, "err");
  else if (stopped) setStatus(status, `Stopped at the limit with ${generated.length} new tokens. Nothing past position ${ctx - 1} was computed.`, "err");
  else if (policy === "slide") setStatus(status, `${generated.length} tokens. Forgot the oldest half ${events.filter((e) => e.type === "slide").length} time(s). Mean probability of picks: ${avg(first)} inside the context, ${avg(later)} after it.`, "ok");
  else setStatus(status, `${generated.length} tokens with the guard off. Mean probability of picks: ${avg(first)} inside the context, ${avg(later)} after it. The model never saw positions beyond ${ctx - 1} in training.`, later.length ? "err" : "ok");
}

function runScore() {
  const ids = state.fixture.heldout.ids;
  const ctx = state.model.config.context;
  const mean = (a) => (a.length ? (a.reduce((x, y) => x + y, 0) / a.length).toFixed(2) : "none");
  const rows = OVERFLOW_POLICIES.map((p) => {
    const r = scoreWithPolicy(state.model, ids, p);
    const inside = r.nll.slice(0, ctx), after = r.nll.slice(ctx);
    const outcome = r.error ? `refused at token ${ctx}` : r.stoppedAt ? `stopped at token ${r.stoppedAt}` : "scored all";
    return h("tr", {}, h("td", {}, p), h("td", { class: "num" }, mean(inside)), h("td", { class: "num" }, mean(after)), h("td", {}, outcome));
  });
  $("scoreOut").replaceChildren(
    h("p", { class: "stats" }, `Held-out text, ${ids.length} tokens, loss in nats per token (lower is better).`),
    h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "policy"), h("th", { class: "num" }, `inside the ${ctx} positions`), h("th", { class: "num" }, "past them"), h("th", {}, "outcome"))), h("tbody", {}, ...rows)));
}

function runTemps() {
  const promptIds = state.tok.encode($("prompt").value);
  if (!promptIds.length) return;
  const seed = controls().seed;
  const cols = [[0, 0, "temperature 0 (greedy)"], [2, 0, "temperature 2, no top-k"]].map(([temperature, topK, title]) => {
    const { generated } = generate(state.model, { promptIds, maxNew: 60, temperature, topK, seed, policy: "stop" });
    const box = h("div", { class: "out" });
    renderChips(box, [...promptIds.map((id) => ({ id, text: tokText(id), prob: null, prompt: true })), ...generated.map((g) => ({ id: g.id, text: tokText(g.id), prob: g.prob, prompt: false }))], { onHover() {}, onPin() {} });
    const distinct = new Set(generated.map((g) => g.id)).size;
    const mean = generated.reduce((a, g) => a + g.prob, 0) / Math.max(1, generated.length);
    return h("div", {}, h("h3", {}, title), box, h("p", { class: "stats" }, `${distinct} distinct tokens in ${generated.length}. Mean probability of its own picks: ${pct(mean)}.`));
  });
  $("cmpOut").replaceChildren(...cols);
}

function runParity() {
  const f = state.fixture, V = state.model.config.vocab_size;
  const { logits } = state.model.forward(f.ids);
  let worst = 0;
  f.logits.forEach((row, t) => row.forEach((v, j) => (worst = Math.max(worst, Math.abs(logits[t * V + j] - v)))));
  const greedy = generate(state.model, { promptIds: f.ids, maxNew: f.greedy_continuation.length, temperature: 0, topK: 0, seed: 0, policy: "stop" }).generated.map((g) => g.id);
  const same = greedy.length === f.greedy_continuation.length && greedy.every((id, i) => id === f.greedy_continuation[i]);
  const ok = worst < 1e-4;
  const tokOk = f.tokenizer.every((c) => JSON.stringify(state.tok.encode(c.text)) === JSON.stringify(c.ids));
  const mark = (good) => h("span", { class: good ? "pass" : "fail" }, good ? "PASS" : "FAIL");
  $("parity").replaceChildren(
    h("div", {}, mark(ok), ` logits for a fixed ${f.ids.length}-token prompt, all ${f.ids.length} positions x ${V} tokens: largest difference from Python ${worst.toExponential(1)} (limit 1e-4).`),
    h("div", {}, mark(same), ` greedy continuation of ${greedy.length} tokens identical to Python's.`),
    h("div", {}, mark(tokOk), ` tokenizer: ${f.tokenizer.length} fixed strings give exactly Python's token ids.`));
  return ok && same && tokOk;
}

function wire() {
  for (const id of ["temp", "topk", "maxnew", "breakN"]) $(id).addEventListener("input", () => { syncLabels(); if (id === "temp" || id === "topk") applyFocus(); });
  for (const [label, text] of EXAMPLES) $("examples").append(h("button", { class: "small", onclick: () => { $("prompt").value = text; runGenerate(); } }, label));
  $("go").addEventListener("click", runGenerate);
  $("breakGo").addEventListener("click", runBreak);
  $("scoreGo").addEventListener("click", runScore);
  $("tempGo").addEventListener("click", runTemps);
  for (const id of ["go", "breakGo", "scoreGo", "tempGo"]) $(id).disabled = false;
}

async function init() {
  syncLabels();
  attView = new AttentionView($("attViz"), setHover);
  try {
    await load();
  } catch (err) {
    $("facts").textContent = `Could not load the model files: ${err.message}. Serve this folder over HTTP (npm run dev).`;
    return;
  }
  showFacts();
  buildSegs();
  drawCurve($("curve"), state.results);
  drawResultsTable($("resTable"), $("resNote"), state.results);
  wire();
  const ok = runParity();
  document.documentElement.dataset.parity = ok ? "pass" : "fail";
  if (params.get("prompt")) $("prompt").value = params.get("prompt");
  if (params.get("seed")) $("seed").value = params.get("seed");
  if (params.get("temp")) $("temp").value = params.get("temp");
  syncLabels();
  if (wants("generate") || wants("attention")) {
    await runGenerate();
    if (params.has("layer")) { setSeg("layer", Number(params.get("layer"))); }
    if (params.has("head")) { setSeg("head", Number(params.get("head"))); }
    drawAttention();
    if (wants("attention")) {
      state.pinned = Number(params.get("focus") ?? Math.min(state.tokens.length - 1, state.tokens.findIndex((t) => !t.prompt) + 3));
      applyFocus();
      markChips($("chipsOut"), null, state.pinned);
    }
  }
  if (wants("break")) {
    if (params.has("policy")) $("policy").value = params.get("policy");
    if (params.has("n")) { $("breakN").value = params.get("n"); syncLabels(); }
    await runBreak();
    runScore();
  }
  if (wants("temps")) runTemps();
  document.documentElement.dataset.ready = "1";
}

init();
