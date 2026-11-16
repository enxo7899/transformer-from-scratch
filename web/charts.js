import { h, s, plainToken, pct } from "./dom.js";
import { samplingDistribution } from "./sampling.js";

/** Training and validation loss with the two baselines drawn as reference lines. Reads docs/results.json. */
export function drawCurve(root, results) {
  const W = 640, H = 330, m = { l: 44, r: 118, t: 28, b: 36 };
  const curve = results.curve;
  const base = results.results;
  const maxStep = curve[curve.length - 1].step;
  const losses = [...curve.map((c) => c.val_loss), ...curve.map((c) => c.train_loss), base.unigram.val_loss, base.bigram.val_loss];
  const lo = Math.floor(Math.min(...losses) * 2) / 2 - 0.0;
  const hi = Math.ceil(Math.max(...losses) * 2) / 2;
  const X = (v) => m.l + (v / maxStep) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Training and validation loss against two baselines" });
  const text = (x, y, str, o = {}) => {
    const t = s("text", { x, y, "font-size": 11.5, fill: "#4b463d", ...o });
    t.textContent = str;
    return t;
  };
  for (let v = lo; v <= hi + 1e-9; v += 0.5) {
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "#d8cfbc", "stroke-width": 0.8 }));
    svg.append(text(m.l - 6, Y(v) + 4, v.toFixed(1), { "text-anchor": "end" }));
  }
  const xticks = 4;
  for (let i = 0; i <= xticks; i++) {
    const v = Math.round((maxStep * i) / xticks);
    svg.append(text(X(v), H - m.b + 16, String(v), { "text-anchor": "middle" }));
  }
  svg.append(text((m.l + W - m.r) / 2, H - 4, "training step", { "text-anchor": "middle" }));
  svg.append(text(4, 12, "loss (nats per token)"));
  const line = (key, color, width, dash) => s("polyline", { points: curve.map((c) => `${X(c.step).toFixed(1)},${Y(c[key]).toFixed(1)}`).join(" "), fill: "none", stroke: color, "stroke-width": width, "stroke-dasharray": dash ?? "none", "stroke-linejoin": "round" });
  const ref = (v, label) => {
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "#1e1c18", "stroke-width": 1.2, "stroke-dasharray": "5 4" }));
    svg.append(text(W - m.r + 6, Y(v) + 4, `${label} ${v.toFixed(2)}`, { fill: "#1e1c18" }));
  };
  ref(base.unigram.val_loss, "unigram");
  ref(base.bigram.val_loss, "bigram");
  svg.append(line("train_loss", "#9d9482", 1.6));
  svg.append(line("val_loss", "#b5441f", 2.4));
  const last = curve[curve.length - 1];
  svg.append(text(W - m.r + 6, Y(last.val_loss) + 4, `val ${last.val_loss.toFixed(2)}`, { fill: "#b5441f", "font-weight": 700 }));
  svg.append(text(W - m.r + 6, Y(last.train_loss) + 18, `train ${last.train_loss.toFixed(2)}`, { fill: "#6a645a" }));
  root.replaceChildren(svg);
}

export function drawResultsTable(table, note, results) {
  const r = results.results;
  const rows = [["transformer", r.transformer], ["bigram", r.bigram], ["unigram", r.unigram]];
  table.replaceChildren(
    h("thead", {}, h("tr", {}, h("th", {}, "model"), h("th", { class: "num" }, "val loss"), h("th", { class: "num" }, "perplexity"), h("th", { class: "num" }, "bits/byte"))),
    h("tbody", {}, ...rows.map(([name, v]) => h("tr", {}, h("td", {}, name), h("td", { class: "num" }, v.val_loss.toFixed(3)), h("td", { class: "num" }, v.perplexity.toFixed(1)), h("td", { class: "num" }, v.bits_per_byte.toFixed(2))))),
  );
  const d = results.data;
  note.textContent = `Scored on ${d.val_tokens_scored.toLocaleString("en")} tokens from ${d.documents_val} documents the model never trained on. Perplexity is exp(loss) over a vocabulary of ${results.model.vocab_size}. Baselines use the same tokens and the same targets. The curve above is scored on a fixed 48-window subset for speed, so its last point differs slightly from the full-set row.`;
}

/** Top-n bars for one position: the model's own distribution and the one the sampler draws from. */
export function drawDistribution(root, tokText, record, temperature, topK, n = 10) {
  if (!record || !record.probs) {
    root.replaceChildren(h("div", { class: "empty" }, "The first token has no earlier text to predict from. Hover a later token."));
    return;
  }
  const probs = record.probs;
  const logProbs = Float64Array.from(probs, (p) => Math.log(Math.max(p, 1e-300)));
  const sampler = samplingDistribution(logProbs, temperature, topK);
  const order = Array.from(probs, (p, id) => id).sort((a, b) => probs[b] - probs[a]).slice(0, n);
  if (!order.includes(record.id)) order.push(record.id);
  const max = Math.max(probs[order[0]], ...order.map((id) => sampler[id]));
  root.replaceChildren(h("div", { class: "bars" }, ...order.map((id) => {
    const w = (p) => `width:${Math.max(0.5, (p / max) * 100).toFixed(1)}%`;
    return h("div", { class: `bar${id === record.id ? " chosen" : ""}` },
      h("span", { class: "t" }, `${id === record.id ? "▸ " : ""}${plainToken(tokText(id))}`),
      h("span", { class: "track" }, h("i", { style: w(probs[id]) }), h("i", { style: w(sampler[id]) })),
      h("span", { class: "p" }, pct(probs[id])));
  })));
}
