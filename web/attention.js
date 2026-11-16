import { h, s, plainToken, pct } from "./dom.js";

const GAP = 6;
const ARC_H = 78;
const LABEL_H = 58;

/** Draws one attention head: arcs from the focused token, then the T x T weight matrix with the causal mask shown. */
export class AttentionView {
  /** @param {HTMLElement} root @param {(i: number|null) => void} onHover */
  constructor(root, onHover) {
    this.root = root;
    this.onHover = onHover;
    this.data = null;
    this.focus = null;
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(root);
  }

  /** @param {{att: Float64Array, T: number, labels: string[], offset: number}|null} data */
  set(data, focus) {
    this.data = data;
    this.focus = focus;
    this.draw();
  }

  setFocus(focus) {
    this.focus = focus;
    this.draw();
  }

  layout() {
    const W = Math.max(280, this.root.clientWidth);
    const { T } = this.data;
    const cell = (W - 2 * GAP) / T;
    return { W, cell, matrixH: cell * T, x: (i) => GAP + (i + 0.5) * cell };
  }

  draw() {
    if (!this.data) return;
    const { att, T, labels } = this.data;
    const { W, cell, matrixH, x } = this.layout();
    const showLabels = cell >= 11;
    const top = ARC_H + (showLabels ? LABEL_H : 8);
    const q = this.focus;
    const svg = this.ensure("svg", `${W}|${top}`, () => s("svg", { viewBox: `0 0 ${W} ${top}`, height: top, role: "img", "aria-label": "Attention arcs from the hovered token" }));
    svg.replaceChildren();
    svg.append(s("line", { x1: GAP, x2: W - GAP, y1: ARC_H, y2: ARC_H, stroke: "#d8cfbc" }));
    for (let i = 0; i < T; i++) svg.append(s("circle", { cx: x(i), cy: ARC_H, r: i === q ? 3.5 : 1.8, fill: i === q ? "#1e1c18" : "#9d9482" }));
    if (q != null) {
      for (let k = 0; k < q; k++) {
        const w = att[q * T + k];
        if (w < 0.03) continue;
        const dx = x(q) - x(k);
        const height = Math.min(ARC_H - 6, 10 + Math.abs(dx) * 0.9);
        svg.append(s("path", { d: `M${x(k)},${ARC_H} C${x(k)},${ARC_H - height} ${x(q)},${ARC_H - height} ${x(q)},${ARC_H}`, fill: "none", stroke: "#b5441f", "stroke-width": (0.8 + 5 * w).toFixed(2), "stroke-opacity": (0.25 + 0.75 * Math.sqrt(w)).toFixed(2), "stroke-linecap": "round" }));
      }
      const self = att[q * T + q];
      if (self >= 0.03) svg.append(s("circle", { cx: x(q), cy: ARC_H - 9, r: 4 + 8 * self, fill: "none", stroke: "#b5441f", "stroke-width": 1.5, "stroke-opacity": 0.8 }));
    }
    if (showLabels) {
      for (let i = 0; i < T; i++) {
        const t = s("text", { transform: `translate(${x(i) + 3.5},${ARC_H + 8}) rotate(90)`, "font-size": 10.5, fill: i === q ? "#b5441f" : "#4b463d", "font-weight": i === q ? 700 : 400 });
        t.textContent = plainToken(labels[i]).slice(0, 8);
        svg.append(t);
      }
    }
    const dpr = window.devicePixelRatio || 1;
    const canvas = this.ensure("canvas", `${W}|${T}|${dpr}`, () => {
      const c = h("canvas", { "aria-label": "Attention weight matrix" });
      c.width = Math.round(W * dpr);
      c.height = Math.round(matrixH * dpr);
      c.style.height = `${matrixH}px`;
      c.addEventListener("mousemove", (e) => {
        const r = c.getBoundingClientRect();
        const row = Math.floor(((e.clientY - r.top) / r.height) * this.data.T);
        if (row >= 0 && row < this.data.T && row !== this.focus) this.onHover(row);
      });
      c.addEventListener("mouseleave", () => this.onHover(null));
      return c;
    });
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, matrixH);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < T; k++) {
        g.fillStyle = k > t ? "#ebe4d4" : `rgba(181,68,31,${Math.min(1, Math.sqrt(att[t * T + k])).toFixed(3)})`;
        g.fillRect(GAP + k * cell, t * cell, cell + 0.4, cell + 0.4);
      }
    }
    if (q != null) {
      g.strokeStyle = "#1e1c18";
      g.lineWidth = 1.5;
      g.strokeRect(GAP, q * cell, T * cell, cell);
    }
    if (svg.parentNode !== this.root || canvas.parentNode !== this.root) {
      this.ro.disconnect();
      this.root.replaceChildren(svg, canvas);
      this.ro.observe(this.root);
    }
  }

  /** Reuse an element while its key is unchanged so hovering never replaces the node under the pointer. */
  ensure(kind, key, make) {
    this.els ??= {};
    if (!this.els[kind] || this.els[kind].key !== key) this.els[kind] = { key, el: make() };
    return this.els[kind].el;
  }

  /** Text for the readout line: the strongest places the focused token reads from. */
  readout(q) {
    if (!this.data || q == null) return "Hover a token (or a row of the matrix) to see where it reads from.";
    const { att, T, labels } = this.data;
    const top = Array.from({ length: q + 1 }, (_, k) => [k, att[q * T + k]]).sort((a, b) => b[1] - a[1]).slice(0, 4);
    const parts = top.map(([k, w]) => `${pct(w)} on "${plainToken(labels[k])}" (${k})`);
    return `"${plainToken(labels[q])}" (${q}) reads from: ${parts.join(", ")}.`;
  }
}
