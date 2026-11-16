// Tiny DOM helpers shared by the page modules.

/** @param {string} tag @param {Record<string, any>} [attrs] @param {...(Node|string)} kids */
export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  el.append(...kids);
  return el;
}

export const SVGNS = "http://www.w3.org/2000/svg";

/** @param {string} tag @param {Record<string, any>} [attrs] @param {...(Node|string)} kids */
export function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  el.append(...kids);
  return el;
}

/** Show a token's text so whitespace is visible: spaces become a dim middle dot, newlines a return arrow. */
export function tokenNodes(text) {
  const nodes = [];
  for (const part of text.split(/( |\n)/)) {
    if (part === " ") nodes.push(h("i", {}, "·"));
    else if (part === "\n") nodes.push(h("i", {}, "↵"), "\n");
    else if (part) nodes.push(part);
  }
  return nodes;
}

export function plainToken(text) {
  return text.replace(/\n/g, "↵").replace(/ /g, "·");
}

export const pct = (p) => (p >= 0.995 ? "99%+" : p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`);
