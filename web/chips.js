import { h, tokenNodes } from "./dom.js";

/** Background tint for a token: the less probable it was, the stronger the accent. */
export function surpriseTint(prob) {
  if (prob == null) return "transparent";
  const a = 0.05 + 0.72 * Math.pow(1 - prob, 1.15);
  return `rgba(181, 68, 31, ${a.toFixed(3)})`;
}

/**
 * Render tokens as chips.
 * @param {HTMLElement} root @param {{text: string, prob: number|null, prompt: boolean}[]} tokens
 * @param {{onHover: (i: number|null) => void, onPin: (i: number) => void}} handlers
 * @param {number} [boundary] draw a marker before this token index
 */
export function renderChips(root, tokens, handlers, boundary = -1) {
  root.replaceChildren();
  const box = h("div", { class: "chips" });
  tokens.forEach((t, i) => {
    if (i === boundary) box.append(h("span", { class: "boundary" }, `trained context ends here (position ${boundary})`));
    const title = t.prob == null ? `position ${i}` : `position ${i}, probability ${(t.prob * 100).toFixed(1)}%`;
    const chip = h("span", { class: `chip${t.prompt ? " prompt" : ""}`, "data-i": i, title, style: `background:${surpriseTint(t.prob)}` }, ...tokenNodes(t.text));
    chip.addEventListener("mouseenter", () => handlers.onHover(i));
    chip.addEventListener("mouseleave", () => handlers.onHover(null));
    chip.addEventListener("click", () => handlers.onPin(i));
    box.append(chip);
  });
  root.append(box);
}

/** Mark which chip is hovered and which is pinned. */
export function markChips(root, focus, pinned) {
  for (const c of root.querySelectorAll(".chip")) {
    const i = Number(c.dataset.i);
    c.classList.toggle("focus", i === focus);
    c.classList.toggle("pin", i === pinned);
  }
}
