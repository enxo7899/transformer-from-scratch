// Reproducible README screenshots. Run `npm run dev` in another terminal, then `npm run screenshots`.
import { createRequire } from "node:module";
const { chromium, devices } = createRequire(import.meta.url)("playwright");
const base = process.env.BASE || "http://127.0.0.1:5187/";
const out = process.env.OUT || "docs";
const browser = await chromium.launch();
const errors = [];
async function shot(name, url, { device, viewport = { width: 1400, height: 1000 }, full = false, clip, before } = {}) {
  const ctx = await browser.newContext(device ? { ...devices[device], deviceScaleFactor: 2 } : { viewport, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  page.on("console", (m) => m.type() === "error" && errors.push(`${name}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
  await page.goto(base + url, { waitUntil: "networkidle" });
  await page.waitForSelector("html[data-ready='1']");
  await page.waitForTimeout(500);
  if (before) await before(page);
  const target = clip ? page.locator(clip) : page;
  await target.screenshot({ path: `${out}/${name}.png`, fullPage: full && !clip });
  await ctx.close();
}
await shot("overview", "?run=generate,attention", { viewport: { width: 1400, height: 1000 }, full: true });
await shot("attention", "?run=attention&layer=2&head=1", { clip: "#inside" });
await shot("context-limit", "?run=break&policy=unguarded&n=200", { clip: "#break" });
await shot("evidence", "?run=generate", { clip: "#evidence" });
await shot("mobile", "?run=generate,attention", { device: "iPhone 13", full: true });
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("no console errors");
