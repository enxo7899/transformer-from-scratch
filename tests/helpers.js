import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { BPETokenizer } from "../web/bpe.js";
import { GPT } from "../web/model.js";
import { parseWeights } from "../web/weights.js";

const dir = process.env.MODEL_DIR ?? fileURLToPath(new URL("../model/", import.meta.url));

export function loadAll() {
  const buf = readFileSync(`${dir}/weights.bin`);
  const weights = parseWeights(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const tok = new BPETokenizer(JSON.parse(readFileSync(`${dir}/tokenizer.json`, "utf8")));
  const fixture = JSON.parse(readFileSync(`${dir}/parity.json`, "utf8"));
  return { weights, model: new GPT(weights), tok, fixture };
}

export function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
