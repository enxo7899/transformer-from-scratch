import { test } from "node:test";
import assert from "node:assert/strict";
import { BPEError } from "../web/bpe.js";
import { loadAll } from "./helpers.js";

const { tok, fixture } = loadAll();

test("JS tokenizer produces exactly the ids Python produced for every fixed string", () => {
  assert.ok(fixture.tokenizer.length >= 10);
  for (const { text, ids } of fixture.tokenizer) assert.deepEqual(tok.encode(text), ids, JSON.stringify(text));
});

test("JS tokenizer round-trips text, code, whitespace and non-Latin scripts", () => {
  for (const { text } of fixture.tokenizer) assert.equal(tok.decode(tok.encode(text)), text);
});

test("JS tokenizer round-trips random strings", () => {
  const alphabet = [..."abcde fgh\n\t.,;:()0123456789THE", "é", "中"];
  let seed = 12345;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 200; i++) {
    const text = Array.from({ length: Math.floor(rand() * 40) }, () => alphabet[Math.floor(rand() * alphabet.length)]).join("");
    assert.equal(tok.decode(tok.encode(text)), text);
  }
});

test("an unknown token id is rejected when decoding", () => {
  assert.throws(() => tok.decode([tok.vocabSize]), BPEError);
});
