// Byte-level BPE tokenizer: the same pre-tokenizer and merge procedure as src/tiny_gpt/bpe.py.
// tests/bpe.test.js checks token-for-token parity against ids exported from Python.

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: false });

export class BPEError extends Error {
  constructor(message) {
    super(message);
    this.name = "BPEError";
  }
}

export class BPETokenizer {
  /** @param {{merges: number[][], pretoken_pattern: string}} spec contents of model/tokenizer.json */
  constructor(spec) {
    this.pattern = new RegExp(spec.pretoken_pattern, "g");
    this.merges = spec.merges;
    this.ranks = new Map(spec.merges.map(([a, b], i) => [a * 65536 + b, i]));
    this.vocab = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
    for (const [a, b] of spec.merges) this.vocab.push(Uint8Array.from([...this.vocab[a], ...this.vocab[b]]));
    this.eotId = this.vocab.length;
    this.vocabSize = this.eotId + 1;
    this.cache = new Map();
  }

  /** @param {string} piece one pre-token @returns {number[]} */
  encodePiece(piece) {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    let seq = Array.from(encoder.encode(piece));
    while (seq.length > 1) {
      let bestRank = Infinity;
      let bestPair = null;
      for (let i = 0; i + 1 < seq.length; i++) {
        const rank = this.ranks.get(seq[i] * 65536 + seq[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = [seq[i], seq[i + 1]];
        }
      }
      if (!bestPair) break;
      const merged = [];
      for (let i = 0; i < seq.length; i++) {
        if (i + 1 < seq.length && seq[i] === bestPair[0] && seq[i + 1] === bestPair[1]) {
          merged.push(256 + bestRank);
          i++;
        } else merged.push(seq[i]);
      }
      seq = merged;
    }
    this.cache.set(piece, seq);
    return seq;
  }

  /** @param {string} text @returns {number[]} */
  encode(text) {
    const ids = [];
    for (const piece of text.match(this.pattern) ?? []) ids.push(...this.encodePiece(piece));
    return ids;
  }

  /** @param {number[]} ids @returns {Uint8Array} */
  decodeBytes(ids) {
    const parts = [];
    for (const id of ids) {
      if (id === this.eotId) parts.push(encoder.encode("<|endoftext|>"));
      else if (Number.isInteger(id) && id >= 0 && id < this.eotId) parts.push(this.vocab[id]);
      else throw new BPEError(`unknown token id ${id}`);
    }
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }

  /** @param {number[]} ids @returns {string} */
  decode(ids) {
    return decoder.decode(this.decodeBytes(ids));
  }
}
