// Reader for the weight file written by src/tiny_gpt/weights.py:
// "TGPT", uint32 header length, JSON header, then little-endian float32 tensors.

export class WeightFileError extends Error {
  constructor(message) {
    super(message);
    this.name = "WeightFileError";
  }
}

/** @param {ArrayBuffer} buffer @returns {{config: object, tensors: Map<string, Float32Array>, shapes: Map<string, number[]>}} */
export function parseWeights(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 8 || String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== "TGPT") {
    throw new WeightFileError("not a tiny_gpt weight file");
  }
  const headerLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength)));
  const base = 8 + headerLength;
  const tensors = new Map();
  const shapes = new Map();
  for (const t of header.tensors) {
    if (base + t.offset + t.length > buffer.byteLength) throw new WeightFileError(`file truncated while reading ${t.name}`);
    const bytes = new Uint8Array(buffer.slice(base + t.offset, base + t.offset + t.length)); // copy: guarantees 4-byte alignment
    tensors.set(t.name, new Float32Array(bytes.buffer));
    shapes.set(t.name, t.shape);
  }
  return { config: header.config, tensors, shapes };
}
