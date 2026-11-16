"""Weight file format shared with the browser.

Layout: b"TGPT", uint32 little-endian header length, UTF-8 JSON header, zero
padding to a 4-byte boundary, then every tensor as little-endian float32 in
header order. The header holds the config and, per tensor, name/shape/offset.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .model import GPT, GPTConfig

MAGIC = b"TGPT"


class WeightFileError(ValueError):
    """Raised when a weight file is malformed."""


def save_weights(model: GPT, path: str | Path, extra: dict | None = None) -> int:
    tensors, blobs, offset = [], [], 0
    for name, p, _ in model.named_parameters():
        data = np.ascontiguousarray(p, dtype="<f4").tobytes()
        tensors.append({"name": name, "shape": list(p.shape), "offset": offset, "length": len(data)})
        blobs.append(data)
        offset += len(data)
    header = json.dumps({"format": 1, "dtype": "float32", "config": model.config.to_dict(), "tensors": tensors, **(extra or {})},
                        separators=(",", ":")).encode("utf-8")
    header += b" " * (-(len(header) + 8) % 4)
    payload = MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs)
    Path(path).write_bytes(payload)
    return len(payload)


def load_weights(path: str | Path, dtype=np.float32) -> GPT:
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise WeightFileError("not a tiny_gpt weight file")
    (hlen,) = struct.unpack("<I", raw[4:8])
    header = json.loads(raw[8:8 + hlen].decode("utf-8"))
    base = 8 + hlen
    model = GPT(GPTConfig(**header["config"]), dtype=dtype)
    state = {}
    for t in header["tensors"]:
        chunk = raw[base + t["offset"]: base + t["offset"] + t["length"]]
        if len(chunk) != t["length"]:
            raise WeightFileError(f"file truncated while reading {t['name']}")
        state[t["name"]] = np.frombuffer(chunk, dtype="<f4").reshape(t["shape"]).astype(dtype)
    try:
        model.load_state(state)
    except (KeyError, ValueError) as err:
        raise WeightFileError(str(err)) from err
    return model
