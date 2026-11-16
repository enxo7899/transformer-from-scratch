"""Seeded training run: python -m tiny_gpt.train --corpus data/corpus.txt --model-dir model --results docs/results.json"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from .baselines import Unigram, fit_bigram, perplexity
from .bpe import BPETokenizer
from .data import Windows, fixed_windows, load_documents, random_batch, split_documents, tokenize_documents
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm, cosine_lr
from .weights import save_weights


def evaluate(model: GPT, w: Windows, batch: int = 16, limit: int | None = None) -> float:
    n = len(w.x) if limit is None else min(limit, len(w.x))
    losses, counts = [], []
    for i in range(0, n, batch):
        x, y = w.x[i:i + batch], w.y[i:i + batch]
        _, loss = model.forward(x, y)
        losses.append(loss * x.size)
        counts.append(x.size)
    return float(sum(losses) / sum(counts))


def hardware() -> dict:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__}


def train(args: argparse.Namespace) -> dict:
    config = GPTConfig(vocab_size=args.vocab, context=args.context, n_layers=args.layers, n_heads=args.heads, dim=args.dim)
    docs = load_documents(args.corpus)
    train_docs, val_docs = split_documents(docs, seed=args.seed)
    tok = BPETokenizer.train("\n".join(train_docs), args.vocab)  # merges learned from training documents only
    train_tok, val_tok = tokenize_documents(tok, train_docs), tokenize_documents(tok, val_docs)
    val_w = fixed_windows(val_tok, config.context)
    print(f"docs {len(train_docs)} train / {len(val_docs)} val; tokens {len(train_tok)} / {len(val_tok)}", flush=True)

    model = GPT(config, seed=args.seed)
    opt = AdamW(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    curve, window_losses = [], []
    start = time.time()
    for step in range(args.steps):
        opt.lr = cosine_lr(step, args.steps, args.lr, args.warmup)
        x, y = random_batch(train_tok, args.batch, config.context, rng)
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        clip_grad_norm([g for _, _, g in model.named_parameters()], args.clip)
        opt.step()
        window_losses.append(loss)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val = evaluate(model, val_w, limit=args.eval_windows)
            curve.append({"step": step + 1, "train_loss": round(float(np.mean(window_losses)), 4), "val_loss": round(val, 4),
                          "lr": round(opt.lr, 6), "seconds": round(time.time() - start, 1)})
            window_losses = []
            print(curve[-1], flush=True)
    seconds = time.time() - start

    val_loss = evaluate(model, val_w)
    uni, bi = Unigram(train_tok, config.vocab_size), fit_bigram(train_tok, config.vocab_size)
    val_chars = len(tok.decode_bytes(val_tok.tolist()))
    bytes_per_token = val_chars / len(val_tok)
    scored = val_w.y.size
    entry = lambda nll: {"val_loss": round(nll, 4), "perplexity": round(perplexity(nll), 2),
                         "bits_per_byte": round(nll / math.log(2) / bytes_per_token, 3)}
    results = {
        "model": {**config.to_dict(), "parameters": model.n_parameters(), "dtype": "float32"},
        "training": {"seed": args.seed, "steps": args.steps, "batch_size": args.batch, "tokens_seen": args.steps * args.batch * config.context,
                     "learning_rate": args.lr, "warmup_steps": args.warmup, "weight_decay": args.weight_decay, "grad_clip": args.clip,
                     "seconds": round(seconds, 1), "hardware": hardware()},
        "data": {"documents_train": len(train_docs), "documents_val": len(val_docs), "tokens_train": int(len(train_tok)),
                 "tokens_val": int(len(val_tok)), "val_tokens_scored": int(scored), "bytes_per_token_val": round(bytes_per_token, 3)},
        "results": {"transformer": entry(val_loss), "bigram": {**entry(bi.nll(val_w)), "alpha": bi.alpha}, "unigram": entry(uni.nll(val_w))},
        "curve": curve,
    }
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    size = save_weights(model, Path(args.model_dir) / "weights.bin", extra={"seed": args.seed})
    tok.save(Path(args.model_dir) / "tokenizer.json")
    results["model"]["weights_bytes"] = size
    Path(args.results).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["results"], indent=2), flush=True)
    return results


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/corpus.txt")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--results", default="docs/results.json")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-windows", type=int, default=48)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
