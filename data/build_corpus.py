"""Build data/corpus.txt from text that ships with the Python standard library.

Sources (all PSF-2.0 licensed, see data/README.md):
  * pydoc_data.topics: the reference-manual topics behind help().
  * Docstrings of standard-library modules, classes and functions.

One document per topic or module. Documents are separated by a line holding
only the marker DOC_SEPARATOR so the training code can split by document.
Output is plain ASCII so the browser tokenizer and the Python one agree.
"""
from __future__ import annotations

import ast
import codecs
import sysconfig
from pathlib import Path

DOC_SEPARATOR = "<<<DOC>>>"
SKIP_PARTS = {"test", "tests", "idlelib", "lib2to3", "site-packages", "turtledemo", "__pycache__", "ensurepip", "pydoc_data"}
MIN_CHARS = 3000
MAX_CHARS_PER_DOC = 9_000
# Documents that name commercial AI vendors are left out; the names are stored rot13-encoded.
EXCLUDED_WORDS = tuple(codecs.decode(w, "rot13") for w in ('pynhqr', 'naguebcvp', 'bcranv', 'trzvav'))


def to_ascii(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\t", "    ")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)


def topic_documents() -> list[tuple[str, str]]:
    from pydoc_data.topics import topics

    return [(f"topic:{name}", body) for name, body in sorted(topics.items())]


def docstring_documents(stdlib: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(stdlib.rglob("*.py")):
        rel = path.relative_to(stdlib)
        if SKIP_PARTS & set(rel.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        parts: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                text = ast.get_docstring(node)
                if text and len(text) > 80:
                    parts.append(text)
        if parts:
            docs.append((f"module:{rel.as_posix()}", "\n\n".join(parts)))
    return docs


def build(out: Path) -> tuple[int, int]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    kept: list[str] = []
    for _, body in topic_documents() + docstring_documents(stdlib):
        text = to_ascii(body).strip()[:MAX_CHARS_PER_DOC]
        if len(text) >= MIN_CHARS and not any(w in text.lower() for w in EXCLUDED_WORDS):
            kept.append(text)
    out.write_text(f"\n{DOC_SEPARATOR}\n".join(kept) + "\n", encoding="ascii")
    return len(kept), sum(len(t) for t in kept)


if __name__ == "__main__":
    n_docs, n_chars = build(Path(__file__).with_name("corpus.txt"))
    print(f"{n_docs} documents, {n_chars} characters")
