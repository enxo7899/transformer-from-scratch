# Training corpus

`corpus.txt` is built by `build_corpus.py` from text that ships inside the Python standard library of the machine that ran it (CPython 3.13). Nothing is downloaded.

| Source | What was taken |
|---|---|
| `pydoc_data.topics` | The language reference topics behind `help()` ("assert", "for", "class", ...). |
| Standard-library `*.py` files | Module, class and function docstrings (tests, `idlelib`, `lib2to3` and similar are skipped). |

**Licence.** The Python documentation and standard library are distributed under the Python Software Foundation License Version 2 (PSF-2.0), which permits copying and redistribution with the licence notice kept. Text is quoted unchanged apart from being reduced to plain ASCII (smart quotes and dashes mapped to ASCII, tabs to four spaces). Copyright remains with the Python Software Foundation and the individual contributors; see https://docs.python.org/3/license.html.

**Shape.** 621 documents (one per topic or module), at most 9,000 characters each, about 4.1 MB. Any document that names an AI vendor is left out. Documents are separated by a line containing `<<<DOC>>>`.

**Split.** `tiny_gpt.data.split_documents` shuffles whole documents with a fixed seed and holds out documents until 8% of the characters are reached. Tokenizer merges are learned from the training documents only. No document appears on both sides (there is a test for it).

To rebuild: `python3 data/build_corpus.py`. The output depends on the Python version, which is why the built `corpus.txt` is committed.
