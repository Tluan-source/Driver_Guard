"""Convert a '# %%' cell script into a Jupyter notebook (no extra dependencies).

    python scripts/py_to_ipynb.py kaggle/01_extract_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    cells, cur, kind = [], [], "code"

    def flush():
        text = "\n".join(cur).strip("\n")
        if not text:
            return
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in text.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": text})

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            cur, kind = [], ("markdown" if "[markdown]" in line else "code")
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(convert(Path(a)))
