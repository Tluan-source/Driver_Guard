"""Feature file I/O: FrameSignals <-> parquet (preferred) or csv(.gz) fallback."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from .schemas import FrameSignals


def write_features(rows: Iterable[FrameSignals], path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [r.to_row() for r in rows]
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.DataFrame(dicts)
        df.to_parquet(path, index=False)
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", newline="", encoding="utf-8") as f:
            if dicts:
                w = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
                w.writeheader()
                w.writerows(dicts)
    if meta is not None:
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _coerce(v: str):
    if v == "" or v == "None":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        f = float(v)
        return f
    except ValueError:
        return v


def read_features(path: str | Path) -> Iterator[FrameSignals]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for row in df.to_dict(orient="records"):
            yield FrameSignals.from_row(row)
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield FrameSignals.from_row({k: _coerce(v) for k, v in row.items()})
