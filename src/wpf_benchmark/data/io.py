"""Parquet dataset and preprocessing metadata access."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ..paths import ProjectPaths


def load_clean(paths: Optional[ProjectPaths] = None,
               columns: Optional[Sequence[str]] = None,
               filters: Optional[Sequence[tuple]] = None) -> pd.DataFrame:
    paths = paths or ProjectPaths.resolve()
    source = paths.processed / "sdwpf_clean.parquet"
    if not source.is_file():
        raise FileNotFoundError("Clean Parquet data is missing: " + str(source))
    return pd.read_parquet(source,
                           columns=list(columns) if columns is not None else None,
                           filters=list(filters) if filters is not None else None)


def load_meta(paths: Optional[ProjectPaths] = None) -> Dict[str, Any]:
    paths = paths or ProjectPaths.resolve()
    source = paths.processed / "sdwpf_meta.json"
    return json.loads(source.read_text(encoding="utf-8"))
