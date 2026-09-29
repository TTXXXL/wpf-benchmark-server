"""Explicit project paths shared by commands and data loaders."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def resolve(cls, root: Optional[PathLike] = None) -> "ProjectPaths":
        if root is not None:
            candidate = Path(root)
        elif os.environ.get("WPF_BENCHMARK_ROOT"):
            candidate = Path(os.environ["WPF_BENCHMARK_ROOT"])
        else:
            candidate = cls._discover()
        candidate = candidate.expanduser().resolve()
        if not (candidate / "data").is_dir():
            raise FileNotFoundError("Project root must contain a data directory: " + str(candidate))
        return cls(candidate)

    @staticmethod
    def _discover() -> Path:
        for candidate in (Path.cwd(), *Path.cwd().parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
                return candidate
        # Works when invoked from a source checkout outside its working directory.
        return Path(__file__).resolve().parents[2]

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / "sdwpf"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figs"

    @property
    def evaluation(self) -> Path:
        return self.reports / "eval"
