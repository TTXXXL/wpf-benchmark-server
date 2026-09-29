"""Generic entry point for saved-run paired bootstrap comparisons."""
from __future__ import annotations

from wave1_paired_bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main(default_preset="auto"))
