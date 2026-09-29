"""表 2：SDWPF 数据集和标记比例。"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..paths import ProjectPaths
from .io import MissingMaterial, latest_cohort, load_index, load_metrics
from .table_utils import write_table


FLAGS = ("m_missing", "m_imputed", "m_outlier", "f_fault", "f_curtail", "f_farm")


def render(paths: ProjectPaths, out: Path) -> List[Path]:
    source = paths.processed / "sdwpf_clean.parquet"
    if not source.is_file():
        raise MissingMaterial("clean Parquet not found")
    frame = pd.read_parquet(source, columns=["Day", "TurbID"] + list(FLAGS))
    records = latest_cohort(load_index(paths), paths)
    config = load_metrics(paths, records[0])["config"] if records else {"train_days": 196, "val_days": 25}
    days = int(frame["Day"].nunique())
    train_days, val_days = int(config["train_days"]), int(config["val_days"])
    items = [
        ("Turbines", "{:,}".format(frame["TurbID"].nunique())),
        ("Days", "{:,}".format(days)),
        ("Turbine-time rows", "{:,}".format(len(frame))),
        ("Train / validation / test (days)", "{} / {} / {}".format(
            train_days, val_days, days - train_days - val_days)),
    ]
    items += [(flag.replace("_", " ").title() + " (%)", "{:.2f}".format(100 * frame[flag].mean()))
              for flag in FLAGS]
    return write_table(out, "tab02_dataset", ["Statistic", "Value"], items, "lr",
                       "Flag categories can overlap; their percentages should not be added.")
