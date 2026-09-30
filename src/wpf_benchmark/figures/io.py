"""论文图件使用的结果索引、预测数组与跨模型读取。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..evaluation.config import ProtocolConfig
from ..paths import ProjectPaths


class MissingMaterial(Exception):
    """缺少某张图所需的输入，CLI 会跳过该图并说明原因。"""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    model: str
    table: str
    seed: int
    config_digest: str
    json_path: str
    created: str
    experiment: str = "standard"
    data_digest: str = ""
    code_digest: str = ""


MAIN_ABLATION_MODELS = frozenset(("ours", "barest"))


def result_provenance(paths: ProjectPaths) -> Tuple[str, str]:
    """Fingerprint the cleaned inputs and result-producing source code."""
    data_hash = hashlib.sha256()
    for relative in ("data/processed/sdwpf_clean.parquet",
                     "data/processed/sdwpf_meta.json"):
        source = paths.root / relative
        data_hash.update(relative.encode("utf-8"))
        if source.is_file():
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    data_hash.update(chunk)
        else:
            data_hash.update(b"<missing>")
    code_hash = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        if ("figures" in relative.parts and relative.name != "io.py") or (
                "__pycache__" in relative.parts):
            continue
        code_hash.update(relative.as_posix().encode("utf-8"))
        code_hash.update(source.read_bytes())
    return data_hash.hexdigest()[:16], code_hash.hexdigest()[:16]


def config_digest(config: ProtocolConfig, model_config: Dict[str, Any]) -> str:
    raw = json.dumps({"protocol": asdict(config), "model": model_config},
                     ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def append_index(paths: ProjectPaths, run_id: str, model: str, table: str,
                 seed: int, digest: str, json_path: str, created: str,
                 experiment: str = "standard", data_digest: str = "",
                 code_digest: str = "") -> None:
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    source = Path(json_path)
    try:
        relative = source.resolve().relative_to(paths.root.resolve()).as_posix()
    except ValueError:
        relative = source.as_posix()
    row = RunRecord(run_id, model, table, seed, digest, relative, created,
                    experiment, data_digest, code_digest)
    with (paths.evaluation / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _scan_legacy(paths: ProjectPaths) -> List[RunRecord]:
    records: List[RunRecord] = []
    pattern = re.compile(r"^(.*?)_(main|all)_(\d{8}_\d{6}(?:_\d{6})?)$")
    for source in paths.evaluation.glob("*.json"):
        matched = pattern.match(source.stem)
        if not matched or matched.group(1).startswith("selftest"):
            continue
        try:
            result = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        model = result.get("model", "")
        if not model or model.endswith("_selftest"):
            continue
        stamp = matched.group(3)
        cfg = result.get("config", {})
        digest_source = json.dumps({"protocol": cfg, "model": result.get("model_config", {})},
                                   sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
        records.append(RunRecord(
            result.get("run_id", "{}_{}".format(matched.group(1), stamp)),
            model, matched.group(2), int(result.get("seed", 0)), digest,
            source.relative_to(paths.root).as_posix(), stamp.replace("_", "T", 1),
            result.get("experiment", "standard"),
            result.get("data_digest", ""), result.get("code_digest", "")))
    return records


def load_index(paths: ProjectPaths, table: str = "main") -> List[RunRecord]:
    """按实验、版本、模型、种子和配置去重；旧索引仍可读取。"""
    index_path = paths.evaluation / "index.jsonl"
    if index_path.is_file():
        records = []
        for line_no, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(RunRecord(**json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError("index.jsonl 第 {} 行无效：{}".format(line_no, exc)) from exc
    else:
        records = _scan_legacy(paths)
    latest: Dict[Tuple[str, str, str, str, int, str], RunRecord] = {}
    for record in records:
        if record.table != table:
            continue
        key = (record.experiment, record.data_digest, record.code_digest,
               record.model, record.seed, record.config_digest)
        if key not in latest or record.created >= latest[key].created:
            latest[key] = record
    return sorted(latest.values(), key=lambda record: (record.model, record.seed))


def latest_cohort(records: List[RunRecord],
                  paths: Optional[ProjectPaths] = None) -> List[RunRecord]:
    """Select one data/code version, keeping legacy unversioned runs isolated."""
    if not records:
        return []
    versioned = [record for record in records
                 if record.data_digest and record.code_digest]
    if versioned and paths is not None:
        data_digest, code_digest = result_provenance(paths)
        records = [record for record in versioned
                   if record.data_digest == data_digest and
                   record.code_digest == code_digest]
        if not records:
            raise MissingMaterial("No results match the current data and code version")
    newest = max(records, key=lambda record: record.created)
    return [record for record in records
            if (record.data_digest, record.code_digest) ==
            (newest.data_digest, newest.code_digest)]


def model_groups(paths: ProjectPaths, experiment: str = "standard") -> Dict[str, List[RunRecord]]:
    """Group one result version; final ours/barest ablations enter main figures."""
    selected: Dict[str, List[RunRecord]] = {}
    eligible = [record for record in load_index(paths)
                if record.experiment == experiment or
                (experiment == "standard" and record.experiment == "ablation"
                 and record.model in MAIN_ABLATION_MODELS)]
    for record in latest_cohort(eligible, paths):
        if record.experiment == experiment or experiment == "standard":
            selected.setdefault(record.model, []).append(record)
    groups: Dict[str, List[RunRecord]] = {}
    for model, records in selected.items():
        if experiment == "standard" and model in MAIN_ABLATION_MODELS:
            final_runs = [record for record in records
                          if record.experiment == "ablation"]
            if final_runs:
                records = final_runs
        newest = max(records, key=lambda record: record.created)
        groups[model] = [record for record in records
                         if record.config_digest == newest.config_digest]
    return groups


def metric_values(paths: ProjectPaths, records: List[RunRecord], *keys: str) -> np.ndarray:
    """读取同配置多个 seed 的同一数值，用于均值与标准差。"""
    values = []
    for record in records:
        value: Any = load_metrics(paths, record)
        for key in keys:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def load_metrics(paths: ProjectPaths, record: RunRecord) -> Dict[str, Any]:
    source = Path(record.json_path)
    if not source.is_absolute():
        source = paths.root / source
    if not source.is_file():
        raise MissingMaterial("JSON not found: {}".format(source))
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("model") != record.model or result.get("table") != record.table:
        raise ValueError("Index and JSON disagree: {}".format(source))
    if record.data_digest and (result.get("data_digest") != record.data_digest or
                               result.get("code_digest") != record.code_digest):
        raise ValueError("Index and JSON provenance disagree: {}".format(source))
    return result


def save_run_arrays(paths: ProjectPaths, run_id: str, forecasts: np.ndarray,
                    evaluator: Any) -> Path:
    """保存画时序图所需的同一预测网格及主表有效位。"""
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    truth = evaluator.truth_for(evaluator.cfg.eval_mask)
    valid = evaluator._valid_stack(evaluator.valid_main)
    speed = evaluator._valid_stack(evaluator.dte["Wspd"])
    last = evaluator.dte["Patv"][:evaluator.T_eff, :, None]
    ramp = valid & (np.abs(truth - last) >= evaluator.cfg.ramp_threshold_kw())
    time_ns = np.asarray(evaluator.dte["times"], dtype="timedelta64[ns]").astype("int64")
    curtail_source = evaluator.dte.get("f_curtail")
    curtail = (evaluator._valid_stack(curtail_source) if curtail_source is not None
               else np.zeros_like(valid))
    source = paths.evaluation / "{}_arrays.npz".format(run_id)
    np.savez_compressed(source, forecasts=forecasts.astype(np.float32),
                        truth=truth.astype(np.float32), valid=valid,
                        valid_m1=evaluator._valid_stack(evaluator.valid_m1),
                        valid_m2=evaluator._valid_stack(evaluator.valid_m2),
                        truth_m1=evaluator.truth_for("m1").astype(np.float32),
                        truth_m2=evaluator.truth_for("m2").astype(np.float32),
                        target_mask=np.asarray(evaluator.cfg.target_mask),
                        eval_mask=np.asarray(evaluator.cfg.eval_mask),
                        times=time_ns, turbine_ids=evaluator.dte["tids"],
                        ramp_mask=ramp, wind_speed=speed.astype(np.float32),
                        curtail_mask=curtail, last_power=last[:, :, 0].astype(np.float32))
    return source


def load_run(paths: ProjectPaths, record: RunRecord) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """加载 JSON 和 npz，并核对时间、机组、预测步维度。"""
    metrics = load_metrics(paths, record)
    source = paths.evaluation / "{}_arrays.npz".format(record.run_id)
    if not source.is_file():
        raise MissingMaterial("arrays not found: {}".format(source.name))
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key in ("forecasts", "truth", "valid", "times", "turbine_ids", "ramp_mask"):
        if key not in arrays:
            raise ValueError("{} missing array {}".format(source, key))
    shape = arrays["forecasts"].shape
    if len(shape) != 3 or any(arrays[key].shape != shape
                              for key in ("truth", "valid", "ramp_mask")):
        raise ValueError("Prediction, truth and mask shapes differ in {}".format(source))
    grid = metrics.get("grid", {})
    if shape[2] != metrics["config"]["horizon"] or (grid and shape != (
            grid["T_eff"], grid["N"], grid["H"])):
        raise ValueError("Array shape differs from JSON protocol/grid: {}".format(source))
    if arrays["times"].shape != (shape[0] + shape[2],):
        raise ValueError("Time array length differs from forecast grid: {}".format(source))
    if arrays["turbine_ids"].shape != (shape[1],):
        raise ValueError("Turbine ID count differs from forecast grid: {}".format(source))
    for key in ("wind_speed", "curtail_mask"):
        if key in arrays and arrays[key].shape != shape:
            raise ValueError("{} shape differs from forecast grid".format(key))
    if "last_power" in arrays and arrays["last_power"].shape != shape[:2]:
        raise ValueError("last_power shape differs from forecast grid")
    return metrics, arrays


def aligned_array_runs(paths: ProjectPaths) -> Dict[str, Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """每个模型取最新 run，并确认画在一起的模型使用同一测试网格。"""
    groups = model_groups(paths)
    if not groups:
        raise MissingMaterial("main evaluation JSON not found")
    runs = {}
    reference = None
    for model, records in sorted(groups.items()):
        record = max(records, key=lambda item: item.created)
        metrics, arrays = load_run(paths, record)
        grid = (arrays["forecasts"].shape, arrays["times"], arrays["turbine_ids"])
        if reference is not None and (grid[0] != reference[0] or
                                      not np.array_equal(grid[1], reference[1]) or
                                      not np.array_equal(grid[2], reference[2])):
            raise ValueError("Model arrays use different test grids")
        reference = grid
        runs[model] = (metrics, arrays)
    return runs
