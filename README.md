# SDWPF 风电功率预测基准

这个项目将 SDWPF 的数据检查、清洗、模型预测和统一评估组织为一个 Python 包。预测目标是 134 台机组未来 12 个 10 分钟步长的有功功率；默认使用前 196 天训练、随后 25 天验证、最后 24 天测试。

首次使用请按[使用指南](docs/使用指南.md)依次完成数据清洗、评估协议自检和首个基线运行。

上传到服务器时使用精简版项目；本地验证、同步方法见[服务器部署与同步](docs/服务器部署与同步.md)。

新增规则、线性、GBDT 与三个深度基线的配置、安装和运行命令见[模型基线使用说明](docs/model_baselines.md)。

## 安装与数据

目标环境为 Python 3.8。进入项目目录并在所用环境中执行：

```bash
python -m pip install -e .
```

原始数据放在 `data/raw/sdwpf/`。`preprocess` 生成 `data/processed/sdwpf_clean.parquet` 和 `sdwpf_meta.json`；评估只读取 Parquet，因此环境需要 `pyarrow`。数据文件不纳入版本控制。

## 命令

```bash
wpf-benchmark eda
wpf-benchmark calibrate-k
wpf-benchmark preprocess
wpf-benchmark fit-power-curve
wpf-benchmark eval selftest
wpf-benchmark models
wpf-benchmark run --model persistence --config configs/persistence.json
wpf-benchmark run --model linear --config configs/linear.json --no-plots
wpf-benchmark plots --all
```

不安装包时可在项目目录执行 `PYTHONPATH=src python -m wpf_benchmark ...`。从其他目录执行时，在子命令前加 `--root /path/to/wpf-benchmark`。

`eval selftest` 同时生成主表和附表 JSON 及 `reports/eval_selftest.md`。`run` 使用同一预测数组生成 `reports/eval/{tag}_main_*.json` 和 `{tag}_all_*.json`。主表排除配置中的缺失、插补和异常标记；附表按评估规格只要求目标 Patv 有限，因此可能包含插补行。

清洗时当前连续缺失的前 3 步只用过去有效值填补，第 4 步起使用同时刻邻机值；空间邻居数按前 196 天的两个缺失场景标定为 k=12。功率曲线与残差阈值也仅从训练段估计。`preprocess` 会重写元数据，之后须重新运行 `fit-power-curve` 才能把方法模型的训练段曲线持久化。

看不懂持续性基线的 JSON 时，运行 `python scripts/analyze_persistence.py`。脚本会读取 `reports/eval/` 中最新的一对 `persistence_main_*.json` 和 `persistence_all_*.json`，生成 `reports/eval/persistence_analysis.md`。也可以用 `--main`、`--all` 指定两个文件，用 `--output` 指定输出位置。

比较已有模型运行的误差差异，可用 `python scripts/paired_bootstrap.py --experiment <标签> --out <目录>`；模型筛选、比较方向和种子数选项见[配对检验脚本说明](docs/配对检验脚本.md)。

## 目录与扩展点

| 目录或文件 | 职责 |
|---|---|
| `src/wpf_benchmark/analysis/` | EDA、空间邻居数标定 |
| `src/wpf_benchmark/preprocessing/` | SDWPF 清洗 |
| `src/wpf_benchmark/data/` | Parquet 读取、按天划分、滑窗、训练段缩放统计 |
| `src/wpf_benchmark/models/` | 模型基类、注册表与各类预测基线 |
| `src/wpf_benchmark/runner.py` | 模型训练、批量预测与评估入口 |
| `src/wpf_benchmark/evaluation/` | 协议配置、指标、评估器、报告和自检 |
| `configs/` | 实验参数 JSON |
| `tests/` | 数据对齐与模型接口测试 |

新增模型时，在 `models/` 下新建文件，继承 `BaseForecaster` 并使用 `@register_model("name")`；然后在 `models/__init__.py` 导入该类以完成注册。设置 `features` 指明所需的清洗后列，实现 `fit(train, valid)` 与 `predict(history)`。`history` 是 `(N,F,input_window)` 数组，输出必须是 kW 的 `(N,horizon)` 数组。需要批量推理的模型可覆盖 `predict_batch(histories, phases=None)`；需按日内时刻预测时设置 `needs_phases=True`。

运行器把训练段统计 `self.scaler` 提供给模型。模型若设置 `history_scale = "normalized"`，推理历史会用训练段 min/max 缩放；模型输出仍须转换回 kW。配置文件中的 `protocol` 字段传给 `ProtocolConfig`，`model` 字段传给模型构造函数。新增模型不需要修改评估指标代码。

## 验证

```bash
python -m unittest discover -s tests -v
wpf-benchmark eval selftest
```

现有数据的持续性回归基准：主表 4,948,656 个有效评分格，附表 5,537,952 个；主表 MAE 约 66.46 kW，真实值和持续性预测的功率曲线违背率约为 0.73% 和 17.70%。详细定义见 [评估协议](docs/spec_eval_protocol.md)。

模块间的数据流、模型契约与时间对齐约定见 [架构说明](docs/architecture.md)。

论文图件的命令、输出和后续实验输入格式见[论文图件说明](docs/paper_figures.md)。

方法模型、风向标定、验证集选参和消融实验的步骤见[方法模型与消融指南](docs/method_ablation.md)。
