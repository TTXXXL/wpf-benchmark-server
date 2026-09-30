# 模型基线使用说明

> **2026-09-30 口径更新**：新运行默认 M1，也可选 M2。下文旧版各模型 MAE 和 4,948,656 格基准属于历史 M0，须在新口径重跑后才能作为论文结果。定义见[目标掩码口径](目标掩码口径_M1_M2.md)。

所有模型都读取同一份 `data/processed/sdwpf_clean.parquet`。默认用前 196 天训练、后 25 天验证、最后 24 天测试；发布时刻使用过去 144 个十分钟值，预测未来 12 个值。`src/wpf_benchmark/runner.py` 负责切分与评估，模型只接收训练段、验证段和历史窗口。

## 先看有哪些模型

```bash
wpf-benchmark models
```

| 类型 | 模型名 | 文件 | 用到的数据 |
|---|---|---|---|
| 参照 | `persistence` | `models/persistence.py` | 最新功率 |
| 简单规则 | `seasonal_persistence` | `models/naive.py` | 一天前同一时刻的功率 |
| 简单规则 | `climatology` | `models/naive.py` | 训练段内每台机组、每个时刻的平均功率 |
| 简单规则 | `trend_persistence` | `models/naive.py` | 近一小时单机功率变化 |
| 简单规则 | `farm_mean_persistence` | `models/naive.py` | 近一小时全场平均功率变化 |
| 传统学习 | `linear` | `models/linear.py` | 各机组过去功率的岭回归 |
| 传统学习 | `gbdt` | `models/gbdt.py` | 功率、风速、风向分量的窗口统计量 |
| 深度学习 | `lstm_seq2seq` | `models/lstm_seq2seq.py` | 风速、风向分量、功率 |
| 深度学习 | `patchtst` | `models/patchtst.py` | 同上，切成小时间片 |
| 深度学习 | `agcrn_lite` | `models/agcrn_lite.py` | 原项目简化图循环模型，6 步池化 |
| 深度学习 | `agcrn` | `models/agcrn.py`、`models/agcrn_original.py` | 论文 AGCRN 架构：完整时序、节点自适应参数与自适应图 |

方法模型另有 `ours`、`ours_no_consist`、`ours_no_wake`、`barest` 四个注册名，位于 `models/pin.py`。它们的风速辅助目标、尾流标定和消融运行顺序见[方法模型与消融指南](method_ablation.md)。

这些模型都在 `src/wpf_benchmark/models/__init__.py` 注册。新模型可按 `models/base.py` 的接口添加；训练窗口的统一实现放在 `models/training.py`，深度学习共用的训练循环放在 `models/neural.py`。

## 安装依赖

规则模型和 `linear` 只需要项目基础依赖：

```bash
python -m pip install -e .
```

运行 `gbdt` 优先安装 LightGBM：

```bash
python -m pip install -e ".[gbdt]"
```

如果 LightGBM 在服务器上装不上，可改装 XGBoost；`models/gbdt.py` 会自动选可用的后端：

```bash
python -m pip install -e ".[gbdt-fallback]"
```

深度模型（包括方法模型）需要 PyTorch。服务器已有 `torch 1.11+cu113` 时直接使用；若没有 PyTorch，按服务器 CUDA 环境安装对应版本后再运行。项目的可选依赖在 `pyproject.toml` 的 `deep` 分组中声明。普通安装不会强制下载 PyTorch。

## 运行并看结果

例如：

```bash
wpf-benchmark run --model seasonal_persistence --config configs/seasonal_persistence.json --no-plots
wpf-benchmark run --model linear --config configs/linear.json --no-plots
wpf-benchmark run --model lstm_seq2seq --config configs/lstm_seq2seq.json --seed 0 --repeat 3 --no-plots
wpf-benchmark plots --all
```

在 Linux 服务器上依次完成深度模型的 3 种子实验：

```bash
for model in lstm_seq2seq patchtst agcrn_lite agcrn; do
  wpf-benchmark run --model "$model" --config "configs/$model.json" --repeat 3 --no-plots
done
wpf-benchmark plots --all
```

每个 `configs/<模型名>.json` 都包含 `protocol`、`model` 和 `seed`。`protocol` 留空即默认时间划分；改窗口或划分时，在自己的实验配置里写明。命令行 `--seed` 会覆盖配置里的 seed；`--repeat 3` 顺序使用 seed、seed+1、seed+2。一次运行产生一对 `reports/eval/<模型名>_main_*.json`、`*_all_*.json`，并将路径追加到 `reports/eval/index.jsonl`。结果中的 `model_config` 是实际使用的参数，`timing` 是耗时，`model_size.n_params` 是参数量。主表多次运行的均值和标准差由图表读取索引时计算。

深度模型按训练段每 6 步取一个窗口，最多训练 30 轮，验证段连续 5 轮没有进步就停止，并使用验证误差最小的一轮做测试预测。训练曲线保存在 `reports/train/<run_id>.log`。`agcrn_lite` 将 144 步历史按 6 步平均后送入图循环层，默认使用 MSE；`agcrn` 保留全部 144 步，使用论文的 DAGG、NAPL 和直接多时距输出头，默认使用 MAE。两者因此不是只改一处结构的受控消融。当前 `agcrn` 沿用本项目的数据、输入特征和 30 轮预算，并非复现原论文数据集及完整训练设置。

两模型现在均支持 `model.loss="mae"` / `"mse"` 以及 `model.current_power_skip=true`。开启直通时预测为 P(t)+Δh，残差输出头初始化为零；原 `agcrn.json` / `agcrn_lite.json` 保留关闭直通的对照。新四配置与完整八组消融可用 `bash scripts/rerun_graph_ablation.sh` 运行，详见[图模型残差与损失消融](图模型残差与损失消融.md)。

Lite 另提供可选低功率状态混合头：`configs/agcrn_lite_low_power_mae.json` 使用 MAE 加状态分类辅助损失，将低功率分支与当前功率残差分支按未来低功率概率融合。原 Lite+MAE 与无辅助损失的结构消融均保留，运行命令及验证说明见 [Lite 低功率状态头](Lite低功率状态头.md)。

命名迁移：此前 Wave 1 以 `agcrn` 写出的预测数组和指标，来自现在的 `agcrn_lite` 架构。旧产物不会自动更名；绘图或汇总时应按运行配置确认身份。新版 `agcrn` 须重新训练，不能引用旧版 `agcrn` 的分数。

## 功率曲线模块

运行 `wpf-benchmark fit-power-curve` 会用前 196 天的有效风速和功率为每台机组拟合四参数 logistic 曲线，并把参数、样本数、R² 写入 `data/processed/sdwpf_meta.json` 的 `power_curves` 字段。当前数据有 6 台机组没有足够的有效训练样本；它们使用全场曲线，`source` 标为 `global_fallback`，R² 留空。`models/power_curve.py` 提供 NumPy 计算函数和可求导的 PyTorch 函数；后续物理模型可以直接调用。这个命令只用训练数据，当前不改变任何基线预测。

## 检查结果

```bash
python -m unittest discover -s tests -v
wpf-benchmark eval selftest
```

`selftest` 应显示主表样本数 4,948,656，persistence 主表 MAE 约 66.46 kW。这些是因果填补、训练段拟合、k=12 后的参考值。本版完整数据已重跑轻量基线：trend_persistence 85.85、farm_mean_persistence 101.66、linear 84.15、seasonal_persistence 293.44、climatology 344.00 kW（均为主表 MAE）；JSON 见开发版 `reports/eval/`。当前 `climatology` 将发布时刻相位均值重复到全部 12 步，其定义是否改为目标相位仍待决定。GBDT 后端未安装，仍待重跑。
