# 方法模型与消融实验怎么跑

这部分分为 W0（核对风向）、W1（准备尾流图）、W2（只用训练和验证集选参数）、W3（冻结配置后跑测试集与作图）。代码已实现这些入口。**本版清洗与 k=12 后的 W0 已重跑，四变体尚未重跑**；后续决策以当前 `reports/wake_calibration.json` 为准。

## 1. 清洗与 W0

`src/wpf_benchmark/preprocessing/sdwpf.py` 会额外保存四个逐通道标记：风速和功率各自的“原始缺失”“经过插补”。它们不会改变已有主表功率掩码。风速监督目标由 `models/pin.py` 使用风速标记，并排除 `f_stuck`。

```powershell
wpf-benchmark preprocess
wpf-benchmark fit-power-curve
wpf-benchmark calibrate-wake
```

`analysis/wake_calibration.py` 只读取前 196 天；前 60% 训练日选 8 个风向约定，后 40% 训练日确认。它保存 `reports/wake_calibration.md` 和 `.json`。风速与功率亏损都应为正；若共同可用分箱、共同分层或确认样本不足，结果是“无法判定”。φ 扫描只在选择集上做，峰值离最近 90° 超过 15° 时结果为“方位未确认”。

本版 `calibrate-wake` 选中 `plus-north-from`，确认集风速亏损为 **0.175 m/s**（95% CI 0.146–0.203），功率亏损为额定功率的 **1.232%**（95% CI 1.016–1.579）；均低于预注册的 0.3 m/s 与 2% 锚点，因此 `decision=weak`。完整样本数、候选比较和不确定性见当前 `reports/wake_calibration.md`。

只有 JSON 的 `decision` 为 `significant` 时，尾流才进入主方法。`weak`、`none`、`undetermined` 或 `orientation_unconfirmed` 时，主实验只比较带一致性损失的 `ours` 与不带该损失的 `barest`，且两者 `lambda_wake` 都设为 0。此时用 `configs/ours_consist_only.json` 运行注册名 `ours`，用 `configs/barest.json` 运行 `barest`；不要把尾流称为已验证的增益。

## 2. W1：尾流先验

仅在 W0 判为 `significant` 后运行：

```powershell
wpf-benchmark prepare-wake-prior
```

`models/wake.py` 生成 `data/processed/wake_prior.npz`。矩阵的**行是目标机组，列是上风向来源机组**。每个方向的图按行归一化；再按训练段风向出现频率求静态平均，并对非零行重新归一化。`models/networks.py` 中的 `AGCRNLiteNetwork` 是 `agcrn_lite` 与方法模型共用的图循环骨干；论文架构的 `agcrn` 位于 `models/agcrn_original.py`，与方法模型不共用骨干。此前 Wave 1 中名为 `agcrn` 的运行属于现称 `agcrn_lite` 的旧实现。

## 3. W2：只在验证集选 λ

`models/pin.py` 定义同一双输出网络与四个注册名：`ours`、`ours_no_consist`、`ours_no_wake`、`barest`。网络同时预测未来功率和风速，但未来真实风速仅作为训练目标，不进入历史输入。四者网络结构相同，只改变一致性损失与尾流损失的权重。

先核对本版 W0 的 `decision`，再跑一个 `lambda_consist=0.1` 的预实验，查看 `reports/train/<run_id>.log` 的四项原始损失。随后按训练与验证结果确定 λ 候选范围，并把范围和“验证 MAE 为主、直接一致性误差为辅”的选择规则记录下来。当前 `decision=weak`，使用 `configs/lambda_sweep_no_wake/consist_*.json`（λ_wake=0）作为初始候选；若后续另版 W0 显著，才使用 `configs/lambda_sweep/consist_*.json`。实际范围以首跑日志为准。

```powershell
wpf-benchmark run --model ours --config configs/ours_consist_only.json --experiment lambda_sweep --validation-only --no-plots
```

对每个候选配置重复上述命令。`--validation-only` 会在训练结束后直接保存 `*_validation_*.json`，不读取测试段。`figures/fig10_lambda.py` 只读这些验证结果：

```powershell
wpf-benchmark plots --figures 10
```

## 4. W3：冻结配置，测试一次

确定 λ 后，把最终值写入要使用的配置文件，再运行完整评估。**本版 W0 为弱证据**，主实验跑两变体：

```powershell
wpf-benchmark run --model ours --config configs/ours_consist_only.json --experiment ablation --repeat 3 --no-plots
wpf-benchmark run --model barest --config configs/barest.json --experiment ablation --repeat 3 --no-plots
wpf-benchmark plots --figures 9
```

若新版 W0 显著，再跑四变体与图 12，每个变体三个随机种子：

```powershell
wpf-benchmark run --model ours --config configs/ours.json --experiment ablation --repeat 3 --no-plots
wpf-benchmark run --model ours_no_consist --config configs/ours_no_consist.json --experiment ablation --repeat 3 --no-plots
wpf-benchmark run --model ours_no_wake --config configs/ours_no_wake.json --experiment ablation --repeat 3 --no-plots
wpf-benchmark run --model barest --config configs/barest.json --experiment ablation --repeat 3 --no-plots
wpf-benchmark plots --figures 9,12
```

`figures/fig09_ablation.py` 画主表 MAE、RMSE、功率曲线违背率的均值 ± 标准差，并在 `.tex` 表附上预测功率与预测风速对应曲线的直接一致性误差。最终 `ours` 与 `barest` 还会进入表 1、图 6/7/13；其余消融变体只用于图 9。本版 W0 为弱证据，不把图 12 放进正文。

`figures/fig12_graph.py` 默认取注册名**精确等于** `ours` 的最新主表运行，对比实际学习图和训练时用的静态尾流先验；逐风向图另存补充图。指定某次运行可执行 `wpf-benchmark plots --figures 12 --run-id <run_id>`。

本地环境没有 PyTorch 时可以运行除深度模型训练外的测试与图件样图。真实的 W2/W3 模型训练应在安装 PyTorch 的环境执行。
