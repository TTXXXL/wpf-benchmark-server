# Lite 低功率状态头

本功能先用于 `agcrn_lite`，通过 `low_power_head=true` 启用。原 `agcrn_lite_mae.json` 保留为当前功率残差对照；`agcrn` 和 PIN 模型尚未接入这个头。当前已完成实现及合成数据验证，SDWPF 完整重训练的性能增益待验证。

## 预测结构

对每台机组、每个未来时距 h，模型计算：

```text
P_normal,h = clip(P_anchor + Delta_h, 0, P_rated)
P_low,h    = tau × sigmoid(a_h)
q_h        = sigmoid(state_logit_h)
P_hat,h    = q_h × P_low,h + (1 − q_h) × P_normal,h
```

`P_anchor` 是历史窗口中最新的有限功率，完全缺失时回退为 0 kW。每个时距的正常分支都相对同一锚点预测有正负号的残差；融合后的功率作为下一时距的解码输入。启用新头时，正常分支在融合前限制到物理功率范围，低功率分支限制在 `[0, tau]`。关闭新头时保留原 Lite 网络的参数、初始化与前向计算。

状态头与低功率分支共享一个小型 MLP，输入为图解码器隐藏状态、相对预测时距 `h/H`，以及**时间池化前**近期本机历史的 8 个统计量：最新有限功率、最新有限风速、两者的端点斜率、平均功率、末尾连续低功率步数占窗口的比例、功率/风速各自的有限值比例。斜率按观测端点间的步数归一化。默认取最近 12 步（2 小时），不足 12 步则使用整个历史窗口。

缺失或无穷值不计作低功率，且会打断连续低功率时长；整段缺失的功率回退到归一化后的 0 kW，风速回退到归一化值 0，并通过有效值比例表示缺失。状态头不会接收未来功率、未来风速、未来状态标记或未来掩码。阈值和两个分支的边界都按训练段功率 scaler 转换，不能假定归一化值 0 就是物理 0 kW。

正常残差输出头沿用零初始化。状态输出层初始权重为 0、偏置为 −4（初始低功率概率约 0.018），使初始融合接近残差分支；它是可学习的软概率，没有按当前功率强制置零的规则。

## 损失、掩码与早停

```text
z_h = 1[y_h < tau]
L = mean_valid(|P_hat − y|) + state_loss_weight × mean_valid(BCEWithLogits(logit, z))
```

功率误差在训练段 min/max 归一化空间计算。分类和回归使用同一 `target_mask` 及有限目标过滤，在训练/验证各自的时间段内构造标签；等于 tau 的目标属于正常状态。也支持 `loss="mse"`，但首轮配置使用 MAE。

训练日志分别记录 `train_mae` / `valid_mae`、`train_state_bce` / `valid_state_bce` 和 `train_total` / `valid_total`。**最佳 checkpoint 和早停始终按验证功率 MAE 选择**，MSE 配置则按验证功率 MSE；总损失用于反向传播，不用于替代功率指标选择模型。

结果 JSON 的 `model_config` 记录开关、阈值、辅助权重及历史长度；`validation_metrics` 记录最佳轮的各项损失、`MAE_kW`、最佳轮/停止轮及早停指标。这里的验证误差在每 `stride` 步取窗、窗口完整落在验证段的目标上计算；它不等于测试报告的逐发布时刻评估。普通模型的验证 `MAE_kW` 对应未裁剪网络输出；本混合头已在网络内约束到物理范围。

## 首轮配置

| 配置 | 作用 |
|---|---|
| `configs/agcrn_lite_mae.json` | 原 Lite + MAE + 当前功率残差对照 |
| `configs/agcrn_lite_low_power_mae.json` | 新混合头，tau=10 kW，辅助权重 0.05，历史长度 12 |
| `configs/agcrn_lite_low_power_no_aux.json` | 同一混合结构，辅助权重为 0，仅通过最终功率误差训练 |

三组公共超参数和 M1 口径一致。后两组新增参数必须满足：`low_power_head` 为布尔值，启用时 `current_power_skip=true`，`0 < low_power_threshold_kw < rated_power_kw`，`state_loss_weight >= 0`，`state_history_steps` 为正整数；阈值和权重必须有限。

10 kW 与 0.05 是首轮起点，不是测试集调优后的最优值。需要调参时，先只跑训练/验证：

```bash
wpf-benchmark run --model agcrn_lite --config configs/agcrn_lite_low_power_mae.json --validation-only --experiment lite_low_power_validation --tag lite_low_power_validation --seed 0 --no-plots
```

可以复制配置，仅在训练/验证数据上比较阈值与辅助权重。确认配方后，在已完成 `preprocess` 和 `fit-power-curve` 的服务器项目中跑测试：

```bash
# 快速首轮：同一个 seed，先比较原残差与新混合头。
wpf-benchmark run --model agcrn_lite --config configs/agcrn_lite_mae.json --experiment lite_low_power --tag lite_mae_control --seed 0 --save-arrays --no-plots
wpf-benchmark run --model agcrn_lite --config configs/agcrn_lite_low_power_mae.json --experiment lite_low_power --tag lite_low_power_mae --seed 0 --save-arrays --no-plots

# 结构与辅助监督的区分：按需要追加第三组。
wpf-benchmark run --model agcrn_lite --config configs/agcrn_lite_low_power_no_aux.json --experiment lite_low_power --tag lite_low_power_no_aux --seed 0 --save-arrays --no-plots
```

正式对照在每条命令加 `--repeat 5`，使用配对的 seed 0–4。新配置通过以上独立命令运行，不纳入原本冻结的 `rerun_graph_ablation` 计划。结果仍以 `agcrn_lite` 为模型名，比较时用 tag 和 `model_config` 区分，不能把三组混成同一模型的种子重复。

## 结果应怎样检查

沿用已有总体 MAE/RMSE、逐时距、爬坡和日电量报告，并利用 `--save-arrays` 保存的最终功率预测检查：当前和未来都低功率、当前正常后转低功率、当前低功率后回升（含未来至少 50 kW 子组）的 MAE 与平均偏差。低功率误报下降时，也要检查真实回升和日电量负偏差是否恶化。

代码层可调用 `model.network(x, return_aux=True)` 取得 `power`、`state_logits`、`low_power`、`normal_power`，形状均为 `(B,N,H)`；功率值处于训练归一化空间，状态概率为 `sigmoid(state_logits)`。该接口只接收历史 x。标准 CLI 当前保存最终功率数组及损失日志，不额外保存逐样本状态概率或分类校准报告。

开发测试：`tests/test_lite_low_power.py` 覆盖门控两端、分支边界、非零 scaler 下限、缺失历史、池化前特征、递归反馈、分类阈值/掩码、梯度流、关闭开关的等价性、按功率损失恢复 checkpoint，以及 CLI 训练/验证/测试链路。使用开发版已有 `.runtime/` 执行测试，之后按[服务器部署与同步](服务器部署与同步.md)刷新服务器目录与 ZIP。
