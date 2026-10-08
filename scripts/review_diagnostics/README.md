# 冻结预测的离线诊断

这些脚本只读已有预测和数据，不启动训练。正式实现放在开发版 `scripts/review_diagnostics/`，由同步脚本带入服务器版；原实验目录里的两个入口只是兼容包装。新增结果必须写到新目录，非空输出目录和同名文件都会拒绝覆盖。

在项目根目录，使用已安装 numpy/pandas/pyarrow 的 Python。开发版可用已有 `.runtime`；以下命令中的 `python` 必须指向兼容该运行时的解释器。本次使用 Python 3.12，核心逻辑也按项目 Python 3.8 语法约束编写。

## 运行清单

`--runs` 是 UTF-8 JSON 对象，值为 NPZ 去掉 `_arrays.npz` 的完整 run_id，保留原时间戳。示例：

```json
{"A": "lite_control_mae_20261002_073721_911589", "C": "lite_low_power_20261002_061816_059203"}
```

P0 必须包含七个标签：`agcrn_noskip`、`agcrn_skip`、`lite_noskip_mse`、`lite_skip_mse`、`lite_skip_mae`、`B`、`C`。其他 manifest 脚本可以使用任意非空标签集合。每个 run 必须有对应主表 JSON 和 NPZ。

```powershell
python scripts/review_diagnostics/diagnose_low_wind.py --reports '<已有 reports/eval>' --runs '<七标签清单.json>' --out '<新的 P0 输出目录>'
python scripts/review_diagnostics/audit_labels.py --reports '<已有 reports/eval>' --run-id '<参考 run_id>' --project-root . --out '<新的标签审计目录>'
python scripts/review_diagnostics/rescore_m2.py --reports '<已有 reports/eval>' --runs '<清单.json>' --out '<新的 M2 复核目录>'
python scripts/review_diagnostics/analyze_history_groups.py --reports '<已有 reports/eval>' --runs '<清单.json>' --project-root . --history-steps 6 --threshold 1 --sensitivity --out '<新的历史分组目录>'
python scripts/review_diagnostics/analyze_history_groups.py --reports '<已有 reports/eval>' --runs '<A_s0..4/B_s0..4/C_s0..4 清单.json>' --history-source saved-wind --paired-seeds --history-steps 6 --threshold 1 --sensitivity --out '<新的原实验风速历史分组目录>'
python scripts/review_diagnostics/audit_versions.py --repository '<服务器版仓库>' --source-root src/wpf_benchmark --reports '<已有 reports/eval>' --compare-runs '<两个运行的清单.json>' --out '<新的版本审计目录>'
python scripts/review_diagnostics/analyze_anchor_ablation.py --reports '<十组锚点实验的 reports/eval>' --experiment anchor_ablation --out '<新的锚点分析目录>'
```

开发版没有把 `.runtime` 装到解释器环境时，可用如下入口，其他脚本替换模块名即可：

```powershell
python -c "import sys; sys.path[:0]=['.runtime','src']; from scripts.review_diagnostics.rescore_m2 import main; main()" --reports '<已有 reports/eval>' --runs '<清单.json>' --out '<新目录>'
```

## 校验与解释

读取时检查 `(issue,turbine,horizon)` 形状、JSON/NPZ 口径、连续 10 分钟纳秒时间轴、机组唯一性、有效目标及预测的有限性。跨运行逐元素检查 `truth_m1/valid_m1/truth_m2/valid_m2/times/turbine_ids/wind_speed/last_power`，并核对完整 protocol 配置和非空、相同的 data_digest。允许 NaN 只存在于相同的未评分位置；预测始终必须有限。每份结果保存来源路径与指纹。

锚点分析额外强制 code_digest 相同、每格唯一 seed 0–4、模型配置只差 `current_power_skip`。缺指纹、混指纹、重复 seed、缺 seed、错位数组均报错退出。差值统一为 **有锚点 MAE − 无锚点 MAE**，负值有益，只有五个差值全负才输出 `anchor_all_seeds_better=true`。全同向只是描述训练随机性，不能替代时间块不确定性分析。

P0/M2/历史诊断可以复用经版本取证的跨版本预测，code_digest 会逐项记录；它们不能自动把版本差异当作一个因子的因果效应。版本分类按改动路径保守标注，仍需执行路径核验。源文件指纹沿用 `result_provenance` 的路径加原始字节算法，排除图件源码但保留 `figures/io.py`。Git 历史通过 `cat-file --batch` 读对象原始字节，避免 `git archive` 的导出换行转换；另报 LF/CRLF 指纹。当前 checkout 的等价性逐文件比较换行规范化后的完整字节，并不只依赖短哈希。

P0 日界为 SDWPF 相对时间 `floor(issue_time_ns/86400e9)+1`，00:00 切日，不引入时区偏移；按起报日汇总 h=1 的低风速误差，SSE 排序与 SSE 占比一致，输出所有日的 n/ME/MAE/SSE。逐时距 MAE 只使用该时距的预测与掩码。空组指标为 null；零总 SSE 占比为 null；包络样本不足单列，不能当作工况识别。`truth_zero_frac` 为实际等于零的比例，`truth_below_1kw_frac` 另报低于 1 kW 的比例。

M1 和 M2 均使用非负目标。M2 复评分只检验**有效目标群体与掩码敏感性**，同时输出交集、仅 M2、仅 M1；它不能回答“不截零之后还剩多少误差”。persistence 从各数组的同一 `last_power` 构造。

标签审计按 `(TurbID,target_time)` 去重，分开报告预测单元与不同目标数、原始负/零/正功率、清洗功率取值。连接后将本地值转换到 NPZ dtype 再做精确相等检查，没有宽松数值容差。整文件 data_digest 不同会被保留；切片值相等只能认证该切片，不认证全部历史。

## 历史分组

主群体 `H_calm`：起报时刻及此前连续 K 个清洗风速均有限、非负且 `<v`。主分析按规格固定 K=6、v=1 m/s；敏感性为 K=6/12/36 与 v=0.5/1/1.5 的完整网格，不挑选最有利的一格。该阈值不是在查看本测试结果前注册的研究终点；后续确认性验证需要独立时段或数据。若阈值在验证集选定，使用 `--threshold-source validation --selection-period '<实际验证段和选择规则>'`。

缺失、负风速、历史不足不会被当成静风。清洗数据可能包含因果填充，本分析不据此声称原始传感器有效。测试通过改变未来数组，确认 H_calm 不变。

未来整段均有限非负且 `<v` 为未来持续静风；任一步达到阈值即非持续静风。诊断子组 S1=历史静风且未来持续、S2=历史非静风且未来持续、S3=历史静风且未来非持续、S4=二者均非静风；未知单列。S1/S2/S3/S4/unknown 互斥完备，H_calm 与 S1/S3 重叠。未来分组只用于诊断，不能在起报时选择模型。

默认本地与实验 data_digest 必须相同，同时验证全部目标风速和起报功率。`--allow-data-digest-mismatch` 是明确的**探索性本地历史重建**模式：仍精确核验这些值，报告两份指纹及限制，不能宣称训练时历史完全相同。本次历史结果采用该模式，确认性复核需找回原数据快照。

补充 `--history-source saved-wind`：从原 NPZ 重建目标时间轴上的风速，先精确核验所有重叠时距的同一目标风速一致。对于当前起报时刻，以前起报保存的目标风速已是历史观测，H_calm 仍只读取 `<=issue_time` 的时间点。NPZ 未保存首个起报时刻及此前的风速，将它们保留为 NaN；每个 K 的最前 K 个起报时刻历史不足，归 unknown，绝不借用未来填补。该模式无需本地 Parquet，不接受 `--allow-data-digest-mismatch`，证据状态为 `saved_experiment_wind_history`。它认证原实验保存的风速历史，不能恢复其他特征或完整预处理快照。

`--paired-seeds` 要求标签严格为 A_s0..4/B_s0..4/C_s0..4，与元数据种子一致，15 组代码/数据指纹相同，各模型配置跨种子相同。额外输出每群体 MAE/ME 的跨种子均值/样本 SD、配对 C−A、B−A、C−persistence 的五个差值和摘要。未知历史/未来的比例单列，`overall` 为完整 M1，不能与互斥诊断子组相加。原始 experiment 标签在来源里保留，不把三种模型人为改成同一个实验标签。

## 验证

```powershell
python -c "import sys,pathlib,tempfile,unittest; sys.path[:0]=['.runtime','src','.']; p=pathlib.Path('.test-tmp').resolve(); p.mkdir(exist_ok=True); tempfile.tempdir=str(p); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); sys.exit(not r.wasSuccessful())"
```

核心合成测试不依赖 GPU 或 SDWPF 原始数据。锚点符号测试曾在旧脚本上因 `False is not True` 失败，在新实现上通过。原分析输出保留，新产物、红绿回归证据和本次完整运行清单位于工作区 `experiment-analysis/实现交付_20261008/`，不随服务器 ZIP/Git 发布。
