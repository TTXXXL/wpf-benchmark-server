
# 脚本索引

所有命令默认从项目根目录执行。本文区分批量训练、冻结预测的复核和本地维护；单模型训练、清洗、自检与作图使用 `wpf-benchmark` CLI。

## 本轮训练与结果复核

| 入口 | 用途 |
|---|---|
| [rerun_low_power_review.py](rerun_low_power_review.py) | 当前 8 组 × seed 0–4，共 40 次；冻结数据/代码并严格核验续跑结果 |
| [review_diagnostics/](review_diagnostics/README.md) | P0、原始标签、M1/M2 同预测复评分、历史群体、锚点与版本审计；只读已有结果 |
| [paired_bootstrap.py](paired_bootstrap.py) | 通用日块/种子配对检验，默认 standard/auto；`--preset wave1` 提供历史固定预设 |
| [analyze_persistence.py](analyze_persistence.py) | 将 persistence 主表与附表 JSON 生成为中文解读 |

本轮 A/B/C 同为 agcrn_lite，通用 bootstrap 不自动按 tag 区分配置。先按 [40 次说明](../docs/低功率40次统一重跑.md)使用专用运行清单；模型和统计口径见 [评估分析](../docs/评估与结果分析.md)。

## 其他仍可用的实验套件

| 入口 | 用途与保留理由 |
|---|---|
| [rerun_m1.py](rerun_m1.py)、[rerun_m1.sh](rerun_m1.sh) | 旧 13 模型、41 次 M1 基线套件；Python 文件还提供当前两个批跑入口依赖的执行与指纹工具 |
| [rerun_graph_ablation.py](rerun_graph_ablation.py)、[rerun_graph_ablation.sh](rerun_graph_ablation.sh) | 架构 × 损失 × current_power_skip 的图模型实验，包含 AGCRN+MSE；矩阵与本轮低功率八组不同 |
| [official_mask_rescore.py](official_mask_rescore.py) | 从原始观测重评历史 M0/Wave 1 数组，复核旧结果中的官方规则影响；不接收新 M1/M2 运行 |
| [mask_rule_ablation.py](mask_rule_ablation.py) | 历史 M0 的逐规则、重叠及新增群体诊断；依赖前一项，保留历史证据复现能力 |

两个 `.sh` 文件负责定位项目、设置 PYTHONPATH 并使用已激活的解释器，不含另一套训练逻辑。M1/图模型旧套件首次运行会重建数据；本轮入口按说明复用数据或显式 `--prepare`，不能混用准备策略。

## 本地维护

[sync_server_project.py](sync_server_project.py) 从开发版生成上传副本与 ZIP，并提供 `--check`。此维护脚本不随上传包分发；本地验证后再运行，步骤见 [部署与同步](../docs/服务器部署与同步.md)。

2026-10-08 删除一次性合成样图生成脚本，将配对检验的实现与入口合并到 `paired_bootstrap.py`。合成图的测试夹具与图件测试保留在开发版 `tests/`。历史 Wave 1 调用统一改为 `python scripts/paired_bootstrap.py --preset wave1 ...`，数值计算和统计定义保持原实现。
