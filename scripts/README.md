# 训练脚本索引

单模型训练、预处理、权重保存和标准评估使用wpf-benchmark CLI。这里保留需要在服务器运行的批跑入口；更新代码本身不会启动训练。

| 文件 | 用途 |
|---|---|
| [rerun_low_power_review.py](rerun_low_power_review.py) | 已有八组×五种子，复用冻结处理数据，按需重跑与续跑 |
| [rerun_graph_ablation.py](rerun_graph_ablation.py)、[rerun_graph_ablation.sh](rerun_graph_ablation.sh) | AGCRN/Lite、损失和当前功率开关的匹配消融套件 |
| [rerun_m1.py](rerun_m1.py)、[rerun_m1.sh](rerun_m1.sh) | 旧13模型41次M1套件及批跑公共执行工具 |
| [artifact_checks.py](artifact_checks.py) | 批跑依赖的结果读取、完整性和数组对齐函数，无独立分析命令 |

运行参数与准备条件见[批跑说明](../docs/低功率40次统一重跑.md)和[模型与消融](../docs/模型与消融.md)。批跑内的身份、哈希和完成项检查保留，避免错误续跑。

离线审计、bootstrap、M1/M2复评分、历史分组及阶段2/3验证交付入口集中在开发版scripts/local/；其说明位于本地docs/local/。这些文件不进入服务器Git或上传ZIP。

sync_server_project.py也仅在开发版保留。它使用文档与脚本白名单生成上传副本，操作见[部署与同步](../docs/服务器部署与同步.md)。新增本地文件不会自动被同步到服务器。
