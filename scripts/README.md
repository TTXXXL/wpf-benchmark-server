# 训练脚本索引

单模型训练、预处理、权重保存和标准评估使用wpf-benchmark CLI。这里保留需要在服务器运行的批跑入口；更新代码本身不会启动训练。

| 文件 | 用途 |
|---|---|
| [run_study.py](run_study.py) | 当前验证实验的通用入口；读取集中计划，核验续跑，自动生成回传 ZIP |
| [rerun_low_power_review.py](rerun_low_power_review.py) | 已有八组×五种子，复用冻结处理数据，按需重跑与续跑 |
| [rerun_graph_ablation.py](rerun_graph_ablation.py)、[rerun_graph_ablation.sh](rerun_graph_ablation.sh) | AGCRN/Lite、损失和当前功率开关的匹配消融套件 |
| [rerun_m1.py](rerun_m1.py)、[rerun_m1.sh](rerun_m1.sh) | 旧13模型41次M1套件及批跑公共执行工具 |
| [artifact_checks.py](artifact_checks.py) | 批跑依赖的结果读取、完整性和数组对齐函数，无独立分析命令 |

当前命令、结果路径与准备条件见[实验训练](../docs/实验训练.md)。历史入口的模型信息见[模型与消融](../docs/模型与消融.md)。批跑内的身份、哈希和完成项检查保留，避免错误续跑。

通用离线审计、bootstrap、M1/M2复评分及历史分组工具集中在开发版 scripts/local/。已经替代的阶段 2/3 训练交付脚本与说明归档到项目外，不进入服务器 Git 或上传 ZIP。

sync_server_project.py也仅在开发版保留。它使用文档与脚本白名单生成上传副本，操作见[部署与同步](../docs/服务器部署与同步.md)。新增本地文件不会自动被同步到服务器。
