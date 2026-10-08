# SDWPF 风电功率预测基准

本项目将 SDWPF 数据检查、因果清洗、模型预测和统一评估组织为 Python 包。默认预测 134 台机组未来 12 个十分钟有功功率值，使用过去 144 步，按 196/25/24 天划分训练、验证与测试。

当前任务是 **8 组对照 × seed 0–4，共 40 次统一重跑**。新副本先看 [执行与续跑说明](docs/低功率40次统一重跑.md)，用 `--prepare` 完成一次数据准备；已有匹配清洗数据则不传该参数。

## 文档入口

完整导航和合并说明见 [docs 文档索引](docs/README.md)。

| 任务 | 文档 |
|---|---|
| 首次使用、清洗、模型接入 | [使用指南](docs/使用指南.md) |
| 开发验证、上传副本、ZIP 和 Git | [部署与同步](docs/服务器部署与同步.md) |
| M1/M2、指标、结果核验、配对统计、图件 | [评估与结果分析](docs/评估与结果分析.md) |
| 基线、当前功率开关、状态头、方法模型 | [模型与消融](docs/模型与消融.md) |
| 本轮 40 次及旧 M1 41 次套件 | [实验批跑](docs/低功率40次统一重跑.md) |
| 既有证据与尚未实现的 D 设计 | [评审与后续方向](docs/低功率诊断与实现规格评审.md) |

## 首次运行

```bash
python -m pip install -e .
wpf-benchmark preprocess
wpf-benchmark fit-power-curve
wpf-benchmark eval selftest
wpf-benchmark run --model persistence --config configs/persistence.json --save-arrays --no-plots
```

Python 支持 3.8–3.12，深度模型另需 PyTorch。原始 CSV 位于 `data/raw/sdwpf/`；清洗数据与报告由运行生成。数据和报告不纳入 Git。开发版 `wpf-benchmark/` 是唯一修改入口，验证后同步到上传副本，操作见部署说明。
