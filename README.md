# SDWPF 风电功率预测基准

默认使用过去144个十分钟观测，预测134台机组未来12步有功功率。训练、验证、测试按196/25/24天划分，默认M1目标与评分。

模型与训练接口已恢复到TCN改动前的版本（24f862b）。本轮TCN、共享线性增量模型与密集验证导出已撤回；已有AGCRN/Lite、低功率头和验证最佳权重功能保留。原40次及阶段3结果无需因本次整理而重跑。

## 文档

| 内容 | 入口 |
|---|---|
| 安装、数据准备与单模型运行 | [使用指南](docs/使用指南.md) |
| 服务器部署、开发版同步与Git | [部署与同步](docs/服务器部署与同步.md) |
| 模型、损失、锚点和消融配置 | [模型与消融](docs/模型与消融.md) |
| 最佳权重保存与加载 | [模型权重](docs/模型权重保存与加载.md) |
| M1/M2、指标、结果文件与图件 | [评估与结果分析](docs/评估与结果分析.md) |
| 已有40次与旧M1套件的重跑入口 | [实验批跑](docs/低功率40次统一重跑.md) |

完整导航见[文档索引](docs/README.md)，训练入口见[脚本索引](scripts/README.md)。离线验证、诊断和阶段性记录只保存在本地开发版的scripts/local/与docs/local/，不进入服务器Git或上传ZIP。

## 首次部署

```bash
python -m pip install -e '.[deep]'
wpf-benchmark preprocess
wpf-benchmark fit-power-curve
wpf-benchmark eval selftest
wpf-benchmark models
wpf-benchmark run --model persistence --config configs/persistence.json --save-arrays --no-plots
```

已有冻结处理数据的服务器更新代码时，保留data/processed/与reports/，不重复执行首次预处理步骤。Python支持3.8–3.12；GBDT另需对应后端。

开发版wpf-benchmark/是唯一修改入口，使用已有.runtime/和tests/验证后同步到wpf-benchmark-server/与ZIP。Git只提交代码、配置、运行说明及目录占位，不提交数据、报告或ZIP。
