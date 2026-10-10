# SDWPF 风电功率预测基准

默认使用过去144个十分钟观测，预测134台机组未来12步有功功率。训练、验证、测试按196/25/24天划分，默认M1目标与评分。

当前实验使用现有 C 模型比较 M1/M2 训练与早停口径，各三个种子。服务器更新后直接运行：

```bash
git pull --ff-only
python -u scripts/run_study.py --study m2_robustness
```

完成后交回 `reports/studies/m2_core_20261010_v1/m2_core_20261010_v1_delivery.zip`，由本地统一分析。中断后执行同一命令可核验并续跑。数据与模型源码必须匹配计划冻结指纹，详见[实验训练](docs/实验训练.md)。

## 文档

| 内容 | 入口 |
|---|---|
| 安装、数据准备与单模型运行 | [使用指南](docs/使用指南.md) |
| 服务器部署、开发版同步与Git | [部署与同步](docs/服务器部署与同步.md) |
| 模型、损失、锚点和消融配置 | [模型与消融](docs/模型与消融.md) |
| 最佳权重保存与加载 | [模型权重](docs/模型权重保存与加载.md) |
| M1/M2、指标、结果文件与图件 | [评估与结果分析](docs/评估与结果分析.md) |
| 当前训练命令、续跑与结果交付 | [实验训练](docs/实验训练.md) |

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
