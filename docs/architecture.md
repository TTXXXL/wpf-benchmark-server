# 工程结构与扩展约定

```mermaid
flowchart LR
    A[原始 SDWPF CSV] --> B[preprocessing/sdwpf.py]
    B --> C[clean Parquet + train-only meta]
    C --> D[runner.py]
    D --> E[models/ 中注册的模型]
    E --> F[预测 T_eff × N × H]
    F --> G[evaluation/evaluator.py]
    G --> H[JSON + Markdown 报告]
```

`cli.py` 仅负责解析命令和配置；`paths.py` 解析项目目录；`data/` 负责 Parquet 读取、时间划分、机组对齐和滑窗。清洗、模型与评估之间通过明确的数组形状连接。

## 模型契约

- `features` 是清洗后数据中的输入列，按声明顺序组成 `F` 轴。训练 DataFrame 还包含 `Patv` 目标和所有数据质量标记。
- `fit(train, valid)` 接收严格按天划分的数据。模型可使用运行器提供的 `self.scaler`；其统计量来自前 196 天的 `sdwpf_meta.json`，调整训练天数时改从新的训练段计算。
- `predict(history)` 接收 `(N,F,input_window)`；`predict_batch(histories)` 接收 `(B,N,F,input_window)`。默认输入是物理单位，模型设置 `history_scale="normalized"` 后改用训练段 min/max 缩放的输入。
- 模型输出必须为物理单位 kW，形状分别为 `(N,horizon)` 或 `(B,N,horizon)`。运行器负责合成 `(T_eff,N,H)`，并检查形状与有限性。

## 时间与评估不变量

- 划分按 Day 单调进行，不打乱数据。首个测试预测使用测试前训练／验证段最后 `input_window-1` 个时间步及测试段当前时间步作历史，不读取目标时刻及更晚的信息。
- 样本起点 `t` 的第 `h` 列预测目标 `t+h`，其中 `h=1..H`；`T_eff=T_test-H`。
- 训练段单独拟合功率曲线。主表按配置排除质量标记；附表仅检查目标 Patv 有限。
- 统计结果由同一 `Evaluator` 对相同预测分别生成主表与附表。配置、模型参数、输入特征和缩放来源写入结果 JSON。

新增模型的最小改动是增加 `models/<name>.py`、在 `models/__init__.py` 导入它，并增加一个配置文件和形状测试。指标代码保持不变。
