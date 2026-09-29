# 论文图件管道

每次 `wpf-benchmark run` 完成后会自动刷新图表；也可在项目根目录单独运行 `wpf-benchmark plots --all`。输出位于 `reports/figs/paper/`。图 1（研究框架）和图 4（模型架构）需手绘；其余图和表由 [`src/wpf_benchmark/figures/`](../src/wpf_benchmark/figures/) 的独立函数生成。画面文字为英文，PDF 中的文字保留为可编辑字体，PNG 为 300 dpi。

## 日常用法

```powershell
wpf-benchmark run --model persistence --config configs/persistence.json
wpf-benchmark plots --all
wpf-benchmark plots --figures 6,7,13 --tables 1,2
wpf-benchmark plots --all --out reports/figs/paper --wind-direction 270
```

最后一个命令用传入的气象风向（度）覆盖图 2 根据清洗数据计算的主导风向。其他命令中的输入位置和输出位置相对于项目根目录。某图缺少素材时会打印 `SKIP figXX: 原因`，已经具备素材的图继续生成；绘图代码错误仍会报错，便于发现问题。

## 结果是怎样找到的

[`runner.py`](../src/wpf_benchmark/runner.py) 一次运行会保存 `main` 和 `all` JSON，另存一份 `<run_id>_arrays.npz`，并向 `reports/eval/index.jsonl` 追加两行索引。索引包含模型名、表名、种子、配置摘要、JSON 路径、创建时间、实验类型及清洗数据/结果代码摘要。图件只聚合当前数据与代码版本的结果；同模型同配置的多个种子可计算均值和标准差。没有索引的旧结果仍可通过文件名扫描读取，但不会与带版本摘要的新结果混合。

NPZ 中必需的数组：`forecasts`、`truth`、`valid`、`ramp_mask` 都是 `(T_eff, N, H)`；`times` 是测试段全部时刻的 `int64` 纳秒时间偏移，长度为 `T_eff + H`；`turbine_ids` 长度为 `N`。另外保存 `wind_speed`、`curtail_mask`（同三维网格）和 `last_power`（`T_eff, N`），分别用于图 5、8、11。读取时会与 JSON 的 `config.horizon` 和 `grid` 核对形状；不一致会报错。

`run` 默认保存数组；对普通模型可用 `--no-save-arrays` 关闭。持续性基线始终保存。`standard` 运行自动进入图 6、7、13 和表 1；W3 的 `ablation` 运行中，最终 `ours` 与 `barest` 也进入这些主结果图表，其余消融变体只进入图 9。新模型的固定颜色和英文显示名可在 [`style.py`](../src/wpf_benchmark/figures/style.py) 的 `MODEL_STYLE` 登记。

只想运行评估、不刷新图件时可给 `run` 加 `--no-plots`。图件会在下一次 `plots` 命令运行时更新。

## 未来实验的输入约定

| 图或表 | 所需输入 |
| --- | --- |
| 图 9 | 主表运行结果带 `experiment="ablation"`；`model` 是四个独立注册的消融名称。图和 `.tex` 表给出多种子的均值 ± 标准差。 |
| 图 10 | 用 `run --validation-only --experiment lambda_sweep` 产生验证结果；`model_config.lambda_consist` 大于 0。只聚合同一数据/代码版本、同一完整配置的种子；相同 λ 存在多种配置时取最新配置。图中只画验证 MAE，不读取测试结果。 |
| 图 12 | `ours` 主表运行的 `reports/interpret/<run_id>_graph.npz` 包含 `A_learned`、`A_wake_prior`、`A_wake`、`wind_dirs`、`w_k`、`turbine_ids`、`convention`。默认精确选择最新的 `ours`；`plots --figures 12 --run-id <run_id>` 可指定。逐风向图另存补充图。 |
| 表 4 | 运行结果带 `experiment="fcurve_sensitivity"`，`model_config.f_curve_source` 是曲线类型名称。 |

生成这些输入后再次执行 `wpf-benchmark plots --all` 即可更新相应图表。图 9 的消融表也随图生成。各图的代码 docstring 提供英文 caption 草稿；投稿前仍应结合最终实验结果核对文字结论。
