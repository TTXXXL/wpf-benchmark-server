# 实现规格书：评估协议模块 `src/eval_protocol.py`

> **口径更新（2026-09-30）**：本文原有主表闸门和 4,948,656 格参考数字是历史 M0；新运行只允许 M1/M2，默认 M1，详见[目标掩码口径](目标掩码口径_M1_M2.md)。下文旧数字不作为当前验收标准。

> 交给 Codex 实现。本文档保留历史协议说明；目标掩码以 2026-09-30 口径更新为准。
> 背景文档（为什么这样设计，可读可不读）：`reports/eval_protocol_evidence.md`、`reports/cleaning_report.md`
> 撰写：ClawsGO Science Agent · 2026-09-24

---

## 0. 项目现状（不要重做的部分）

repo 位于 `wpf-benchmark/`，以下已完成并验证，**不要改动其逻辑**：

- `src/eda_sdwpf.py` — 数据 EDA（已完成）
- `src/preprocess_sdwpf.py` — 清洗管道（已完成），产出 `data/processed/sdwpf_clean.parquet` 与 `sdwpf_meta.json`
- `src/k_selection.py` — 邻居数标定（已完成）
- 现有 `src/eval_protocol.py` 是我手写的半成品初版，**可以推倒重写**，但其中 `_fit_power_curves`、`to_wide`、`_valid_stack` 的逻辑经调试已可用，可参考
- `src/selftest_eval.py` — 自检脚本雏形，实现完成后应改为模块 CLI 子命令（见 §9）

## 1. 运行环境（硬约束）

| 项 | 值 | 对实现的影响 |
|---|---|---|
| Python | **3.8.12** | 所有含类型标注的模块顶部加 `from __future__ import annotations`；禁用 `X \| Y` 联合类型语法（3.10+）、match 语句 |
| PyTorch | 1.11.0+cu113 | 模型代码按 1.11 API 写（本模块本身不用 torch） |
| GPU | RTX 3090 24GB | 与本模块无关，供后续模型用 |
| pyarrow | **服务器未安装** | 数据读写必须兼容：优先 parquet，`ImportError` 时降级 csv.gz |
| numpy / pandas | 1.21.2 / 2.0.3 | 不要用 numpy 1.22+ 才有的 API |
| 执行环境 | gVisor 沙箱，**会硬杀高内存进程** | 大数组用 float32、及时 `del` 中间量（见 §8.5） |

## 2. 输入数据规格

### 2.1 `data/processed/sdwpf_clean.parquet`（读取失败则读同目录 `sdwpf_clean.csv.gz`）

4,727,520 行 = 245 天 × 144 步 × 134 台机，长表，列：

| 列 | 类型 | 含义 |
|---|---|---|
| `ts` | timedelta64 | 绝对时间索引（0 起，10min 步长，全局唯一） |
| `Day` | int | 1–245 |
| `TurbID` | int | 1–134 |
| `Wspd Wdir Etmp Itmp Ndir Pab1 Pab2 Pab3 Prtv Patv` | float | 物理单位特征（已清洗：缺失已插补、温度坏点已置NaN重补、Patv<0 已截0、Wdir 已回绕到[-180,180)） |
| `Wsin Wcos` | float | 风向 sin/cos 特征 |
| `m_missing` | bool | 原始缺失行 |
| `m_imputed` | bool | 该行任一特征被填补过 |
| `m_outlier` | bool | Patv 曾被判功率曲线离群并重补 |
| `f_fault` | bool | 停机/故障（Patv≤0 且 Wspd≥3） |
| `f_curtail` | bool | 限电/降额（残差 z≤−3 且 Wspd≥5） |
| `f_stuck` | bool | 风速计卡死 |
| `f_farm` | bool | 当日停机的机组×时刻格占比 >50%（本版 Day 17–20） |

单位：功率 kW，额定功率约 1550（实测饱和 1520–1567）；风速 m/s。

### 2.2 `data/processed/sdwpf_meta.json`

训练段（前 196 天）各特征 min/max，供**训练脚本**归一化用（防数据泄漏）。本模块不需要读它，但不要破坏它。

## 3. 任务设定与数据划分

- **划分**（按时间，严禁 shuffle）：`Day 1–196` 训练 / `Day 197–221` 验证（25 天）/ `Day 222–245` 测试（24 天）。天数写成配置参数 `train_days=196`、`val_days=25`
- **预测任务**：给定过去 `input_window=144` 步（1 天）历史，预测未来 `horizon=12` 步（2 h）的 Patv，10min 分辨率
- **预测数组约定**：任何模型的预测统一为 numpy 数组 `(T_eff, N, H)`，其中样本 `t` 的第 `h` 列是对目标时刻 `t+h`（1-based h）的预测；`T_eff = T_test − H`；N=134
- **真值数组**：同样 `(T_eff, N, H)`，由测试段宽表 Patv 沿时间堆叠而来。堆叠切片务必写对（见 §8.1）

## 4. 配置类 `ProtocolConfig`（dataclass，全部可调，这是用户明确要求的）

```python
@dataclass
class ProtocolConfig:
    # 任务设定
    input_window: int = 144
    horizon: int = 12
    train_days: int = 196
    val_days: int = 25
    steps_per_day: int = 144
    # 数据卫生：主表排除的行级 flag（可增删）
    exclude_flags_main: tuple = ("m_missing", "m_imputed", "m_outlier",
                                 "f_fault", "f_curtail", "f_farm")
    # A 层
    rated_power_kw: float = 1550.0
    farm_min_valid_frac: float = 0.85   # 场站级汇总的最低有效率（实测每时刻有效率上限 ~0.948，见 §8.2）
    # B 层
    wind_bins: tuple = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 40))
    stability_block_days: int = 7
    # C 层
    ramp_threshold_frac: float = 0.10       # 爬坡阈值 = 额定功率的 10% ≈ 155 kW
    ramp_window_steps: int = 6              # 预留给未来事件级扩展；当前逐格判别不使用
    high_wind_speed: float = 18.0           # 大风段下限
    # D 层
    violation_sigma: float = 3.0
    curve_bin_width: float = 1.0
    curve_min_samples: int = 30
    # 物理界（留作后续物理约束/分段实验接口）
    cut_in: float = 3.0
    cut_out: float = 25.0
```

## 5. 数据卫生规则（每个指标先过这道闸）

- **主表**（默认）：仅使用 `Patv` 有限 且 所有 `exclude_flags_main` 均为 False 的 (t, j) 行
- **附表**（`table="all"`）：仅要求清洗后的 `Patv` 有限，可能包含插补、停机和限电目标；这是宽松口径参考，不等于部署误差
- `m_imputed` 行不进入主表评分；附表仍可能计入插补目标，报告时须标明
- 同一 `Evaluator` 实例可对同一预测分别出 main/all 两张表

## 6. 指标定义（五层）

记号：`e = P̂ − P`（kW）；"有效样本"指通过 §5 闸门的 (t, j, h) 三元组。

### A 层 · 精度主表

| 指标 | 定义 | 聚合口径 |
|---|---|---|
| MAE | mean(\|e\|) | 双口径都要：单机级（池化全部有效样本）+ 场站级 |
| RMSE | sqrt(mean(e²)) | 同上 |
| NMAE | 单机级：MAE / `rated_power_kw` × 100 (%)；场站级：逐格求 `100 × |Σ有效机组误差| / (有效机组数 × rated_power_kw)` 后取均值 | 分母与各格实际计入的机组一致 |
| 技巧分 SS | 100 × (1 − MAE_模型 / MAE_持续性)（单机级口径） | 持续性基准 = P̂(t+h) = P(t)（输入窗最后一步实测），在模块内部计算 |

**场站级定义**：某发布时刻与时距的场站误差 = Σ_j e(t,j,h)（无效风机的误差按 0 计）；仅当该格有效率至少为 `farm_min_valid_frac` 且至少有一台有效机组时计入。场站 MAE/RMSE 使用该有效子集的误差和；场站 NMAE 按该格有效机组的额定容量归一化后再平均。因此该口径是“有效机组子集”的场站误差，不等于全场实际注入功率误差。**不要要求全部风机有效——那样一个时刻都不会通过**（见 §8.2）。

### B 层 · 误差结构（单机级口径）

1. **逐时域误差曲线**：h = 1..H 各自的 MAE/RMSE
2. **分风速段误差**：按目标时刻的**实测** Wspd 落入 `wind_bins` 分箱，各箱 MAE/RMSE/样本数
3. **偏差分解**：ME = mean(e)；高估占比 = mean(e>0)；低估占比 = mean(e<0)（高估/低估不对称风险，Chen 2022）
4. **分块稳定性**：测试段按 `stability_block_days` 天分块，各块单机级 MAE

### C 层 · 工业工况

1. **爬坡逐格判别**：在每个发布时刻×机组×时距格，真实阳性 = |P(t+h) − P(t)| ≥ `ramp_threshold_frac × rated_power_kw`；预测阳性 = |P̂(t+h) − P(t)| ≥ 同阈值。输出阳性格数、逐格 recall/precision/F1、真实阳性格上的 MAE。重叠格不代表独立事件。
2. **合格机组子集的逐日电量误差**：仅汇总 `h=1` 的合格目标格，按测试日求 Σ(预测/真值功率) × (10min/60min) / 1000 → MWh；输出 MAE_MWh 与 MAPE_pct。首尾日只统计有预测的时段，不是完整风场或结算电量
3. **大风段**：实测 Wspd ≥ `high_wind_speed` 的样本单独报 MAE/RMSE/n；若 n ≤ 100 只报 n 并注明"样本不足"

### D 层 · 物理合理性

功率曲线拟合（**只在训练段拟合**，防泄漏；拟合时排除停机行与 Patv≤0）：
- 逐机、按 `curve_bin_width` 风速箱取 Patv 中位数；每箱至少 `curve_min_samples` 个样本，不足的箱用有效箱线性插值平滑
- 残差稳健 σ：逐机逐箱 MAD × 1.4826（箱内样本 <100 时回退到整机 σ），σ 下限 5 kW

指标：
1. **功率曲线违背率**：有效样本中 |P̂ − curve(Wspd_实测)| > `violation_sigma × σ` 的比例 (%)。**同时报告真实值的违背率作为参照**（本版主表实测约 0.73%；模型违背率下降仍需与 MAE 联合解读）
2. **负功率率**：mean(P̂ < 0)
3. **超额定率**：mean(P̂ > `rated_power_kw`)

### E 层 · 概率化（本轮只留占位）

结果 JSON 中放 `"E_probabilistic": null`，注释说明论文②启用（pinball / PICP / PINAW / CRPS）。

## 7. 模型扩展接口（用户明确要求"以后可继续加模型"）

```python
MODEL_REGISTRY: Dict[str, type] = {}

def register_model(name: str):
    """装饰器。@register_model("lstm_seq2seq") class Lstm(BaseForecaster): ..."""

class BaseForecaster:
    name: str = "base"
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame = None) -> None: ...
    def predict(self, history: np.ndarray) -> np.ndarray:
        """history: (N, F, input_window) 物理单位特征窗；返回 (N, horizon) kW 预测"""
```

```python
def eval_forecaster(model, cfg=None, tag="") -> Dict:
    """统一入口：split_by_day → 训练段归一化（用 sdwpf_meta.json 的 min/max）
    → 滑窗构造 history → 批量 predict 组装 (T_eff, N, H) → Evaluator.evaluate
    → save_result。首个基线实现时填充；接口签名现在就固定。"""
```

要求：任何模型接入后**不改本模块任何指标代码**即可得到 §6 全部指标。

## 8. 实现注意事项（我在调试中实际踩过的坑，务必遵守）

1. **时域堆叠差一错误**：`(T_eff, N, H)` 堆叠必须写 `base[h : T−H+h] for h in 1..H`。写成 `T−(H−1)+h` 会在 h=H 时越界/形状不一致
2. **场站级全有效率不可能达成**：每时刻平均约 5% 风机被排除项覆盖（实测有效率上限 ~0.948），要求 100% 或 ≥95% 都会得到空集。用 `farm_min_valid_frac=0.85`
3. **除零护栏**：persistence 预测永远不会触发爬坡预测事件（|P̂−P(t)|≡0），precision 分母为 0；MAPE、F1、SS 同理。所有比值必须 `if denominator else NaN`
4. **空切片**：`_metrics` 在选择为空时返回 `{"MAE_kW": nan, ...}`，不得抛异常或让 numpy 发 RuntimeWarning（验收标准包含无警告）
5. **内存**：沙箱会硬杀高内存进程。(T_eff,N,H)=(3444,134,12) 的 float64 数组约 44MB，评估过程会同时存在 6–10 个——一律转 float32（功率值精度足够），每层算完 `del` 不再用的中间量（pers/perr、ramp 掩码、last 等）
6. **JSON 序列化**：结果含 numpy 标量/NaN，`json.dumps(..., default=float)`；建议把 NaN 转成 `None` 保证严格合法 JSON
7. **功率曲线/σ 只能从训练段拟合**——这是数据泄漏防线，不要图省事用全数据
8. **D 层风速用实测值**：违背率查曲线时用目标时刻的**实测** Wspd（模型预测的风速不参与，D 层评估的是"给定真实风况下预测功率的物理合理性"）
9. **Python 3.8 兼容**：`from __future__ import annotations`；不用 PEP 604 / match
10. **数据读取双路径**：parquet → 失败(ImportError) → csv.gz；不要硬编码 parquet

## 9. 交付物与 CLI

```
src/eval_protocol.py     # 本规格的全部实现
```

CLI：
```
python src/eval_protocol.py --selftest     # 运行 §10 自检，写 reports/eval/selftest_*.json 与 reports/eval_selftest.md
```

输出：
- `save_result(R, tag)` → `reports/eval/{tag}_{时间戳}.json`（完整结果 + `asdict(config)`，保证可复现）
- `result_to_markdown(R)` → 人类可读报告（A 主表 4 列 × 双口径、B 时域曲线与分风速段表、C 三项、D 三项）

## 10. 验收标准（persistence 自检必须全部满足）

用持续性预测 `P̂(t+h)=P(t)` 跑 main 表：

| 检查项 | 预期值 |
|---|---|
| 进程存活、端到端完成 | 无警告（尤其不得出现 "Mean of empty slice"），< 1 分钟 |
| 主表有效样本数 | ≈ 4,948,656（因果填补、训练段拟合、k=12 后实测；容差 ±0.1%） |
| 附表有效样本数 | ≈ 5,537,952 |
| A 层 SS | = 0（persistence 对 persistence） |
| A 层单机级 MAE | ≈ 66.46 kW（新管道 persistence 实测） |
| C 层爬坡 | recall = 0，precision = NaN/None（persistence 不报阳性格，属正确行为），n_positive_cells > 0 |
| D 层真实值违背率 | ≈ 0.73%（新管道实测） |
| D 层 persistence 违背率 | ≈ 17.70%（新管道实测） |
| JSON | 严格合法（无裸 NaN），含完整 config |
| farm 级 | 有非空样本（≥0.85 阈值下约 90%+ 时刻有效） |

以上实测值来自本机（同样数据同样配置）调试运行，可作为 Codex 实现后的回归校验数字。

## 11. 明确不在本轮范围

- 任何具体预测模型（LSTM/Transformer/GNN）——接口固定，模型后续加
- E 层概率指标实现——只留 `"E_probabilistic": null` 占位
- 训练/归一化逻辑——属于未来 `eval_forecaster` 与模型侧，`sdwpf_meta.json` 已备好统计量
