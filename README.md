# llm-eval-lab

**LLM 应用的评测与质量回归平台**——把"这版能不能上线"从拍脑袋变成可复现、可阻断的工程流程。

一个项目覆盖三块工程能力：AI 测试（评测集 + 自动化评测 + 版本回归）、
数据开发（入湖校验 + 幂等回填 + 漂移治理）、平台工程（跑批 + 埋点 + CI 门禁）。

> **当前状态**：96 个 pytest 全绿（约 2.6 秒），CI 上能真跑通整条链路。
> **核心产出是两次拦截**，见 `REPORT.md`。
> **没有跑过真实大模型**——生成层是抽取式降级策略，judge 是确定性 StubJudge。
> 详见文末「诚实边界」。

---

## 一、它拦住了什么

```
v4 → v5（放宽安全策略）
  宏平均 0.6014 → 0.5194，置信区间 [-0.241, 0.0] 跨过 0 → 总分说不清
  但 refusal_edge 类 1.000 → 0.000，McNemar p = 0.000488（12 变坏 / 0 变好）
  → 阻断

v3 → v4（"提升召回"的改动）
  宏平均 0.6605 → 0.6014，置信区间 [-0.0961, -0.0206] 不含 0 → 真退化
  coupon 类 -0.1667 但 McNemar p=1.0（0 变坏 / 0 变好）→ 标注为噪声，不据此拦
  → 阻断
```

---

## 二、被测应用（不是 demo）

上一版被测对象是 10 条硬编码 if-else，diff 出来全是 0，平台就是空壳。
这一版是**真的 RAG 客服助手**：

| 组件 | 实现 |
|---|---|
| 语料 | `src/kb.py`，66 篇客服文档，10 个类目，分布刻意不均 |
| 检索 | 自写 BM25，中文 2-gram，无 jieba 依赖 |
| 生成 | 抽取式（可插拔，换成真模型只需替换一个函数） |
| 工具 | 11 条工具，Agent 轨迹评测 |
| 安全 | strict / loose 两套拒答策略 |

五个版本的差异来自**配置项**（top_k / query 改写 / 拒答严格度 / 工具），
不是开关能力位——所以 diff 才有东西可拦。

| v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| 0.4271 | 0.4554 | **0.6605** | 0.6014 | 0.5194 |

（宏平均，326 条评测集）

---

## 三、四层链路

```
数据层  store     入湖 schema 校验 → 幂等 upsert → PSI 漂移检测
执行层  runner    被测应用（协议）× 多版本 → 跑批 → trace 落库 → 断点续跑
评分层  scorers   规则评分器 + LLM-as-judge（协议）→ 解析容错 → golden 校准
判定层  gate      版本 diff → McNemar + bootstrap CI → 退出码 → CI 阻断
```

**贯穿全程的原则**：凡依赖外部模型的地方都做成**协议 + 可插拔实现**。
本地没有可用模型，所以先落能**手算正确答案**的实现，反过来验证评分器和门禁对不对。
等有真模型，换一个实现类接进去，上层一行不改。

### 目录

```
config.json               路径与阈值（不硬编码）
src/
  kb.py                   客服知识库 + 两套拒答策略
  rag.py                  被测应用：BM25 检索 + 可插拔生成 + 版本矩阵
  apps.py                 应用协议与构造（rag / mock / rule / openai）
  store.py                DuckDB 数据层：schema、幂等 upsert、attempt、漂移
  runner.py               跑批：run_id 幂等 + 断点续跑 + 失败不中断
  scorers.py              规则评分器 + judge 协议 + StubJudge
  calibration.py          裁判三类偏差探针 + Wilson 置信区间
  golden.py               人工标注参照系 + 一致率 + Cohen's kappa
  stats.py                McNemar 精确检验 + bootstrap 置信区间
  gate.py                 版本 diff + 统计判定
scripts/
  make_seed.py            生成 326 条评测集（固定 seed，可复现）
  ingest.py               入湖 + 漂移检测
  run_all.py              一键跑通整条链路 ★
  run_batch.py            跑批（--force 新开 attempt）
  score_run.py            评分
  compare.py              成对比较
  calibrate.py            裁判偏差探针
  calibrate_golden.py     评分器 vs 人工标注
  check_regression.py     回归门禁
data/raw/golden.jsonl     34 条人工标注
tests/                    96 个 pytest
```

---

## 四、快速开始

```bash
pip install -r requirements.txt

python scripts/run_all.py                   # 入湖 → 跑批 → 评分 → 比较 → 门禁
python -m pytest tests/ -q                  # 96 passed
python scripts/calibrate_golden.py --judge  # 评分器 vs 人工（kappa 0.954）
python scripts/calibrate.py                 # 裁判偏差探针
```

`run_all.py` 是必需的：改一行 src 代码，run_id 就变，之前跑过的批次全部失效。

### 关键设计

- **run_id = 数据集 + 应用指纹 + 代码指纹**。三者任一变化都落到新批次，
  不会误复用旧结果（代码指纹这条是踩坑加的）
- **存储 append-only**：主表只追加，主键含 `attempt`；查询走 `*_latest` 视图。
  跑完再调用 = 不动；跑到一半 = 补缺失；`--force` = 新开 attempt，旧数据原封不动
- **判定用配对统计检验**：McNemar（对错翻转）+ bootstrap CI（分数差），
  效应量与显著性分开报

---

## 五、诚实边界

- **没跑过真模型**。生成层是抽取式降级，`OpenAIGenerator` 接口留好了
  （填 `base_url + model` 即可接 ollama / vLLM）。judge 是 `StubJudge`。
- **被测应用是自己写的**，为让版本 diff 有东西可拦而造。
- **评测集是脚本生成的**，326 条，不是真实线上流量。
- **golden set 只有 34 条**，标注者是开发者本人，主观性无法排除。
- **宏平均只有 13 个类目**参与，bootstrap CI 宽达 24 个百分点。
- **没做服务化与压测**，没有线上延迟/成本监控。

---

## 六、环境

- Python 3.10.9（Anaconda 发行版实测可用，其他发行版未测）
- duckdb 1.5.6、pandas 1.5.3、numpy 1.23.5、scikit-learn 1.2.1
- ⚠️ 不要升 numpy 到 2.x，会打崩 pandas 1.5.3 的二进制兼容
