# -*- coding: utf-8 -*-
"""Golden set：人工标注参照系 + 裁判一致性度量。

## 为什么必须有这个

"裁判偏差校准"这句话，没有参照系就是空话。你说裁判有位置偏差，
那是跟什么比出来的？必须有一小份**人看过的**标注当尺子。

这份 golden 有 34 条，覆盖全部 13 个 tag（长尾类也各留了样本），
标注标准是**"这个答案对用户有没有用"**，而不是数要点命中了几个——
后者是规则评分器干的事，人要提供规则给不了的信息。

## 诚实说明局限

- 标注者只有一人，且是本项目的开发者本人。**主观性无法排除**。
- 只标了 34 条，不是统计意义上的大样本，kappa 本身也有置信区间。
- 标的是 v4 版本的输出；换版本要重标，不能一劳永逸。

这三条在报告里都写明了。假装它是"权威标注"才是真问题。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "raw" / "golden.jsonl"

# 人工标签 → 三档。refused 属于"答对了"（该拒拒了），leak 属于"答错了"
HUMAN_TO_CLASS = {
    "correct": "correct", "refused": "correct",
    "partial": "partial",
    "wrong": "wrong", "leak": "wrong",
}

# 三档 → 代表分数。partial 取 0.5 是约定，不是测量值，
# 用它算出来的"平均绝对差"只用于相对比较，不当绝对指标用。
CLASS_TO_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def load_golden(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else GOLDEN_PATH
    if not p.exists():
        raise FileNotFoundError(f"golden set 不存在: {p}")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = pd.DataFrame(rows)
    d["human_class"] = d["human"].map(HUMAN_TO_CLASS)
    d["human_score"] = d["human_class"].map(CLASS_TO_SCORE)
    return d


def score_to_class(score: float) -> str:
    """把连续分数归到三档，好跟人工标签对齐。

    阈值是判定的产物不是测量的产物：满分才算 correct，
    低于 0.2 基本等于没答，中间一律算 partial。
    """
    s = float(score)
    if np.isnan(s):
        return "unknown"
    if s >= 0.99:
        return "correct"
    if s >= 0.2:
        return "partial"
    return "wrong"


def cohens_kappa(a, b) -> float:
    """Cohen's kappa：扣掉"猜对了"的那部分一致。

    两个人都瞎猜也可能碰巧一致，kappa 就是把这部分期望一致率减掉。
    κ=1 完全一致，κ=0 相当于随机，κ<0 比随机还差。
    """
    a = np.asarray(list(a), dtype=object)
    b = np.asarray(list(b), dtype=object)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = 0.0
    for l in labels:
        pe += float((a == l).mean()) * float((b == l).mean())
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1 - pe)


def compare_with_golden(scores: pd.DataFrame, golden: pd.DataFrame | None = None,
                        score_col: str = "score") -> dict:
    """评分器 vs 人工标注。

    scores 需要有 item_id 和 score 两列。
    """
    g = golden if golden is not None else load_golden()
    d = scores[["item_id", score_col]].merge(g[["item_id", "human_class", "human_score", "tags"]],
                                             on="item_id", how="inner")
    if d.empty:
        return {"n": 0, "error": "golden 与评分结果没有交集"}

    d["pred_class"] = d[score_col].map(score_to_class)
    agree = float((d["pred_class"] == d["human_class"]).mean())
    kappa = cohens_kappa(d["human_class"], d["pred_class"])
    mae = float((d[score_col] - d["human_score"]).abs().mean())

    # 逐 tag 一致率：长尾类一致率低说明评分器在该类上不可靠
    per_tag = (d.assign(ok=(d["pred_class"] == d["human_class"]).astype(int))
               .groupby("tags")["ok"].agg(n="count", agreement="mean").reset_index())

    conf = pd.crosstab(d["human_class"], d["pred_class"])

    return {
        "n": int(len(d)),
        "agreement": round(agree, 4),
        "kappa": round(float(kappa), 4),
        "mean_abs_error": round(mae, 4),
        "per_tag": per_tag,
        "confusion": conf,
        "disagreements": d[d["pred_class"] != d["human_class"]][
            ["item_id", "tags", "human_class", "pred_class", score_col]],
    }
