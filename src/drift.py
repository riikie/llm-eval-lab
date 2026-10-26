# -*- coding: utf-8 -*-
"""数据漂移检测（PSI）。

从上一个项目 sentiment_eval 继承的两条硬经验：

1. **PSI 在某个类别占比掉到 0 时会爆炸式放大**。
   均匀四分类 -> [0.5,0.5,0,0]，朴素估计 2×0.25×ln2 ≈ 0.35，**实际算出 6.56**。
   原因是 ln(1e-6/0.25) ≈ -12.4 把结果抬了一个数量级。
   所以：必须做 EPS 截断，而且**不能只看 PSI 一个数**——
   本模块同时输出「偏移幅度」(占比差的绝对值之和 / 2)，两个一起看才不会误判。

2. **检测器本身也要被检测**。写了 detector 不去验证它真能报警，等于没写。
   `inject_drift` 就是干这个的：人为制造漂移，看它报不报。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6

PSI_WARN = 0.10
PSI_ALERT = 0.25


def psi(expected, actual, eps: float = EPS) -> float:
    """群体稳定性指数 PSI。入参是**占比**向量（会自动归一化）。"""
    e = np.asarray(expected, dtype=float) + eps
    a = np.asarray(actual, dtype=float) + eps
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def shift_magnitude(expected, actual) -> float:
    """偏移幅度：占比差绝对值之和的一半，取值 [0,1]。

    PSI 没有上界、对零概率过敏，这个量有上界、直观。配合 PSI 一起看。
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.sum() <= 0 or a.sum() <= 0:
        return 0.0
    e = e / e.sum()
    a = a / a.sum()
    return float(np.abs(a - e).sum() / 2)


def proportions(values, categories) -> np.ndarray:
    """按给定类别顺序统计占比。缺失类别记 0（PSI 那边会做 EPS 截断）。"""
    s = pd.Series(values).astype(str)
    counts = s.value_counts()
    return np.array([float(counts.get(c, 0)) for c in categories])


def length_buckets(lengths, edges) -> np.ndarray:
    """把长度切成桶，返回每桶占比。edges 是**上界**序列，最后一桶兜底。"""
    idx = np.digitize(np.asarray(lengths, dtype=float), np.asarray(edges, dtype=float))
    n = len(edges) + 1
    out = np.zeros(n)
    for i in idx:
        out[i] += 1
    return out / max(len(lengths), 1)


def compute_drift(ref: pd.DataFrame, cur: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """对比两份数据，逐维度算 PSI。

    ref / cur 需要有 prompt 列；tags / source 列可选。
    返回列：dimension, category, psi, shift, level
    """
    import json
    cfg = cfg or json.load(open(ROOT / "config.json", encoding="utf-8"))["drift"]
    rows = []

    # 维度 1：prompt 长度分桶
    edges = cfg.get("length_buckets", [0, 32, 64, 128, 256, 512, 100000])
    r = length_buckets(ref["prompt"].astype(str).str.len(), edges)
    c = length_buckets(cur["prompt"].astype(str).str.len(), edges)
    rows.append(("prompt_length", "bucket", psi(r, c), shift_magnitude(r, c)))

    # 维度 2：tags 分布
    if "tags" in ref.columns and "tags" in cur.columns:
        cats = sorted(set(ref["tags"].dropna().astype(str)) | set(cur["tags"].dropna().astype(str)))
        if cats:
            r = proportions(ref["tags"], cats)
            c = proportions(cur["tags"], cats)
            rows.append(("tags", "all", psi(r, c), shift_magnitude(r, c)))

    # 维度 3：source 分布
    if "source" in ref.columns and "source" in cur.columns:
        cats = sorted(set(ref["source"].dropna().astype(str)) | set(cur["source"].dropna().astype(str)))
        if cats:
            r = proportions(ref["source"], cats)
            c = proportions(cur["source"], cats)
            rows.append(("source", "all", psi(r, c), shift_magnitude(r, c)))

    df = pd.DataFrame(rows, columns=["dimension", "category", "PSI", "shift"])
    df["level"] = pd.cut(
        df["PSI"], [-1, cfg.get("psi_warn", PSI_WARN), cfg.get("psi_alert", PSI_ALERT), 1e9],
        labels=["无漂移", "中等", "显著"],
    ).astype(str)
    return df.sort_values("PSI", ascending=False).reset_index(drop=True)


def has_alert(df: pd.DataFrame) -> bool:
    return bool((df["level"] == "显著").any())


def inject_drift(df: pd.DataFrame, seed: int = 0,
                 drop_frac: float = 0.8, lengthen_frac: float = 0.6,
                 repeat: int = 4, new_channel_frac: float = 0.35) -> pd.DataFrame:
    """人为制造漂移，用来验证检测器真的会报警。

    三刀，对应三种真实漂移场景：
      1. 某类问题锐减（多数 tag 砍掉 drop_frac）→ tags 分布漂移
      2. 用户提问变长（lengthen_frac 比例的 prompt 重复 repeat 遍）→ 长度分布漂移
      3. 新渠道涌入（new_channel_frac 的 source 换成 new_channel）→ 来源分布漂移

    【改过一次】第 2 刀原本是"截短到 40 字"，但种子集 prompt 全都短于 40 字，
    这一刀完全无效（PSI 恒为 0）——检测器没坏，是注入器没打到点上。改成拉长。
    这个教训本身值得记：验证工具时要确认"注入的变化真的发生了"。
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)

    # 刀 1：多数 tag 类别锐减
    if "tags" in out.columns and out["tags"].notna().any():
        top = out["tags"].astype(str).value_counts().idxmax()
        idx = np.where((out["tags"].astype(str) == top).values)[0]
        if len(idx):
            drop = rng.choice(idx, int(len(idx) * drop_frac), replace=False)
            out = out.drop(out.index[drop])

    out = out.reset_index(drop=True)
    n = len(out)

    # 刀 2：把一部分 prompt 拉长
    if n:
        k = rng.choice(n, int(n * lengthen_frac), replace=False)
        prompts = out["prompt"].astype(str).values.copy()
        for i in k:
            prompts[i] = " ".join([prompts[i]] * repeat)
        out["prompt"] = prompts

    # 刀 3：新渠道涌入
    if n and "source" in out.columns:
        k = rng.choice(n, int(n * new_channel_frac), replace=False)
        src = out["source"].astype(str).values.copy()
        src[k] = "new_channel"
        out["source"] = src

    return out
