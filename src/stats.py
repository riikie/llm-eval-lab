# -*- coding: utf-8 -*-
"""判定层的统计依据：配对检验，替代"下降超过 X% 就拦"。

## 为什么需要这个

同一份评测集，同一对模型版本，跑两次结果都可能不一样（LLM 采样有随机性，
timeout / 重试也会变）。只看"分数掉了 2%"就拦，会被噪声牵着走：
有时候白拦（其实没退化），有时候漏放（其实退化了）。

配对检验的思路：**不看两个版本各自多少分，只看同一道题上谁对谁错**。
同一道题的难度、歧义、标注噪声，在两个版本上是共有的，配对之后抵消掉了。

## 三个工具

    mcnemar_exact(b, c)        二分类（对/错）的配对检验，精确版
    bootstrap_ci(diffs)        连续分数差的置信区间，不假设正态分布
    paired_verdict(...)        上面两个合起来，给出一个可解释的结论

## 手算校验（写进测试）

    b=0, c=7（7 道题只在候选版上错，0 道只在基线版上错）
    n=7, p = 2 × C(7,0) / 2^7 = 2/128 = 0.0156  → 显著
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双边）。

    b = 基线对、候选错（变坏）的题数
    c = 基线错、候选对（变好）的题数

    只用 b 和 c（不一致对），一致的题数再多也不提供信息 —— 这是配对检验
    的关键：1000 道题里 990 道两版都对，真正的证据只在剩下 10 道里。

    H0: 两个版本错误率相同 → b 和 c 应当接近。
    p = 2 × Σ_{i=0}^{min(b,c)} C(n, i) / 2^n
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0          # 两版完全一致，没有任何证据说有差异
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail / (2.0 ** n)))


def bootstrap_ci(diffs: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """配对差值的 bootstrap 百分位置信区间。

    为什么用 bootstrap 而不是 t 检验：分数分布在 0/1 附近堆着，
    既不正态也不对称，t 检验的前提不成立。bootstrap 只假设"样本能代表总体"。

    重采样的是**样本**（整条 (base, cand) 对一起抽），保持配对结构不被破坏。
    """
    d = np.asarray(list(diffs), dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_verdict(base: Sequence[float], cand: Sequence[float],
                   pass_threshold: float = 0.5, alpha: float = 0.05,
                   n_boot: int = 5000, seed: int = 0) -> dict:
    """对同一批样本的两个版本做配对判定。

    返回三件事，缺一不可：
      - **效应量**：分数掉了多少（delta），以及置信区间
      - **显著性**：掉得是不是超出噪声（CI 是否跨过 0、McNemar p 值）
      - **方向**：变坏的题多还是变好的题多（b vs c）

    只有效应量和显著性分开报，才能避免"显著但无所谓"和"很大但不确定"两种误判。
    """
    b_arr = np.asarray(list(base), dtype=float)
    c_arr = np.asarray(list(cand), dtype=float)
    m = np.isfinite(b_arr) & np.isfinite(c_arr)
    b_arr, c_arr = b_arr[m], c_arr[m]
    n = len(b_arr)
    if n == 0:
        return {"n": 0, "error": "没有可比较的样本"}

    diffs = c_arr - b_arr
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)

    b_ok, c_ok = b_arr >= pass_threshold, c_arr >= pass_threshold
    n_worse = int((b_ok & ~c_ok).sum())   # b：基线对、候选错
    n_better = int((~b_ok & c_ok).sum())  # c：基线错、候选对
    p = mcnemar_exact(n_worse, n_better)

    delta = float(diffs.mean())
    ci_excludes_zero = bool((lo > 0) or (hi < 0))
    significant = bool(p < alpha)
    if significant and ci_excludes_zero and delta < 0:
        conclusion = "显著退化"
    elif significant and ci_excludes_zero and delta > 0:
        conclusion = "显著提升"
    else:
        conclusion = "无统计显著差异（变化在噪声范围内）"

    return {
        "n": int(n),
        "delta": round(delta, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "ci_excludes_zero": ci_excludes_zero,
        "n_worse": n_worse,
        "n_better": n_better,
        "mcnemar_p": round(p, 9),
        "significant": significant,
        "conclusion": conclusion,
    }
