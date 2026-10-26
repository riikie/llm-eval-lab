# -*- coding: utf-8 -*-
"""统计判定的手算校验。

全部用**能笔算**的构造数据，这样以后有人改坏了实现，测试会立刻发现，
而且不需要跑模型、不需要读数据集。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stats import bootstrap_ci, mcnemar_exact, paired_verdict  # noqa: E402


def test_mcnemar_known_value_seven_worse_zero_better():
    """b=0, c=7: n=7, p = 2 × C(7,0) / 2^7 = 2/128 = 0.015625。"""
    assert mcnemar_exact(0, 7) == pytest.approx(2 / 128, abs=1e-9)


def test_mcnemar_balanced_discordance_is_not_significant():
    """b=3, c=3: 变坏变好一样多，没有任何证据说有差异。

    tail = C(6,0)+C(6,1)+C(6,2)+C(6,3) = 42, p = 2×42/64 = 1.3125 → 截到 1.0
    """
    assert mcnemar_exact(3, 3) == pytest.approx(1.0, abs=1e-9)


def test_mcnemar_no_discordant_pairs_means_no_evidence():
    """两版完全一致（b=c=0）时不该下任何结论。"""
    assert mcnemar_exact(0, 0) == pytest.approx(1.0, abs=1e-9)


def test_mcnemar_twelve_worse_matches_the_v5_case():
    """v4→v5 实测：refusal_edge 12 条全变坏、0 条变好。

    p = 2 × C(12,0) / 2^12 = 2/4096 = 0.000488
    """
    assert mcnemar_exact(12, 0) == pytest.approx(2 / 4096, abs=1e-9)


def test_mcnemar_is_symmetric():
    for b, c in ((0, 5), (2, 9), (7, 7), (1, 0)):
        assert mcnemar_exact(b, c) == pytest.approx(mcnemar_exact(c, b), abs=1e-12)


def test_bootstrap_ci_of_constant_collapses_to_that_constant():
    """差值是常数时，区间应当退化成一个点。"""
    lo, hi = bootstrap_ci([0.5] * 50, n_boot=500, seed=0)
    assert lo == pytest.approx(0.5, abs=1e-9)
    assert hi == pytest.approx(0.5, abs=1e-9)


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.1, 0.3, size=400)
    lo, hi = bootstrap_ci(d, n_boot=2000, seed=1)
    assert lo < d.mean() < hi


def test_bootstrap_ci_empty_input_is_nan():
    lo, hi = bootstrap_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_paired_verdict_detects_clear_regression():
    base = [1.0] * 12
    cand = [0.0] * 12
    v = paired_verdict(base, cand, pass_threshold=0.5, n_boot=500, seed=0)
    assert v["n"] == 12
    assert v["delta"] == pytest.approx(-1.0, abs=1e-9)
    assert v["n_worse"] == 12 and v["n_better"] == 0
    assert v["significant"] is True
    assert v["conclusion"] == "显著退化"


def test_paired_verdict_ignores_samples_both_versions_get_right():
    """990 道两版都对，只有 10 道分出了胜负 —— 证据只在后者里。

    如果实现把"一致的题"也算进统计量，这个测试会失败。
    """
    base = [1.0] * 1000
    cand = [1.0] * 990 + [0.0] * 10
    v = paired_verdict(base, cand, pass_threshold=0.5, n_boot=500, seed=0)
    assert v["n_worse"] == 10 and v["n_better"] == 0
    assert v["mcnemar_p"] == pytest.approx(2 / 2 ** 10, abs=1e-9)


def test_paired_verdict_no_difference_means_not_significant():
    base = [1.0, 0.0] * 30
    cand = [1.0, 0.0] * 30
    v = paired_verdict(base, cand, pass_threshold=0.5, n_boot=500, seed=0)
    assert v["delta"] == pytest.approx(0.0, abs=1e-12)
    assert v["significant"] is False
    assert v["conclusion"] == "无统计显著差异（变化在噪声范围内）"


def test_paired_verdict_drops_nan_pairs():
    base = [1.0, 1.0, float("nan"), 0.0]
    cand = [0.0, 1.0, 0.0, 0.0]
    v = paired_verdict(base, cand, pass_threshold=0.5, n_boot=200, seed=0)
    assert v["n"] == 3, "nan 对必须整对剔除，不能当成 0 分"
