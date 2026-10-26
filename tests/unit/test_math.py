# -*- coding: utf-8 -*-
"""纯数学断言：不读任何数据、不连库、不需要模型。

放在 tests/unit/ 下是刻意的——CI 上没有数据集和权重，这一层必须能独立跑通，
否则 CI 就是个摆设（上一个项目 sentiment_eval 就吃过这个亏）。

原则：**用能手算的基准值做断言，而不是"上次跑出来的数"**。
后者只防回归，不防一开始就错。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drift import psi, shift_magnitude, proportions, length_buckets  # noqa: E402


# ---------------------------------------------------------------- PSI

def test_psi_of_identical_distributions_is_zero():
    p = np.array([0.6, 0.1, 0.1, 0.2])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    base = np.array([0.25, 0.25, 0.25, 0.25])
    small = np.array([0.28, 0.24, 0.24, 0.24])
    big = np.array([0.55, 0.15, 0.15, 0.15])
    assert 0 < psi(base, small) < psi(base, big)


def test_psi_known_value():
    """手算校验（无零概率项，EPS 影响可忽略）：
    e=[.4,.3,.2,.1] -> a=[.5,.3,.15,.05]
      0.1·ln(1.25) + 0 + (-0.05)·ln(0.75) + (-0.05)·ln(0.5) ≈ 0.0714
    """
    e = np.array([0.4, 0.3, 0.2, 0.1])
    a = np.array([0.5, 0.3, 0.15, 0.05])
    expected = 0.1 * np.log(1.25) - 0.05 * np.log(0.75) - 0.05 * np.log(0.5)
    assert psi(e, a) == pytest.approx(expected, abs=1e-3)


def test_psi_explodes_when_a_category_disappears():
    """PSI 的已知陷阱：某类别占比掉到 0（或新类别从 0 冒出来）时，PSI 爆炸式放大。

    均匀四分类 -> [0.5,0.5,0,0]，直觉估计 2×0.25×ln2 ≈ 0.35，**实际 6.56**。
    因为 ln(1e-6/0.25) ≈ -12.4 把结果抬了一个数量级。

    所以本模块同时输出 shift_magnitude（有界的偏移量），两个一起看。
    这个测试就是把这个陷阱钉死，防止以后有人"优化"掉 EPS 截断。
    """
    e = np.array([0.25, 0.25, 0.25, 0.25])
    a = np.array([0.5, 0.5, 0.0, 0.0])
    v = psi(e, a)
    assert v > 1.0, "本应爆炸放大"
    assert v > 10 * (2 * 0.25 * np.log(2)), "远大于忽略零概率项时的朴素估计"


def test_psi_is_normalized_so_raw_counts_are_ok():
    """传计数而不是占比，结果应基本一致。

    注意：只是"基本一致"，不是严格相等。EPS=1e-6 加在计数（60）和占比（0.6）上
    的相对影响不同，差约 5e-6。也就是说 **EPS 截断让 PSI 不满足严格尺度不变性**。
    这是可以接受的代价——不做截断的话，一个占比为 0 的类别会让 PSI 直接爆炸
    （见上面那条测试）。容忍度取 1e-4，比这个误差大两个数量级。
    """
    counts_e = np.array([60, 10, 10, 20])
    counts_a = np.array([50, 30, 15, 5])
    assert psi(counts_e, counts_a) == pytest.approx(
        psi(counts_e / counts_e.sum(), counts_a / counts_a.sum()), abs=1e-4
    )


# ------------------------------------------------- shift_magnitude

def test_shift_magnitude_is_bounded_and_symmetric():
    e = np.array([0.5, 0.3, 0.2])
    a = np.array([0.2, 0.3, 0.5])
    v = shift_magnitude(e, a)
    assert 0.0 <= v <= 1.0
    assert v == pytest.approx(shift_magnitude(a, e), abs=1e-12), "应当对称"


def test_shift_magnitude_of_identical_is_zero_and_disjoint_is_one():
    e = np.array([0.5, 0.5, 0.0])
    assert shift_magnitude(e, e) == pytest.approx(0.0, abs=1e-12)
    a = np.array([0.0, 0.0, 1.0])
    assert shift_magnitude(e, a) == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------- 辅助函数

def test_proportions_follows_given_category_order_and_fills_zero():
    vals = ["a", "b", "a"]
    p = proportions(vals, ["a", "b", "c"])
    assert p.tolist() == [2.0, 1.0, 0.0]


def test_length_buckets_sums_to_one():
    lens = [10, 40, 100, 300, 900, 5000]
    b = length_buckets(lens, [32, 64, 128, 256, 512])
    assert b.sum() == pytest.approx(1.0, abs=1e-12), "返回的是占比，和必须为 1"


def test_length_buckets_puts_values_in_the_right_bucket():
    # edges=[10, 20] -> 桶0: <10, 桶1: [10,20), 桶2: >=20
    b = length_buckets([5, 15, 25], [10, 20])
    assert b.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-12)
