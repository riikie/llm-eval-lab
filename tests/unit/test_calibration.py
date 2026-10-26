# -*- coding: utf-8 -*-
"""校准与成对比较的断言。全部可手算，不依赖模型和数据集。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from calibration import (  # noqa: E402
    wilson_ci, verdict_from_ci, pairwise_compare,
    probe_position_bias, probe_length_bias, probe_self_preference,
    make_probes, make_probe_items,
)
from scorers import StubJudge  # noqa: E402


# ------------------------------------------------------------ Wilson CI

def test_wilson_is_centred_on_half_for_a_tie():
    lo, hi = wilson_ci(5, 10)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-6)


def test_wilson_narrows_as_n_grows():
    a = wilson_ci(50, 100)
    b = wilson_ci(500, 1000)
    assert (b[1] - b[0]) < (a[1] - a[0]), "样本越多区间越窄"


def test_wilson_stays_inside_unit_interval_at_the_edges():
    """正态近似在 p=0 或 1 时会给出 [0,0] 这种"确定无疑"的假象，Wilson 不会。"""
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi > 0.0, "全败也要给出上界，不能假装确定"
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_wilson_of_zero_samples_is_uninformative():
    assert wilson_ci(0, 0) == (0.0, 0.0)


# ------------------------------------------------------------ 结论判定

def test_zero_comparisons_never_yields_a_conclusion():
    """一场都没比就下结论是最危险的错。曾经这里判成"B 显著更好"。"""
    v = verdict_from_ci(0, 0)
    assert v["conclusive"] is False
    assert "无法判定" in v["conclusion"] or "样本不足" in v["conclusion"]


def test_close_scores_are_inconclusive():
    """3:2 的胜负跟抛硬币没区别，必须说无法判定。"""
    v = verdict_from_ci(3, 5)
    assert v["conclusive"] is False
    assert "无法判定" in v["conclusion"]


def test_a_clear_sweep_is_conclusive():
    v = verdict_from_ci(18, 20)
    assert v["conclusive"] is True and "A 显著更好" in v["conclusion"]


def test_ci_covering_half_means_inconclusive():
    lo, hi = wilson_ci(12, 20)
    v = verdict_from_ci(12, 20)
    assert v["conclusive"] == bool(lo > 0.5 or hi < 0.5)


# ------------------------------------------------------------ 三类探针

def _items(n=12):
    return make_probe_items(n)


def test_clean_judge_shows_no_bias():
    """无偏差的裁判，三个探针都必须报"无"。"""
    j = StubJudge()
    short, long, own, other = make_probes()
    it = _items()
    assert "无" in probe_position_bias(j, it, short, long)["verdict"]
    assert "无" in probe_length_bias(j, it, short, long)["verdict"]
    assert "无" in probe_self_preference(j, it, own, other)["verdict"]


def test_position_bias_is_detected():
    j = StubJudge(position_bias=0.3)
    short, long, _, _ = make_probes()
    r = probe_position_bias(j, _items(), short, long)
    assert "存在位置偏差" in r["verdict"]
    assert r["choose_position_a_rate"] > 0.9, "偏差裁判几乎总选 A 位"


def test_length_bias_is_detected():
    j = StubJudge(length_bias=0.4)
    short, long, _, _ = make_probes()
    r = probe_length_bias(j, _items(), short, long)
    assert "存在长度偏差" in r["verdict"]
    assert r["long_win_rate"] > 0.9


def test_self_preference_is_detected():
    j = StubJudge(self_preference=0.3)
    _, _, own, other = make_probes()
    r = probe_self_preference(j, _items(), own, other)
    assert "存在自偏好" in r["verdict"]
    assert r["own_win_rate"] > 0.9


def test_probes_do_not_cross_contaminate():
    """关键性质：注入 A 偏差时，B、C 探针必须保持干净。

    如果三个探针会互相串扰，说明它们测的不是各自的偏差，
    而是笼统的"答案有变化"——那校准就没有意义了。
    """
    it = _items()
    short, long, own, other = make_probes()

    j = StubJudge(position_bias=0.3)
    assert "无" in probe_length_bias(j, it, short, long)["verdict"]
    assert "无" in probe_self_preference(j, it, own, other)["verdict"]

    j = StubJudge(length_bias=0.4)
    assert "无" in probe_position_bias(j, it, short, long)["verdict"]
    assert "无" in probe_self_preference(j, it, own, other)["verdict"]

    j = StubJudge(self_preference=0.3)
    assert "无" in probe_position_bias(j, it, short, long)["verdict"]
    assert "无" in probe_length_bias(j, it, short, long)["verdict"]


def test_all_ties_are_not_read_as_a_loss():
    """全平局时，胜率的分母应当只有"分出胜负"的场次。

    平局被当成"输"的话，会得到"胜率 0.00，B 显著更好"——
    明明是裁判觉得两者一样好，却被读成裁判极度偏袒 B。
    """
    j = StubJudge()  # 无偏差 → 两个等价答案必然全平局
    short, long, _, _ = make_probes()
    r = probe_length_bias(j, _items(), short, long)
    assert r["n_decided"] == 0
    assert r["ties"] > 0
    assert "无长度偏差" in r["verdict"]


# ------------------------------------------------------------ 成对比较

def _scores(a: list[float], b: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.DataFrame({"item_id": [f"i{i}" for i in range(len(a))], "score": a}),
            pd.DataFrame({"item_id": [f"i{i}" for i in range(len(b))], "score": b}))


def test_pairwise_counts_win_loss_tie():
    # 1>0 胜, 1>0 胜, 0==0 平, 0.5==0.5 平
    da, db = _scores([1, 1, 0, 0.5], [0, 0, 0, 0.5])
    r = pairwise_compare(None, da, db)
    assert r["win"] == 2 and r["loss"] == 0 and r["tie"] == 2


def test_pairwise_only_compares_common_items():
    da = pd.DataFrame({"item_id": ["a", "b", "c"], "score": [1.0, 1.0, 1.0]})
    db = pd.DataFrame({"item_id": ["a", "b"], "score": [0.0, 0.0]})
    r = pairwise_compare(None, da, db)
    assert r["n_common"] == 2


def test_pairwise_raises_when_nothing_overlaps():
    da = pd.DataFrame({"item_id": ["a"], "score": [1.0]})
    db = pd.DataFrame({"item_id": ["z"], "score": [0.0]})
    with pytest.raises(ValueError):
        pairwise_compare(None, da, db)


def test_pairwise_reports_delta():
    da, db = _scores([1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    r = pairwise_compare(None, da, db)
    assert r["delta"] == pytest.approx(0.75)
    assert r["mean_a"] == pytest.approx(0.75) and r["mean_b"] == 0.0
