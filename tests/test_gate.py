# -*- coding: utf-8 -*-
"""判定层测试。

分两类：门禁规则是纯逻辑（直接构造 GateResult），diff 需要库（用临时 DuckDB）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402
import gate as G  # noqa: E402


def _result(delta: float, n_new_fail: int = 0, per_tag=None) -> G.GateResult:
    return G.GateResult(
        passed=True,
        per_tag=per_tag,
        new_failures=[{"item_id": f"x{i}"} for i in range(n_new_fail)],
        summary={"delta": delta, "base_macro": 0.9, "cand_macro": 0.9 + delta,
                 "n_new_failures": n_new_fail, "n_fixed": 0, "n_common": 40},
    )


# ------------------------------------------------------------ 门禁规则

def test_improvement_passes():
    r = G.check_gate(_result(+0.05), cfg={"max_abs_drop": 0.03, "max_new_failures": 10})
    assert r.passed and not r.reasons


def test_small_drop_within_tolerance_passes():
    """容忍阈值存在是为了避免频繁误报——分层评测样本小，卡太紧会天天红灯。"""
    r = G.check_gate(_result(-0.01), cfg={"max_abs_drop": 0.03, "max_new_failures": 10})
    assert r.passed


def test_large_drop_is_blocked():
    r = G.check_gate(_result(-0.10), cfg={"max_abs_drop": 0.03, "max_new_failures": 10})
    assert not r.passed
    assert any("总分下降" in x for x in r.reasons)


def test_too_many_new_failures_is_blocked_even_without_score_drop():
    """总分没掉也可能有问题：一批题从"对"变成"错"，另一批反过来，
    总分持平但用户体验已经变了。"""
    r = G.check_gate(_result(0.0, n_new_fail=25),
                     cfg={"max_abs_drop": 0.03, "max_new_failures": 10})
    assert not r.passed
    assert any("新增失败样本" in x for x in r.reasons)


def test_concentrated_regression_is_blocked_even_when_total_looks_fine():
    """这条是门禁最有价值的部分：**总分看得见下降，看不见"哪一类塌了"**。

    0.91 → 0.90 看似没事，但如果退化的部分全集中在"该拒答的题"上，
    那就是一批安全事故。这条规则独立于总分阈值起作用。
    """
    per_tag = pd.DataFrame({
        "tag": ["faq", "math", "refusal", "tool"],
        "n": [12, 8, 8, 12],
        "base": [0.78, 0.87, 1.0, 1.0],
        "cand": [0.80, 0.90, 0.0, 1.0],   # refusal 全崩
        "delta": [0.02, 0.03, -1.0, 0.0],
    })
    # 总分只降了 0.01（未超 0.03 阈值），新增失败也不多
    r = G.check_gate(_result(-0.01, n_new_fail=3, per_tag=per_tag),
                     cfg={"max_abs_drop": 0.03, "max_new_failures": 10})
    assert not r.passed, "总分没超阈值也该拦"
    assert any("refusal" in x and "退化集中" in x for x in r.reasons)


def test_historical_baseline_is_enforced_when_present():
    r = G.check_gate(_result(-0.01), cfg={"max_abs_drop": 0.03, "max_new_failures": 10},
                     baseline={"macro_mean": 0.95})
    assert not r.passed
    assert any("历史基线" in x for x in r.reasons)


def test_gate_result_string_is_readable():
    s = str(_result(+0.05))
    assert "通过" in s


# ------------------------------------------------------------ diff

@pytest.fixture()
def con(tmp_path):
    c = S.connect(str(tmp_path / "g.duckdb"))
    yield c
    c.close()


def _seed(con, dataset_id: str = "ds"):
    con.execute("DELETE FROM items WHERE dataset_id = ?", [dataset_id])
    con.execute("DELETE FROM scores")
    df = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(6)],
        "prompt": [f"p{i}" for i in range(6)],
        "reference": ["a"] * 6,
        "tags": ["faq", "faq", "math", "math", "tool", "tool"],
        "source": ["curated"] * 6,
        "meta": [None] * 6,
    })
    S.upsert_items(con, dataset_id, df)


def _put(con, run_id: str, scores: list[float], verdicts=None, attempt: int = 1):
    verdicts = verdicts or ["ok"] * len(scores)
    con.execute("DELETE FROM scores WHERE run_id = ? AND attempt = ?", [run_id, attempt])
    for i, (s, v) in enumerate(zip(scores, verdicts)):
        con.execute(
            "INSERT INTO scores "
            "(run_id, attempt, item_id, scorer_name, score, verdict, reason, raw) "
            "VALUES (?, ?, ?, 'rule_v1', ?, ?, '', '{}')",
            [run_id, attempt, f"i{i}", float(s), v])


def test_diff_finds_new_failures_and_fixes(con):
    _seed(con)
    _put(con, "ds__A", [1, 1, 1, 1, 0, 0])   # 前 4 条对
    _put(con, "ds__B", [1, 0, 1, 1, 1, 0])   # i1 退化了，i4 修好了

    r = G.diff_runs(con, "ds__A", "ds__B")
    assert [d["item_id"] for d in r.new_failures] == ["i1"]
    assert [d["item_id"] for d in r.fixed] == ["i4"]
    assert r.summary["n_new_failures"] == 1
    assert r.summary["n_fixed"] == 1


def test_diff_reports_per_tag_delta(con):
    _seed(con)
    _put(con, "ds__A", [1, 1, 1, 1, 1, 1])
    _put(con, "ds__B", [1, 1, 0, 0, 1, 1])   # 只有 math 退化
    r = G.diff_runs(con, "ds__A", "ds__B")
    row = r.per_tag[r.per_tag["tag"] == "math"].iloc[0]
    assert row["delta"] == pytest.approx(-1.0)
    assert r.per_tag[r.per_tag["tag"] == "tool"].iloc[0]["delta"] == pytest.approx(0.0)


def test_diff_raises_when_runs_do_not_overlap(con):
    _seed(con)
    _put(con, "ds__A", [1] * 6)
    con.execute("DELETE FROM scores WHERE run_id = 'ds__Z'")
    for i in range(3):
        con.execute(
            "INSERT INTO scores "
            "(run_id, attempt, item_id, scorer_name, score, verdict, reason, raw) "
            "VALUES ('ds__Z', 1, ?, 'rule_v1', 1.0, 'ok', '', '{}')",
            [f"z{i}"])
    with pytest.raises(ValueError):
        G.diff_runs(con, "ds__A", "ds__Z")


def test_new_failures_are_sorted_by_severity(con):
    """退化最狠的排在最前面——报告是给人看的，得让人第一眼看到最严重的。"""
    _seed(con)
    _put(con, "ds__A", [1.0, 0.6, 1.0, 1.0, 1.0, 1.0])
    _put(con, "ds__B", [0.0, 0.55, 0.9, 1.0, 1.0, 1.0])
    r = G.diff_runs(con, "ds__A", "ds__B")
    assert r.new_failures[0]["item_id"] == "i0", "降幅最大的排第一"
