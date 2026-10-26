# -*- coding: utf-8 -*-
"""数据层契约测试。

用 tmp_path 建临时 DuckDB，不污染 data/eval.duckdb。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402
import drift as Dr  # noqa: E402


@pytest.fixture()
def con(tmp_path):
    c = S.connect(str(tmp_path / "t.duckdb"))
    yield c
    c.close()


def _frame(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n)],
        "prompt": [f"问题{i}" for i in range(n)],
        "reference": [f"答案{i}" for i in range(n)],
        "tags": ["faq"] * n,
        "source": ["curated"] * n,
        "meta": [None] * n,
    })


# ------------------------------------------------------------ 校验

def test_validate_passes_on_a_clean_frame():
    assert S.validate_frame(_frame()) == []


def test_validate_reports_every_problem_at_once():
    """一次报全，不是遇到第一个就抛——半夜对着一条报错反复重跑最浪费时间。"""
    bad = pd.DataFrame({"item_id": ["a", "a", None], "prompt": ["x", None, "z"]})
    errs = S.validate_frame(bad)
    assert len(errs) >= 2
    joined = " | ".join(errs)
    assert "重复" in joined
    assert "空值" in joined


def test_validate_catches_missing_required_column():
    df = pd.DataFrame({"prompt": ["只有 prompt"]})
    assert any("缺少必填列" in e for e in S.validate_frame(df))


def test_validate_honours_enum_constraint():
    df = _frame()
    cfg = {"required_columns": ["item_id", "prompt"], "enum_columns": {"tags": ["faq", "math"]}}
    assert S.validate_frame(df, cfg) == []
    df.loc[0, "tags"] = "nope"
    assert any("非枚举值" in e for e in S.validate_frame(df, cfg))


def test_upsert_refuses_invalid_data_instead_of_silently_fixing(con):
    df = pd.DataFrame({"item_id": ["a", "a"], "prompt": ["x", "y"]})
    with pytest.raises(S.SchemaError):
        S.upsert_items(con, "ds", df)
    n = con.execute("SELECT count(*) FROM items").fetchone()[0]
    assert n == 0, "校验失败就一行都不该进去"


# ------------------------------------------------------------ 幂等

def test_content_hash_is_stable_and_content_sensitive():
    r1 = {"item_id": "i", "prompt": "p", "reference": "r"}
    r2 = {"item_id": "i", "prompt": "p", "reference": "r"}
    r3 = {"item_id": "i", "prompt": "p", "reference": "改了"}
    assert S.content_hash(r1) == S.content_hash(r2)
    assert S.content_hash(r1) != S.content_hash(r3)


def test_reimporting_the_same_data_changes_nothing(con):
    """幂等的核心：同一份数据导两次，行数不变、updated 为 0。"""
    df = _frame(5)
    a = S.upsert_items(con, "ds", df)
    assert a == {"inserted": 5, "updated": 0, "unchanged": 0}

    b = S.upsert_items(con, "ds", df)
    assert b == {"inserted": 0, "updated": 0, "unchanged": 5}

    n = con.execute("SELECT count(*) FROM items WHERE dataset_id='ds'").fetchone()[0]
    assert n == 5, "重复导入不该产生重复行"


def test_changed_content_is_backfilled_not_duplicated(con):
    """回填：内容变了就更新，不是插一条新的。"""
    df = _frame(5)
    S.upsert_items(con, "ds", df)

    df2 = df.copy()
    df2.loc[0, "reference"] = "新的答案"
    stat = S.upsert_items(con, "ds", df2)
    assert stat == {"inserted": 0, "updated": 1, "unchanged": 4}

    n = con.execute("SELECT count(*) FROM items WHERE dataset_id='ds'").fetchone()[0]
    assert n == 5
    got = con.execute("SELECT reference FROM items WHERE item_id='i0'").fetchone()[0]
    assert got == "新的答案"


def test_adding_new_items_keeps_the_old_ones(con):
    S.upsert_items(con, "ds", _frame(5))
    stat = S.upsert_items(con, "ds", _frame(8))
    assert stat == {"inserted": 3, "updated": 0, "unchanged": 5}


def test_datasets_are_isolated_by_id(con):
    S.upsert_items(con, "a", _frame(3))
    S.upsert_items(con, "b", _frame(7))
    assert con.execute("SELECT count(*) FROM items WHERE dataset_id='a'").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM items WHERE dataset_id='b'").fetchone()[0] == 7


def test_load_items_roundtrips(con):
    df = _frame(4)
    S.upsert_items(con, "ds", df)
    got = S.load_items(con, "ds")
    assert len(got) == 4
    assert set(got["item_id"]) == set(df["item_id"])


# ------------------------------------------------------------ 漂移

def test_drift_detector_stays_quiet_on_identical_data(con):
    df = _frame(20)
    df["tags"] = ["faq"] * 10 + ["math"] * 10
    rep = Dr.compute_drift(df, df.copy())
    assert (rep["level"] == "无漂移").all()
    assert not Dr.has_alert(rep)


def test_drift_detector_alarms_on_injected_drift(con):
    """检测器本身也要被检测。注入漂移后必须报警，否则它只是个装饰。"""
    df = _frame(40)
    df["tags"] = ["faq"] * 20 + ["math"] * 20
    rep = Dr.compute_drift(df, Dr.inject_drift(df))
    assert Dr.has_alert(rep), "注入了漂移却没报警，检测器有问题"


def test_injected_drift_actually_changes_the_length_distribution():
    """曾经踩过：注入器"截短到 40 字"对本身就很短的 prompt 完全无效，
    PSI 恒为 0——检测器没坏，是注入器没打到点上。这条把它钉死。
    """
    df = _frame(40)
    df["tags"] = ["faq"] * 20 + ["math"] * 20
    drifted = Dr.inject_drift(df)
    before = df["prompt"].astype(str).str.len().mean()
    after = drifted["prompt"].astype(str).str.len().mean()
    assert after > before * 1.5, "注入后平均长度必须明显变长"
