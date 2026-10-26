# -*- coding: utf-8 -*-
"""attempt / append-only 的契约测试。

要保证的三件事（第四条建议：trace 落库要 append-only + run_id 隔离）：
  1. 重跑不覆盖旧数据
  2. 查询默认拿到最新一次
  3. 断点续跑与重跑不再互相打架
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402
import runner as R  # noqa: E402
import apps as A  # noqa: E402


@pytest.fixture
def con(tmp_path):
    """每个测试用独立的临时库，互不干扰。"""
    c = S.connect(str(tmp_path / "attempt.duckdb"))
    yield c
    c.close()


def _items(con, dataset_id: str = "ds", n: int = 5) -> None:
    df = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n)],
        "prompt": [f"p{i}" for i in range(n)],
        "reference": ["七天无理由退货"] * n,
        "tags": ["refund"] * n,
        "source": ["curated"] * n,
        "meta": [None] * n,
    })
    S.upsert_items(con, dataset_id, df)


def test_repeated_run_does_not_overwrite_previous_attempt(con):
    """第二次跑（force）之后，第一次的输出必须还在。"""
    _items(con)
    app = A.build_app("rag:v1")
    rid = R.make_run_id("ds", app)

    s1 = R.run_batch(con, "ds", app)
    s2 = R.run_batch(con, "ds", app, force=True)

    assert s1["attempt"] == 1
    assert s2["attempt"] == 2
    assert S.attempt_size(con, rid, 1) == 5, "第一次的输出被覆盖了"
    assert S.attempt_size(con, rid, 2) == 5
    assert S.latest_attempt(con, rid) == 2


def test_latest_view_returns_only_the_newest_attempt(con):
    """视图必须每个 item 只返回最新一条，否则下游会看到重复样本。"""
    _items(con)
    app = A.build_app("rag:v1")
    rid = R.make_run_id("ds", app)
    R.run_batch(con, "ds", app)
    R.run_batch(con, "ds", app, force=True)

    df = con.execute(
        "SELECT item_id, attempt FROM outputs_latest WHERE run_id = ?", [rid]).df()
    assert len(df) == 5, "视图返回了重复行"
    assert set(df["attempt"]) == {2}


def test_plain_repeat_is_a_noop(con):
    """跑完后再跑一次，默认什么都不做（既不重跑也不新增 attempt）。"""
    _items(con)
    app = A.build_app("rag:v1")
    rid = R.make_run_id("ds", app)
    R.run_batch(con, "ds", app)
    s = R.run_batch(con, "ds", app)
    assert s["n_new"] == 0 and s["n_skipped"] == 5
    assert S.latest_attempt(con, rid) == 1


def test_resume_fills_the_gap_within_the_same_attempt(con):
    """跑到一半中断，再跑只补缺失的，且仍在同一 attempt 里。"""
    _items(con)
    app = A.build_app("rag:v1")
    rid = R.make_run_id("ds", app)
    R.run_batch(con, "ds", app, limit=2)      # 只跑了 2 条
    assert S.attempt_size(con, rid, 1) == 2

    s = R.run_batch(con, "ds", app)           # 接着跑
    assert s["attempt"] == 1, "续跑不该另起 attempt"
    assert s["n_new"] == 3 and s["n_skipped"] == 2
    assert S.attempt_size(con, rid, 1) == 5


def test_history_is_readable_by_attempt_number(con):
    """append-only 的意义在于能复盘：按 attempt 号能取回当时的输出。"""
    _items(con)
    app = A.build_app("rag:v1")
    rid = R.make_run_id("ds", app)
    R.run_batch(con, "ds", app)
    first = R.load_outputs(con, rid, attempt=1)
    R.run_batch(con, "ds", app, force=True)
    assert len(first) == 5
    assert len(R.load_outputs(con, rid, attempt=1)) == 5, "历史取不回来了"
