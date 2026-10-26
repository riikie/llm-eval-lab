# -*- coding: utf-8 -*-
"""执行层：跑批。

三件要紧事：
  1. **run_id 天然幂等** —— 由 (dataset_id, app, version) 决定，同参数重跑落到同一批记录
  2. **断点续跑** —— 已跑过的 (run_id, item_id) 跳过，中断后重跑只补缺的
  3. **失败不中断整批** —— 单条报错记进 error 列，别让一条坏数据毁掉一小时跑批

关于并发：默认 max_workers=1 是**刻意的**。本地规则应用是纯 CPU（正则匹配），
受 GIL 限制，开线程只会变慢。并发参数留给接真实模型时（那时是 IO 密集）才有效。
不写清楚这点，别人会以为并发没实现。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402


def app_fingerprint(app) -> str:
    """应用的完整标识，**必须覆盖所有影响输出的参数**。

    【踩过一次】早期版本用 f"{name}_{version}" 当标识，结果
    MockApp(v9, accuracy=1.0) 和 MockApp(v9, accuracy=0.0) 撞成同一个 run_id，
    断点续跑把后者整个跳过了，三个不同正确率跑出三个一样的满分。
    幂等键漏掉任何一个影响输出的参数，都会导致"看起来跑过了，其实没跑"。
    """
    fp = getattr(app, "fingerprint", None)
    return fp or f"{getattr(app, 'name', 'app')}_{getattr(app, 'version', 'v0')}"


_CODE_FP: str | None = None


def code_fingerprint() -> str:
    """src/ 下所有源码的内容哈希。

    【为什么必须进 run_id】改了 rag.py 的生成逻辑或 kb.py 的拒答规则之后，
    应用配置一个字没变，fingerprint 也就不变，run_id 也就没变 ——
    断点续跑于是把整批跳过，跑出来的是**旧代码的结果**，而屏幕上一切正常。
    实测踩到过：修完句子打分 bug 后五个版本分数一个没动，就是因为这。

    代码变了就必须重跑，这是断点续跑唯一的正确语义。
    """
    global _CODE_FP
    if _CODE_FP is None:
        h = hashlib.md5()
        for p in sorted((ROOT / "src").glob("*.py")):
            h.update(p.read_bytes())
        _CODE_FP = h.hexdigest()[:8]
    return _CODE_FP


def make_run_id(dataset_id: str, app) -> str:
    """run_id = 数据集 + 应用指纹 + 代码指纹。

    三者任一变化都会落到新的一批，不会误复用旧结果。
    """
    return f"{dataset_id}__{app_fingerprint(app)}__c{code_fingerprint()}"


def _already_done(con, run_id: str, attempt: int) -> set[str]:
    try:
        rows = con.execute(
            "SELECT item_id FROM outputs WHERE run_id = ? AND attempt = ?",
            [run_id, attempt]).fetchall()
    except Exception:
        return set()
    return {r[0] for r in rows}


def _write_outputs(con, run_id: str, rows: list[dict], attempt: int) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["attempt"] = attempt
    con.register("_outs", df)
    # 纯 INSERT，不是 INSERT OR REPLACE：attempt 是主键的一部分，
    # 同一 (run_id, attempt, item_id) 不会重复写，历史 attempt 也不会被覆盖。
    con.execute("""
        INSERT INTO outputs (run_id, attempt, item_id, text, tool_calls, latency_ms, error)
        SELECT run_id, attempt, item_id, text, tool_calls, latency_ms, error FROM _outs
    """)
    con.unregister("_outs")


def run_batch(con, dataset_id: str, app, resume: bool = True,
              max_workers: int = 1, limit: int = 0, force: bool = False) -> dict:
    """跑一批。返回统计字典。

    ## attempt 语义（重跑 vs 续跑，两者不再打架）

        上次跑完了   + 默认    → 不动（n_new=0）
        上次跑到一半 + 默认    → 接着跑，补缺失的 item（同一 attempt）
        任意情况     + force   → 新开一个 attempt，全部重跑

    旧数据永远留在库里（主表只追加），查历史用 attempt 号取。
    """
    items = S.load_items(con, dataset_id)
    if not len(items):
        raise ValueError(f"数据集 {dataset_id} 为空或不存在")
    if limit:
        items = items.head(limit)

    run_id = make_run_id(dataset_id, app)
    cur = S.latest_attempt(con, run_id)
    if cur == 0 or force:
        attempt = cur + 1
    else:
        attempt = cur                      # 续跑同一批，不另起炉灶
    done = _already_done(con, run_id, attempt) if resume else set()
    todo = items[~items["item_id"].isin(done)]

    recs = todo.to_dict("records")
    t0 = time.time()
    rows: list[dict] = []

    def one(item: dict) -> dict:
        try:
            out = app.predict(item)
            row = {"run_id": run_id, "item_id": item["item_id"], **out.to_row()}
        except Exception as e:  # noqa: BLE001
            # 单条失败不能毁掉整批
            row = {"run_id": run_id, "item_id": item["item_id"], "text": "",
                   "tool_calls": "[]", "latency_ms": 0.0, "error": f"{type(e).__name__}: {e}"}
        return row

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(one, recs))
    else:
        rows = [one(r) for r in recs]

    _write_outputs(con, run_id, rows, attempt)

    n_err = sum(1 for r in rows if r.get("error"))
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    stat = {
        "run_id": run_id,
        "attempt": attempt,
        "dataset_id": dataset_id,
        "app": f"{app.name}_{app.version}",
        "n_items": int(len(items)),
        "n_new": len(rows),
        "n_skipped": int(len(done)),
        "n_error": n_err,
        "elapsed_s": round(time.time() - t0, 2),
        "p50_ms": round(float(pd.Series(lat).quantile(0.5)), 3) if lat else 0.0,
        "p95_ms": round(float(pd.Series(lat).quantile(0.95)), 3) if lat else 0.0,
    }

    con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
    con.execute(
        "INSERT INTO runs (run_id, dataset_id, app_name, app_version, n_items, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [run_id, dataset_id, app.name, app.version, len(items),
         "ok" if n_err == 0 else f"partial({n_err} errors)"],
    )
    return stat


def load_outputs(con, run_id: str, attempt: int | None = None) -> pd.DataFrame:
    """取输出。默认走 outputs_latest 视图（每个 item 的最新 attempt）。

    想看历史就显式传 attempt —— 这就是 append-only 带来的能力：
    旧批次不会被覆盖，随时能调出来复盘。
    """
    if attempt is None:
        return con.execute(
            "SELECT item_id, text, tool_calls, latency_ms, error FROM outputs_latest "
            "WHERE run_id = ? ORDER BY item_id", [run_id]).df()
    return con.execute(
        "SELECT item_id, text, tool_calls, latency_ms, error FROM outputs "
        "WHERE run_id = ? AND attempt = ? ORDER BY item_id", [run_id, attempt]).df()


def join_items_outputs(con, dataset_id: str, run_id: str) -> pd.DataFrame:
    """把评测集与输出对齐，评分层的输入就是这一张表。"""
    items = S.load_items(con, dataset_id)
    outs = load_outputs(con, run_id)
    if not len(outs):
        raise ValueError(f"run {run_id} 没有任何输出")
    df = items.merge(outs, on="item_id", how="inner")
    if df.empty:
        raise ValueError(f"run {run_id} 的输出与数据集 {dataset_id} 对不上（item_id 无交集）")
    return df
