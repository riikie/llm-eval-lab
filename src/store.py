# -*- coding: utf-8 -*-
"""数据层：DuckDB 单文件湖仓。

三件事：入湖（schema 校验）、幂等 upsert（支持回填）、漂移检测准备。

设计原则（从上一个项目 sentiment_eval 继承）：
  - 校验失败就中止，绝不静默 fillna。脏数据混进去，后面所有指标都不可信。
  - content_hash 是幂等的唯一依据：同一条样本内容变了才算更新，没变就跳过。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return json.load(open(ROOT / "config.json", encoding="utf-8"))


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

DDL_ITEMS = """
CREATE TABLE IF NOT EXISTS items (
    dataset_id   VARCHAR NOT NULL,
    item_id      VARCHAR NOT NULL,
    prompt       VARCHAR NOT NULL,
    reference    VARCHAR,
    tags         VARCHAR,
    source       VARCHAR,
    meta         VARCHAR,
    content_hash VARCHAR NOT NULL,
    ingested_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (dataset_id, item_id)
)
"""

DDL_DATASETS = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id   VARCHAR PRIMARY KEY,
    name         VARCHAR,
    version      VARCHAR,
    n_items      INTEGER,
    created_at   TIMESTAMP DEFAULT current_timestamp
)
"""

DDL_OUTPUTS = """
CREATE TABLE IF NOT EXISTS outputs (
    run_id       VARCHAR NOT NULL,
    attempt      INTEGER NOT NULL,
    item_id      VARCHAR NOT NULL,
    text         VARCHAR,
    tool_calls   VARCHAR,
    latency_ms   DOUBLE,
    error        VARCHAR,
    created_at   TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (run_id, attempt, item_id)
)
"""

DDL_SCORES = """
CREATE TABLE IF NOT EXISTS scores (
    run_id        VARCHAR NOT NULL,
    attempt       INTEGER NOT NULL,
    item_id       VARCHAR NOT NULL,
    scorer_name   VARCHAR NOT NULL,
    score         DOUBLE,
    verdict       VARCHAR,
    reason        VARCHAR,
    raw           VARCHAR,
    created_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (run_id, attempt, item_id, scorer_name)
)
"""

# 只暴露每个 (run_id, item_id) 的**最新一次**尝试。
#
# 【为什么要有 attempt 这一维】早期版本主键是 (run_id, item_id)，重跑就是
# INSERT OR REPLACE —— 旧数据被直接覆盖。两个后果：
#   1. 断点续跑和幂等打架：无法区分"这次没跑"和"这次跑过但被覆盖了"
#   2. 出事后没法复盘：v5 那批结果被 v6 覆盖后，再也拿不回当时的输出
# 加上 attempt 之后主表**只追加不修改**，历史天然留着；查询一律走下面这两个
# 视图，上层代码不用关心 attempt。
DDL_VIEW_OUTPUTS = """
CREATE OR REPLACE VIEW outputs_latest AS
SELECT run_id, item_id, text, tool_calls, latency_ms, error, attempt
FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY run_id, item_id ORDER BY attempt DESC) AS rn
    FROM outputs
) WHERE rn = 1
"""

DDL_VIEW_SCORES = """
CREATE OR REPLACE VIEW scores_latest AS
SELECT run_id, item_id, scorer_name, score, verdict, reason, raw, attempt
FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY run_id, item_id, scorer_name ORDER BY attempt DESC) AS rn
    FROM scores
) WHERE rn = 1
"""

DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,
    dataset_id  VARCHAR,
    app_name    VARCHAR,
    app_version VARCHAR,
    n_items     INTEGER,
    status      VARCHAR,
    started_at  TIMESTAMP DEFAULT current_timestamp
)
"""


def connect(db_path: str | None = None):
    """连库并建表。db_path 为空时用 config.json 的 db_path（相对 ROOT）。"""
    import duckdb

    cfg = load_config()
    if db_path is None:
        db_path = str(ROOT / cfg["db_path"])
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    for ddl in (DDL_DATASETS, DDL_ITEMS, DDL_RUNS, DDL_OUTPUTS, DDL_SCORES):
        con.execute(ddl)
    for ddl in (DDL_VIEW_OUTPUTS, DDL_VIEW_SCORES):
        con.execute(ddl)
    return con


# --------------------------------------------------------------------------
# attempt：同一次 run_id 的第几次跑批
# --------------------------------------------------------------------------

def latest_attempt(con, run_id: str) -> int:
    """该 run_id 当前最大的 attempt。没跑过返回 0。"""
    try:
        r = con.execute(
            "SELECT COALESCE(MAX(attempt), 0) FROM outputs WHERE run_id = ?", [run_id]
        ).fetchone()
    except Exception:
        return 0
    return int(r[0]) if r and r[0] is not None else 0


def next_attempt(con, run_id: str) -> int:
    return latest_attempt(con, run_id) + 1


def attempt_size(con, run_id: str, attempt: int) -> int:
    r = con.execute(
        "SELECT COUNT(*) FROM outputs WHERE run_id = ? AND attempt = ?", [run_id, attempt]
    ).fetchone()
    return int(r[0]) if r else 0


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------

class SchemaError(ValueError):
    """入湖校验失败。带完整错误清单，不抛第一个就跑。"""


def validate_frame(df: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    """返回错误清单。空列表 = 通过。

    刻意返回**全部**错误而不是遇到第一个就 raise——半夜对着一条报错反复重跑最浪费时间。
    """
    cfg = cfg or load_config()["ingest"]
    errs: list[str] = []

    required = cfg.get("required_columns", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        errs.append(f"缺少必填列: {missing}")

    if "item_id" in df.columns:
        if df["item_id"].isna().any():
            errs.append(f"item_id 有 {int(df['item_id'].isna().sum())} 个空值")
        dup = df["item_id"][df["item_id"].duplicated()].unique()
        if len(dup):
            errs.append(f"item_id 重复 {len(dup)} 个，例: {list(dup[:3])}")

    if "prompt" in df.columns:
        n_null = int(df["prompt"].isna().sum())
        ratio = n_null / max(len(df), 1)
        if ratio > cfg.get("max_null_ratio", 0.0):
            errs.append(f"prompt 空值 {n_null} 条（{ratio:.2%}），超过上限")

    for col, allowed in (cfg.get("enum_columns") or {}).items():
        if col in df.columns:
            bad = set(df[col].dropna().unique()) - set(allowed)
            if bad:
                errs.append(f"{col} 出现非枚举值: {sorted(bad)[:5]}")

    return errs


# --------------------------------------------------------------------------
# 幂等 upsert
# --------------------------------------------------------------------------

def content_hash(row: dict) -> str:
    """对样本内容取稳定哈希。只哈希**内容字段**，不含 dataset_id / 时间戳。"""
    payload = json.dumps(
        {k: row.get(k) for k in ("item_id", "prompt", "reference", "tags", "source", "meta")},
        ensure_ascii=False, sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def upsert_items(con, dataset_id: str, df: pd.DataFrame) -> dict:
    """幂等写入。返回 {'inserted', 'updated', 'unchanged'}。

    同一份数据导入两次 → 第二次全部 unchanged，行数不变。
    内容变了 → updated（这就是回填，新版本数据覆盖旧版本）。
    """
    errs = validate_frame(df)
    if errs:
        raise SchemaError("入湖校验失败:\n  - " + "\n  - ".join(errs))

    d = df.copy()
    for c in ("reference", "tags", "source", "meta"):
        if c not in d.columns:
            d[c] = None
    d["dataset_id"] = dataset_id
    d["content_hash"] = [
        content_hash(r) for r in d[["item_id", "prompt", "reference", "tags", "source", "meta"]]
        .to_dict("records")
    ]
    d = d[["dataset_id", "item_id", "prompt", "reference", "tags", "source", "meta", "content_hash"]]

    try:
        old = con.execute(
            "SELECT item_id, content_hash FROM items WHERE dataset_id = ?", [dataset_id]
        ).df()
    except Exception:
        old = pd.DataFrame(columns=["item_id", "content_hash"])
    old_map = dict(zip(old["item_id"], old["content_hash"])) if len(old) else {}

    new_map = dict(zip(d["item_id"], d["content_hash"]))
    inserted = [k for k in new_map if k not in old_map]
    updated = [k for k in new_map if k in old_map and old_map[k] != new_map[k]]
    unchanged = [k for k in new_map if k in old_map and old_map[k] == new_map[k]]

    con.register("_incoming", d)
    if updated or inserted:
        ids = inserted + updated
        placeholders = ",".join(["?"] * len(ids))
        con.execute(
            f"DELETE FROM items WHERE dataset_id = ? AND item_id IN ({placeholders})",
            [dataset_id] + ids,
        )
        con.execute("""
            INSERT INTO items
                (dataset_id, item_id, prompt, reference, tags, source, meta, content_hash)
            SELECT dataset_id, item_id, prompt, reference, tags, source, meta, content_hash
            FROM _incoming
            WHERE item_id IN ({})
        """.format(",".join(["?"] * len(ids))), ids)
    con.unregister("_incoming")

    con.execute("DELETE FROM datasets WHERE dataset_id = ?", [dataset_id])
    con.execute(
        "INSERT INTO datasets (dataset_id, name, version, n_items) VALUES (?, ?, ?, ?)",
        [dataset_id, dataset_id, "1", len(d)],
    )
    return {"inserted": len(inserted), "updated": len(updated), "unchanged": len(unchanged)}


def load_items(con, dataset_id: str) -> pd.DataFrame:
    return con.execute(
        "SELECT item_id, prompt, reference, tags, source, meta FROM items "
        "WHERE dataset_id = ? ORDER BY item_id", [dataset_id]
    ).df()
