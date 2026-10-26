# -*- coding: utf-8 -*-
"""入湖：文件 -> 校验 -> 幂等 upsert ->（可选）漂移检测 -> 报告。

用法:
    python scripts/ingest.py --file data/raw/seed_evalset.jsonl --dataset-id seed_v1
    python scripts/ingest.py --file new.jsonl --dataset-id seed_v2 --baseline seed_v1
    python scripts/ingest.py --file new.jsonl --dataset-id seed_v2 --inject-drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402
import drift as Dr  # noqa: E402


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"文件不存在: {p}")
    suf = p.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(p, encoding="utf-8-sig")
    if suf == ".jsonl":
        return pd.read_json(p, lines=True, encoding="utf-8")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise SystemExit(f"不支持的格式: {suf}（只认 csv / jsonl / parquet）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--baseline", default="", help="与之对比漂移的旧数据集 id")
    ap.add_argument("--inject-drift", action="store_true", help="人为注入漂移，验证检测器")
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    df = read_any(args.file)
    print(f"读入 {len(df)} 行，列: {list(df.columns)}")

    errs = S.validate_frame(df)
    if errs:
        print("[中止] 入湖校验失败:")
        for e in errs:
            print("   -", e)
        return 2
    print("[通过] schema 校验")

    if args.inject_drift:
        df = Dr.inject_drift(df)
        print(f"[注入漂移] 处理后剩 {len(df)} 行")

    con = S.connect(args.db or None)
    stat = S.upsert_items(con, args.dataset_id, df)
    print(f"[写入] 新增 {stat['inserted']}  更新 {stat['updated']}  未变 {stat['unchanged']}")

    total = con.execute(
        "SELECT count(*) FROM items WHERE dataset_id = ?", [args.dataset_id]
    ).fetchone()[0]
    print(f"[核对] 库中 {args.dataset_id} 共 {total} 条")

    rc = 0
    if args.baseline:
        ref = S.load_items(con, args.baseline)
        cur = S.load_items(con, args.dataset_id)
        if not len(ref):
            print(f"[跳过] 基线 {args.baseline} 不存在")
        else:
            rep = Dr.compute_drift(ref, cur)
            print("\n=== 漂移检测 (基线 %s vs 当前 %s) ===" % (args.baseline, args.dataset_id))
            print(rep.to_string(index=False))
            out = ROOT / "reports"
            out.mkdir(exist_ok=True)
            rep.to_csv(out / f"drift_{args.baseline}_vs_{args.dataset_id}.csv",
                       index=False, encoding="utf-8-sig")
            if Dr.has_alert(rep):
                print("\n[漂移!] 存在显著漂移维度，退出码 1")
                rc = 1
            else:
                print("\n[正常] 无显著漂移")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
