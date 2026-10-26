# -*- coding: utf-8 -*-
"""CLI：跑一批。

用法:
    python scripts/run_batch.py --dataset seed_v1 --app rule:v3
    python scripts/run_batch.py --dataset seed_v1 --app mock:v1:1.0
    python scripts/run_batch.py --dataset seed_v1 --app rule:v4 --no-resume   # 强制重跑
    python scripts/run_batch.py --dataset seed_v1 --app rule:v3 --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402
import runner as R  # noqa: E402
from apps import build_app  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--app", required=True, help="mock:v1:0.8 / rule:v3 / openai:URL:model")
    ap.add_argument("--db", default="")
    ap.add_argument("--workers", type=int, default=1,
                    help="默认 1：本地规则应用受 GIL 限制，并发无收益；接真模型时才有用")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true", help="忽略已有结果，全部重跑")
    ap.add_argument("--force", action="store_true",
                    help="另起一个 attempt 全部重跑，旧批次保留不覆盖")
    args = ap.parse_args()

    app = build_app(args.app)
    con = S.connect(args.db or None)
    stat = R.run_batch(con, args.dataset, app, resume=not args.no_resume,
                       max_workers=args.workers, limit=args.limit, force=args.force)
    print(json.dumps(stat, ensure_ascii=False, indent=2))

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"run_{stat['run_id']}.json").write_text(
        json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
