# -*- coding: utf-8 -*-
"""一键跑通整条链路：入湖 → 跑批 → 评分 → 成对比较 → 门禁。

存在的理由：改一行 src 代码，code_fingerprint 就变，run_id 跟着变，
之前跑过的批次全部失效。手工敲五条命令迟早会漏一条，然后对着
"两个 run 没有共同样本"发呆。这个脚本保证每步都按依赖顺序执行。

用法：
    python scripts/run_all.py                    # 全套
    python scripts/run_all.py --versions v3 v4 v5
    python scripts/run_all.py --dataset cs_v1 --gate v4 v5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str], title: str = "") -> int:
    if title:
        print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")
    r = subprocess.run([PY] + args, cwd=str(ROOT))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cs_v1")
    ap.add_argument("--file", default="data/raw/evalset_v2.jsonl")
    ap.add_argument("--versions", nargs="*", default=["v1", "v2", "v3", "v4", "v5"])
    ap.add_argument("--gate", nargs=2, default=["v4", "v5"], metavar=("BASE", "CAND"))
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    if not args.skip_ingest:
        rc = run(["scripts/ingest.py", "--file", args.file, "--dataset-id", args.dataset],
                 "1/4 入湖")
        if rc:
            return rc

    print(f"\n{'=' * 66}\n2/4 跑批 + 评分（{len(args.versions)} 个版本）\n{'=' * 66}")
    for v in args.versions:
        rc = run(["scripts/run_batch.py", "--dataset", args.dataset, "--app", f"rag:{v}"])
        if rc:
            return rc
        rc = run(["scripts/score_run.py", "--dataset", args.dataset, "--app", f"rag:{v}"])
        if rc:
            return rc

    b, c = args.gate
    run(["scripts/compare.py", "--dataset", args.dataset,
         "--a", f"rag:{c}", "--b", f"rag:{b}"], f"3/4 成对比较 {c} vs {b}")

    rc = run(["scripts/check_regression.py", "--dataset", args.dataset,
              "--base", f"rag:{b}", "--cand", f"rag:{c}"], f"4/4 回归门禁 {b} → {c}")
    print(f"\n门禁退出码 = {rc}（0 通过 / 1 阻断）")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
