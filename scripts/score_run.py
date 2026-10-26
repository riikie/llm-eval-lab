# -*- coding: utf-8 -*-
"""CLI：对一个 run 打分。

用法:
    python scripts/score_run.py --run-id seed_v1__rule_v4 --scorer rule
    python scripts/score_run.py --dataset seed_v1 --app rule:v3 --scorer rule
    python scripts/score_run.py --run-id seed_v1__rule_v4 --scorer stub --position-bias 0.2
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
import runner as R  # noqa: E402
import scorers as Sc  # noqa: E402


def build_scorer(args):
    if args.scorer == "rule":
        return Sc.RuleScorer()
    if args.scorer == "stub":
        return Sc.StubJudge(position_bias=args.position_bias, length_bias=args.length_bias)
    raise SystemExit(f"未知评分器 {args.scorer}（可选 rule / stub）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--app", default="")
    ap.add_argument("--scorer", default="rule")
    ap.add_argument("--position-bias", type=float, default=0.0)
    ap.add_argument("--length-bias", type=float, default=0.0)
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    if not args.run_id:
        if not (args.dataset and args.app):
            raise SystemExit("需要 --run-id，或同时给 --dataset 和 --app")
        from apps import build_app
        app = build_app(args.app)
        args.run_id = R.make_run_id(args.dataset, app)

    con = S.connect(args.db or None)
    # fetchall 而非 fetchone：聚合/查找结果若不读完，duckdb 1.x 会把
    # 连接标记为"有未完成结果"，下一个 execute/register 直接报错。
    run = con.execute(
        "SELECT dataset_id, app_name, app_version FROM runs WHERE run_id = ?", [args.run_id]
    ).fetchall()
    if not run:
        raise SystemExit(f"找不到 run {args.run_id}，先跑 scripts/run_batch.py")
    dataset_id = run[0][0]

    df = R.join_items_outputs(con, dataset_id, args.run_id)
    scorer = build_scorer(args)

    rows = []
    for _, r in df.iterrows():
        item = r.to_dict()
        out = {"text": r.get("text"), "tool_calls": r.get("tool_calls")}
        s = scorer.score(item, out) if hasattr(scorer, "score") else scorer.judge(item, out)
        rows.append({"item_id": r["item_id"], "tags": r["tags"], **s.to_row()})

    res = pd.DataFrame(rows)
    res.insert(0, "scorer_name", f"{scorer.name}_{getattr(scorer, 'version', 'v1')}")
    res.insert(0, "attempt", S.latest_attempt(con, args.run_id))
    res.insert(0, "run_id", args.run_id)

    con.register("_sc", res)
    # 纯 INSERT：attempt 进主键，同一 (run, attempt, item, scorer) 不会重复写，
    # 旧 attempt 的分数也保留着。想覆盖请重跑 run_batch.py --force。
    con.execute("""
        INSERT INTO scores
            (run_id, attempt, item_id, scorer_name, score, verdict, reason, raw)
        SELECT run_id, attempt, item_id, scorer_name, score, verdict, reason, raw FROM _sc
    """)
    con.unregister("_sc")

    per_tag, overall = Sc.aggregate(res.rename(columns={"tags": "tags"}))
    print(f"\n=== {args.run_id} / {scorer.name} ===")
    print(per_tag.to_string(index=False))
    print("\n总体:", json.dumps(overall, ensure_ascii=False))

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    tag_name = f"{scorer.name}_{getattr(scorer, 'version', 'v1')}"
    per_tag.to_csv(out / f"scores_{args.run_id}_{tag_name}_bytag.csv",
                   index=False, encoding="utf-8-sig")
    (out / f"scores_{args.run_id}_{tag_name}.json").write_text(
        json.dumps({"run_id": args.run_id, "scorer": tag_name, **overall},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(out / f"samples_{args.run_id}_{tag_name}.csv",
               index=False, encoding="utf-8-sig")
    print(f"\n已保存 reports/scores_{args.run_id}_{tag_name}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
