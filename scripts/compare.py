# -*- coding: utf-8 -*-
"""CLI：两个 run 的成对比较。

两种用法：
    # 直接给 run_id
    python scripts/compare.py --a seed_v1__rule-v4-1111 --b seed_v1__rule-v3-0111

    # 给数据集 + 应用规格，run_id 自动推导（推荐）
    python scripts/compare.py --dataset seed_v1 --a rule:v4 --b rule:v3

【为什么推荐第二种】run_id 里含应用指纹，指纹规则一变（比如以后把 temperature
也算进去），手写的 run_id 就全部失效。让程序推导，改了指纹也不用改脚本。
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
import calibration as C  # noqa: E402


def resolve(con, dataset: str, spec_or_id: str) -> str:
    """把 run_id 或应用规格都解析成 run_id。"""
    hit = con.execute("SELECT 1 FROM runs WHERE run_id = ?", [spec_or_id]).fetchall()
    if hit:
        return spec_or_id
    if not dataset:
        raise SystemExit(f"找不到 run {spec_or_id}；若想用应用规格请加 --dataset")
    from apps import build_app
    return R.make_run_id(dataset, build_app(spec_or_id))


def load_run_scores(con, run_id: str, scorer_prefix: str = "rule") -> "object":
    rows = con.execute(
        "SELECT item_id, score FROM scores_latest WHERE run_id = ? AND scorer_name LIKE ?",
        [run_id, f"{scorer_prefix}%"],
    ).df()
    if rows.empty:
        raise SystemExit(f"run {run_id} 没有 {scorer_prefix} 的评分，先跑 score_run.py")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="候选版本：run_id 或应用规格(如 rule:v4)")
    ap.add_argument("--b", required=True, help="基线版本：run_id 或应用规格")
    ap.add_argument("--dataset", default="", help="给了它，--a/--b 就按应用规格解析")
    ap.add_argument("--scorer", default="rule")
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    con = S.connect(args.db or None)
    id_a = resolve(con, args.dataset, args.a)
    id_b = resolve(con, args.dataset, args.b)

    da = load_run_scores(con, id_a, args.scorer)
    db = load_run_scores(con, id_b, args.scorer)

    res = C.pairwise_compare(None, da, db)
    print("=" * 66)
    print(f"成对比较: A={id_a}\n          B={id_b}")
    print("=" * 66)
    print(f"  共同样本      {res['n_common']}")
    print(f"  平均分        A={res['mean_a']:.4f}   B={res['mean_b']:.4f}   "
          f"差 {res['delta']:+.4f}")
    print(f"  胜负平        {res['win']} 胜 / {res['loss']} 负 / {res['tie']} 平")
    print(f"  A 胜率        {res['win_rate']:.4f}   95%CI "
          f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print(f"  结论          {res['conclusion']}")

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"compare_{id_a}_vs_{id_b}.json").write_text(
        json.dumps({"a": id_a, "b": id_b, **res}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n已保存 reports/compare_{id_a}_vs_{id_b}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
