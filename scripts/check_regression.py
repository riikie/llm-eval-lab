# -*- coding: utf-8 -*-
"""CLI：回归门禁。

用法:
    python scripts/check_regression.py --base <run_id> --cand <run_id>          # 判定
    python scripts/check_regression.py --base <run_id> --cand <run_id> --update # 把这版设为基线
    python scripts/check_regression.py --base ... --cand ... --show 20          # 列出更多新增失败
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import store as S  # noqa: E402
import gate as G  # noqa: E402


def baseline_path(cand_id: str) -> Path:
    return ROOT / "reports" / f"baseline_{cand_id}.json"


def _resolve(con, dataset: str, spec_or_id: str) -> str:
    """把 run_id 或应用规格都解析成 run_id。

    不手写 run_id 是有理由的：run_id 里含应用指纹，指纹规则一变
    （比如以后把 temperature 也算进去），硬编码的 run_id 就全部失效。
    """
    if con.execute("SELECT 1 FROM runs WHERE run_id = ?", [spec_or_id]).fetchone():
        return spec_or_id
    if not dataset:
        raise SystemExit(f"找不到 run {spec_or_id}；若想用应用规格请加 --dataset")
    from apps import build_app
    import runner as R
    return R.make_run_id(dataset, build_app(spec_or_id))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基线：run_id 或应用规格(如 rule:v3)")
    ap.add_argument("--cand", required=True, help="候选：run_id 或应用规格")
    ap.add_argument("--dataset", default="", help="给了它，--base/--cand 就按应用规格解析")
    ap.add_argument("--scorer", default="rule")
    ap.add_argument("--show", type=int, default=10, help="列出前 N 条新增失败")
    ap.add_argument("--update", action="store_true", help="把当前结果写为基线")
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    con = S.connect(args.db or None)
    base_id = _resolve(con, args.dataset, args.base)
    cand_id = _resolve(con, args.dataset, args.cand)
    res = G.diff_runs(con, base_id, cand_id, args.scorer)
    res = G.check_gate(res, baseline=json.loads(baseline_path(cand_id).read_text(encoding="utf-8"))
                       if baseline_path(cand_id).exists() else None)

    print("=" * 68)
    print(f"回归门禁: 基线 {base_id}\n          候选 {cand_id}")
    print("=" * 68)
    s = res.summary
    print(f"  共同样本      {s['n_common']}")
    print(f"  总分          {s['base_macro']:.4f} → {s['cand_macro']:.4f}   ({s['delta']:+.4f})")
    print(f"  新增失败      {s['n_new_failures']}      修复 {s['n_fixed']}")

    if res.per_tag is not None and len(res.per_tag):
        print("\n  分类变化:")
        print("   " + res.per_tag.to_string(index=False).replace("\n", "\n   "))

    if res.new_failures:
        print(f"\n  新增失败样本（前 {args.show} 条，按退化幅度排序）:")
        for d in res.new_failures[:args.show]:
            print(f"    [{d['tags']:8s}] {d['item_id']:10s} "
                  f"{d['base_score']:.2f} → {d['cand_score']:.2f}  ({d['verdict']})")

    print()
    print(res)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"gate_{base_id}_to_{cand_id}.json").write_text(
        json.dumps({"summary": s, "reasons": res.reasons, "passed": res.passed,
                    "new_failures": res.new_failures, "fixed": res.fixed},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if args.update:
        baseline_path(cand_id).write_text(
            json.dumps({"run_id": cand_id, "macro_mean": s["cand_macro"],
                        "n": s["n_common"]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n已把 {cand_id} 设为基线（macro_mean={s['cand_macro']:.4f}）")

    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
