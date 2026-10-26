# -*- coding: utf-8 -*-
"""用 golden set 校准评分器/裁判。

    python scripts/calibrate_golden.py                # 规则评分器 vs 人工
    python scripts/calibrate_golden.py --judge        # 再叠加一个带偏差的裁判做对照

输出三件事：
  1. 一致率 + Cohen's kappa（扣掉碰巧猜对的部分）
  2. 混淆矩阵 + 分歧清单（人工和机器到底在哪儿不一样）
  3. 位置交换翻转率（同一对答案，A/B 顺序对调后裁判改判的比例）

退出码：kappa 低于 --min-kappa 返回 1，可直接当 CI 门禁。
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
import apps as A  # noqa: E402
import golden as G  # noqa: E402
import calibration as C  # noqa: E402
from scorers import StubJudge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cs_v1")
    ap.add_argument("--app", default="rag:v4")
    ap.add_argument("--scorer", default="rule")
    ap.add_argument("--judge", action="store_true", help="额外跑带偏差裁判的对照")
    ap.add_argument("--min-kappa", type=float, default=0.5)
    ap.add_argument("--db", default="")
    ap.add_argument("--out", default="reports/golden_calibration.json")
    args = ap.parse_args()

    g = G.load_golden()
    print("=" * 68)
    print(f"Golden set: {len(g)} 条人工标注，覆盖 {g['tags'].nunique()} 个 tag")
    print(g["human"].value_counts().to_string())
    print("=" * 68)

    con = S.connect(args.db or None)
    app = A.build_app(args.app)
    run_id = R.make_run_id(args.dataset, app)
    scores = con.execute(
        "SELECT item_id, score, verdict FROM scores_latest WHERE run_id = ? AND scorer_name LIKE ?",
        [run_id, f"{args.scorer}%"]).df()
    if scores.empty:
        print(f"没有找到 {run_id} 的评分结果，先跑 score_run.py")
        return 2

    res = G.compare_with_golden(scores, g)
    if res.get("n", 0) == 0:
        print("golden 与评分结果没有交集")
        return 2

    print(f"\n【{args.scorer} 评分器 vs 人工标注】共同样本 {res['n']} 条")
    print(f"  一致率 accuracy : {res['agreement']:.4f}")
    print(f"  Cohen's kappa   : {res['kappa']:.4f}   （扣掉碰巧猜对的部分）")
    print(f"  平均绝对误差    : {res['mean_abs_error']:.4f}")
    print("\n  混淆矩阵（行=人工，列=机器）:")
    print(res["confusion"].to_string().replace("\n", "\n  "))
    print("\n  逐 tag 一致率:")
    print(res["per_tag"].to_string(index=False).replace("\n", "\n  "))

    nd = len(res["disagreements"])
    print(f"\n  分歧 {nd} 条（人工与机器判断不同）:")
    for _, r in res["disagreements"].head(8).iterrows():
        print(f"    [{r['tags']:<13}] {r['item_id']:<22} 人工={r['human_class']:<8} "
              f"机器={r['pred_class']:<8} score={r['score']:.2f}")

    out = {"scorer": args.scorer, "run_id": run_id,
           "n": res["n"], "agreement": res["agreement"],
           "kappa": res["kappa"], "mean_abs_error": res["mean_abs_error"],
           "n_disagreements": nd}

    if args.judge:
        print("\n" + "=" * 68)
        print("【裁判偏差对照】同一批 golden 样本，换不同偏差设置的裁判")
        print("=" * 68)
        # 用 golden 里的 correct 样本和 wrong 样本配成对，做位置交换测试
        # 探针构造：两个答案**质量等价**（都命中 3/4 要点 = 0.75），
        # 只有位置偏差能打破平衡。若一个明显更好，裁判选它是对的，
        # 那测到的就不是偏差——上一个项目在这里误报过。
        items = [{"item_id": f"g{i}", "tags": "faq",
                  "reference": "七天,无理由,运费,包装完好"} for i in range(40)]
        a_out = {"text": "支持七天无理由退货，运费由卖家承担"}
        b_out = {"text": "支持七天无理由退货，包装需完好"}
        print("  （两个答案等价：都命中 3/4 要点 = 0.75，只有偏差能打破平衡）")
        for prob in (0.0, 0.2, 0.5, 0.8, 1.0):
            j = StubJudge(position_bias=0.15, bias_prob=prob)
            pr = C.probe_position_bias(j, items, a_out, b_out)
            flip = 1 - pr["consistency"]
            print(f"  偏差发生概率 {prob:.1f} → 位置交换翻转率 {flip:.3f}  "
                  f"选A位比例 {pr['choose_position_a_rate']:.3f}")
            out.setdefault("position_probe", []).append({
                "bias_prob": prob, "consistency": pr["consistency"],
                "flip_rate": round(flip, 4)})

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {p}")

    if res["kappa"] < args.min_kappa:
        print(f"\n[阻断] kappa {res['kappa']:.4f} 低于下限 {args.min_kappa}，"
              f"评分器与人工判断不一致，不能拿它的分数做门禁")
        return 1
    print(f"\n[通过] kappa {res['kappa']:.4f} ≥ {args.min_kappa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
