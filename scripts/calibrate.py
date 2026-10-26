# -*- coding: utf-8 -*-
"""CLI：裁判偏差校准。

用法:
    python scripts/calibrate.py                          # 无偏差的裁判，应当全绿
    python scripts/calibrate.py --position-bias 0.3      # 注入位置偏差，应当被抓出来
    python scripts/calibrate.py --length-bias 0.4
    python scripts/calibrate.py --self-preference 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scorers import StubJudge  # noqa: E402
import calibration as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--position-bias", type=float, default=0.0)
    ap.add_argument("--length-bias", type=float, default=0.0)
    ap.add_argument("--self-preference", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    judge = StubJudge(position_bias=args.position_bias,
                      length_bias=args.length_bias,
                      self_preference=args.self_preference)
    items = C.make_probe_items(args.n)

    short, long, own, other = C.make_probes()

    print("=" * 68)
    print(f"裁判偏差校准  n={args.n}")
    print(f"注入: 位置 {args.position_bias} / 长度 {args.length_bias} / 自偏好 {args.self_preference}")
    print("=" * 68)

    p = C.probe_position_bias(judge, items, short, long)
    print("\n【位置偏差】")
    print(f"  一致性率        {p['consistency']:.4f}   （两轮指向同一答案的比例）")
    print(f"  选 A 位比例      {p['choose_position_a_rate']:.4f}   （理想 0.5）")
    print(f"  → {p['verdict']}")

    lb = C.probe_length_bias(judge, items, short, long)
    print("\n【长度偏差】")
    print(f"  长答案胜率      {lb['long_win_rate']:.4f}   "
          f"95%CI [{lb['ci_low']:.3f}, {lb['ci_high']:.3f}]")
    print(f"  → {lb['verdict']}")

    sp = C.probe_self_preference(judge, items, own, other)
    print("\n【自偏好】")
    print(f"  同源答案胜率    {sp['own_win_rate']:.4f}   "
          f"95%CI [{sp['ci_low']:.3f}, {sp['ci_high']:.3f}]")
    print(f"  → {sp['verdict']}")

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    report = {"injected": {
                  "position": args.position_bias, "length": args.length_bias,
                  "self_pref": args.self_preference},
              "position": p, "length": lb, "self_preference": sp}
    (out / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 reports/calibration.json")

    # 有偏差就该报出来。三个探针里只要有一个"存在偏差"，退出 1。
    biased = any("存在" in v for v in (p["verdict"], lb["verdict"], sp["verdict"]))
    return 1 if biased else 0


if __name__ == "__main__":
    raise SystemExit(main())
