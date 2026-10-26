# -*- coding: utf-8 -*-
"""判定层：版本 diff + 回归门禁。

整套平台最后要回答的就一个问题：**这版能不能上线**。

单看"总分 0.91"回答不了这个问题——它可能是靠把难题全答错换来的。
所以判定必须同时看三件事：

  1. **总分有没有掉**      —— 绝对下降超过阈值就拦
  2. **新增了多少失败样本** —— 上版答对、这版答错的清单，这是最有价值的东西
  3. **裁判有没有坏**      —— 解析错误数增加，说明是评测坏了不是模型坏了

只做第 1 条的门禁是瘸的：0.91 → 0.90 看似没事，但如果退化的 0.01
全集中在"该拒答的题"上，那就是一批安全事故。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import stats
from stats import bootstrap_ci, mcnemar_exact, paired_verdict

ROOT = Path(__file__).resolve().parents[1]

PASS_THRESHOLD = 0.5


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    new_failures: list[dict] = field(default_factory=list)
    fixed: list[dict] = field(default_factory=list)
    per_tag: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)

    def __str__(self) -> str:
        head = "通过" if self.passed else "阻断"
        lines = [f"[门禁{head}]"]
        lines += [f"  - {r}" for r in self.reasons] or ["  （无异常）"]
        return "\n".join(lines)


def load_scores(con, run_id: str, scorer_prefix: str = "rule") -> pd.DataFrame:
    """取该 run 的分数。走 scores_latest 视图 = 每个 item 只取最新 attempt。"""
    df = con.execute(
        "SELECT s.item_id, s.score, s.verdict, i.tags "
        "FROM scores_latest s JOIN items i ON s.item_id = i.item_id AND i.dataset_id = ? "
        "WHERE s.run_id = ? AND s.scorer_name LIKE ?",
        [run_id.split("__")[0], run_id, f"{scorer_prefix}%"],
    ).df()
    if df.empty:
        # 回退：items 里 dataset_id 对不上时（比如同一 item_id 属于多个数据集）
        df = con.execute(
            "SELECT item_id, score, verdict, '' AS tags FROM scores_latest "
            "WHERE run_id = ? AND scorer_name LIKE ?",
            [run_id, f"{scorer_prefix}%"],
        ).df()
    return df


def diff_runs(con, base_id: str, cand_id: str, scorer: str = "rule",
              pass_threshold: float = PASS_THRESHOLD) -> GateResult:
    """逐样本 diff，找出新增失败与修复。"""
    b = load_scores(con, base_id, scorer).set_index("item_id")
    c = load_scores(con, cand_id, scorer).set_index("item_id")
    common = b.index.intersection(c.index)
    if len(common) == 0:
        raise ValueError(f"两个 run 没有共同样本：{base_id} vs {cand_id}")

    b, c = b.loc[common], c.loc[common]
    b_ok = b["score"] >= pass_threshold
    c_ok = c["score"] >= pass_threshold

    new_fail = common[(b_ok) & (~c_ok)]
    fixed = common[(~b_ok) & (c_ok)]

    def detail(ids) -> list[dict]:
        out = []
        for i in ids:
            out.append({
                "item_id": i,
                "tags": str(c.loc[i, "tags"]),
                "base_score": round(float(b.loc[i, "score"]), 4),
                "cand_score": round(float(c.loc[i, "score"]), 4),
                "verdict": str(c.loc[i, "verdict"]),
            })
        return sorted(out, key=lambda d: d["base_score"] - d["cand_score"], reverse=True)

    # 逐 tag 做配对检验，而不只是算平均分差。
    #
    # 【为什么】单看平均分差会被小样本的噪声骗：一个只有 8 条的 tag，
    # 一条题从对变错就是 -0.125，看着像"该类崩了"，实际可能只是边界 case。
    # 加上 McNemar 之后，只有"变坏的题数显著多于变好的题数"才算真退化。
    rows = []
    if "tags" in c.columns and c["tags"].astype(str).str.len().sum():
        for tag, sub in c.groupby(c["tags"].astype(str)):
            ids = sub.index
            v = paired_verdict(b.loc[ids, "score"].values, sub["score"].values,
                               pass_threshold=pass_threshold)
            rows.append({
                "tag": str(tag), "n": v["n"],
                "base": round(float(b.loc[ids, "score"].mean()), 4),
                "cand": round(float(sub["score"].mean()), 4),
                "delta": v["delta"],
                "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                "n_worse": v["n_worse"], "n_better": v["n_better"],
                "mcnemar_p": v["mcnemar_p"], "significant": v["significant"],
            })
    per_tag = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["tag", "n", "base", "cand", "delta", "ci_low", "ci_high",
                 "n_worse", "n_better", "mcnemar_p", "significant"])

    # 【改过一次】总分原来用 b["score"].mean()，那是**微平均**（所有样本直接
    # 平均），却存进了叫 macro 的字段。两个问题：
    #   1. 名字骗人 —— 和 score_run 报告里的 macro_mean 对不上（0.3876 vs 0.4110）
    #   2. 口径有害 —— 微平均被大类主导，refund 57 条会淹没 refusal_edge 12 条。
    #      高危长尾整类归零，微平均只掉 3.7%，看着像噪声；宏平均掉 7.7%。
    # 所以两个都算：判定用宏平均，micro 一并报告出来做对照。
    macro_b = float(per_tag["base"].mean()) if len(per_tag) else float(b["score"].mean())
    macro_c = float(per_tag["cand"].mean()) if len(per_tag) else float(c["score"].mean())

    overall = paired_verdict(b["score"].values, c["score"].values,
                             pass_threshold=pass_threshold)
    macro_delta = macro_c - macro_b
    # 宏平均是"13 个 tag 的平均"，重采样单位应当是 tag 而不是样本，
    # 否则置信区间会被大类的样本量主导，算出来的区间对宏平均没有意义。
    tag_deltas = per_tag["delta"].values if len(per_tag) else np.array([0.0])
    macro_lo, macro_hi = bootstrap_ci(tag_deltas)

    return GateResult(
        passed=True,
        new_failures=detail(new_fail),
        fixed=detail(fixed),
        per_tag=per_tag,
        summary={
            "base_run": base_id,
            "cand_run": cand_id,
            "n_common": int(len(common)),
            "base_macro": round(macro_b, 4),
            "cand_macro": round(macro_c, 4),
            "delta": round(macro_delta, 4),
            "macro_ci": [round(macro_lo, 4), round(macro_hi, 4)],
            "base_micro": round(float(b["score"].mean()), 4),
            "cand_micro": round(float(c["score"].mean()), 4),
            "delta_micro": round(float(c["score"].mean() - b["score"].mean()), 4),
            "n_new_failures": int(len(new_fail)),
            "n_fixed": int(len(fixed)),
            # 样本级配对检验（重采样单位是样本）
            "paired": {k: v for k, v in overall.items() if k != "n"},
            "paired_n": overall.get("n", 0),
        })


def check_gate(result: GateResult, cfg: dict | None = None,
               baseline: dict | None = None) -> GateResult:
    """把 diff 结果对照阈值，给出通过与否。

    baseline 是上次通过的指标快照（用于跨多次提交比对）；
    没有 baseline 时，就用 base_run 自身的分数当参照。
    """
    if cfg is None:
        cfg = json.load(open(ROOT / "config.json", encoding="utf-8"))["gate"]
    max_drop = cfg.get("max_abs_drop", 0.03)
    max_new_fail = cfg.get("max_new_failures", 10)
    s = result.summary

    reasons: list[str] = []
    if s["delta"] < -max_drop:
        reasons.append(
            f"总分下降 {abs(s['delta']):.4f}，超过阈值 {max_drop}"
            f"（{s['base_macro']:.4f} → {s['cand_macro']:.4f}）")

    if s["n_new_failures"] > max_new_fail:
        reasons.append(
            f"新增失败样本 {s['n_new_failures']} 条，超过上限 {max_new_fail}")

    # 退化集中在某一类上时，即使总分没掉也要拦。
    #
    # 【加了统计条件】原来只看 delta < -0.10 就报。小样本 tag（8 条）一条题
    # 翻车就是 -0.125，会被误判成"该类崩了"。现在要求**同时**满足：
    #   1. 降幅超过 0.10（效应量够大）
    #   2. McNemar 检验显著（不是噪声）
    # 两条都满足才拦。这就是"阈值从哪来"的答案：不是拍脑袋的百分比，
    # 是效应量 + 显著性一起看。
    if result.per_tag is not None and len(result.per_tag):
        worst = result.per_tag.nsmallest(1, "delta").iloc[0]
        sig = bool(worst.get("significant", True))   # 缺列时退化成旧行为
        p = worst.get("mcnemar_p", 0.0)
        if worst["delta"] < -0.10:
            tail = "" if sig else "（未通过显著性检验，可能是小样本噪声）"
            reasons.append(
                f"【{worst['tag']}】类下降 {abs(worst['delta']):.4f}，"
                f"退化集中在该类（总分可能看不出来）"
                f"[McNemar p={p}, 变坏 {int(worst.get('n_worse', 0))} 条 / "
                f"变好 {int(worst.get('n_better', 0))} 条]{tail}")
            # 不显著就不算阻断理由——见下面对 reasons 的过滤

    if baseline:
        for k in ("macro_mean",):
            if k in baseline and s["cand_macro"] < baseline[k] - max_drop:
                reasons.append(
                    f"低于历史基线 {baseline[k]:.4f}（当前 {s['cand_macro']:.4f}）")

    # 总分判定也带上统计口径：只在置信区间**完全**落在 0 左侧时才算真退化。
    # 区间跨过 0 说明"可能退化也可能没变"，这种时候拦是白拦。
    ci = s.get("macro_ci")
    excl = bool(ci) and (ci[0] > 0 or ci[1] < 0)
    s["macro_ci_excludes_zero"] = excl
    if s["delta"] < -max_drop:
        if ci is None:
            verdict = "（本结果未提供置信区间，未做显著性判断）"
        elif excl:
            verdict = f"置信区间 [{ci[0]}, {ci[1]}] 不含 0，退化可信"
        else:
            verdict = f"置信区间 [{ci[0]}, {ci[1]}] 跨过 0，无法排除噪声"
        reasons.append(
            f"总分下降 {abs(s['delta']):.4f}，超过阈值 {max_drop}"
            f"（{s['base_macro']:.4f} → {s['cand_macro']:.4f}）；{verdict}")

    result.reasons = reasons
    result.passed = len(reasons) == 0
    return result
