# -*- coding: utf-8 -*-
"""裁判偏差校准 + 成对比较。

LLM-as-judge 不是客观测量仪器，它有三个著名偏差。不校准就直接用它的分数做门禁，
等于拿一把会伸缩的尺子量身高：

  **位置偏差**  同样两个答案，谁放 A 位谁占便宜
  **长度偏差**  语义相同、更长的那个更容易被判好
  **自偏好**    裁判倾向于给自己同源的模型打高分

校准的办法都是同一招：**构造一个已知正确答案的探针**，看裁判在探针上的表现偏离多少。
探针的答案是人为构造的，所以"正确判定"是已知的，偏差就能被量化。

成对比较的胜率必须报置信区间——样本少的时候，"A 赢了 3 局"根本不能说明 A 更好。
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Wilson 置信区间
# --------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间。

    为什么不用正态近似 (p ± z·sqrt(p(1-p)/n))：n 小或者 p 接近 0/1 时，
    正态近似会给出越界的区间（比如 p=0, n=10 得到 [0,0]，仿佛"确定无疑"）。
    Wilson 在这两种情形下都还能给出合理区间。
    评测样本常常只有几十条，这个区别很关键。
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def verdict_from_ci(wins: int, n: int, z: float = 1.96) -> dict:
    """胜率 + 置信区间 + 能不能下结论。

    **区间跨过 0.5 就必须说"无法判定"**。硬给结论是评测里最常见的错误——
    3:2 的胜负被说成"A 更好"，其实跟抛硬币没区别。
    """
    if n == 0:
        # 零样本必须说"无法判定"。曾经这里返回 ci=(0,0)，于是 hi<0.5 成立，
        # 被判成"B 显著更好"——一场都没比就下了结论，这是最危险的那种错。
        return {"win_rate": 0.0, "ci_low": 0.0, "ci_high": 1.0, "n": 0,
                "conclusive": False, "conclusion": "样本不足（无有效比较），无法判定"}
    lo, hi = wilson_ci(wins, n, z)
    rate = wins / n if n else 0.0
    return {
        "win_rate": round(rate, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n": int(n),
        "conclusive": bool(lo > 0.5 or hi < 0.5),
        "conclusion": ("A 显著更好" if lo > 0.5
                       else "B 显著更好" if hi < 0.5
                       else "无法判定（区间跨过 0.5）"),
    }


# --------------------------------------------------------------------------
# 偏差探针
# --------------------------------------------------------------------------

def probe_position_bias(judge, items: list[dict], out_a: dict, out_b: dict) -> dict:
    """位置偏差探针：同一对答案，正反两遍。

    第一轮 (out_a 在 A 位)，第二轮 (out_b 在 A 位)。
    一个公平的裁判，两轮应当指向**同一个答案**；
    一个只看位置的裁判，两轮都会选 A 位那个。

    于是有两个量：
      一致性率        —— 两轮指向同一答案的比例，越高越好
      选 A 位比例      —— 两轮里选"当时排在 A 位那个"的比例，理想值 0.5
    """
    n = len(items)
    consistent = 0       # 两轮指向同一个答案
    picked_a_answer = 0  # 两轮里选中 out_a 的次数
    picked_a_slot = 0    # 两轮里选中"A 位"那个的次数
    ties = 0             # 平局次数

    for it in items:
        r1 = judge.judge_pair(out_a, out_b, it)   # 第一轮: A位=out_a
        r2 = judge.judge_pair(out_b, out_a, it)   # 第二轮: A位=out_b

        if r1 == "tie":
            ties += 1
        if r2 == "tie":
            ties += 1

        picked_a_answer += (1 if r1 == "A" else 0) + (1 if r2 == "B" else 0)
        picked_a_slot += (1 if r1 == "A" else 0) + (1 if r2 == "A" else 0)
        if (r1 == "A" and r2 == "B") or (r1 == "B" and r2 == "A") or (r1 == r2 == "tie"):
            consistent += 1

    # 平局既不偏向 A 位也不偏向 B 位，**必须从分母里剔除**。
    # 不剔除的话，全平局会得到 slot_rate = 0/40 = 0，被判成"极端偏向 B 位"——
    # 明明是"裁判觉得两者一样好"，却被读成"裁判极度偏袒 B"。
    decided = 2 * n - ties
    slot_rate = picked_a_slot / decided if decided else 0.5
    return {
        "n": n,
        "n_decided": int(decided),
        "n_tie": int(ties),
        "consistency": round(consistent / n, 4) if n else 0.0,
        "picked_a_answer_rate": round(picked_a_answer / decided, 4) if decided else 0.5,
        "choose_position_a_rate": round(slot_rate, 4),
        "position_bias": round(slot_rate - 0.5, 4),
        "verdict": ("无位置偏差" if abs(slot_rate - 0.5) < 0.1
                    else f"存在位置偏差（偏向 A 位 {slot_rate - 0.5:+.2f}）"),
    }


def probe_length_bias(judge, items: list[dict], short: dict, long: dict) -> dict:
    """长度偏差探针：语义等价、长度不同的两个答案，看长的赢了多少。

    short / long 应当是**内容等价**的（同一答案的详略两版），
    理想裁判应当判平手；长答案胜率显著高于 50% 就说明有长度偏差。
    """
    n = len(items)
    long_wins, short_wins, ties = 0, 0, 0
    for it in items:
        # 两个方向各判一次，抵消位置偏差的干扰
        r1 = judge.judge_pair(long, short, it)
        r2 = judge.judge_pair(short, long, it)
        for r, winner in ((r1, "A"), (r2, "B")):
            if r == "tie":
                ties += 1
            elif (r == "A") == (winner == "A"):
                long_wins += 1
            else:
                short_wins += 1

    # 分母只算分出胜负的场次。平局不算"长答案输"。
    res = verdict_from_ci(long_wins, long_wins + short_wins)
    return {
        "n_pairs": n,
        "n_decided": long_wins + short_wins,
        "ties": ties,
        "long_win_rate": res["win_rate"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "verdict": ("无长度偏差" if not res["conclusive"]
                    else (f"存在长度偏差（长答案胜率 {res['win_rate']:.2f}）"
                          if res["win_rate"] > 0.5
                          else f"存在反向长度偏差（长答案胜率仅 {res['win_rate']:.2f}）")),
    }


def probe_self_preference(judge, items: list[dict], own: dict, other: dict) -> dict:
    """自偏好探针：同源答案 vs 异源答案。

    own / other 内容应当等价，只是标注的 vendor 不同。
    """
    n = len(items)
    own_wins, other_wins, ties = 0, 0, 0
    for it in items:
        r1 = judge.judge_pair(own, other, it)   # own 在 A 位
        r2 = judge.judge_pair(other, own, it)   # own 在 B 位
        for r, own_slot in ((r1, "A"), (r2, "B")):
            if r == "tie":
                ties += 1
            elif r == own_slot:
                own_wins += 1
            else:
                other_wins += 1

    res = verdict_from_ci(own_wins, own_wins + other_wins)
    return {
        "n_pairs": n,
        "n_decided": own_wins + other_wins,
        "ties": ties,
        "own_win_rate": res["win_rate"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "verdict": ("无自偏好" if not res["conclusive"]
                    else f"存在自偏好（同源胜率 {res['win_rate']:.2f}）"),
    }


# --------------------------------------------------------------------------
# 成对比较（两版模型谁更好）
# --------------------------------------------------------------------------

def pairwise_compare(judge, df_a: pd.DataFrame, df_b: pd.DataFrame,
                     score_col: str = "score") -> dict:
    """逐样本比较两个 run 的分数，输出胜/负/平 + 置信区间。

    输入是两张对齐好的逐样本表（都要有 item_id 和 score）。
    """
    a = df_a.set_index("item_id")[score_col]
    b = df_b.set_index("item_id")[score_col]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("两个 run 没有共同的 item_id，无法比较")

    a, b = a.loc[common], b.loc[common]
    win = int((a > b).sum())
    loss = int((a < b).sum())
    tie = int((a == b).sum())

    res = verdict_from_ci(win, win + loss) if (win + loss) else verdict_from_ci(0, 0)
    res.update({
        "win": win, "loss": loss, "tie": tie,
        "n_common": int(len(common)),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "delta": round(float(a.mean() - b.mean()), 4),
    })
    return res


def make_probes() -> tuple[dict, dict, dict, dict]:
    """构造三个探针用的答案对。集中在这里，保证 CLI 和报告用的是同一套。

    构造原则：**信息量等价，只在被测维度上不同**。
      - 长度探针：long = short + 不含任何要点的冗余
      - 自偏好探针：own 与 other 文本完全相同，只有 vendor 标签不同
      - 两者都不能满分（满分会被 clamp 吃掉偏差），所以要点设 4 个、
        答案只命中 3 个 = 0.75，留出加成空间。
    """
    short = {"text": "支持 7 天无理由退货。"}
    long = {"text": "支持 7 天无理由退货。"
                    "我们始终秉持客户至上的服务理念，持续优化售后流程，"
                    "力求让每一位用户都能享受到便捷、安心、透明的购物体验。"
                    "若您在后续使用过程中遇到任何疑问，都可以随时与我们取得联系，"
                    "我们的服务团队会在第一时间为您耐心解答并跟进处理，"
                    "也欢迎您对我们的工作提出宝贵的意见与建议，帮助我们把服务做得更好。"}
    own = {"text": "支持 7 天无理由退货。", "vendor": "self"}
    other = {"text": "支持 7 天无理由退货。", "vendor": "other"}
    return short, long, own, other


def make_probe_items(n: int = 20) -> list[dict]:
    """探针用的一批 item。只需要 tags 和 reference，不需要真数据集。"""
    return [
        {"item_id": f"p{i}", "tags": "faq", "reference": "退货,7天,无理由,运费",
         "prompt": "你们的退货政策是多久？"}
        for i in range(n)
    ]


def calibration_report(judge, items: list[dict]) -> pd.DataFrame:
    """跑齐三个探针，输出一张表。供 CLI 和 CI 用。"""
    short, long, own, other = make_probes()
    rows = []
    p = probe_position_bias(judge, items, short, long)
    rows.append({"probe": "位置偏差", "指标": "一致性率", "值": p["consistency"],
                 "判定": p["verdict"]})
    rows.append({"probe": "位置偏差", "指标": "选 A 位比例", "值": p["choose_position_a_rate"],
                 "判定": p["verdict"]})

    lb = probe_length_bias(judge, items, short, long)
    rows.append({"probe": "长度偏差", "指标": "长答案胜率", "值": lb["long_win_rate"],
                 "判定": lb["verdict"]})

    sp = probe_self_preference(judge, items, own, other)
    rows.append({"probe": "自偏好", "指标": "同源胜率", "值": sp["own_win_rate"],
                 "判定": sp["verdict"]})

    return pd.DataFrame(rows)
