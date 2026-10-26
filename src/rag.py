# -*- coding: utf-8 -*-
"""被测应用：一个真的 RAG 客服问答助手（不是 demo）。

上一版的 RuleApp 是 10 条硬编码 if-else，那不是应用——diff 出来全是 0，
平台就是空壳。这一版是真的：

  - **检索是真的**：在 kb.py 的 66 篇文档上做 BM25 检索（自己实现，字符 2-gram）
  - **生成是真的**：从检索到的文档里抽取句子拼接。这是无 LLM 时的降级策略，
    换成真模型只需替换 generator，上层不动
  - **版本差异是真的**：来自配置项（top_k / query 改写 / 拒答严格度 / 工具调用），
    不是靠开关能力位。这样 diff 才有东西可拦

## 版本矩阵

    v1  top_k=1  rewrite=off  refusal=off     tools=off    基线
    v2  top_k=3  rewrite=off  refusal=off     tools=off    召回改善
    v3  top_k=3  rewrite=off  refusal=strict  tools=on     安全 + 工具
    v4  top_k=5  rewrite=on   refusal=strict  tools=on     查询改写
    v5  top_k=5  rewrite=on   refusal=LOOSE   tools=on     ← 事故版本

v1→v4 能力单调不降（可手算验证，是回归门禁的素材）。
v5 是**人为制造的真实事故**：线上误拒率被投诉，于是把拒答策略从 strict 放宽到
loose。直白的违规请求还能拦住，但隐蔽的社工话术全漏。而隐蔽请求只占评测集 4%，
宏平均几乎看不出变化——只有按 tag 单独检查的门禁能抓到。这就是要讲的故事。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from kb import DOCS, REFUSAL_MODES

# ==========================================================================
# BM25 检索
# ==========================================================================

_TOKEN = re.compile(r"[0-9A-Za-z]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中文字符 2-gram + 单字，英文数字按词。

    为什么不用 jieba：环境里没装，而且 2-gram 对短查询的中文检索效果够好，
    还少一个依赖。真要上生产换分词器，只改这一个函数。
    """
    s = str(text or "")
    out: list[str] = []
    for m in _TOKEN.finditer(s):
        t = m.group()
        if "\u4e00" <= t[0] <= "\u9fff":
            out.append(t)
            # 中文串内部再切 2-gram
            for i in range(len(t) - 1):
                out.append(t[i:i + 2])
        else:
            out.append(t.lower())
    # 中文整体串也要切（正则会把连续中文当一个 token，上面已处理）
    return out


class BM25:
    """极简 BM25。语料只有几十篇，纯 Python 足够快。"""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tok = [tokenize(d["text"] + " " + d.get("topic", "")) for d in docs]
        self.dl = [len(t) for t in self.tok]
        self.avgdl = sum(self.dl) / max(len(self.dl), 1)
        df: dict[str, int] = {}
        for t in self.tok:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for i, t in enumerate(self.tok):
            tf: dict[str, int] = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                idf = self.idf.get(w, 0.0)
                s += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            out.append(s)
        return out

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [(i, sc[i]) for i in order if sc[i] > 0]


_SENT = re.compile(r"[^。；;！!？?]+[。；;！!？?]?")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(str(text or "")) if s.strip()]


# ==========================================================================
# query 改写：同义/口语词扩展
# ==========================================================================

REWRITE_MAP: list[tuple[str, str]] = [
    (r"多久|要多长|几天|啥时候到", "时效 时间 天"),
    (r"怎么退|想退|不要了|退货", "退货 退款 无理由"),
    (r"运费|邮费|快递费", "运费 包邮"),
    (r"能不能|可以吗|行不行", ""),
    (r"咋|怎么|如何|怎样", ""),
    (r"钱什么时候|什么时候到账|几号到账", "退款 到账 工作日"),
    (r"票|开票", "发票 开具 税号"),
    (r"券|优惠", "优惠券 使用 门槛"),
]


def rewrite_query(q: str) -> str:
    extra = []
    for pat, add in REWRITE_MAP:
        if re.search(pat, q) and add:
            extra.append(add)
    return q + " " + " ".join(extra) if extra else q


# ==========================================================================
# 生成器（可插拔）
# ==========================================================================

def extractive_generator(query: str, hits: list[tuple[int, float]], docs: list[dict],
                         bm25: BM25, max_sent: int | None = None,
                         per_doc_cap: int = 2, rel_floor: float = 0.25) -> str:
    """抽取式生成：从检索结果里挑最贴题的句子拼成答案。

    这是**真实的降级策略**，不是玩具：没有 LLM 时，从语料里抽句回答是工业界
    常见做法。换成真模型只要替换这个函数即可，上层不动。

    两个约束是刻意的，也是版本差异的来源：
      - max_sent = min(4, 2*len(hits))：召回几篇就最多用几篇的量。
        top_k=1 时只能给出 2 句，跨两篇文档的组合题必然答不全。
      - per_doc_cap = 2：同一篇最多出 2 句，避免三句话全来自同一篇，
        把另一篇的要点挤掉。真实 RAG 也要做这种多样性控制。

    【修过一次】最初用 bm25.scores(sent)[idx] 给句子排序，那是
    "句子与它自己所在文档的相关度"，跟用户问了什么完全无关，
    等于按句子长短排序。结果 top_k 越大反而分越低（噪声句被优先选中）。
    正确做法是算**句子与 query** 的相关度。
    """
    qt = set(tokenize(query))
    per_doc: dict[int, list[tuple[str, float]]] = {}
    all_scores: list[float] = []
    for idx, _dscore in hits:
        sents: list[tuple[str, float]] = []
        for sent in split_sentences(docs[idx]["text"]):
            st = set(tokenize(sent))
            s = sum(bm25.idf.get(w, 0.0) for w in qt & st)
            if s > 0:
                sents.append((sent, s))
                all_scores.append(s)
        if sents:
            sents.sort(key=lambda x: -x[1])
            per_doc[idx] = sents[:per_doc_cap]

    # 相关度下限：top_k 调大能帮召回，但也会带进一堆擦边文档。
    # 没有这道闸，top_k=5 反而比 top_k=3 差（实测每个 tag 掉 0.02~0.17）——
    # 多召回的好处被噪声句子吃掉了。真实 RAG 也得配这个闸门。
    if all_scores:
        cut = max(all_scores) * rel_floor
        per_doc = {i: [(s, v) for s, v in lst if v >= cut] for i, lst in per_doc.items()}
        per_doc = {i: lst for i, lst in per_doc.items() if lst}

    if max_sent is None:
        max_sent = min(4, 2 * len(hits))

    # 轮转取：先每篇取第 1 句，再每篇取第 2 句 —— 保证多文档覆盖
    order = [idx for idx, _ in hits if idx in per_doc]
    picked: list[str] = []
    round_i = 0
    while len(picked) < max_sent:
        added = False
        for idx in order:
            if round_i < len(per_doc[idx]) and len(picked) < max_sent:
                if per_doc[idx][round_i][0] not in picked:
                    picked.append(per_doc[idx][round_i][0])
                    added = True
        if not added:
            break
        round_i += 1
    return "".join(picked)


# 检索分数高于这个值才走 RAG，否则认为是"要办事"而不是"要问政策"，
# 转给工具路由。这个阈值是实测分布定的：FAQ 类 top1 中位数 15~38，
# 工具类中位数 9.9，取 14 能把两类分开。
MIN_RAG_SCORE = 14.0


@dataclass
class Output:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0

    def to_row(self) -> dict:
        return {
            "text": self.text,
            "tool_calls": json.dumps(self.tool_calls, ensure_ascii=False),
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


# ==========================================================================
# 工具调用（Agent 轨迹）
# ==========================================================================

TOOL_RULES: list[tuple[str, str, str]] = [
    (r"订单\s*\d+|我的订单|到哪了|订单.*状态|物流.*查", "query_order", "正在查询您的订单"),
    (r"账单|消费统计|花了多少|这个月.*消费", "query_bill", "正在拉取您的账单"),
    (r"退掉|退货|退款|不要了", "create_refund", "正在为您创建退货申请"),
    (r"改.*地址|修改地址|换地址", "update_address", "正在修改收货地址"),
    (r"库存|还有货|有货吗|现货", "query_stock", "正在查询库存"),
    (r"取消订单|不要这个订单", "cancel_order", "正在取消订单"),
    (r"积分", "query_points", "正在查询您的积分"),
    (r"投诉|举报|不满", "create_complaint", "正在为您登记投诉"),
    (r"预约|上门安装|安装", "book_install", "正在为您预约上门安装"),
    (r"会员.*到期|什么时候.*会员|会员.*还有多久", "query_membership", "正在查询会员有效期"),
    (r"转人工|人工客服|找人工", "transfer_human", "正在为您转接人工客服"),
]


# ==========================================================================
# 被测应用
# ==========================================================================

VERSIONS: dict[str, dict] = {
    "v1": {"top_k": 1, "rewrite": False, "refusal": "off", "tools": False},
    "v2": {"top_k": 3, "rewrite": False, "refusal": "off", "tools": False},
    "v3": {"top_k": 3, "rewrite": False, "refusal": "strict", "tools": True},
    "v4": {"top_k": 5, "rewrite": True, "refusal": "strict", "tools": True},
    "v5": {"top_k": 5, "rewrite": True, "refusal": "loose", "tools": True},
}


class RagApp:
    """RAG 客服助手。配置决定能力，版本只是配置的具名组合。"""

    name = "rag"

    def __init__(self, version: str = "v1", top_k: int = 1, rewrite: bool = False,
                 refusal: str = "off", tools: bool = False,
                 generator: Callable | None = None, docs: list[dict] | None = None):
        self.version = version
        self.top_k = int(top_k)
        self.rewrite = bool(rewrite)
        self.refusal = refusal
        self.tools = bool(tools)
        self._docs = docs if docs is not None else DOCS
        self.bm25 = BM25(self._docs)
        self._gen = generator or extractive_generator

    @classmethod
    def from_version(cls, version: str, **override) -> "RagApp":
        if version not in VERSIONS:
            raise ValueError(f"未知版本 {version}，可选 {sorted(VERSIONS)}")
        cfg = dict(VERSIONS[version])
        cfg.update(override)
        return cls(version=version, **cfg)

    @property
    def fingerprint(self) -> str:
        """所有影响输出的配置都必须进指纹。

        上一版踩过的坑：MockApp 的 accuracy 没进指纹，三个不同正确率的实例
        撞同一个 run_id，断点续跑把后面两次全跳过了。
        """
        return (f"rag-{self.version}-k{self.top_k}"
                f"-{'rw' if self.rewrite else 'norw'}"
                f"-{self.refusal}"
                f"-{'tool' if self.tools else 'notool'}")

    # -- 检索 ------------------------------------------------------------
    def retrieve(self, prompt: str) -> list[tuple[int, float]]:
        """检索。开了 rewrite 时做**真正的 query expansion**。

        【改过一次】最初把扩展词直接拼在原 query 后面，结果扩展词也参与
        BM25 打分，把原 query 的权重稀释了——改写本该帮助召回，实测反而
        让每个 tag 都掉了 0.02~0.17。正确的做法是：原 query 和扩展 query
        **各检索一次，同一篇文档取两次的较高分**，再合并排序。
        这样扩展只会帮召回，永远不会拖后腿。
        """
        hits = self.bm25.topk(prompt, self.top_k)
        if not self.rewrite:
            return hits
        q2 = rewrite_query(prompt)
        if q2 == prompt:
            return hits
        merged: dict[int, float] = dict(hits)
        for i, s in self.bm25.topk(q2, self.top_k):
            if s > merged.get(i, -1.0):
                merged[i] = s
        return sorted(merged.items(), key=lambda x: -x[1])[: self.top_k]

    # -- 拒答 ------------------------------------------------------------
    def _should_refuse(self, prompt: str) -> bool:
        if self.refusal == "off" or self.refusal not in REFUSAL_MODES:
            return False
        return any(re.search(p, prompt) for p in REFUSAL_MODES[self.refusal])
        if self.refusal == "off" or self.refusal not in REFUSAL_MODES:
            return False
        return any(re.search(p, prompt) for p in REFUSAL_MODES[self.refusal])

    # -- 预测 ------------------------------------------------------------
    def predict(self, item: dict) -> Output:
        t0 = time.perf_counter()
        prompt = str(item.get("prompt") or "")

        # 1) 安全优先：拒答判断必须排在所有逻辑之前
        if self._should_refuse(prompt):
            return Output(text="抱歉，这个请求我无法协助。",
                          meta={"rule": "refuse", "mode": self.refusal},
                          latency_ms=(time.perf_counter() - t0) * 1000)

        # 2) 先检索。路由顺序是**检索优先**，不是工具优先。
        #
        # 【改过一次】最初把工具匹配放在检索前面，结果 refund 类 57 题里
        # 49 题被 "退掉|退货|退款" 正则截胡去调 create_refund —— 用户问的是
        # "退货政策怎么算"，系统却直接给他办了退货，答案里一个要点都没有，
        # 该类得分 0.023。真实客服系统里这叫**意图路由冲突**，是很常见的缺陷。
        hits = self.retrieve(prompt)
        if hits and hits[0][1] >= MIN_RAG_SCORE:
            text = self._gen(prompt, hits, self._docs, self.bm25)
            if text:
                return Output(text=text,
                              meta={"rule": "rag", "n_hits": len(hits),
                                    "top1": round(hits[0][1], 2),
                                    "doc_ids": [self._docs[i]["doc_id"] for i, _ in hits]},
                              latency_ms=(time.perf_counter() - t0) * 1000)

        # 3) 工具调用（Agent 轨迹）：检索没答上来，再看是不是要办件事
        if self.tools:
            for pat, tool, saying in TOOL_RULES:
                if re.search(pat, prompt):
                    return Output(text=saying, tool_calls=[{"name": tool, "args": {}}],
                                  meta={"rule": "tool"},
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        # 4) 兜底
        return Output(text="没有找到相关资料，已为您转接人工客服。",
                      meta={"rule": "no_hit"},
                      latency_ms=(time.perf_counter() - t0) * 1000)


# ==========================================================================
# 真模型适配器（接进来就换成真 RAG）
# ==========================================================================

class OpenAIGenerator:
    """把检索到的文档塞进 prompt，让真模型生成。

    本地无模型时跑不起来，但接口在这里——有 ollama / vLLM / 任意 OpenAI 兼容端点
    之后，构造时传进来即可：

        RagApp.from_version("v4", generator=OpenAIGenerator("http://127.0.0.1:11434/v1", "qwen2.5:7b"))
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, query: str, hits: list[tuple[int, float]], docs: list[dict],
                 bm25: BM25, max_sent: int = 2) -> str:
        import urllib.request

        ctx = "\n".join(f"[{i+1}] {docs[idx]['text']}" for i, (idx, _) in enumerate(hits))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是客服助手。只依据给定资料回答，资料里没有就说不知道。"},
                {"role": "user", "content": f"资料:\n{ctx}\n\n问题: {query}"},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"].get("content") or ""


def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()
