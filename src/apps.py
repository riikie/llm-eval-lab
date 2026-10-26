# -*- coding: utf-8 -*-
"""执行层：被测应用（App）。

**关键设计：App 是一个协议，不是具体实现。**

为什么这么设计：本地没有可用的大模型（ollama 未运行、无 API key），
如果代码写死"调某个模型"，这台机器上一行都跑不了，也没法验证链路对不对。
做成协议后，先用能**手算正确答案**的实现把链路跑通，等有真模型，
换一个实现类接进去，上层一行不改。

三种实现：
  MockApp              作弊器，答案可精确控制 → 用来验证评分器和门禁
  RuleApp              真实的规则客服系统，四个能力档位 → 用来产生真实的版本差异
  OpenAICompatibleApp  适配器，填 base_url + model 就能接真模型（含 ollama / vLLM）
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Output:
    """应用输出。tool_calls 是 Agent 轨迹评测的关键——光看文本判断不出能力。"""
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


class App(Protocol):
    name: str
    version: str

    def predict(self, item: dict) -> Output:
        ...


# ==========================================================================
# 1. MockApp —— 用来验证链路，不作弊就不叫 Mock
# ==========================================================================

class MockApp:
    """按指定正确率输出答案。

    **它直接读 reference 来构造答案，这是作弊**——但目的就是作弊：
    当 accuracy=1.0 时，评分器必须给出满分；accuracy=0.0 时必须给零分。
    这些值能手算，于是可以反过来验证评分器和门禁实现得对不对。

    上一个项目 sentiment_eval 用同样的手法（perfect / all_positive / uniform_random
    三个假预测器）当场抓出一个 np.corrcoef 零方差返回 nan 的 bug。
    """

    name = "mock"

    def __init__(self, version: str = "v1", accuracy: float = 1.0, seed: int = 0):
        self.version = version
        self.accuracy = accuracy
        self.seed = seed

    @property
    def fingerprint(self) -> str:
        """accuracy 和 seed 都会改变输出，必须进指纹。

        漏掉它们会导致不同正确率的 MockApp 撞同一个 run_id，
        断点续跑把后来的整批跳过——三个不同正确率跑出三个一样的分数。
        """
        return f"mock-{self.version}-a{self.accuracy:g}-s{self.seed}"

    def _rng(self, item_id: str) -> float:
        """按 (seed, item_id) 生成稳定的 [0,1) 随机数。必须是**跨进程稳定**的，
        不能直接用内置 hash()——Python 的字符串哈希按进程随机（PYTHONHASHSEED），
        重启后同输入结果会变，评测就不可复现了。上一个项目踩过这个坑。"""
        h = hashlib.md5(f"{self.seed}:{item_id}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def predict(self, item: dict) -> Output:
        t0 = time.perf_counter()
        ref = str(item.get("reference") or "")
        tags = str(item.get("tags") or "")

        if self._rng(str(item.get("item_id"))) < self.accuracy:
            # 答"对"：直接把 reference 当成答案吐出来
            text = ref
            tool_calls = []
            meta = json.loads(item["meta"]) if item.get("meta") else {}
            if meta.get("expect_tool"):
                tool_calls = [{"name": meta["expect_tool"], "args": {}}]
            if ref == "__REFUSE__":
                text = "抱歉，我无法协助这个请求。"
        else:
            # 答"错"：给一个明显不对的输出
            text = "我不太清楚，建议您咨询人工客服。"
            tool_calls = [{"name": "transfer_human", "args": {}}] if tags == "tool" else []

        return Output(text=text, tool_calls=tool_calls,
                      meta={"mock": True}, latency_ms=(time.perf_counter() - t0) * 1000)


# ==========================================================================
# 2. RuleApp —— 真实的规则客服系统，四个能力档位
# ==========================================================================

FAQ_KB: list[tuple[str, str]] = [
    (("退货", "退款"), "支持 7 天无理由退货，商品需保持完好，运费由卖家承担"),
    (("地址", "收货"), "未发货的订单可在订单详情里修改收货地址，已发货需联系快递"),
    (("支付", "付款", "怎么付"), "支持微信、支付宝、银行卡三种支付方式"),
    (("发票",), "可在订单完成后申请开具电子发票，需提供税号"),
    (("会员", "权益"), "会员享专属折扣、积分加倍、生日礼券等权益"),
    (("注销",), "账号注销需联系人工客服核验身份后处理"),
    (("快递", "多久", "物流", "几天"), "一般在付款后 48 小时内发货，物流 3-5 天送达"),
    (("人工", "客服", "联系"), "可拨打 400-000-0000 或在 App 内点击在线客服"),
    (("优惠券", "券"), "优惠券在结算页自动抵扣，注意使用门槛和有效期"),
    (("质保", "保修", "售后"), "商品享一年质保，非人为损坏免费维修"),
]

REFUSE_PATTERNS = [
    r"别人的|他人|内部员工|名单",
    r"骂人|骂回|难听",
    r"刷单|刷量|虚假交易",
    r"密码|数据库|后台权限",
    r"伪造|造假|假的发票",
    r"绕过|私下交易|逃单",
    r"诈骗|骗取|骗钱",
]

TOOL_RULES: list[tuple[str, str, str]] = [
    (r"订单\s*\d+|我的订单|到哪了|订单.*状态", "query_order", "正在查询您的订单"),
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


def _extract_numbers(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def solve_math(prompt: str) -> str | None:
    """极小的一套中文算术模式匹配。

    刻意写成"能解一部分题"而不是全解：真实应用本来就有能力边界，
    评分器要能反映出这个边界。解不了就返回 None。
    """
    nums = _extract_numbers(prompt)
    if not nums:
        return None

    # 打几折：打折后 X 元，原价 Y 元
    m = re.search(r"打折后\s*([\d.]+)\s*元.*?原价\s*([\d.]+)\s*元", prompt)
    if m:
        return f"{float(m.group(1)) / float(m.group(2)) * 10:.1f}"

    # 分别是 a、b、c → 求和
    if "分别是" in prompt:
        parts = re.findall(r"[\d.]+", prompt.split("分别是")[-1])
        if parts:
            return f"{sum(float(p) for p in parts):g}"

    # 满 A 减 B，买了 C
    m = re.search(r"满\s*(\d+(?:\.\d+)?)\s*减\s*(\d+(?:\.\d+)?)", prompt)
    if m:
        threshold, discount = float(m.group(1)), float(m.group(2))
        rest = prompt[m.end():]
        rest_nums = _extract_numbers(rest)
        if rest_nums:
            amount = rest_nums[0]
            return f"{amount - discount if amount >= threshold else amount:g}"

    # 打 X 折（含"买 N 件"）
    m = re.search(r"打\s*(\d+(?:\.\d+)?)\s*折", prompt)
    if m:
        rate = float(m.group(1)) / 10
        m2 = re.search(r"买\s*(\d+)\s*件", prompt)
        qty = float(m2.group(1)) if m2 else 1.0
        unit = nums[0]
        return f"{unit * qty * rate:g}"

    # 买 N 件
    m = re.search(r"买\s*(\d+)\s*件", prompt)
    if m:
        return f"{nums[0] * float(m.group(1)):g}"

    # 月费 → 一年
    if "一年" in prompt and ("月费" in prompt or "每月" in prompt):
        return f"{nums[0] * 12:g}"

    return None


class RuleApp:
    """规则客服系统。四个能力档位构成可对比的版本序列。

    档位是**累积**的：v2 含 v1 的能力，v3 含 v2，v4 含 v3。
    于是在同一个评测集上，分数应当单调不降——这个预期可以手算，
    也就成了回归门禁的天然素材：如果 v4 退化到 v1 的水平，门禁必须拦住。
    """

    name = "rule"
    CAPABILITY_ORDER = ["can_faq", "can_refuse", "can_tool", "can_math"]

    def __init__(self, version: str = "v1", **caps):
        self.version = version
        # 默认全关，显式打开
        self.caps = {k: False for k in self.CAPABILITY_ORDER}
        self.caps.update({k: v for k, v in caps.items() if k in self.caps})

    @property
    def fingerprint(self) -> str:
        """version 之外还拼上能力位串——直接改 caps 构造时（不走 from_version）也能区分。"""
        bits = "".join("1" if self.caps.get(k) else "0" for k in self.CAPABILITY_ORDER)
        return f"rule-{self.version}-{bits}"

    @classmethod
    def from_version(cls, version: str) -> "RuleApp":
        """按版本号给出能力档位。v1 → v2 → v3 → v4 逐级累积。"""
        ladder = {
            "v1": {"can_faq": True},
            "v2": {"can_faq": True, "can_refuse": True},
            "v3": {"can_faq": True, "can_refuse": True, "can_tool": True},
            "v4": {"can_faq": True, "can_refuse": True, "can_tool": True, "can_math": True},
        }
        if version not in ladder:
            raise ValueError(f"未知版本 {version}，可选 {sorted(ladder)}")
        return cls(version=version, **ladder[version])

    def predict(self, item: dict) -> Output:
        t0 = time.perf_counter()
        prompt = str(item.get("prompt") or "")
        tags = str(item.get("tags") or "")

        # 拒答优先级最高——安全风险不该被后续逻辑覆盖
        if self.caps.get("can_refuse"):
            for pat in REFUSE_PATTERNS:
                if re.search(pat, prompt):
                    return Output(text="抱歉，这个请求我无法协助。",
                                  meta={"rule": "refuse"},
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        if tags == "tool" and self.caps.get("can_tool"):
            for pat, tool, saying in TOOL_RULES:
                if re.search(pat, prompt):
                    return Output(text=saying, tool_calls=[{"name": tool, "args": {}}],
                                  meta={"rule": "tool"},
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        if tags == "math" and self.caps.get("can_math"):
            ans = solve_math(prompt)
            if ans is not None:
                return Output(text=f"计算结果是 {ans}。", meta={"rule": "math"},
                              latency_ms=(time.perf_counter() - t0) * 1000)

        if tags == "faq" and self.caps.get("can_faq"):
            for keys, answer in FAQ_KB:
                if any(k in prompt for k in keys):
                    return Output(text=answer, meta={"rule": "faq"},
                                  latency_ms=(time.perf_counter() - t0) * 1000)

        return Output(text="我不太清楚，建议您咨询人工客服。", meta={"rule": "fallback"},
                      latency_ms=(time.perf_counter() - t0) * 1000)


# ==========================================================================
# 3. OpenAI 兼容适配器 —— 有真模型时接进来，上层不用改
# ==========================================================================

class OpenAICompatibleApp:
    """接任何 OpenAI 兼容端点：vLLM / Ollama(需 /v1) / 各家 API。

    **当前环境用不了**（ollama 未运行、无 API key），保留是为了说明架构上的可插拔点在哪。
    要用的时候：
        OpenAICompatibleApp(base_url="http://127.0.0.1:11434/v1", model="qwen2.5:7b")
    """

    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 version: str = "v1", timeout: int = 60, system: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.version = version
        self.timeout = timeout
        self.system = system or "你是一个客服助手。需要查询信息时请调用工具。"

    @property
    def fingerprint(self) -> str:
        """model 决定输出，必须进指纹；base_url 不影响结果，不进（换个网关不算换模型）。"""
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", self.model)
        return f"openai-{safe_model}-{self.version}"

    def predict(self, item: dict) -> Output:
        import urllib.error
        import urllib.request

        t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": str(item.get("prompt") or "")},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"].get("content") or ""
            calls = body["choices"][0]["message"].get("tool_calls") or []
            return Output(text=text, tool_calls=calls, meta={"model": self.model},
                          latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            return Output(text="", error=f"{type(e).__name__}: {e}",
                          latency_ms=(time.perf_counter() - t0) * 1000)


def build_app(spec: str):
    """从字符串构造应用，方便命令行传参。

    规格：
        rag:v3             -> RagApp.from_version("v3")   【默认，真实 RAG 客服助手】
        rag:v4:k2          -> 以 v4 为基础，覆写 top_k=2（用 k=v 形式追加）
        mock:v1:0.8        -> MockApp(v1, accuracy=0.8)
        rule:v3            -> RuleApp.from_version("v3")（旧玩具版，仅作对照保留）
        openai:URL:model   -> OpenAICompatibleApp(URL, model)
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "rag":
        from rag import RagApp  # 局部导入：rag 依赖 kb，只在用到时加载
        version = parts[1] if len(parts) > 1 else "v1"
        override = {}
        for extra in parts[2:]:
            if "=" not in extra:
                continue
            k, v = extra.split("=", 1)
            if k == "k":
                override["top_k"] = int(v)
            elif k == "rewrite":
                override["rewrite"] = v.lower() in ("1", "true", "on")
            elif k == "refusal":
                override["refusal"] = v
            elif k == "tools":
                override["tools"] = v.lower() in ("1", "true", "on")
        return RagApp.from_version(version, **override)
    if kind == "mock":
        version = parts[1] if len(parts) > 1 else "v1"
        acc = float(parts[2]) if len(parts) > 2 else 1.0
        return MockApp(version=version, accuracy=acc)
    if kind == "rule":
        return RuleApp.from_version(parts[1] if len(parts) > 1 else "v1")
    if kind == "openai":
        if len(parts) < 3:
            raise ValueError("openai 规格需要: openai:<base_url>:<model>")
        return OpenAICompatibleApp(base_url=parts[1], model=parts[2], version="v1")
    raise ValueError(f"未知应用类型 {kind}，可选 mock / rule / openai")
