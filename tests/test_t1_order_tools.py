"""测试：T1 订单工具层 + general_agent 工具调用循环（2026-09-16）。纯本地密封测试。

覆盖（具名风险）：
  R1 种子幂等播种        —— 重复连接不产生重复订单
  R2 查订单三分支        —— 存在/不存在/订单号格式错（不可信输入白名单）
  R3 查物流              —— 有轨迹/未发货暂无物流（KB 口径）
  R4 改地址政策代码化    —— 成功/已发货拒/限1次拒/已取消拒/地址过短（KB「未发货改地址」）
  R5 分派器兜底          —— 未知工具/坏 JSON/缺参/工具内部异常（C5：不硬答可转述）
  R6 DB 路径隔离         —— ORDERS_DB_PATH 覆盖生效（评估隔离先例）
  L1 tool_calls 执行回填 —— 消息结构（assistant+tool dict）与 tools_used 标记
  L2 无工具直接回复      —— first.content 即回复
  L3 KB miss + 无工具    —— C5 兜底升级（不硬答）
  L4 KB miss + 有工具    —— 工具结果也是事实来源，正常应答
  L5/L6 首轮通道故障降级 —— KB 命中走纯文本 / miss 走升级
  L7 终答故障            —— 系统错误文案（与原路径一致）
  C1-C3 client 兼容      —— 默认 tool_calls=None / payload 无 tools 不带键 / dict 消息透传
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multi_agents.general_agent as _ga
from multi_agents import GeneralAgent
import tools.orders as _orders
from llm.client import CustomResponse, OpenAICompatibleClient


# ---------------------------------------------------------------------------
# 公共设施
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """每个测试独立订单库（tmp 目录），不碰 data/orders.db。"""
    db = tmp_path / "orders.db"
    monkeypatch.setenv("ORDERS_DB_PATH", str(db))
    yield str(db)


def _make_state(query):
    return {
        "session_id": "t1-test",
        "customer_query": query,
        "tools_used": [],
        "persisted_dialogue": [],
        "escalation_summary": "",
        "escalated": False,
    }


class _FakeResp:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _ScriptedLLM:
    """按脚本依次返回响应，记录每次调用的 messages/kwargs。"""

    def __init__(self, script):
        self.script = list(script)  # [(content, tool_calls), ...]
        self.calls = []

    def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        content, tool_calls = self.script.pop(0)
        return _FakeResp(content, tool_calls)


def _patch_kb(monkeypatch, hit_text):
    """密封隔离：替换 general_agent 的 KB 检索（不加载真模型/GPU）。"""
    monkeypatch.setattr(_ga, "kb_retrieve", lambda domain, q: hit_text)


TOOL_CALL = [{
    "id": "call_test_1", "type": "function",
    "function": {"name": "query_order", "arguments": json.dumps({"order_id": "XH202609010001"})},
}]


# ---------------------------------------------------------------------------
# R 系列：订单数据层
# ---------------------------------------------------------------------------

def test_r1_seed_idempotent():
    _orders._connect().close()
    conn = _orders._connect()
    n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert n == 8
    conn.close()


def test_r2_query_order_branches():
    ok = _orders.query_order("XH202609010001")
    assert "XH202609010001" in ok and "SF1234567890" in ok and "4999" in ok

    miss = _orders.query_order("XH202609019999")
    assert "未找到" in miss

    bad = _orders.query_order("abc")
    assert "格式不正确" in bad


def test_r3_query_logistics():
    hit = _orders.query_logistics("XH202609010006")
    assert "JD5550001111" in hit and "派送中" in hit

    pending = _orders.query_logistics("XH202609010002")
    assert "暂无物流信息" in pending and "待发货" in pending


def test_r4_update_address_policy():
    # 成功（待发货 + 首次修改）
    ok = _orders.update_order_address("XH202609010002", "北京市海淀区中关村大街1号")
    assert "修改成功" in ok
    conn = _orders._connect()
    row = conn.execute(
        "SELECT address, address_updated FROM orders WHERE order_id='XH202609010002'"
    ).fetchone()
    assert row["address"] == "北京市海淀区中关村大街1号" and row["address_updated"] == 1
    conn.close()

    # 已发货 → 拒绝（KB：仅待发货可改）
    assert "修改失败" in _orders.update_order_address("XH202609010001", "上海市黄浦区南京东路100号")
    # 已用 1 次额度 → 拒绝（KB：每单限 1 次）
    assert "修改失败" in _orders.update_order_address("XH202609010005", "广州市天河区体育西路55号")
    # 已取消 → 拒绝
    assert "修改失败" in _orders.update_order_address("XH202609010004", "某省某市某区路1号")
    # 地址过短 → 拒绝
    assert "无效" in _orders.update_order_address("XH202609010008", "北京")


def test_r5_dispatcher_guards():
    ok = _orders.execute_tool_call("query_order", json.dumps({"order_id": "XH202609010001"}))
    assert "XH202609010001" in ok

    assert "未知工具" in _orders.execute_tool_call("no_such_tool", "{}")
    assert "参数格式错误" in _orders.execute_tool_call("query_order", "not-json{")
    assert "参数不匹配" in _orders.execute_tool_call("query_order", json.dumps({"wrong": 1}))

    # 工具内部异常 → 可转述错误（不抛出）
    orig = _orders._TOOL_FUNCS["query_logistics"]
    _orders._TOOL_FUNCS["query_logistics"] = lambda **kw: (_ for _ in ()).throw(RuntimeError("db boom"))
    try:
        out = _orders.execute_tool_call("query_logistics", json.dumps({"order_id": "XH202609010001"}))
        assert "执行出错" in out
    finally:
        _orders._TOOL_FUNCS["query_logistics"] = orig


def test_r6_db_path_isolation(tmp_path, monkeypatch):
    db2 = tmp_path / "orders2.db"
    monkeypatch.setenv("ORDERS_DB_PATH", str(db2))
    _orders.query_order("XH202609010001")  # 触发建库播种
    assert db2.exists()
    # 原库不存在于此路径命名空间（隔离生效：不同文件）
    import sqlite3
    conn = sqlite3.connect(str(db2))
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 8
    conn.close()


# ---------------------------------------------------------------------------
# L 系列：general_agent 工具循环（mock LLM + mock KB）
# ---------------------------------------------------------------------------

def test_l1_tool_call_executed_and_backfilled(monkeypatch):
    """tool_calls → 执行 + 回填（assistant dict + tool dict）+ 流式终答。"""
    _patch_kb(monkeypatch, "服务信息：物流单号查询…")
    llm = _ScriptedLLM([
        ("I'll look up.", TOOL_CALL),          # 首轮：返回 tool_calls
        ("您的订单已发货，单号 SF1234567890。", None),  # 终答（call_llm 无 sink → invoke）
    ])
    agent = GeneralAgent()
    agent.set_llm(llm)
    result = agent.process(_make_state("查一下订单 XH202609010001 到哪了"))

    assert "query_order" in result["tools_used"]
    assert "SF1234567890" in result["response"]
    assert result["escalated"] is False

    # 第二次调用（终答）消息结构：尾两条 = assistant(tool_calls) + tool 结果
    final_msgs = llm.calls[1]["messages"]
    assert final_msgs[-2]["role"] == "assistant" and final_msgs[-2]["tool_calls"] == TOOL_CALL
    assert final_msgs[-1]["role"] == "tool" and final_msgs[-1]["tool_call_id"] == "call_test_1"
    assert "XH202609010001" in final_msgs[-1]["content"]
    # 完整 loop（mt-605 教训修复）：终答轮也带 tools——堵死通道会致 DSML 标记泄漏
    assert llm.calls[0]["tools"] is not None and llm.calls[1]["tools"] is not None


def test_l2_no_tool_direct_reply(monkeypatch):
    _patch_kb(monkeypatch, "服务信息：发货时效…")
    llm = _ScriptedLLM([("下单后 48 小时内发货。", None)])
    agent = GeneralAgent()
    agent.set_llm(llm)
    result = agent.process(_make_state("一般下单后多久发货"))

    assert result["response"] == "下单后 48 小时内发货。"
    assert len(llm.calls) == 1  # 单次调用直接产出
    assert not any(t in result["tools_used"] for t in ("query_order", "query_logistics", "update_order_address"))


def test_l3_kb_miss_no_tool_escalates(monkeypatch):
    """KB miss + 模型不调工具 → C5 兜底升级（不硬答语义保持）。"""
    _patch_kb(monkeypatch, "")
    llm = _ScriptedLLM([("", None)])
    agent = GeneralAgent()
    agent.set_llm(llm)
    result = agent.process(_make_state("快艇怎么修"))

    assert result["escalated"] is True
    assert "知识库无匹配" in result["escalation_reason"]


def test_l4_kb_miss_with_tool_answers(monkeypatch):
    """KB miss + tool_calls → 工具结果也是事实来源（C5 语义扩展）。"""
    _patch_kb(monkeypatch, "")
    llm = _ScriptedLLM([
        ("", TOOL_CALL),
        ("您的订单 XH202609010001 已发货。", None),
    ])
    agent = GeneralAgent()
    agent.set_llm(llm)
    result = agent.process(_make_state("订单 XH202609010001 到哪了"))

    assert result["escalated"] is False
    assert "query_order" in result["tools_used"]


def test_l5_first_call_failure_degrades_to_text(monkeypatch):
    """首轮通道故障 + KB 命中 → 降级纯文本链路（原行为）。"""

    class _FailingThenOk:
        def __init__(self):
            self.n = 0

        def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
            self.n += 1
            if tools:
                raise RuntimeError("channel down")
            return _FakeResp("降级回复。", None)

    _patch_kb(monkeypatch, "服务信息：发货时效…")
    llm = _FailingThenOk()
    agent = GeneralAgent()
    agent.set_llm(llm)
    result = agent.process(_make_state("多久发货"))
    assert result["response"] == "降级回复。"


def test_l6_first_call_failure_kb_miss_escalates(monkeypatch):
    _patch_kb(monkeypatch, "")

    class _AlwaysFail:
        def invoke(self, messages, **kw):
            raise RuntimeError("channel down")

    agent = GeneralAgent()
    agent.set_llm(_AlwaysFail())
    result = agent.process(_make_state("随便什么"))
    assert result["escalated"] is True


def test_l7_loop_round_exhaustion(monkeypatch):
    """轮次耗尽（模型每轮都想调工具）→ 确定性兜底文案（不硬答、不假装执行）。"""

    class _AlwaysToolCall:
        def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
            return _FakeResp("", TOOL_CALL)

    _patch_kb(monkeypatch, "服务信息：…")
    agent = GeneralAgent()
    agent.set_llm(_AlwaysToolCall())
    result = agent.process(_make_state("查订单 XH202609010001"))
    assert "步数上限" in result["response"]
    assert "query_order" in result["tools_used"]


def test_l7b_loop_failure_degrades(monkeypatch):
    """loop 中途通道故障 + KB 命中 → 降级纯文本链路。"""

    class _FailOnSecond:
        def __init__(self):
            self.n = 0

        def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
            self.n += 1
            if self.n == 1:
                return _FakeResp("", TOOL_CALL)
            if tools:
                raise RuntimeError("loop down")
            return _FakeResp("降级回复。", None)

    _patch_kb(monkeypatch, "服务信息：…")
    agent = GeneralAgent()
    agent.set_llm(_FailOnSecond())
    result = agent.process(_make_state("查订单 XH202609010001"))
    assert result["response"] == "降级回复。"


# ---------------------------------------------------------------------------
# C 系列：client 兼容（回归——现有调用方行为零变化）
# ---------------------------------------------------------------------------

def test_c1_custom_response_default():
    r = CustomResponse("hello")
    assert r.content == "hello" and r.tool_calls is None


def test_c2_payload_tools_optional():
    c = OpenAICompatibleClient("k", "https://x.example", "m")
    base = c._build_payload([{"role": "user", "content": "hi"}])
    assert "tools" not in base and "tool_choice" not in base
    with_tools = c._build_payload(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="none",
    )
    assert with_tools["tools"] and with_tools["tool_choice"] == "none"


def test_c3_format_messages_dict_passthrough():
    c = OpenAICompatibleClient("k", "https://x.example", "m")
    tool_msg = {"role": "tool", "tool_call_id": "t1", "content": "result"}
    asst = {"role": "assistant", "content": "", "tool_calls": TOOL_CALL}
    out = c._format_messages([asst, tool_msg])
    assert out == [asst, tool_msg]  # 原样透传（同对象语义）


def test_c3b_format_messages_langchain_unchanged():
    """LangChain 消息转换行为不变（回归）。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    c = OpenAICompatibleClient("k", "https://x.example", "m")
    out = c._format_messages([
        SystemMessage(content="sys"),
        HumanMessage(content="q"),
        "raw string",
    ])
    assert out == [
        {"role": "user", "content": "System instruction: sys"},
        {"role": "user", "content": "q"},
        {"role": "user", "content": "raw string"},
    ]


# ---------------------------------------------------------------------------
# S 系列：流式 tool_calls 聚合（_ToolStream + call_llm_with_tools）
# ---------------------------------------------------------------------------

from llm.client import _ToolStream  # noqa: E402
import llm.stream_sink as _sink_mod  # noqa: E402
from llm.stream_sink import call_llm_with_tools, set_sink, clear_sink  # noqa: E402


class _FakeStreamClient:
    """带 _stream_events 的最小客户端（SSE 事件脚本驱动）。"""

    def __init__(self, events):
        self._script = list(events)
        self.invoke_calls = []

    def _stream_events(self, messages, response_format=None, tools=None):
        for ev in self._script:
            yield ev

    def invoke_stream_tools(self, messages, tools):
        return _ToolStream(self, messages, tools)

    def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
        self.invoke_calls.append({"messages": list(messages), "tools": tools})
        return _FakeResp("非流式兜底。", None)


def _tok(t):
    return {"kind": "token", "text": t}


def _td(**kw):
    return {"kind": "tool_delta", "delta": kw}


def test_s1_tool_call_aggregation_and_lead_suppression():
    """英文前导 + tool_calls 分片：前导被吞、arguments 跨分片拼接正确。"""
    events = [
        _tok("I'll look "), _tok("up."),  # 英文前导（<16 字符，入缓冲）
        _td(index=0, id="call_1", function={"name": "query_order", "arguments": '{"order_i'}),
        _td(index=0, function={"arguments": 'd": "XH202609010001"}'}),
        _tok("done internally"),  # 工具模式：静默丢弃
    ]
    ts = _FakeStreamClient(events).invoke_stream_tools([], tools=[])
    out = list(ts)
    assert out == []  # 前导+工具模式 content 均不产出
    assert ts.tool_calls == [{
        "id": "call_1", "type": "function",
        "function": {"name": "query_order", "arguments": '{"order_id": "XH202609010001"}'},
    }]


def test_s2_chinese_text_passthrough():
    """中文正文：无 tool_delta，token 流保留（打字机），tool_calls=None。"""
    events = [_tok("您好"), _tok("，订单"), _tok("已发货。")]
    ts = _FakeStreamClient(events).invoke_stream_tools([], tools=[])
    out = list(ts)
    assert out == ["您好", "，订单", "已发货。"]
    assert ts.tool_calls is None


def test_s3_ascii_overflow_flush():
    """纯英文正文超 16 字符：兜底 flush（不丢失）。"""
    events = [_tok("Hello "), _tok("world, "), _tok("this is "), _tok("a long reply.")]
    ts = _FakeStreamClient(events).invoke_stream_tools([], tools=[])
    out = list(ts)
    assert "".join(out) == "Hello world, this is a long reply."
    assert ts.tool_calls is None


def test_s4_stream_end_flush_short_ascii():
    """短英文正文（<16 字符）无 tool_delta：流结束收尾产出。"""
    events = [_tok("ok"), _tok(" done")]
    ts = _FakeStreamClient(events).invoke_stream_tools([], tools=[])
    out = list(ts)
    assert out == ["ok done"]


def test_s5_call_llm_with_tools_no_sink(monkeypatch):
    """sink 未注册（评估/单测）→ invoke(tools) 直通（T1 首版行为）。"""
    clear_sink()
    c = _FakeStreamClient([])
    r = call_llm_with_tools(c, [{"role": "user", "content": "q"}], tools=["schema"])
    assert r.content == "非流式兜底。"
    assert c.invoke_calls[0]["tools"] == ["schema"]


def test_s6_call_llm_with_tools_streaming(monkeypatch):
    """sink 注册（web）→ 流式：中文 token 推送 + tool_calls 聚合返回。"""
    events = [
        _tok("I'll check."),  # 前导（吞）
        _td(index=0, id="call_9", function={"name": "query_order", "arguments": '{"order_id": "XH202609010006"}'}),
    ]
    c = _FakeStreamClient(events)
    pushed = []
    set_sink(lambda kind, payload: pushed.append((kind, payload)))
    try:
        r = call_llm_with_tools(c, [{"role": "user", "content": "q"}], tools=["schema"])
    finally:
        clear_sink()
    assert pushed == []  # 工具轮零 token 推送（前导被吞）
    assert r.content == ""
    assert r.tool_calls[0]["function"]["name"] == "query_order"

    # 纯文本轮：token 逐个推送
    events2 = [_tok("您"), _tok("好")]
    c2 = _FakeStreamClient(events2)
    set_sink(lambda kind, payload: pushed.append((kind, payload)))
    try:
        r2 = call_llm_with_tools(c2, [{"role": "user", "content": "q"}], tools=["schema"])
    finally:
        clear_sink()
    assert r2.content == "您好" and r2.tool_calls is None
    assert [p for _, p in pushed] == ["您", "好"]


def test_s7_legacy_client_fallback(monkeypatch):
    """客户端无 invoke_stream_tools 方法（mock/旧实现）→ 回退 invoke。"""
    set_sink(lambda *a: None)  # sink 已注册但客户端不支持流式工具

    class _LegacyClient:
        def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
            return _FakeResp("非流式兜底。", None)

    try:
        r = call_llm_with_tools(_LegacyClient(), [], tools=["schema"])
        assert r.content == "非流式兜底。"
    finally:
        clear_sink()
