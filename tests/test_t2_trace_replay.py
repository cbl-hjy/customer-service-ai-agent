"""测试：T2 生产 trace 回流（采集回填 + 生产池聚合 + 多轮聚合）。纯本地密封测试。

覆盖（具名风险）：
  A1 finish_run 落 response/tools_used —— T2 回流数据源字段可写可读
  A2 旧库幂等迁移                     —— T2 之前建的 trace.db 补列不炸、旧行兼容
  B1 生产池过滤+聚合                  —— eval-/v12-/v14-/bench- 排除、unique 聚合、label 分布
  C1 _traced 出口回填                 —— final_response 节点 response/tools_used 进 trace_runs
  S1 web 层 session_id 注入           —— invoke 输入缺 session_id → trace thread_id
                                         fallback query 前缀 + 上下文回退 "default" 池串话
  D1 多轮池过滤+提取                  —— 前缀 thread/垃圾轮/重复型（压测同句）误入多轮池
"""

import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))

import trace_store as _ts
import trace_replay as _tr


@pytest.fixture(autouse=True)
def _isolated_trace_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "trace.db"))
    yield str(tmp_path / "trace.db")


def test_a1_finish_run_persists_replay_fields():
    _ts.start_run("r1", "t1", "查订单", "2026-09-16 10:00:00")
    _ts.finish_run("r1", 100.0, 10, 20, {"query_type": "general_inquiry"},
                   response="已发货，单号 SF123。", tools_used=["query_order", "综合客服_processing"])
    conn = sqlite3.connect(os.environ["TRACE_DB_PATH"])
    row = conn.execute("SELECT response, tools_used FROM trace_runs WHERE run_id='r1'").fetchone()
    conn.close()
    assert row[0] == "已发货，单号 SF123。"
    assert json.loads(row[1]) == ["query_order", "综合客服_processing"]


def test_a2_legacy_db_migration(tmp_path, monkeypatch):
    """T2 之前的旧 schema（无 response/tools_used 列）→ _connect 自动补列，旧行可读。"""
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy))
    conn.execute(
        """CREATE TABLE trace_runs (
            run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_query TEXT NOT NULL,
            ts TEXT NOT NULL, total_ms REAL, prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0, decision TEXT)"""
    )
    conn.execute("INSERT INTO trace_runs (run_id, thread_id, user_query, ts, decision) "
                 "VALUES ('old1','t','q','2026-01-01','{}')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("TRACE_DB_PATH", str(legacy))
    _ts.start_run("new1", "t", "q2", "2026-09-16")  # 触发迁移
    _ts.finish_run("new1", 1.0, 0, 0, {}, response="ok")
    rows = _ts.fetch_runs(10)
    ids = {r["run_id"] for r in rows}
    assert {"old1", "new1"} <= ids  # 旧行兼容 + 新行带 response


def _seed_pool_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE trace_runs (
            run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, user_query TEXT NOT NULL,
            ts TEXT NOT NULL, total_ms REAL, prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0, decision TEXT)"""
    )
    rows = [
        ("p1", "web-abc", "多少钱", "2026-09-10", '{"query_type": "product_info"}'),
        ("p2", "web-abc", "多少钱", "2026-09-11", '{"query_type": "billing"}'),      # 同 query 摇摆
        ("e1", "eval-mt-201", "多少钱", "2026-09-12", '{"query_type": "complaint"}'),  # 评估轮（排除）
        ("b1", "v12-curve-A", "压测", "2026-09-12", '{"query_type": "general_inquiry"}'),
        ("p3", "web-def", "写诗", "2026-09-13", '{"query_type": "out_of_scope", "escalated": false}'),
    ]
    conn.executemany(
        "INSERT INTO trace_runs VALUES (?,?,?,?,NULL,0,0,?)",
        [(r[0], r[1], r[2], r[3], r[4]) for r in rows],
    )
    conn.commit()
    conn.close()


def test_b1_pool_filter_and_aggregate(tmp_path):
    db = tmp_path / "prod.db"
    _seed_pool_db(str(db))
    pool = _tr.load_production_pool(str(db))

    assert len(pool) == 2  # 多少钱 + 写诗（eval-/v12- 排除）
    by_q = {p["query"]: p for p in pool}
    assert by_q["多少钱"]["n_turns"] == 2
    assert by_q["多少钱"]["labels"] == {"product_info": 1, "billing": 1}
    assert by_q["多少钱"]["latest_ts"] == "2026-09-11"
    assert by_q["写诗"]["labels"] == {"out_of_scope": 1}
    assert pool[0]["query"] == "多少钱"  # 高频在前


def test_c1_traced_final_response_persists(monkeypatch):
    """_traced 包装 final_response：state 的 response/tools_used 落 trace_runs。"""
    from trace import _traced

    def fake_final_node(state):
        return state  # 透传（同真实 final_response_node）

    def fake_classify(state):
        return {"query_type": "general_inquiry", "query_confidence": 0.9}

    # 手动构造入口→出口序列（classify 建 run，final_response 收口）
    wrapped_classify = _traced("classify_query", fake_classify)
    wrapped_final = _traced("final_response", fake_final_node)
    state = {"session_id": "c1-test", "customer_query": "查订单 XH202609010001",
             "response": "已发货。", "tools_used": ["query_order"]}
    wrapped_classify(state)
    wrapped_final(state)

    runs = _ts.fetch_runs(5)
    assert len(runs) == 1
    # fetch_runs 未含新列——直查
    conn = sqlite3.connect(os.environ["TRACE_DB_PATH"])
    row = conn.execute("SELECT response, tools_used FROM trace_runs").fetchone()
    conn.close()
    assert row[0] == "已发货。"
    assert json.loads(row[1]) == ["query_order"]


def test_s1_web_invoke_injects_session_id():
    """三处 web 入口 invoke 输入必须带 session_id（= tid）。

    风险：缺 session_id → state.session_id 空 → trace thread_id fallback 到
    customer_query 前缀（多轮无法聚合）+ billing/complaint 上下文回退
    session_manager("default") 池跨线程串话。
    """
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "chat_web_service.py")
    src = open(src_path, encoding="utf-8").read()
    injected = src.count('{"customer_query": user_message.strip(), "session_id": tid}')
    bare = src.count('{"customer_query": user_message.strip()}')
    assert injected == 3  # run_chat_sync / run_chat_once_events / run_chat_stream_events._worker
    assert bare == 0      # 不允许漏网的裸输入调用


def test_d1_multi_turn_pool_filter_and_extract(tmp_path):
    """多轮池：真多轮保留；重复型（同句压测）/前缀/单轮/垃圾轮剔除。"""
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from typing import TypedDict

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph

    class _S(TypedDict):
        customer_query: str
        persisted_dialogue: list

    def _node(state):
        pd = list(state.get("persisted_dialogue") or [])
        pd.append({"is_user": True, "content": state["customer_query"], "timestamp": "2026-09-16"})
        return {"persisted_dialogue": pd}

    db = tmp_path / "ckpt.db"
    cp_cm = SqliteSaver.from_conn_string(str(db))
    cp = cp_cm.__enter__()
    try:
        g = StateGraph(_S)
        g.add_node("n", _node)
        g.set_entry_point("n")
        g.add_edge("n", END)
        app = g.compile(checkpointer=cp)

        # web-mt-a：2 个不同 query（真多轮）
        for q in ("手机多少钱", "那电脑呢"):
            app.invoke({"customer_query": q}, {"configurable": {"thread_id": "web-mt-a"}})
        # web-mt-b：同 query ×3（压测重复型 → 剔除）
        for _ in range(3):
            app.invoke({"customer_query": "物流太慢我要投诉"}, {"configurable": {"thread_id": "web-mt-b"}})
        # eval-mt-x：前缀排除
        for q in ("q1", "q2"):
            app.invoke({"customer_query": q}, {"configurable": {"thread_id": "eval-mt-x"}})
        # web-mt-c：单轮 → 剔除
        app.invoke({"customer_query": "单轮问题"}, {"configurable": {"thread_id": "web-mt-c"}})
        # web-mt-d：垃圾轮 'q' 剔除后仅剩 1 个不同 query → 剔除
        for q in ("q", "正常问题"):
            app.invoke({"customer_query": q}, {"configurable": {"thread_id": "web-mt-d"}})
        # web-mt-e：3 个不同 query（真多轮，turns 最多排前）
        for q in ("退耳机", "运费谁出", "发票能补开吗"):
            app.invoke({"customer_query": q}, {"configurable": {"thread_id": "web-mt-e"}})
    finally:
        cp_cm.__exit__(None, None, None)

    pool = _tr.load_multi_turn_threads(str(db))
    tids = [p["thread_id"] for p in pool]
    assert set(tids) == {"web-mt-a", "web-mt-e"}  # b 重复型 / eval- 前缀 / c 单轮 / d 垃圾轮均剔除
    assert pool[0]["thread_id"] == "web-mt-e"      # turns 多在前
    assert pool[0]["turns"] == ["退耳机", "运费谁出", "发票能补开吗"]
    assert pool[1]["turns"] == ["手机多少钱", "那电脑呢"]
