#!/usr/bin/env python3
"""
客服 Web 相关业务逻辑：进程内 LangGraph 调用（fork 适配）、线程/运行、状态解析、会话列表拼装等。
与原仓库的 LangGraph Server (REST) 版解耦：本 fork 的 harness 是进程内直调，
见 multi_agent_customer_service.make_graph()。与 Flask 路由解耦，便于单测与复用。

会话约定：session_id == thread_id（LangGraph SqliteSaver 的线程键），
前端传入的 client_session_id 直接作为 thread_id 复用（跨进程持久化）。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *  # noqa: E402,F401,F403
from multi_agent_customer_service import CHECKPOINT_DIR, get_checkpointer, make_graph  # noqa: E402
from tickets_store import batch_status, ensure_ticket  # noqa: E402（V10：工单闭环状态）

# -----------------------------------------------------------------------------
# 进程内图（单例，复用 checkpointer 与 LLM 客户端，避免每条工单重建）
# -----------------------------------------------------------------------------

_app = None


def get_app():
    """构建/复用进程内 LangGraph 应用（make_graph 内部已绑定 checkpointer 单例）。"""
    global _app
    if _app is None:
        _app = make_graph()
    return _app


def get_checkpointer_ref():
    """复用 multi_agent_customer_service 的 SqliteSaver 单例（同一 DB 文件）。"""
    return get_checkpointer()


# -----------------------------------------------------------------------------
# V11：SQL 直查每线程最新 checkpoint（替代 cp.list 全量遍历 + 支持分页）
# -----------------------------------------------------------------------------


def _checkpoint_db_path() -> str:
    """checkpointer DB 路径（与 multi_agent_customer_service.get_checkpointer 一致）。"""
    return os.environ.get("CHECKPOINT_DB_PATH") or os.path.join(CHECKPOINT_DIR, "checkpoints.db")


def _latest_checkpoint_rows(db_path: str, limit: Optional[int] = None, offset: int = 0) -> List[Tuple[str, str, bytes, Optional[bytes]]]:
    """SQL 直查每线程最新 checkpoint 行（V11）。

    纯 SQL 逻辑（可单测）：每线程最新 = MAX(rowid)（写入顺序单调递增）；
    只读连接（mode=ro）不抢 WAL 写锁（WSL+NTFS 友好）。
    返回 [(thread_id, type, checkpoint_blob, metadata_blob)]，按写入序倒序。
    type 列供 serde.loads_typed((type, blob)) 选择解码器（msgpack/json 等）。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        sql = (
            "SELECT c.thread_id, c.type, c.checkpoint, c.metadata FROM checkpoints c "
            "JOIN (SELECT thread_id, MAX(rowid) AS m FROM checkpoints "
            "      WHERE checkpoint_ns = '' GROUP BY thread_id) latest "
            "  ON c.thread_id = latest.thread_id AND c.rowid = latest.m "
            "ORDER BY c.rowid DESC"
        )
        params: List[Any] = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        return list(conn.execute(sql, params).fetchall())
    finally:
        conn.close()


def _fetch_latest_checkpoints(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """拉取每线程最新 checkpoint 的轻量 channel_values（V11）。

    用 checkpointer 自身的 serde（公开 API）反序列化 BLOB：loads_typed((type, blob))，
    type 列告诉 serde 解码器（msgpack/json），格式自动跟随；
    只取轻量字段，不构建完整对话历史。
    """
    cp = get_checkpointer()
    serde = cp.serde
    rows = _latest_checkpoint_rows(_checkpoint_db_path(), limit=limit, offset=offset)
    out = []
    for tid, typ, cp_blob, _md_blob in rows:
        try:
            cp_data = serde.loads_typed((typ, cp_blob))
            values = cp_data.get("channel_values") or {}
            ts = cp_data.get("ts") or ""
        except Exception as e:
            logger.warning("反序列化线程 %s 失败: %s", tid, e)
            values, ts = {}, ""
        out.append({"thread_id": tid, "values": values, "ts": ts})
    return out


# 当前线程缓存（与 web_app 行为一致，供首页/测试读取）
# V6 修复（2026-08-14）：模块级全局 → thread-local。
# 原因：Flask 多线程下全局变量被并发请求互相覆盖，/api/chat 返回错误的 session_id。
_current_thread_local = threading.local()


def get_current_thread_id() -> Optional[str]:
    return getattr(_current_thread_local, "thread_id", None)


# -----------------------------------------------------------------------------
# 线程 state → 对话列表 / 侧栏预览
# -----------------------------------------------------------------------------

def append_turn_from_state(conversation_history: List[Dict[str, Any]], msg: Dict[str, Any]) -> None:
    """从状态中的单条消息追加到会话历史列表；仅在状态里带有 timestamp 时写入条目。"""
    content = msg.get("content", "") or ""
    if not content:
        return
    is_user = bool(msg.get("is_user", False))
    entry: Dict[str, Any] = {
        "is_user": is_user,
        "content": content,
        "role": "user" if is_user else "assistant",
    }
    ts = msg.get("timestamp")
    if ts is not None and ts != "":
        entry["timestamp"] = ts
    conversation_history.append(entry)


def conversation_history_from_state_data(state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 LangGraph 线程 state JSON 解析对话列表。"""
    conversation_history: List[Dict[str, Any]] = []
    if not isinstance(state_data, dict):
        return conversation_history

    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]

        source_turns = None
        filled_from_turn_list = False
        pd_raw = values.get("persisted_dialogue")
        ch_raw = values.get("conversation_history")
        if isinstance(pd_raw, list) and len(pd_raw) > 0:
            source_turns = pd_raw
        elif isinstance(ch_raw, list) and len(ch_raw) > 0:
            source_turns = ch_raw

        if source_turns is not None:
            filled_from_turn_list = True
            for msg in source_turns:
                if isinstance(msg, dict):
                    append_turn_from_state(conversation_history, msg)

        elif "messages" in values:
            for message in values["messages"]:
                role = message.get("role", "user")
                content = message.get("content", "")
                if content:
                    is_user = role == "user"
                    row: Dict[str, Any] = {
                        "is_user": is_user,
                        "content": content,
                        "role": role,
                    }
                    mt = message.get("timestamp")
                    if mt is not None and mt != "":
                        row["timestamp"] = mt
                    conversation_history.append(row)

        # 已有 persisted_dialogue / conversation_history 时不再追加 values.response：
        # 助手正文已在轮次里；final_response_node 还可能给 response 加前缀导致去重失败、出现双线助手气泡。
        if "response" in values and values["response"]:
            response_content = values["response"]
            if not filled_from_turn_list:
                if not any(
                    msg["content"] == response_content and not msg["is_user"]
                    for msg in conversation_history
                ):
                    conversation_history.append({
                        "is_user": False,
                        "content": response_content,
                        "role": "assistant"
                    })

    elif "messages" in state_data:
        for message in state_data["messages"]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content:
                is_user = role == "user"
                row = {
                    "is_user": is_user,
                    "content": content,
                    "role": role,
                }
                mt = message.get("timestamp")
                if mt is not None and mt != "":
                    row["timestamp"] = mt
                conversation_history.append(row)

    return conversation_history


def last_user_question_from_history(conversation_history: List[Dict[str, Any]]) -> str:
    """取最后一条用户消息的纯文本（用于侧栏预览）。"""
    for msg in reversed(conversation_history):
        if not msg.get("is_user"):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        s = content.strip()
        if s:
            return s
    return ""


def extract_ai_response(thread_state: Dict[str, Any]) -> str:
    """从线程状态中提取 AI 回复文本。"""
    try:
        if "values" in thread_state and isinstance(thread_state["values"], dict):
            values = thread_state["values"]

            if "response" in values and values["response"]:
                return str(values["response"])

            if "messages" in values:
                for message in values["messages"]:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if content:
                            return str(content)

        return "抱歉，我无法理解您的问题。"

    except Exception as e:
        logger.warning(f"提取 AI 回复失败: {e}")
        return "抱歉，处理您的请求时出现了错误。"


# -----------------------------------------------------------------------------
# 助手 / 线程（进程内：无独立 LangGraph Server，线程键即 session_id）
# -----------------------------------------------------------------------------

def seed_thread_if_needed(client_session_id: Optional[str]) -> str:
    """返回有效的线程 ID（= session_id）。空/默认值生成新 UUID 线程。"""
    sid = (client_session_id or "").strip()
    # 已有合法会话键则复用；否则新建
    if sid and sid != "default":
        _current_thread_local.thread_id = sid
        return sid
    import uuid
    new_id = str(uuid.uuid4())
    _current_thread_local.thread_id = new_id
    return new_id


def _checkpoint_values(client_session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """取某线程最新 checkpoint 的 channel_values（无则 None）。"""
    tid = seed_thread_if_needed(client_session_id)
    try:
        tup = get_checkpointer_ref().get_tuple({"configurable": {"thread_id": tid}})
    except Exception as e:
        logger.warning(f"读取线程 {tid} 状态失败: {e}")
        return None
    if tup is None:
        return None
    return tup.checkpoint.get("channel_values") or {}


def _state_payload(values: Dict[str, Any]) -> Dict[str, Any]:
    """把 channel_values 包装成原 REST '/threads/{id}/state' 的 {values: ...} 结构，兼容既有解析函数。"""
    # get_tuple 返回的是 msgpack 序列化后的 dict，保证可被 json 编码
    return {"values": values}


def _normalize_created_at(created_at: Any) -> float:
    if isinstance(created_at, str):
        try:
            dt = _dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return time.time()
    if isinstance(created_at, (int, float)) and created_at > 0:
        return float(created_at)
    return time.time()


def _message_count_from_state_data(state_data: Dict[str, Any]) -> int:
    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]
        # fork 采用 persisted_dialogue 作为权威对话记录（conversation_history 在本 fork 可为空）
        if values.get("persisted_dialogue"):
            return len(values["persisted_dialogue"])
        if values.get("conversation_history"):
            return len(values["conversation_history"])
        if "messages" in values:
            return len(values["messages"])
        if "response" in values and values["response"]:
            return 1
    if "messages" in state_data:
        return len(state_data["messages"])
    return 0


def fetch_sessions_list(limit: Optional[int] = None, offset: int = 0) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    拉取线程列表并拼装前端会话项（V11：SQL 直查每线程最新 checkpoint + 轻量解析 + 分页）。
    成功返回 (sessions, None)，失败返回 (None, error_message)。
    """
    try:
        threads = _fetch_latest_checkpoints(limit=limit, offset=offset)
        from dialogue_archive import count_entries_batch
        archive_counts = count_entries_batch(t["thread_id"] for t in threads)
        sessions: List[Dict[str, Any]] = []
        for info in threads:
            dobj = info.get("values") or {}
            last_user_question = ""
            message_count = 0
            try:
                # 轻量解析：只读 persisted_dialogue，不构建完整对话历史（V11 优化）
                pd = dobj.get("persisted_dialogue") or []
                for msg in reversed(pd):
                    if isinstance(msg, dict) and msg.get("is_user") and msg.get("content"):
                        last_user_question = str(msg["content"])
                        break
                # V12：压缩后 pd 只留尾部，真实消息数 = pd 尾部 + 归档条数
                message_count = _merge_message_count(info["thread_id"], len(pd), archive_counts)
            except Exception:
                message_count = 0
            sessions.append({
                "session_id": info["thread_id"],
                "created_at": _normalize_created_at(info["ts"]),
                "message_count": message_count,
                "last_user_question": last_user_question,
            })

        # 按创建时间倒序（最新在前）
        sessions.sort(key=lambda s: s["created_at"], reverse=True)
        return sessions, None

    except Exception as e:
        logger.exception("获取会话列表失败")
        return None, f"服务器错误: {str(e)}"


def conversation_history_merged(session_id: str, state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """V12：归档历史 + persisted_dialogue 尾部合并，保完整对话顺序（归档在前 = 时间更早）。

    长会话压缩后 pd 只留尾部 DIALOGUE_TAIL_KEEP 条，被归档旧轮从 dialogue_archive
    读回并拼在头部——web 会话详情/工单历史渲染不丢早期对话（V10 工单闭环不退化）。
    """
    from dialogue_archive import fetch_entries
    base = conversation_history_from_state_data(state_data)
    archived = fetch_entries(session_id)
    if not archived:
        return base
    archived_rows: List[Dict[str, Any]] = []
    for msg in archived:
        append_turn_from_state(archived_rows, msg)
    return archived_rows + base


def _merge_message_count(thread_id: str, pd_count: int, archive_counts: Dict[str, int]) -> int:
    """V12：pd 尾部条数 + 归档条数 = 真实消息数（压缩后 pd 不再全量）。"""
    return pd_count + archive_counts.get(thread_id, 0)


def fetch_session_detail(session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """获取单个线程详情 + 对话历史。成功返回 (payload, None)。"""
    try:
        cp = get_checkpointer_ref()
        tup = cp.get_tuple({"configurable": {"thread_id": session_id}})
        if tup is None:
            return None, f"会话不存在: {session_id}"

        values = tup.checkpoint.get("channel_values") or {}
        state_data = _state_payload(values)
        conversation_history = conversation_history_merged(session_id, state_data)

        session_data = {
            "session_id": session_id,
            "created_at": _normalize_created_at(tup.checkpoint.get("ts", time.time())),
            "conversation_history": conversation_history,
        }
        return session_data, None

    except Exception as e:
        logger.exception("获取会话详情失败")
        return None, f"服务器错误: {str(e)}"


def fetch_escalated_tickets(limit: Optional[int] = None, offset: int = 0) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    待人工处理工单队列：SQL 直查每线程最新 checkpoint，筛出 escalated=True 的线程（只读，不做状态流转）。
    每个工单：thread_id、原问题、升级原因、升级摘要、创建时间、消息数、状态（V10/V11）。
    成功返回 (tickets, None)，失败返回 (None, error)。
    """
    try:
        threads = _fetch_latest_checkpoints(limit=limit, offset=offset)

        tickets: List[Dict[str, Any]] = []
        status_map = batch_status()  # V10：全部工单状态（独立实体，与 agent 状态机分离）
        from dialogue_archive import count_entries_batch
        archive_counts = count_entries_batch(t["thread_id"] for t in threads)
        for info in threads:
            tid = info["thread_id"]
            dobj = info.get("values") or {}
            if not bool(dobj.get("escalated")):
                continue

            # V10：lazy 物化工单（首次出现 → pending），并读取状态/时间戳
            ts_float = _normalize_created_at(info["ts"])
            created_str = _dt.datetime.fromtimestamp(ts_float).strftime("%Y-%m-%d %H:%M:%S")
            st = ensure_ticket(tid, created_str)
            st_info = status_map.get(tid, {})

            # 原问题：优先取任务列表里的用户消息首条，否则用 customer_query
            customer_query = str(dobj.get("customer_query", "") or "")
            reason = str(dobj.get("escalation_reason", "") or "")
            summary = str(dobj.get("escalation_summary", "") or "")

            # 轻量解析：只读 persisted_dialogue，不构建完整对话历史（V11 优化）
            last_user_question = customer_query
            message_count = 0
            try:
                pd = dobj.get("persisted_dialogue") or []
                for msg in reversed(pd):
                    if isinstance(msg, dict) and msg.get("is_user") and msg.get("content"):
                        last_user_question = str(msg["content"])
                        break
                message_count = len(pd)
            except Exception:
                message_count = 0

            tickets.append({
                "thread_id": tid,
                "created_at": ts_float,
                "ticket_created_at": st_info.get("created_at") or created_str,
                "status": st,
                "assigned_at": st_info.get("assigned_at") or "",
                "resolved_at": st_info.get("resolved_at") or "",
                "message_count": message_count,
                "customer_query": customer_query,
                "last_user_question": last_user_question,
                "escalation_reason": reason,
                "escalation_summary": summary,
            })

        # 按创建时间倒序（最新升级在前）
        tickets.sort(key=lambda t: t["created_at"], reverse=True)
        return tickets, None

    except Exception as e:
        logger.exception("获取工单队列失败")
        return None, f"服务器错误: {str(e)}"


def delete_remote_thread(thread_id: str) -> Tuple[bool, int]:
    """删除线程（进程内：checkpointer.delete_thread）。成功为 True。"""
    try:
        get_checkpointer_ref().delete_thread(thread_id)
        return True, 200
    except Exception as e:
        logger.warning(f"删除线程 {thread_id} 失败: {e}")
        return False, 500


def clear_thread_and_create_new(thread_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    删除旧线程并新建线程（进程内）。
    成功返回 (new_thread_id, None)，失败返回 (None, error)。
    """
    ok, _ = delete_remote_thread(thread_id)
    if not ok:
        return None, "清空会话失败"

    import uuid
    new_thread_id = str(uuid.uuid4())
    _current_thread_local.thread_id = new_thread_id
    return new_thread_id, None


# -----------------------------------------------------------------------------
# 一次聊天运行（进程内同步 invoke）
# -----------------------------------------------------------------------------

def run_chat_sync(user_message: str, client_session_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    在当前线程上提交一轮用户消息并等待完成（进程内直调 make_graph().invoke()）。
    client_session_id: 前端传入的会话 ID（= 线程 ID）。
    返回 (ai_text, error_text, http_status_optional)。
    """
    if not user_message.strip():
        return None, "消息不能为空", 400

    tid = seed_thread_if_needed(client_session_id)

    try:
        result = get_app().invoke(
            # session_id 注入 state（= thread_id）：trace 聚合键 + billing/complaint
            # agent 会话上下文回退键（此前缺省 "default" 池跨线程共享，w8 挂账修复）
            {"customer_query": user_message.strip(), "session_id": tid},
            {"configurable": {"thread_id": tid}},
        )
        _current_thread_local.thread_id = tid
        ai_text = str(result.get("response") or "")
        if not ai_text:
            return None, "未获取到回复", 500
        return ai_text, None, None

    except Exception as e:
        logger.exception("聊天处理失败")
        return None, f"内部错误: {str(e)}", 500


def run_chat_once_events(user_message: str, client_session_id: Optional[str] = None) -> Iterable[str]:
    """
    一次性聊天返回（SSE 帧格式，V14 改名：非流式）。

    原 stream_chat_events 命名误导——实际是同步 invoke 一次产出完整回复后整体吐给
    SSE，并非逐 token 流式（无流式事件源：OpenAICompatibleClient 仅同步 invoke，
    LangGraph 图为同步执行，LLM 通道韧性层——熔断/重试/信号量——按完整响应设计）。
    前端也未消费该接口（走 /api/chat 同步路径），故改名消除误导，不做流式改造。

    保留 SSE 帧格式（data: JSON + [DONE]）仅为协议兼容，语义 = 一次性返回。
    """
    if not user_message.strip():
        yield "data: " + json.dumps({"error": "消息不能为空"}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    tid = seed_thread_if_needed(client_session_id)

    try:
        result = get_app().invoke(
            {"customer_query": user_message.strip(), "session_id": tid},
            {"configurable": {"thread_id": tid}},
        )
        _current_thread_local.thread_id = tid
        ai_text = str(result.get("response") or "")
        if ai_text:
            yield "data: " + json.dumps({"content": ai_text, "session_id": tid, "thread_id": tid}) + "\n\n"
        else:
            yield "data: " + json.dumps({"error": "未获取到回复"}) + "\n\n"
    except Exception as e:
        logger.exception("一次性返回处理失败")
        yield "data: " + json.dumps({"error": f"一次性返回处理错误: {str(e)}"}) + "\n\n"

    yield "data: [DONE]\n\n"


# -----------------------------------------------------------------------------
# P1 真流式聊天（SSE）：worker 线程跑图 + thread-local sink → Queue → SSE 帧
# -----------------------------------------------------------------------------

def _sse_frame(obj: Dict[str, Any]) -> str:
    """SSE 帧：data: JSON（ensure_ascii=False——中文 token 不转义，帧更小）。"""
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def run_chat_stream_events(user_message: str, client_session_id: Optional[str] = None) -> Iterable[str]:
    """真流式聊天（P1，2026-08-21）：token 流 + 阶段流，逐帧推送。

    架构：Flask 请求线程只做 SSE 帧转发；图 invoke 在 worker 线程执行——
    - worker 注册 thread-local sink（llm.stream_sink），图节点在同线程内执行，
      agent 正文生成逐 token 推送（call_llm）、节点进入时推 stage（_traced）；
    - sink 只往 Queue 投递，请求线程从 Queue 读帧 yield——线程间解耦，
      Flask 生成器异常（客户端断连）不影响 worker 把图跑完（回复照常持久化）。

    帧协议（data: JSON，事件以 \n\n 分隔）：
    - {"type":"meta","session_id":..,"thread_id":..}     首帧（会话键，前端据此更新）
    - {"type":"stage","node":"classify_query"}           阶段流（节点开始）
    - {"type":"token","content":".."}                    token 流（正文增量）
    - {"type":"done","content":..,"agent":..,"query_type":..,"escalated":..}
    - {"type":"error","error":".."}
    结束标记 data: [DONE]。心跳：15s 无事件发 ": ping" 注释行（防代理空闲断连）。

    done.content 是权威全文（含引用脚注——引用在节点内流结束后附加），
    前端应以 done.content 做最终渲染，token 流仅用于渐进展示。
    """
    if not user_message.strip():
        yield _sse_frame({"type": "error", "error": "消息不能为空"})
        yield "data: [DONE]\n\n"
        return

    tid = seed_thread_if_needed(client_session_id)
    yield _sse_frame({"type": "meta", "session_id": tid, "thread_id": tid})

    from llm.stream_sink import clear_sink, set_sink

    q: "queue.Queue" = queue.Queue()
    _DONE = object()  # worker 结束哨兵（与业务帧区分）

    def _worker():
        def sink(kind, payload):
            q.put((kind, payload))

        set_sink(sink)
        try:
            result = get_app().invoke(
                {"customer_query": user_message.strip(), "session_id": tid},
                {"configurable": {"thread_id": tid}},
            )
            q.put(("done", {
                "content": str(result.get("response") or ""),
                "agent": str(result.get("current_agent") or ""),
                "query_type": str(result.get("query_type") or ""),
                "escalated": bool(result.get("escalated")),
            }))
        except Exception as e:  # noqa: BLE001  worker 内兜底：图异常 → error 帧，不静默
            logger.exception("流式聊天处理失败")
            q.put(("error", {"error": f"内部错误: {str(e)}"}))
        finally:
            clear_sink()  # 防线程池/线程残留 sink（V6 同纪律）
            q.put(_DONE)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        try:
            item = q.get(timeout=15)
        except queue.Empty:
            yield ": ping\n\n"  # 心跳：分类/检索阶段无 token 输出时保活连接
            continue
        if item is _DONE:
            break
        kind, payload = item
        if kind == "token":
            yield _sse_frame({"type": "token", "content": payload})
        else:
            yield _sse_frame(dict(payload, type=kind))

    yield "data: [DONE]\n\n"


def fetch_trace_runs(limit: int = 50) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """V18：最近 N 次工单 trace（trace 面板列表页）。成功返回 (runs, None)。"""
    try:
        from trace_store import fetch_runs
        return fetch_runs(limit=limit), None
    except Exception as e:
        logger.exception("获取 trace 列表失败")
        return None, f"服务器错误: {str(e)}"


def fetch_trace_detail(run_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """V18：单次工单完整决策链（trace 面板详情页）。成功返回 (run, None)。"""
    try:
        from trace_store import fetch_run_detail
        run = fetch_run_detail(run_id)
        if run is None:
            return None, f"trace 不存在: {run_id}"
        return run, None
    except Exception as e:
        logger.exception("获取 trace 详情失败")
        return None, f"服务器错误: {str(e)}"


def langgraph_connectivity_test() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    探测进程内 harness 就绪状态（fork 适配：替代原 LangGraph Server REST 探测）。
    成功返回 (result_dict, None)，失败返回 (None, error_message)。
    """
    try:
        cp = get_checkpointer_ref()
        thread_count = 0
        for _ in cp.list(None, limit=1):
            thread_count = 1
        return ({
            "status": "test_completed",
            "health_check": 200,
            "backend": "in_process",
            "threads_found": thread_count,
            "details": {
                "ok_response": "OK",
                "backend": "multi_agent_customer_service.make_graph (进程内)",
            }
        }, None)
    except Exception as e:
        logger.exception("进程内 harness 测试失败")
        return None, f"测试失败: {str(e)}"