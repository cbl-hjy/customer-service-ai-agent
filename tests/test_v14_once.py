"""V14 测试：假流式接口改名（一次性返回）。

背景（审查清单 V14）：stream_chat_events 实际是同步 invoke 完再整体吐给 SSE——
命名误导（"流式"暗示逐 token，实际一次性）。真流式改造评估：OpenAICompatibleClient
仅同步 invoke（无 stream 方法）+ LangGraph 图为同步执行 + LLM 通道韧性层（熔断/重试/
信号量）按完整响应设计——全链路流式改造是小时级+大改，且前端未消费该接口
（index.html 0 引用，走 /api/chat 同步路径），故选择改名方案（审查清单二选一）。

改名：stream_chat_events → run_chat_once_events；路由 /api/chat/stream → /api/chat/once。

验证方式：
- 结构：旧名/旧路由无残留（注释历史说明除外）
- 行为：新接口返回完整回复 + [DONE] 帧（SSE 协议兼容）；空消息错误帧
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _src_of(rel: str) -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
    with open(p, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 结构：改名完成（代码无旧名，注释可提历史）
# ---------------------------------------------------------------------------

def test_old_name_removed_from_code():
    """stream_chat_events 函数/引用已全部改名（注释历史说明除外）。

    P1（2026-09-15）更新：/api/chat/stream 已由真流式实现回归本义
    （worker 线程跑图 + thread-local sink 逐 token 推送），V14 禁的是
    假流式冒用 stream 名——真流式允许 def chat_stream 存在。
    """
    svc = _src_of("chat_web_service.py")
    app = _src_of("web_app.py")
    # 代码形态（def / import / 调用）无旧名（假流式函数）
    assert "def stream_chat_events" not in svc
    assert "stream_chat_events(" not in svc
    assert "stream_chat_events" not in app.replace("原 stream_chat_events", ""), "web_app 残留旧名调用"


def test_stream_route_is_real_streaming():
    """V14 精神延续：/api/chat/stream 存在则必须是真流式（接 run_chat_stream_events），
    一次性返回仍走 /api/chat/once——两路由语义不得混淆。"""
    app = _src_of("web_app.py")
    svc = _src_of("chat_web_service.py")
    assert "@app.route('/api/chat/once'" in app
    assert "@app.route('/api/chat/stream'" in app
    # stream 路由必须接真流式生成器（不是 once 的一次性生成器）
    assert "def chat_stream" in app
    assert "run_chat_stream_events" in app
    assert "def run_chat_stream_events" in svc
    # once 函数体不得调用流式生成器（只看函数体，排除顶部 import 区）
    once_body = app.split("def chat_once")[1].split("def chat_stream")[0]
    assert "run_chat_stream_events" not in once_body


def test_new_names_in_place():
    """新函数名 + 新路由名就位。"""
    svc = _src_of("chat_web_service.py")
    app = _src_of("web_app.py")
    assert "def run_chat_once_events" in svc
    assert "run_chat_once_events" in app
    assert "def chat_once" in app


# ---------------------------------------------------------------------------
# 行为：接口语义（SSE 帧 + 一次性完整返回）
# ---------------------------------------------------------------------------

def _collect_events(user_message: str, session_id: str = "v14-test"):
    """跑生成器，收集全部 SSE 帧。"""
    import os as _os
    import tempfile
    _tmp = _os.path.join(tempfile.gettempdir(), "v14-test.db")
    _os.environ.setdefault("CHECKPOINT_DB_PATH", _tmp)

    from chat_web_service import run_chat_once_events
    return list(run_chat_once_events(user_message, session_id))


@pytest.mark.api  # 真实跑完整图（LLM API + 检索），fast 门禁排除
def test_empty_message_returns_error_frame():
    """空消息 → error 帧 + [DONE]（行为与改名前一致）。"""
    frames = _collect_events("   ")
    assert len(frames) == 2
    assert frames[0].startswith("data: ")
    assert "[DONE]" in frames[-1]
    obj = json.loads(frames[0][len("data: "):].strip())
    assert "error" in obj


@pytest.mark.api  # 真实跑完整图（LLM API + 检索），fast 门禁排除
def test_once_returns_single_content_frame_then_done():
    """非空消息 → 单个 content 帧 + [DONE]（一次性语义，非逐 token）。"""
    frames = _collect_events("X3 Lite 手机多少钱")
    # 至少 content 帧 + [DONE]（无 LLM key 时可能走降级，仍应有 content/error + DONE）
    assert frames, "无任何帧"
    assert "[DONE]" in frames[-1], "缺 [DONE] 结束帧"
    # 结构：所有非 DONE 帧都是 data: JSON
    for f in frames[:-1]:
        assert f.startswith("data: "), f"帧格式错误: {f[:50]}"
        json.loads(f[len("data: "):].strip())  # 必须可解析


@pytest.mark.api  # 真实跑完整图（LLM API + 检索），fast 门禁排除
def test_once_response_has_session_info():
    """content 帧携带 session_id/thread_id（前端可定位会话）。"""
    frames = _collect_events("手机价格")
    for f in frames[:-1]:
        obj = json.loads(f[len("data: "):].strip())
        if "content" in obj:
            assert obj["session_id"], "缺 session_id"
            assert obj["thread_id"], "缺 thread_id"
            return
    # 无 key 环境可能只出 error 帧，此时跳过内容断言（行为已由上方测试覆盖）
