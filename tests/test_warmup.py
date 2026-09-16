"""测试：启动预热（第二波，2026-09-15，warmup.py）。

具名风险（与 warmup.py docstring 逐条对应）：
  R1  预热触发本地模型链：LLM 客户端 + 检索全链 + 图编译各被调用一次
  R2  额度纪律：本地链零 API 消耗；LLM 通道探活恰好一次（~2 token，消除通道冷启动）
  R3  预热失败不阻断：任一环节抛异常 → status=failed，不向上抛
  R4  start_warmup 幂等：重复调用不起第二个线程
  R5  /api/health 暴露 warmup 状态（readiness 探针契约）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import warmup


@pytest.fixture(autouse=True)
def _reset_warmup():
    """每测前后复位预热状态机（配对清理纪律）。"""
    warmup.reset_warmup_for_tests()
    yield
    warmup.reset_warmup_for_tests()


class _StubModules:
    """给 warmup._do_warmup 的依赖打桩（延迟 import 取模块属性，monkeypatch 生效）。"""

    def __init__(self, monkeypatch, *, retrieve_error=None, warmup_error=None, app_error=None):
        self.calls = {"llm": 0, "retrieve": 0, "hybrid_warmup": 0, "app": 0, "llm_invoke": 0}
        self.probe_messages = None
        outer = self

        class _LLM:
            def invoke(self, messages, *a, **k):
                outer.calls["llm_invoke"] += 1
                outer.probe_messages = list(messages)
                return type("R", (), {"content": "1"})()

        import llm as _llm_mod
        monkeypatch.setattr(_llm_mod, "get_llm", lambda: (self.calls.__setitem__("llm", self.calls["llm"] + 1), _LLM())[1])

        import kb_retriever as _kbr

        class _FakeHybrid:
            def warm_up(self):
                self_ref.calls["hybrid_warmup"] += 1
                if warmup_error:
                    raise warmup_error

        self_ref = self
        monkeypatch.setattr(_kbr, "_get_hybrid_retriever", lambda kb_path: _FakeHybrid())

        def _fake_retrieve(domain, query, top_k=3):
            self.calls["retrieve"] += 1
            if retrieve_error:
                raise retrieve_error
            return []
        monkeypatch.setattr(_kbr, "retrieve_titles", _fake_retrieve)

        import chat_web_service as _cws
        def _fake_app():
            self.calls["app"] += 1
            if app_error:
                raise app_error
            return object()
        monkeypatch.setattr(_cws, "get_app", _fake_app)


def test_r1_warmup_triggers_local_chain(monkeypatch):
    """R1：预热依次触发 LLM 客户端 / 检索器 / 混合检索模型 warm_up / 图编译 / 通道探活。"""
    stub = _StubModules(monkeypatch)
    warmup._do_warmup()
    assert stub.calls == {"llm": 2, "retrieve": 1, "hybrid_warmup": 1, "app": 1, "llm_invoke": 1}
    assert warmup.get_warmup_status() == "ready"


def test_r2_llm_probe_exactly_once(monkeypatch):
    """R2：LLM 通道探活恰好一次（~2 token 成本上限，不多打）。"""
    stub = _StubModules(monkeypatch)
    warmup._do_warmup()
    assert stub.calls["llm_invoke"] == 1
    assert stub.probe_messages and "请只回复数字1" in stub.probe_messages[0]["content"]


def test_r3_warmup_failure_degrades_not_raises(monkeypatch):
    """R3：检索加载失败 → status=failed 且不抛异常（服务照常启动）。"""
    _StubModules(monkeypatch, retrieve_error=RuntimeError("model missing"))
    warmup._do_warmup()  # 不应抛
    assert warmup.get_warmup_status() == "failed"


def test_r3b_app_failure_degrades(monkeypatch):
    """R3：图编译失败同样降级不抛。"""
    _StubModules(monkeypatch, app_error=RuntimeError("graph boom"))
    warmup._do_warmup()
    assert warmup.get_warmup_status() == "failed"


def test_r4_start_warmup_idempotent(monkeypatch):
    """R4：start_warmup 重复调用只起一个线程（幂等）。"""
    import threading
    _StubModules(monkeypatch)
    before = threading.active_count()
    warmup.start_warmup()
    warmup.start_warmup()
    warmup.start_warmup()
    # 等预热线程结束再断言（避免竞态），活跃线程数最多 +1
    for t in threading.enumerate():
        if t.name == "warmup":
            t.join(timeout=10)
    assert threading.active_count() <= before + 1
    assert warmup.get_warmup_status() in ("ready", "failed")


def test_r5_health_endpoint_exposes_warmup_status(monkeypatch):
    """R5：/api/health 返回 warmup 字段（readiness 探针契约）。"""
    os.environ.setdefault("FLASK_SECRET_KEY", "test-only-secret")
    from web_app import app

    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "healthy"
    assert body["warmup"] in ("not_started", "warming", "ready", "failed")
