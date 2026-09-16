"""V10 工单状态流转 API 测试（2026-08-14）：claim/resolve/reopen 路由。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import tickets_store


@pytest.fixture
def client(monkeypatch, tmp_path):
    """独立临时 DB + 有效 secret_key 下启动 web_app test client。"""
    monkeypatch.setenv("TICKETS_DB_PATH", str(tmp_path / "api_tickets.db"))
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret-v10")
    import web_app
    web_app.app.config["TESTING"] = True
    return web_app.app.test_client()


def test_claim_api(client):
    tickets_store.ensure_ticket("api-1")
    r = client.post("/api/tickets/api-1/claim")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["status"] == tickets_store.STATUS_PROCESSING


def test_resolve_api(client):
    tickets_store.ensure_ticket("api-1")
    r = client.post("/api/tickets/api-1/resolve")
    assert r.status_code == 200
    assert r.get_json()["status"] == tickets_store.STATUS_RESOLVED


def test_reopen_api(client):
    tickets_store.ensure_ticket("api-1")
    client.post("/api/tickets/api-1/resolve")
    r = client.post("/api/tickets/api-1/reopen")
    assert r.status_code == 200
    assert r.get_json()["status"] == tickets_store.STATUS_PENDING


def test_claim_unknown_ticket_409(client):
    r = client.post("/api/tickets/ghost-1/claim")
    assert r.status_code == 409
    assert "物化" in r.get_json()["message"]


def test_resolve_resolved_409(client):
    tickets_store.ensure_ticket("api-1")
    client.post("/api/tickets/api-1/resolve")
    r = client.post("/api/tickets/api-1/resolve")
    assert r.status_code == 409
