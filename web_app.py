#!/usr/bin/env python3
"""
多智能体客服系统 - Web 入口
基于 LangGraph API 接口，路由与 Flask 会话；业务逻辑见 chat_web_service.py
"""

import os
import time
from typing import Dict, Any, List

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify, session, Response

import logging

logger = logging.getLogger(__name__)

from chat_web_service import (
    run_chat_sync,
    run_chat_once_events,
    run_chat_stream_events,
    fetch_sessions_list,
    fetch_session_detail,
    fetch_escalated_tickets,
    fetch_trace_runs,
    fetch_trace_detail,
    clear_thread_and_create_new,
    langgraph_connectivity_test,
    get_current_thread_id,
)

import tickets_store  # V10：工单闭环状态流转

# 导入配置（与历史行为保持一致）
from config import *  # noqa: E402,F401,F403

app = Flask(__name__)

# Flask 配置
# V2 修复（2026-08-14）：默认密钥 = 会话 cookie 可伪造。生产环境必须显式配置，
# 缺失时启动即失败，不静默兜底。
_FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")
if not _FLASK_SECRET_KEY:
    raise RuntimeError(
        "FLASK_SECRET_KEY 未设置：生产环境必须配置随机密钥，"
        "可用 `python -c \"import secrets;print(secrets.token_hex(32))\"` 生成"
    )
app.secret_key = _FLASK_SECRET_KEY
app.config['SESSION_TYPE'] = 'filesystem'


# --- Flask session 内的本地对话占位（主页模板可能使用）---

def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    if 'conversations' not in session:
        session['conversations'] = {}
    return session['conversations'].get(session_id, [])


def add_conversation_message(session_id: str, role: str, content: str) -> None:
    history = get_conversation_history(session_id)
    history.append({
        'role': role,
        'content': content,
    })
    session['conversations'][session_id] = history


@app.route('/')
def index():
    """主页"""
    current_session_id = session.get('current_session_id', 'default')
    conversation_history = get_conversation_history(current_session_id)
    return render_template('index.html', conversation_history=conversation_history)


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = (data.get('message') or '').strip()
        client_session_id = data.get('session_id', 'default')

        ai_text, err_msg, http_code = run_chat_sync(user_message, client_session_id)
        if err_msg:
            return jsonify({'error': err_msg}), http_code or 500

        tid = get_current_thread_id()
        return jsonify({
            'response': ai_text,
            'session_id': tid,
            'thread_id': tid,
        })
    except Exception as e:
        logger.exception("聊天处理失败")
        return jsonify({'error': f'内部错误: {str(e)}'}), 500


@app.route('/api/chat/once', methods=['POST'])
def chat_once():
    """一次性聊天返回（V14 改名：原 /api/chat/stream 是假流式——同步 invoke 后整体吐 SSE，
    命名误导；改名 /api/chat/once 消除歧义。前端实际走 /api/chat 同步路径，本接口未消费）"""
    try:
        data = request.get_json()
        user_message = (data.get('message') or '').strip()
        client_session_id = data.get('session_id', 'default')

        return Response(
            run_chat_once_events(user_message, client_session_id),
            mimetype='text/event-stream'
        )

    except Exception as e:
        logger.exception("一次性聊天返回失败")
        return jsonify({'error': f'内部错误: {str(e)}'}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """真流式聊天（P1，2026-08-21）：token 流 + 阶段流 SSE。

    V14 曾因假流式把本路由改名 /api/chat/once；本路由为真逐 token 流式
    （worker 线程跑图 + thread-local sink），路由名回归本义。
    帧协议见 chat_web_service.run_chat_stream_events docstring。
    """
    try:
        data = request.get_json()
        user_message = (data.get('message') or '').strip()
        client_session_id = data.get('session_id', 'default')

        return Response(
            run_chat_stream_events(user_message, client_session_id),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # nginx 反代不缓冲（否则 SSE 退化为整段到达）
            },
        )

    except Exception as e:
        logger.exception("流式聊天失败")
        return jsonify({'error': f'内部错误: {str(e)}'}), 500


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """获取会话列表（V11：支持 ?limit=&offset= 分页）"""
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', default=0, type=int)
    sessions, err = fetch_sessions_list(limit=limit, offset=offset)
    if err:
        return jsonify({'error': err}), 500
    return jsonify({'sessions': sessions or []})


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """获取特定会话详情"""
    session_data, err = fetch_session_detail(session_id)
    if err:
        return jsonify({'error': err}), 500
    return jsonify({'session': session_data})


@app.route('/tickets')
def tickets_page():
    """待人工处理工单队列页（V10，只读，复用 checkpointer 数据）。"""
    tickets, err = fetch_escalated_tickets()
    if err:
        return f"获取工单队列失败: {err}", 500
    return render_template('tickets.html', tickets=tickets or [])


@app.route('/trace')
def trace_page():
    """全链路追踪列表页（V18：最近工单决策链汇总）。"""
    runs, err = fetch_trace_runs(limit=50)
    if err:
        return f"获取 trace 列表失败: {err}", 500
    return render_template('trace.html', runs=runs or [])


@app.route('/trace/<run_id>')
def trace_detail_page(run_id):
    """全链路追踪详情页（V18：单次工单完整决策链）。"""
    run, err = fetch_trace_detail(run_id)
    if err:
        return f"获取 trace 详情失败: {err}", 404
    return render_template('trace_detail.html', run=run)


@app.route('/api/tickets', methods=['GET'])
def get_tickets():
    """待人工处理工单队列 API（JSON，V11：支持 ?limit=&offset= 分页）。"""
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', default=0, type=int)
    tickets, err = fetch_escalated_tickets(limit=limit, offset=offset)
    if err:
        return jsonify({'error': err}), 500
    return jsonify({'tickets': tickets or []})


# --- V10：工单闭环状态流转（人工侧操作，独立于 agent 状态机） ---

@app.route('/api/tickets/<thread_id>/claim', methods=['POST'])
def ticket_claim(thread_id):
    """认领工单：pending → processing。"""
    ok, status, msg = tickets_store.claim_ticket(thread_id)
    return jsonify({'ok': ok, 'status': status, 'message': msg}), 200 if ok else 409


@app.route('/api/tickets/<thread_id>/resolve', methods=['POST'])
def ticket_resolve(thread_id):
    """标记解决：pending/processing → resolved。"""
    ok, status, msg = tickets_store.resolve_ticket(thread_id)
    return jsonify({'ok': ok, 'status': status, 'message': msg}), 200 if ok else 409


@app.route('/api/tickets/<thread_id>/reopen', methods=['POST'])
def ticket_reopen(thread_id):
    """重开工单：resolved → pending。"""
    ok, status, msg = tickets_store.reopen_ticket(thread_id)
    return jsonify({'ok': ok, 'status': status, 'message': msg}), 200 if ok else 409


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """删除会话"""
    try:
        ok, status = delete_remote_thread(session_id)
        if ok:
            if 'conversations' in session and session_id in session['conversations']:
                del session['conversations'][session_id]
            return jsonify({'message': '会话删除成功'})
        return jsonify({'error': f'删除会话失败: {status}'}), 500
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/api/sessions/<session_id>/clear', methods=['POST'])
def clear_session(session_id):
    """清空会话"""
    try:
        new_thread_id, err = clear_thread_and_create_new(session_id)
        if err:
            return jsonify({'error': err}), 500

        if 'conversations' in session and session_id in session['conversations']:
            session['conversations'][session_id] = []

        return jsonify({
            'message': '会话清空成功',
            'new_thread_id': new_thread_id
        })
    except Exception as e:
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500


@app.route('/api/new_session', methods=['POST'])
def create_new_session():
    """创建新会话（Flask session 侧）"""
    try:
        import uuid
        new_session_id = str(uuid.uuid4())
        session['current_session_id'] = new_session_id
        if 'conversations' not in session:
            session['conversations'] = {}
        session['conversations'][new_session_id] = []
        return jsonify({
            'session_id': new_session_id,
            'message': '新会话创建成功'
        })
    except Exception as e:
        return jsonify({'error': f'创建会话失败: {str(e)}'}), 500


@app.route('/api/health')
def health_check():
    """健康检查 + 预热就绪状态（第二波：warmup ∈ warming/ready/failed，readiness 探针消费）。"""
    from warmup import get_warmup_status
    return jsonify({
        'status': 'healthy',
        'warmup': get_warmup_status(),
        'timestamp': time.time()
    })


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """运行时指标聚合（第二波可观测性）：四大黄金信号 + 升级率/token成本，只读 trace.db。

    ?window= 分钟数（默认 METRICS_WINDOW_MIN=60）。响应含 alerts 数组（超阈 WARN）。
    """
    import metrics
    window = request.args.get('window', default=METRICS_WINDOW_MIN, type=int)
    return jsonify(metrics.collect(window))


@app.route('/api/test')
def test_langgraph():
    """测试 LangGraph API 调用"""
    result, err = langgraph_connectivity_test()
    if err:
        return jsonify({'error': err}), 500
    return jsonify(result)


def main():
    """主函数"""
    logger.info("多智能体客服系统 Web 应用")
    logger.info("=" * 60)
    logger.info("启动 Web 服务...")
    logger.info("访问地址: http://localhost:5000")
    logger.info("按 Ctrl+C 停止服务")

    # 第二波（2026-09-15）：启动预热——后台线程加载本地模型链（检索/图），
    # 消除冷启动首 token 惩罚（24.2s→稳态 4.8s）；失败回退懒加载不阻断启动。
    if WARMUP_ON_START:
        from warmup import start_warmup
        start_warmup()

    # V1 修复（2026-08-14）：debug 必须显式开启（env），默认关闭。
    # Werkzeug debugger 允许任意代码执行，生产环境严禁 debug=True。
    # 生产部署建议用 wsgi 服务器（gunicorn/waitress）而非 app.run()。
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)


if __name__ == "__main__":
    main()
