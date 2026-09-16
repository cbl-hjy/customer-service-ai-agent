"""checkpointer：SqliteSaver 文件持久化（跨进程/重启保留 persisted_dialogue）。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
LANGGRAPH_STRICT_MSGPACK 安全开关必须在 import SqliteSaver 之前设置（fail-safe：
限制 checkpoint 反序列化为安全类型，防止数据库被篡改时执行任意代码——
负向约束在 harness 层的体现）。
"""

import os

# fail-safe（langgraph-checkpoint-sqlite 官方推荐）：限制 checkpoint 反序列化为安全类型，
# 防止数据库被篡改时执行任意代码——负向约束在 harness 层的体现
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

# checkpointer 文件持久化目录（fork 适配：SqliteSaver 在 langgraph 1.2.x 拆为独立包
# langgraph-checkpoint-sqlite，from_conn_string 接收裸文件路径，源码实测 sqlite/__init__.py:99）
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_checkpointer = None
_checkpointer_cm = None


def get_checkpointer():
    """创建/复用 SqliteSaver（进程级单例，保持连接生命周期=进程生命周期）。

    隔离支持：可通过环境变量 CHECKPOINT_DB_PATH 覆盖 DB 文件路径（加法，默认不变）。
    评估/多进程等 harness 用独立 DB，避免污染生产 checkpoints.db（线上默认路径行为完全不变）。
    """
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        db_path = os.environ.get("CHECKPOINT_DB_PATH")
        if db_path:
            db_dir = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(db_dir, exist_ok=True)
        else:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            db_path = os.path.join(CHECKPOINT_DIR, "checkpoints.db")
        _checkpointer_cm = SqliteSaver.from_conn_string(db_path)
        _checkpointer = _checkpointer_cm.__enter__()
    return _checkpointer
