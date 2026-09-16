"""
会话管理器
使用 LangChain Core 的 BaseChatMessageHistory 实现会话与会话存储
支持多种存储后端和统一的 API 接口
"""

import os
import threading
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import logging

logger = logging.getLogger(__name__)

# 注（2026-08-11 fork 适配）：原实现依赖 langchain-community 的多后端（redis/mongodb/postgres/file），
# 该包已官方停止维护（sunset），且 0.4.x 已移除相关导出。MVP 仅使用 memory 后端，其余后端移除；
# 如需持久化后端，后续按官方迁移指南接入独立包（如 langchain-redis）。

# V13（2026-08-14）：加锁 + 惰性清理 + 移除双写
# - 加锁：sessions dict 全部读写由 _lock 保护（RLock 支持 get_session→create_session 嵌套），
#   add_message 的 message_count 计数原子化（修复多线程丢计数）
# - 惰性清理：memory 后端进程重启即清空，"启动时清理"无意义；改在 add_message 每
#   _CLEANUP_INTERVAL 次调用时触发 cleanup_old_sessions（事件驱动，非持续评估）
# - 双写移除：web 会话渲染/工单/列表全部走 checkpointer persisted_dialogue（V11/V12 确认），
#   session_manager 仅为 base_agent._get_conversation_context 的 state 缺失兜底，不再双写消息
_CLEANUP_INTERVAL = int(os.environ.get("SESSION_CLEANUP_INTERVAL", "100"))


class LangChainSessionManager:
    """基于 LangChain Core BaseChatMessageHistory 的会话管理器
    提供统一的会话管理接口，支持多种存储后端
    """

    def __init__(self, storage_backend: str = "memory", **storage_config):
        """初始化会话管理器

        Args:
            storage_backend: 存储后端类型 ("memory")
            **storage_config: 存储后端配置参数
        """
        self.storage_backend = storage_backend
        self.storage_config = storage_config
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # V13：全方法加锁（RLock 支持嵌套调用）
        self._write_count = 0  # V13：惰性清理计数器

        # 验证存储后端配置
        self._validate_storage_config()

        logger.info(f"会话管理器初始化完成，使用 {storage_backend} 后端")

    def _validate_storage_config(self):
        """验证存储后端配置（fork 适配：MVP 仅支持 memory 后端）"""
        if self.storage_backend != "memory":
            raise ValueError(f"MVP 仅支持 memory 后端，当前: {self.storage_backend}")

    def _create_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        根据存储后端创建对话历史（LangChain Core BaseChatMessageHistory）

        Args:
            session_id: 会话ID

        Returns:
            BaseChatMessageHistory 实例
        """
        if self.storage_backend == "memory":
            return InMemoryChatMessageHistory()

        raise ValueError(f"不支持的存储后端: {self.storage_backend}")

    def create_session(self, session_id: str = None) -> str:
        """
        创建新的会话

        Args:
            session_id: 会话ID，如果为None则自动生成

        Returns:
            会话ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        with self._lock:  # V13
            chat_history = self._create_chat_history(session_id)

            # 记录会话元数据
            self.sessions[session_id] = {
                "memory": chat_history,
                "created_at": time.time(),
                "last_activity": time.time(),
                "message_count": 0,
                "storage_backend": self.storage_backend
            }

        logger.info(f"创建新会话: {session_id} (backend={self.storage_backend})")

        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典
        """
        with self._lock:  # V13：隐式创建 + last_activity 更新原子化
            if session_id not in self.sessions:
                self.create_session(session_id)

            # 更新最后活动时间
            self.sessions[session_id]["last_activity"] = time.time()
            session = self.sessions[session_id]

        return session

    def get_memory(self, session_id: str) -> BaseChatMessageHistory:
        """
        获取会话的对话历史实例

        Args:
            session_id: 会话ID

        Returns:
            BaseChatMessageHistory 实例
        """
        session = self.get_session(session_id)

        with self._lock:  # V13：读 sessions 需在锁内
            if not isinstance(session, dict):
                self.create_session(session_id)
                session = self.sessions[session_id]

            return session["memory"]

    def add_message(self, session_id: str, message: str, is_user: bool = True):
        """
        添加消息到会话历史

        Args:
            session_id: 会话ID
            message: 消息内容
            is_user: 是否为用户消息
        """
        with self._lock:  # V13：计数自增原子化（修复多线程丢计数）
            session = self.get_session(session_id)

            if not isinstance(session, dict):
                logger.warning(f"session 非 dict: session_id={session_id} type={type(session)}")
                return

            memory = session["memory"]

            if is_user:
                memory.add_user_message(message)
            else:
                memory.add_ai_message(message)

            # 更新统计信息（锁内自增，多线程安全）
            session["message_count"] += 1
            session["last_activity"] = time.time()

            # V13：惰性清理（每 _CLEANUP_INTERVAL 次写入触发一次，事件驱动）
            self._write_count += 1
            if self._write_count >= _CLEANUP_INTERVAL:
                self._write_count = 0
                try:
                    cleaned = self.cleanup_old_sessions()
                    if cleaned:
                        logger.info(f"惰性清理过期会话 {cleaned} 个（写计数 {_CLEANUP_INTERVAL}）")
                except Exception as e:  # noqa: BLE001 清理失败不影响主流程
                    logger.warning(f"惰性清理失败: {e}")

        logger.debug(f"会话 {session_id} 写入 {'用户' if is_user else 'AI'} 消息")

    def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """
        获取对话历史（LangChain 标准格式）

        Args:
            session_id: 会话ID

        Returns:
            LangChain 消息列表
        """
        memory = self.get_memory(session_id)
        with self._lock:  # V13：读 memory.messages 在锁内（防并发 clear/写入读到半态）
            return memory.messages

    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        获取对话上下文（带时间戳的格式）

        Args:
            session_id: 会话ID
            max_messages: 最大消息数量

        Returns:
            带时间戳的消息列表
        """
        messages = self.get_conversation_history(session_id)

        # 转换为带时间戳的格式（V13：锁内做快照，防并发修改）
        with self._lock:
            formatted_messages = []
            for msg in messages[-max_messages:]:
                message_data = {
                    "content": msg.content,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "is_user": isinstance(msg, HumanMessage),
                    "message_type": msg.__class__.__name__
                }
                formatted_messages.append(message_data)

        return formatted_messages

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话详细信息

        Args:
            session_id: 会话ID

        Returns:
            会话信息字典
        """
        with self._lock:  # V13：检查+读取原子化（防检查后并发删除）
            if session_id not in self.sessions:
                return {}

            session = self.sessions[session_id]

            if not isinstance(session, dict):
                return {}

            memory = session["memory"]

            return {
                "session_id": session_id,
                "message_count": len(memory.messages),
                "created_at": session["created_at"],
                "last_activity": session["last_activity"],
                "storage_backend": session["storage_backend"],
                "memory_type": type(memory).__name__
            }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话信息

        Returns:
            会话信息列表
        """
        with self._lock:  # V13：快照遍历（防迭代时被并发修改）
            return [
                self.get_session_info(session_id)
                for session_id in list(self.sessions.keys())
            ]

    def clear_session(self, session_id: str):
        """
        清空会话内容

        Args:
            session_id: 会话ID
        """
        with self._lock:  # V13
            if session_id in self.sessions:
                memory = self.sessions[session_id]["memory"]
                memory.clear()

                # 重置统计信息
                self.sessions[session_id]["message_count"] = 0
                self.sessions[session_id]["last_activity"] = time.time()

                logger.debug(f"清空会话: {session_id}")

    def delete_session(self, session_id: str):
        """
        删除会话

        Args:
            session_id: 会话ID
        """
        with self._lock:  # V13
            if session_id in self.sessions:
                # 清空 Memory
                memory = self.sessions[session_id]["memory"]
                memory.clear()

                # 删除会话记录
                del self.sessions[session_id]

                logger.debug(f"删除会话: {session_id}")

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        清理过期会话

        Args:
            max_age_hours: 最大存活时间（小时）

        Returns:
            清理的会话数量
        """
        current_time = time.time()
        with self._lock:  # V13：遍历+删除原子化（防迭代时被并发修改）
            expired_sessions = []

            for session_id, session_data in self.sessions.items():
                age_hours = (current_time - session_data["last_activity"]) / 3600
                if age_hours > max_age_hours:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                self.delete_session(session_id)

            if expired_sessions:
                logger.info(f"清理过期会话 {len(expired_sessions)} 个")

            return len(expired_sessions)

    def get_conversation_summary(self, session_id: str) -> str:
        """
        获取对话摘要

        Args:
            session_id: 会话ID

        Returns:
            对话摘要
        """
        messages = self.get_conversation_history(session_id)

        if not messages:
            return "暂无对话记录"

        user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

        summary = f"对话摘要 (会话ID: {session_id})\n"
        summary += f"总消息数: {len(messages)}\n"
        summary += f"用户消息: {len(user_messages)}\n"
        summary += f"AI回复: {len(ai_messages)}\n"

        if user_messages:
            summary += f"最新用户消息: {user_messages[-1][:100]}...\n"

        return summary

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """
        导出会话数据

        Args:
            session_id: 会话ID

        Returns:
            会话数据字典
        """
        if session_id not in self.sessions:
            return {}

        with self._lock:  # V13
            session = self.sessions[session_id]
            memory = session["memory"]

            return {
                "session_info": self.get_session_info(session_id),
                "messages": [
                    {
                        "content": msg.content,
                        "type": msg.__class__.__name__,
                        "timestamp": datetime.now().isoformat()
                    }
                    for msg in memory.messages
                ]
            }

# 创建默认实例（使用内存存储）
default_session_manager = LangChainSessionManager()

# 便捷函数
def create_session(session_id: str = None) -> str:
    """创建新会话的便捷函数"""
    return default_session_manager.create_session(session_id)

def get_session(session_id: str) -> Dict[str, Any]:
    """获取会话信息的便捷函数"""
    return default_session_manager.get_session(session_id)

def add_message(session_id: str, message: str, is_user: bool = True):
    """添加消息的便捷函数"""
    default_session_manager.add_message(session_id, message, is_user)

def get_conversation_context(session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
    """获取对话上下文的便捷函数"""
    return default_session_manager.get_conversation_context(session_id, max_messages)

def list_sessions() -> List[Dict[str, Any]]:
    """列出所有会话的便捷函数"""
    return default_session_manager.list_sessions()

def clear_session(session_id: str):
    """清空会话的便捷函数"""
    default_session_manager.clear_session(session_id)

def delete_session(session_id: str):
    """删除会话的便捷函数"""
    default_session_manager.delete_session(session_id)
