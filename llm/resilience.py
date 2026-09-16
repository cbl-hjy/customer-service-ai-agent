"""LLM 通道韧性层：token 计量 + 熔断器 + 并发门控 + thinking-only 快照清单。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
模块边界：本层只负责"通道健康与计量"，不含业务节点与图组装。
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional

from config import (
    MAX_CONCURRENCY,
    CIRCUIT_BREAKER_FAIL_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT,
    CIRCUIT_BREAKER_RECOVERY_REQUESTS,
)

logger = logging.getLogger(__name__)

# 全局 API 并发信号量（2026-08-12 生产优化③，前沿做法）
# 限流必须坐在 agent 调用层全局生效（per-tool 限流会被并行绕过——多源共识）；
# 实测 10 并发 API 排队放大延迟（P50 39.8s > 串行 29.3s），默认 5 并发防排队
_API_SEMAPHORE = threading.Semaphore(MAX_CONCURRENCY)

# thinking-only 快照清单（2026-08-16/17）：这些模型服务端强制 enable_thinking=True，
# 传 False 直接 400 InvalidError.Algo.InvalidParameter。切换模型时须核对快照说明。
_THINKING_ONLY_SNAPSHOTS = (
    "qwen3.7-max-2026-05-17",
    "qwen3.7-max-preview",
)


# =============================================================================
# LLM token 用量观测（环外只读，2026-08-13 方向B：延迟/成本基线）
# 纯观测计数器：不改任何 agent 行为，供评估层读取并沉淀为跨版本成本基线。
# =============================================================================
_TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
_TOKEN_LOCK = threading.Lock()


def get_token_usage() -> dict:
    """环外只读：返回 LLM token 累计用量（进程级）。"""
    with _TOKEN_LOCK:
        return dict(_TOKEN_USAGE)


def reset_token_usage() -> None:
    """环外只读：评估进程跑前清零，保证单次运行可归因。"""
    with _TOKEN_LOCK:
        for k in _TOKEN_USAGE:
            _TOKEN_USAGE[k] = 0


def _record_token_usage(usage: Optional[dict]) -> None:
    """环外只读：累加单次 API 响应的 token 用量（usage 缺失不报错）。"""
    if not usage:
        return
    with _TOKEN_LOCK:
        _TOKEN_USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        _TOKEN_USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        _TOKEN_USAGE["total_tokens"] += int(usage.get("total_tokens") or 0)
        _TOKEN_USAGE["calls"] += 1


class CircuitBreaker:
    """熔断器（标准三状态 CLOSED / OPEN / HALF_OPEN）：
    - CLOSED: 正常，失败计数
    - OPEN: 熔断，快速失败（不发请求），保护 API 额度
    - HALF_OPEN: 冷却后，放行探测请求看是否恢复
    遵循生产级熔断器模式，与本项目并发限流光合。
    """
    def __init__(self, fail_threshold: int, timeout: float, recovery_requests: int):
        self.fail_threshold = fail_threshold  # 连续失败次数阈值
        self.timeout = timeout               # 熔断冷却时间（秒）
        self.recovery_requests = recovery_requests  # 半开探测请求数
        self._lock = threading.Lock()

        # 状态：CLOSED / OPEN / HALF_OPEN
        self._state = "CLOSED"
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0

    def allow_request(self) -> bool:
        """是否允许发起请求（熔断 OPEN 时返回 False，快速失败）"""
        with self._lock:
            now = time.time()
            if self._state == "OPEN":
                if now - self._last_failure_time >= self.timeout:
                    # 冷却时间到 → 进入半开探测
                    self._state = "HALF_OPEN"
                    self._success_count = 0
                    return True
                # 仍在冷却 → 不允许
                return False
            # CLOSED / HALF_OPEN → 允许
            return True

    def record_success(self) -> None:
        """记录成功：CLOSED 清计数；HALF_OPEN 成功达到阈值 → CLOSED"""
        with self._lock:
            if self._state == "CLOSED":
                self._failure_count = 0
            elif self._state == "HALF_OPEN":
                self._success_count += 1
                if self._success_count >= self.recovery_requests:
                    # 探测成功 → 恢复正常
                    self._state = "CLOSED"
                    self._failure_count = 0
                    logger.info("熔断器恢复 CLOSED，LLM 通道可用")

    def record_failure(self, status_code: Optional[int] = None) -> None:
        """记录失败：CLOSED 计数 → 达到阈值 → OPEN；HALF_OPEN 失败 → OPEN"""
        with self._lock:
            now = time.time()
            self._last_failure_time = now
            if self._state == "CLOSED":
                self._failure_count += 1
                if self._failure_count >= self.fail_threshold:
                    # 连续失败达到阈值 → 熔断打开
                    self._state = "OPEN"
                    if status_code:
                        logger.warning(f"熔断器熔断 OPEN（{self._failure_count} 次失败，status={status_code}），等待 {self.timeout}s 冷却")
                    else:
                        logger.warning(f"熔断器熔断 OPEN（{self._failure_count} 次失败），等待 {self.timeout}s 冷却")
            elif self._state == "HALF_OPEN":
                # 探测失败 → 回到打开熔断
                self._state = "OPEN"
                logger.warning(f"半开探测失败，回到熔断 OPEN，等待 {self.timeout}s 冷却")


# 全局熔断器（单例，整个服务共享，保护整体 API 额度）
_circuit_breaker = CircuitBreaker(
    fail_threshold=CIRCUIT_BREAKER_FAIL_THRESHOLD,
    timeout=CIRCUIT_BREAKER_TIMEOUT,
    recovery_requests=CIRCUIT_BREAKER_RECOVERY_REQUESTS,
)


# =============================================================================
# 通道状态观测（第二波可观测性，2026-09-16）：只读 getter + 在途请求计量。
# 供 metrics.py 聚合四大黄金信号之 saturation；请求路径行为零变化
# （llm_slot 计数器是纯 ±1 观测，不改变信号量语义）。
# =============================================================================
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_LLM_REQUESTS = 0


@contextmanager
def llm_slot():
    """全局并发槽：信号量门控 + 在途请求计量 ±1。

    替代裸 `with _API_SEMAPHORE:`（llm/client.py 两处调用点）——语义完全一致，
    额外在进入/退出时维护在途计数，供 metrics 计算饱和度（inflight / max）。
    """
    global _INFLIGHT_LLM_REQUESTS
    with _API_SEMAPHORE:
        with _INFLIGHT_LOCK:
            _INFLIGHT_LLM_REQUESTS += 1
        try:
            yield
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT_LLM_REQUESTS -= 1


def get_channel_status() -> dict:
    """环外只读：LLM 通道健康快照（熔断状态 + 并发饱和度）。"""
    with _circuit_breaker._lock:
        state = _circuit_breaker._state
        failure_count = _circuit_breaker._failure_count
    with _INFLIGHT_LOCK:
        inflight = _INFLIGHT_LLM_REQUESTS
    return {
        "circuit_state": state,
        "circuit_failure_count": failure_count,
        "llm_requests_inflight": inflight,
        "max_concurrency": MAX_CONCURRENCY,
    }
