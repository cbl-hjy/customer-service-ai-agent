"""
回复缓存（harness 层，2026-08-12 生产优化①）

客服场景重复问题占比高（"怎么退货"被问 100 次）——相同问题直接返回缓存回复，
不调 LLM：①吞吐大幅提升（重复请求秒回）②降本（省 token）③延迟稳定。

理念（§3.5）：缓存是 harness 加固（不改 agent 行为、不约束思考）；
缓存键=规范化问题文本，命中返回上次回复（含升级决策），未命中走完整图。

⚠️ 定位声明（V7 修复，2026-08-14）：本组件**仅供压测/评估使用，产品路径
（web/CLI 实际请求）未接入、也不应接入**。判据：
  1. 多轮上下文语义依赖：同一句文本在不同轮次语义不同（"多少钱"首轮问 A 产品、
     后续轮问 B 产品），按文本精确匹配复用旧回复会产生答非所问；
  2. 升级是状态性回复：缓存含 escalated 决策，跨用户/跨轮次复用会错误扩散升级状态
     （KB 已更新或上下文已变化时仍返回旧升级结论）；
  3. 收益无支撑：demo 场景重复问题占比无真实数据，KB 仅 56 条、BM25 检索微秒级，
     缓存省下的时间 < 状态一致性维护成本（省<花判据，§18）。
如需产品化：需改为 (thread_id, 上下文指纹, KB 版本) 多维键 + 仅缓存非状态回复。
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional


class ReplyCache:
    """LRU 回复缓存（线程安全）。

    key: 规范化问题文本（小写 + 去空白）；value: 上次回复的关键字段。
    """

    def __init__(self, max_size: int = 200) -> None:
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._max = max_size
        self._lock = threading.Lock()

    def _normalize(self, query: str) -> str:
        """规范化：小写 + 去全部空白（中文空格无意义，"怎么 退货"=“怎么退货”）"""
        return "".join(query.lower().split())

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._normalize(query)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)  # LRU：命中移到末尾
                return self._data[key]
        return None

    def put(self, query: str, reply: Dict[str, Any]) -> None:
        key = self._normalize(query)
        with self._lock:
            self._data[key] = reply
            if len(self._data) > self._max:
                self._data.popitem(last=False)  # 淘汰最久未用

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._data)


def cached_invoke(app, query: str, thread_id: str, cache: ReplyCache) -> Dict[str, Any]:
    """带缓存的图调用：命中直接返回缓存回复（不调 LLM），未命中走完整图并回填缓存。

    返回字典含 ``cached`` 标记（压测统计命中率用）。
    """
    hit = cache.get(query)
    if hit is not None:
        return {"cached": True, "customer_query": query, **hit}
    r = app.invoke(
        {"customer_query": query},
        {"configurable": {"thread_id": thread_id}},
    )
    reply = {
        "response": r["response"],
        "query_type": r["query_type"],
        "escalated": r["escalated"],
    }
    cache.put(query, reply)
    return {"cached": False, "customer_query": query, **r}
