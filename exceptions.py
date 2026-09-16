"""
LLM 通道异常类型（独立模块，避免 tool 与 multi_agent 的循环导入）。
"""

from typing import Optional


class LLMServiceUnavailable(Exception):
    """LLM 通道不可用（额度耗尽/403/5xx 等），需要熔断保护。
    与"分类不确定"明确区分：前者是系统故障，后者是用户问题难分类。
    """
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)