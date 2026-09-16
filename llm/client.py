"""OpenAI 兼容 API 客户端（通义千问 dashscope compatible-mode）。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
模块边界：HTTP 调用 + 重试/退避 + 响应包装 + 延迟初始化单例。
"""

import logging
import json
import random
import time

import requests

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    HTTP_TIMEOUT,
    HTTP_MAX_RETRIES,
    HTTP_HEADERS,
    ENABLE_THINKING,
)
from exceptions import LLMServiceUnavailable
from .resilience import (
    _API_SEMAPHORE,
    _THINKING_ONLY_SNAPSHOTS,
    _circuit_breaker,
    _record_token_usage,
    llm_slot,
)

logger = logging.getLogger(__name__)


class CustomResponse:
    def __init__(self, content):
        self.content = content


# OpenAI兼容API客户端类
class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = HTTP_TIMEOUT
        self.max_retries = HTTP_MAX_RETRIES
        self.headers = HTTP_HEADERS.copy()
        self.headers["Authorization"] = f"Bearer {api_key}"

        # 添加LangChain回调管理器所需的属性
        self.parent_run_id = None
        self.run_id = None
        self.tags = []
        self.metadata = {}
        self.handlers = []
        self.callback_manager = None
        self.inheritable_handlers = []
        self.inheritable_tags = []
        self.inheritable_metadata = {}

    def _format_messages(self, messages) -> list:
        """LangChain 消息对象 → OpenAI 格式（供 invoke / invoke_stream 共用）。"""
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                # 处理LangChain消息对象
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        formatted_messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        formatted_messages.append({"role": "assistant", "content": msg.content})
                    elif msg.type == 'system':
                        # 系统消息转换为用户消息
                        formatted_messages.append({"role": "user", "content": f"System instruction: {msg.content}"})
                    else:
                        formatted_messages.append({"role": "user", "content": msg.content})
                else:
                    # 默认作为用户消息处理
                    formatted_messages.append({"role": "user", "content": msg.content})
            else:
                # 处理字符串或其他类型
                formatted_messages.append({"role": "user", "content": str(msg)})
        return formatted_messages

    def _build_payload(self, formatted_messages: list, response_format=None) -> dict:
        """构造请求 payload（思考模式开关 + response_format，供两条路径共用）。"""
        payload = {
            "model": self.model,
            "messages": formatted_messages
        }
        # 思考模式开关（按端点生态分派，2026-09-15 DeepSeek 切换）：
        # - DeepSeek：官方参数 thinking={"type": enabled/disabled}，服务端默认 enabled——
        #   不显式关闭则分类/常规回复空转思考（Qwen 风格 enable_thinking 被静默忽略，
        #   实测首 token 2.11s 即此因）。开关语义与 config.ENABLE_THINKING 一致。
        # - 百炼（历史生态）：enable_thinking；thinking-only 快照
        #   （qwen3.7-max-2026-05-17）服务端强制 True，传 False 直接 400。
        if "deepseek" in self.base_url:
            payload["thinking"] = {"type": "enabled" if ENABLE_THINKING else "disabled"}
        elif any(s in self.model for s in _THINKING_ONLY_SNAPSHOTS):
            payload["enable_thinking"] = True
        elif not ENABLE_THINKING:
            payload["enable_thinking"] = False
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def invoke(self, messages, response_format=None):
        """调用OpenAI兼容API（response_format 可选：通义千问 JSON 结构化输出）"""
        formatted_messages = self._format_messages(messages)
        payload = self._build_payload(formatted_messages, response_format)

        # 添加调试信息
        # V9：请求日志只记元信息，不记消息内容（消息含用户输入与系统提示词，属敏感信息）
        logger.debug(
            "API request: url=%s model=%s messages=%d roles=%s",
            f"{self.base_url}/chat/completions", self.model, len(formatted_messages),
            [m.get("role") for m in formatted_messages],
        )

        # 熔断器检查：OPEN 时快速失败，不发起网络请求（保护 API 额度）
        if not _circuit_breaker.allow_request():
            raise LLMServiceUnavailable(
                "LLM 通道熔断（并发/额度保护），请稍候重试",
                status_code=None,
            )

        # 重试机制（2026-08-12 增强：信号量限并发 + 429 Retry-After 契约 + jitter 防重试风暴）
        # 2026-08-13 增强：错误分类——4xx 永久故障（401/403/400）不重试，直接抛 LLMServiceUnavailable
        for attempt in range(self.max_retries):
            try:
                with llm_slot():  # 全局并发门控：同时最多 MAX_CONCURRENCY 个在途 LLM 请求（含在途计量）
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout
                    )

                    # 先看状态码：永久故障（4xx 非 429）不重试，直接熔断上报
                    if response.status_code >= 400:
                        status = response.status_code
                        if status == 429:
                            # 429 限流是契约，仍走重试（读 Retry-After）
                            pass
                        elif 400 <= status < 500:
                            # 永久故障：401/403（额度/鉴权）/400 等，重试无用，直接抛
                            _circuit_breaker.record_failure(status)
                            raise LLMServiceUnavailable(
                                f"LLM 通道返回 {status}（{response.text[:200]}）",
                                status_code=status,
                            )
                        elif status >= 500:
                            # 5xx 瞬时故障，可重试
                            pass

                    response.raise_for_status()
                    result = response.json()

                    # 提取响应内容
                    if "choices" in result and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        content = message.get("content", "")
                        _record_token_usage(result.get("usage"))
                        _circuit_breaker.record_success()
                        return CustomResponse(content)
                    else:
                        _circuit_breaker.record_success()
                        return CustomResponse("API response format error")

            except LLMServiceUnavailable:
                # 永久故障（403/401 等）：重试无用，直接向上抛，不降级
                raise

            except requests.exceptions.RequestException as e:
                status = getattr(e.response, "status_code", None)
                logger.warning(f"API 调用第 {attempt + 1} 次失败: {e} (status={status})")

                # 4xx 永久故障（非 429）：重试无用，熔断 + 抛 LLMServiceUnavailable
                if status is not None and 400 <= status < 500 and status != 429:
                    _circuit_breaker.record_failure(status)
                    raise LLMServiceUnavailable(
                        f"LLM 通道返回 {status}（{str(e)[:200]}）",
                        status_code=status,
                    )

                # 瞬时故障重试耗尽（429/5xx/超时）→ 熔断上报 + 抛 LLMServiceUnavailable。
                # 必须先于 429 Retry-After 的 continue 判定：否则末次尝试走 continue 跳过
                # 耗尽检查、for 循环裸结束，invoke 静默返回 None（故障注入 FI-3 复现，2026-09-15）
                if attempt == self.max_retries - 1:
                    _circuit_breaker.record_failure(status)
                    raise LLMServiceUnavailable(f"API call failed: {e}", status_code=status)

                if status == 429 and getattr(e.response, "headers", None):
                    # 429 是契约不是故障：读 Retry-After header 精确等待（+30% jitter）
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = float(retry_after) + random.uniform(0, 0.3 * float(retry_after))
                        logger.info(f"429 限流，Retry-After {retry_after}s（含 jitter）等待 {wait:.1f}s")
                        time.sleep(wait)
                        continue
                # 指数退避 + jitter（±30% 随机，防多实例同时重试形成惊群）
                wait = 2 ** attempt + random.uniform(0, 0.3 * 2 ** attempt)
                time.sleep(wait)

    def invoke_stream(self, messages, response_format=None):
        """流式调用 OpenAI 兼容 API：逐 token yield（P1 真流式 SSE）。

        与 invoke 同一韧性契约（拆解见 invoke 内注释，此处只列流式差异）：
        - 重试仅发生在首 token 之前——已 yield 的内容无法撤回，中途失败直接抛；
        - 信号量在流消费期间持续持有（连接占用即并发占用）；
        - token 计量：stream_options.include_usage（末 chunk 携带 usage），
          服务端不返回 usage 则不记（不估算，防污染成本基线）。

        调用方约束：仅正文生成走本接口（llm.stream_sink.call_llm），
        分类/改写/拆解等结构化中间产物继续走 invoke（response_format=json）。
        """
        formatted_messages = self._format_messages(messages)
        payload = self._build_payload(formatted_messages, response_format)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        logger.debug(
            "API stream request: url=%s model=%s messages=%d roles=%s",
            f"{self.base_url}/chat/completions", self.model, len(formatted_messages),
            [m.get("role") for m in formatted_messages],
        )

        # 熔断器检查：OPEN 时快速失败，不发起网络请求（保护 API 额度）
        if not _circuit_breaker.allow_request():
            raise LLMServiceUnavailable(
                "LLM 通道熔断（并发/额度保护），请稍候重试",
                status_code=None,
            )

        for attempt in range(self.max_retries):
            yielded = False
            try:
                with llm_slot():
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout,
                        stream=True,
                    )
                    if response.status_code >= 400:
                        status = response.status_code
                        if status == 429 or status >= 500:
                            # 瞬时故障：走 raise → 重试分支（与 invoke 同路径）
                            response.raise_for_status()
                        # 4xx 永久故障（401/403/400 等）：重试无用，熔断 + 直接抛
                        _circuit_breaker.record_failure(status)
                        raise LLMServiceUnavailable(
                            f"LLM 通道返回 {status}（{response.text[:200]}）",
                            status_code=status,
                        )

                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = line.decode("utf-8", errors="replace").strip()
                        if not data.startswith("data:"):
                            continue  # SSE 注释行/事件前缀外的噪声
                        data = data[len("data:"):].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue  # 半包/坏行不终止流
                        usage = chunk.get("usage")
                        if usage:
                            _record_token_usage(usage)
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue  # 末 chunk（纯 usage）无 choices
                        delta = (choices[0] or {}).get("delta") or {}
                        token = delta.get("content") or ""
                        if token:
                            yielded = True
                            yield token

                    # 流正常读完：熔断记成功（token 已按 usage chunk 计量）
                    _circuit_breaker.record_success()
                    return

            except LLMServiceUnavailable:
                # 永久故障（403/401 等）：重试无用，直接向上抛，不降级
                raise

            except requests.exceptions.RequestException as e:
                status = getattr(e.response, "status_code", None)
                if yielded:
                    # 首 token 之后失败：内容已交付给上层（sink 已推送），
                    # 重试会造成内容重复——直接熔断上报 + 抛
                    _circuit_breaker.record_failure(status)
                    raise LLMServiceUnavailable(f"API stream failed: {e}", status_code=status)

                logger.warning(f"API 流式调用第 {attempt + 1} 次失败: {e} (status={status})")

                # 4xx 永久故障（非 429）：重试无用，熔断 + 抛 LLMServiceUnavailable
                if status is not None and 400 <= status < 500 and status != 429:
                    _circuit_breaker.record_failure(status)
                    raise LLMServiceUnavailable(
                        f"LLM 通道返回 {status}（{str(e)[:200]}）",
                        status_code=status,
                    )

                # 瞬时故障重试耗尽（429/5xx/超时）→ 熔断上报 + 抛 LLMServiceUnavailable。
                # 必须先于 429 Retry-After 的 continue 判定（同 invoke，FI-3s 复现，2026-09-15）
                if attempt == self.max_retries - 1:
                    _circuit_breaker.record_failure(status)
                    raise LLMServiceUnavailable(f"API stream failed: {e}", status_code=status)

                if status == 429 and getattr(e.response, "headers", None):
                    # 429 是契约不是故障：读 Retry-After header 精确等待（+30% jitter）
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = float(retry_after) + random.uniform(0, 0.3 * float(retry_after))
                        logger.info(f"429 限流，Retry-After {retry_after}s（含 jitter）等待 {wait:.1f}s")
                        time.sleep(wait)
                        continue
                # 指数退避 + jitter（±30% 随机，防多实例同时重试形成惊群）
                wait = 2 ** attempt + random.uniform(0, 0.3 * 2 ** attempt)
                time.sleep(wait)

    def chat(self, messages):
        """兼容LangChain的chat方法"""
        return self.invoke(messages)

    # 添加LangChain回调管理器接口
    def bind(self, **kwargs):
        """绑定参数到客户端"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def with_config(self, config):
        """设置配置"""
        if hasattr(config, 'get'):
            for key, value in config.items():
                setattr(self, key, value)
        return self


# 延迟初始化LLM
_llm_instance = None


def initialize_llm_client():
    """初始化OpenAI兼容API客户端"""
    if not OPENAI_API_KEY:
        raise ValueError("API密钥未设置")

    return OpenAICompatibleClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL
    )


def get_llm():
    """获取LLM实例，延迟初始化"""
    global _llm_instance
    if _llm_instance is None:
        try:
            if not OPENAI_API_KEY:
                logger.warning("API 密钥未设置，无法初始化 LLM")
                _llm_instance = None
            else:
                _llm_instance = initialize_llm_client()
                logger.info("成功初始化 API 客户端")
        except Exception as e:
            logger.warning(f"初始化 API 客户端失败: {e}")
            logger.warning("将使用模拟响应模式")
            _llm_instance = None
    return _llm_instance
