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
    def __init__(self, content, tool_calls=None):
        self.content = content
        # T1 工具调用（2026-09-16）：invoke(tools=...) 时携带模型返回的 tool_calls
        # 原始结构（list[{id, type, function:{name, arguments}}]），None=无工具调用。
        # 默认 None 保证现有调用方（分类/改写/拆解/正文）行为零变化。
        self.tool_calls = tool_calls


# ASCII 前导缓冲上限：工具调用前导（英文独白，如 "I'll look up..."）通常 <16 字符；
# 纯英文正文超过此长度即按正文 flush（客服回复必含中文，中文一出现立即 flush）。
_LEAD_BUF_MAX = 16


class _ToolStream:
    """带 tools 的流式包装（T1）：迭代产出面向用户的 content token，聚合 tool_calls 分片。

    用法：迭代耗尽后读 .tool_calls（None=模型未调工具，此时迭代产出的就是完整正文）。

    前导抑制（三层策略，防工具轮的英文前导泄漏到用户视图）：
    ① prompt 指令（agent 侧："调用工具时直接调用，不要输出说明文字"）；
    ② 一旦收到 tool_delta 即入工具模式——缓冲丢弃、后续 content 静默不产出；
    ③ ASCII 前导缓冲：未见 tool_delta 前的纯 ASCII token 缓冲 ≤16 字符，
       出现非 ASCII（中文正文信号）或超限即 flush 判定为正文。
    残余窗口：英文正文头 16 字符延迟一个缓冲期产出（不丢失）；前导 >16 字符
    且后跟 tool_calls 的罕见形态会泄漏前 16 字符（可接受，V1）。
    """

    def __init__(self, client, messages, tools):
        self._events = client._stream_events(messages, None, tools)
        self.tool_calls = None
        self._done = False
        self._tool_mode = False
        self._text_mode = False  # 正文确认后直通（防英文标点/数字二次入缓冲）
        self._buf = []
        self._buf_len = 0
        self._aggregating = {}  # index -> {"id":…, "name":…, "arguments": str}

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            if self._done:
                raise StopIteration
            try:
                ev = next(self._events)
            except StopIteration:
                self._done = True
                self._finalize()
                if self._buf and not self._tool_mode:
                    # 流结束仍无 tool_delta：缓冲是正文开头（短英文正文），补产出
                    out = "".join(self._buf)
                    self._buf, self._buf_len = [], 0
                    return out
                raise
            if ev["kind"] == "tool_delta":
                self._tool_mode = True
                self._buf, self._buf_len = [], 0  # 丢弃前导缓冲
                self._aggregate(ev["delta"])
                continue
            # token 事件
            text = ev["text"]
            if self._tool_mode:
                continue  # 工具模式：content 静默丢弃
            if self._text_mode:
                return text  # 正文已确认：直通
            if not text.isascii():
                # 缓冲期出现中文/中文标点：正文确认，连同缓冲一起 flush
                self._buf.append(text)
                out = "".join(self._buf)
                self._buf, self._buf_len = [], 0
                self._text_mode = True
                return out
            self._buf.append(text)
            self._buf_len += len(text)
            if self._buf_len > _LEAD_BUF_MAX:
                # 纯英文超限：按正文兜底 flush（英文正文罕见但不可丢）
                out = "".join(self._buf)
                self._buf, self._buf_len = [], 0
                self._text_mode = True
                return out
            continue  # ASCII 前导缓冲中（等中文信号 / tool_delta / 超限 / 流结束）

    def _aggregate(self, td: dict) -> None:
        """聚合 OpenAI 风格 tool_calls 分片：index 分组，arguments 字符串拼接。"""
        idx = td.get("index", 0)
        slot = self._aggregating.setdefault(idx, {"id": "", "name": "", "arguments": ""})
        if td.get("id"):
            slot["id"] = td["id"]
        fn = td.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["arguments"] += fn["arguments"]

    def _finalize(self) -> None:
        """流转结束：把聚合分片定稿为与 invoke 同构的 tool_calls 结构。"""
        if not self._aggregating:
            return
        self.tool_calls = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for _, slot in sorted(self._aggregating.items())
        ]


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
        """LangChain 消息对象 → OpenAI 格式（供 invoke / invoke_stream 共用）。

        T1（2026-09-16）：支持原生 dict 消息透传（含 "role" 键即视为 OpenAI 格式）——
        工具调用循环第二轮需回填 assistant(tool_calls) 与 role:"tool" 消息，这两种
        形态没有 LangChain 消息对象对应，由调用方直接构造 dict 传入。
        """
        formatted_messages = []
        for msg in messages:
            # 原生 OpenAI 格式 dict（tool loop 回填消息）→ 直接透传
            if isinstance(msg, dict) and "role" in msg:
                formatted_messages.append(msg)
                continue
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

    def _build_payload(self, formatted_messages: list, response_format=None, tools=None, tool_choice=None) -> dict:
        """构造请求 payload（思考模式开关 + response_format + tools，供两条路径共用）。"""
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
        # T1 工具调用（2026-09-16，官方 Chat Completion 协议，协议冒烟 4/4 验证）：
        # tools=OpenAI function schema 列表；tool_choice 可选（"none"/"auto"/指定函数）。
        # 流式路径不传 tools（工具轮走非流式 invoke，终答轮不带 tools 走流式）。
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        return payload

    def invoke(self, messages, response_format=None, tools=None, tool_choice=None):
        """调用OpenAI兼容API（response_format 可选：通义千问 JSON 结构化输出；
        tools 可选：OpenAI function calling——返回 .tool_calls（None=模型未调工具））"""
        formatted_messages = self._format_messages(messages)
        payload = self._build_payload(formatted_messages, response_format, tools, tool_choice)

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
                        # T1 工具调用：finish_reason=tool_calls 时 message.tool_calls 非空，
                        # 原样透传给调用方执行（content 可能含前导文本，回填时须一并保留）
                        tool_calls = message.get("tool_calls") or None
                        _record_token_usage(result.get("usage"))
                        _circuit_breaker.record_success()
                        return CustomResponse(content, tool_calls=tool_calls)
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

    def _stream_events(self, messages, response_format=None, tools=None):
        """SSE 流事件生成器（内部核心，2026-09-16 从 invoke_stream 抽取）：
        yield {"kind": "token", "text": str} 或 {"kind": "tool_delta", "delta": dict}。

        韧性契约与原 invoke_stream 完全一致（拆解见 invoke 内注释）：
        - 重试仅发生在首事件之前——已交付的 token/tool_delta 无法撤回，中途失败直接抛；
        - 信号量在流消费期间持续持有（连接占用即并发占用）；
        - token 计量：stream_options.include_usage（末 chunk 携带 usage），
          服务端不返回 usage 则不记（不估算，防污染成本基线）。
        T1 扩展：delta.tool_calls 分片以 tool_delta 事件透传（由 _ToolStream 聚合）。
        """
        formatted_messages = self._format_messages(messages)
        payload = self._build_payload(formatted_messages, response_format, tools)
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
                            yield {"kind": "token", "text": token}
                        for td in (delta.get("tool_calls") or []):
                            yielded = True
                            yield {"kind": "tool_delta", "delta": td}

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

    def invoke_stream(self, messages, response_format=None):
        """流式调用（P1 真流式 SSE）：逐 token yield——_stream_events 的薄包装，行为零变化。

        调用方约束：仅正文生成走本接口（llm.stream_sink.call_llm），
        分类/改写/拆解等结构化中间产物继续走 invoke（response_format=json）。
        带 tools 的流式（tool_calls 聚合）走 invoke_stream_tools。
        """
        for ev in self._stream_events(messages, response_format):
            if ev["kind"] == "token":
                yield ev["text"]

    def invoke_stream_tools(self, messages, tools):
        """带 tools 的流式调用（T1，2026-09-16）：返回 _ToolStream 迭代器。

        - 迭代产出面向用户的 content token（可直接推 sink，打字机效果保留）；
        - 迭代耗尽后 .tool_calls 持有聚合结果（None=模型未调工具）；
        - 前导抑制：见 _ToolStream docstring（三层策略）。
        """
        return _ToolStream(self, messages, tools)

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
