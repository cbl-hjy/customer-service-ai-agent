"""
配置文件
包含系统运行所需的各种配置参数
"""

import logging
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# OpenAI兼容API配置（DeepSeek 官方端点，2026-09-15 生态切换：百炼 glm-5.2 免费额度耗尽）
# 端点：https://api.deepseek.com（🟢 DeepSeek 官方文档 api-docs.deepseek.com）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", DASHSCOPE_API_KEY)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
# V16（2026-08-14）：默认模型统一——与 .env 实际配置、eval 全部基线/报告一致
# （此前默认 qwen3.7-flash 是无 env 兜底值，与真相源漂移；生产/评估实际跑的都是真相源模型）
# 模型定版史：qwen3.7-max-2026-06-08 → glm-5.2（2026-08-17，百炼托管）
# → deepseek-flash（2026-09-15，用户指定；= DeepSeek-V4.1-Flash，官方更新日志确认）。
# 换生态必须过 W4 全量回归验证分类偏移（本项目硬约束）
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-flash")

# 思考模式开关（2026-09-15 适配 DeepSeek：thinking={"type": enabled/disabled}，服务端默认 enabled）
# 默认关闭：分类和常规回复不需要深度思考，关闭后延迟降 5-10 倍（实测数据支撑）
# 复杂/升级场景需要思考时，可设 ENABLE_THINKING=true
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

# 全局并发上限（信号量，2026-08-12 生产优化③——前沿做法：限流在 agent 调用层全局生效）
# 实测：10 并发 API 排队放大延迟（P50 39.8s > 串行 29.3s）；默认 5 并发防排队，可调
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))

# 启动预热（第二波，2026-09-15）：后台线程加载本地模型链（检索/图），
# 消除冷启动首 token 惩罚（实测 24.2s → 稳态 4.8s）。见 warmup.py。
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() == "true"

# V12 对话压缩（2026-08-14）：persisted_dialogue 超 DIALOGUE_CAP 条后，
# 最旧轮次压缩为摘要（dialogue_summary）并转存归档表，防止长会话 checkpoint 线性膨胀。
# 保留尾部条数 = DIALOGUE_CAP - DIALOGUE_SUMMARY_TRIGGER_HEADROOM 的取整策略见
# multi_agent_customer_service._compact_dialogue_if_needed（保留近 DIALOGUE_TAIL_KEEP 条原文，
# 需 ≥ prompt 注入条数 12 + 余量）。
DIALOGUE_CAP = int(os.getenv("DIALOGUE_CAP", "40"))
# 压缩后保留的尾部原文条数（prompt 注入取尾部 12 条，保留 16 条留余量）
DIALOGUE_TAIL_KEEP = int(os.getenv("DIALOGUE_TAIL_KEEP", "16"))

# HTTP请求配置
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "MultiAgentCustomerService/1.0.0"
}

# 熔断器配置（2026-08-13 生产韧性：区分『LLM 通道故障』vs『分类不确定』）
# 连续失败达到阈值 → 熔断 OPEN（快速失败，不发起网络请求，保护 API 额度）
CIRCUIT_BREAKER_FAIL_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAIL_THRESHOLD", "3"))
# 熔断后冷却秒数（OPEN → HALF_OPEN 的等待时间）
CIRCUIT_BREAKER_TIMEOUT = float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30"))
# 半开探测：放行多少个请求试探恢复（成功即 CLOSED，失败回 OPEN）
CIRCUIT_BREAKER_RECOVERY_REQUESTS = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_REQUESTS", "1"))

# 指标与告警（第二波可观测性，2026-09-16）：metrics.py 聚合 trace.db 为四大黄金信号，
# 超阈产出结构化 WARN 日志 + /api/metrics 响应内 alerts 数组（不接外部通知渠道）。
# 告警阈值口径：P50 = 稳态首 token 验收线；错误率 = devops.com LLM 生产实践分级（5% 观察/10% 立即）取观察级。
METRICS_WINDOW_MIN = int(os.getenv("METRICS_WINDOW_MIN", "60"))
# 告警最小样本数：低于该样本量跳过阈值判断（防小样本误报，如窗口内只有 2 个请求且都慢）
METRICS_ALERT_MIN_SAMPLES = int(os.getenv("METRICS_ALERT_MIN_SAMPLES", "5"))
METRICS_ALERT_P50_S = float(os.getenv("METRICS_ALERT_P50_S", "6.0"))
METRICS_ALERT_ERROR_RATE = float(os.getenv("METRICS_ALERT_ERROR_RATE", "0.05"))
# token 单价（元/百万，与 eval_multi_turn 成本口径一致，仅估算用）
METRICS_TOKEN_IN_PRICE_PER_M = float(os.getenv("METRICS_TOKEN_IN_PRICE_PER_M", "0.15"))
METRICS_TOKEN_OUT_PRICE_PER_M = float(os.getenv("METRICS_TOKEN_OUT_PRICE_PER_M", "1.5"))

# 系统配置
SYSTEM_NAME = "多智能体客服系统"
VERSION = "1.0.0"

# 日志配置
LOG_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}


def setup_logging() -> None:
    """初始化根日志配置（V9：print → logging）。

    幂等：basicConfig 只在根 logger 未配置时生效，多模块 import config 安全。
    LOG_LEVEL=DEBUG 时输出请求级脱敏日志（角色列表，不含消息内容）。
    """
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO),
        format=LOG_CONFIG["format"],
    )


setup_logging()
