"""
意图路由与升级决策（C3：harness 层硬边界）

理念（§3.4 Hermes 分层安全模型）：负向约束由**代码硬边界**执行，不靠 prompt 堆规则。
本模块是"Agent/编排层"的调度决策——LLM（模型层）自报的 label/confidence/complexity
不可靠或越界时，由这里的硬规则决定：升级人工（最低代价出口）或继续。

升级 = fail-safe 出口：agent 不确定 / 复杂 / 投诉 → 升级人工，不硬答、不编造。
"""

from typing import Dict, Tuple

# ===== 意图 → 默认复杂度（LLM 自报缺失/不可解析时的 harness 兜底）=====
LABEL_DEFAULT_COMPLEXITY: Dict[str, str] = {
    "product_info": "simple",        # 产品咨询，一句话可答
    "technical_support": "medium",   # 故障诊断，需多步
    "billing": "medium",             # 退款/发票，涉及流程
    "complaint": "complex",          # 投诉需跟进处理
    "general_inquiry": "simple",     # 物流/时间等一般咨询
    "out_of_scope": "simple",        # 直接拒绝，不走升级
}

# ===== 升级硬边界（可配置，不写死）=====
# 置信度低于此值 → 升级（"不确定不硬答"）
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.6
# 复杂度在此集合 → 升级
ESCALATION_COMPLEXITY: Tuple[str, ...] = ("complex",)
# 标签在此集合 → 升级（投诉天然需人工跟进，避免一线硬压）
ESCALATION_LABELS: Tuple[str, ...] = ("complaint",)

# ===== 用户主动求人工信号（v15 修复，代码硬边界）=====
# 用户明确要求转人工 → 强制升级（LLM 无感知，不碰 prompt）。
# 用完整短语而非裸"人工"，避免误伤"人工智能"等子串。
HUMAN_REQUEST_KEYWORDS: Tuple[str, ...] = (
    "转人工", "人工客服", "联系人工", "找经理", "找人工",
    "真人客服", "找真人", "人工服务", "人工处理", "客服经理",
    "人工坐席", "转接人工", "人工接听",
)


def wants_human(query: str) -> bool:
    """检测用户是否主动要求转人工（负向边界：用户求人工 = 必须升级，不硬答）。

    理念（§3.4）：用户主动求人工是最强的升级信号，由代码硬边界接管，
    不依赖分类 LLM 判断——即使分类器把它归为 general_inquiry 也强制升级。
    """
    if not query:
        return False
    return any(kw in query for kw in HUMAN_REQUEST_KEYWORDS)


def should_escalate(label: str, confidence: float, complexity: str) -> Tuple[bool, str]:
    """升级决策（代码硬边界）。

    Args:
        label: 分类标签（LLM 自报或降级结果）
        confidence: 置信度 0-1（LLM 自报；不可解析时上游已强制置 0）
        complexity: 复杂度（LLM 自报或 config 兜底）

    Returns:
        (是否升级, 原因)。升级 = 转人工，不硬答（fail-safe）。
    """
    # 护栏优先：out_of_scope 直接拒绝，不进入升级流程（负向边界不被削弱）
    if label == "out_of_scope":
        return False, "out_of_scope 由护栏直接拒绝"
    # 不确定 → 升级（"不知道就别说"）
    if confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        return True, f"置信度不足 ({confidence:.2f} < {ESCALATION_CONFIDENCE_THRESHOLD})"
    # 复杂 → 升级
    if complexity in ESCALATION_COMPLEXITY:
        return True, f"问题复杂度 {complexity}，需人工处理"
    # 投诉 → 升级（人工跟进，避免一线硬压）
    if label in ESCALATION_LABELS:
        return True, "投诉类问题转人工跟进"
    # 其余：一线直接处理
    return False, ""
