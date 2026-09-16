"""T1 订单工具层（2026-09-16）：模型可调用工具 + 模拟订单库。

与 tools/query_decompose（编排内部件）的区别：本模块是"模型主动调用的业务动作"
（Anthropic 客服 agent 模式的可编程动作层），使 agent 从"只能说"升级为"能查能做"。

订单库：SQLite（data/orders.db，ORDERS_DB_PATH 可覆盖——评估隔离先例
CHECKPOINT_DB_PATH / TICKETS_DB_PATH）。首次连接 lazy 建表 + 播种（幂等，
INSERT OR IGNORE）。种子为虚构演示数据（星环数码，与 KB 同业务域）。

政策代码化（KB 规则 → 可执行边界，口径来源 data/knowledge_base_v2.json）：
- 「未发货改地址」：仅 status=pending 可改；每单限 1 次（address_updated 标记）；
  已发货 → 拒绝并提示走发货后流程
- 「物流单号查询」：未发货（pending）单无物流单号，返回"暂无物流信息"

安全边界：用户输入（订单号/地址）是不可信数据——订单号白名单格式校验、
地址长度截断；查询仅按订单号取数，不提供跨订单检索（防拖库）。
"""

import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

_ORDERS_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "orders.db"
)

# 状态机（与客服话术一致，非英文内部值直接暴露）
STATUS_LABEL = {
    "pending": "待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "cancelled": "已取消",
}

# 订单号格式（虚构业务命名：XH + 日期 + 序号）
_ORDER_ID_RE = re.compile(r"^XH\d{12}$")
_MAX_ADDRESS_LEN = 100

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("ORDERS_DB_PATH", _ORDERS_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS orders (
            order_id        TEXT PRIMARY KEY,
            status          TEXT NOT NULL,
            items           TEXT NOT NULL,
            amount          REAL NOT NULL,
            address         TEXT NOT NULL,
            address_updated INTEGER NOT NULL DEFAULT 0,
            logistics_no    TEXT,
            logistics_events TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        )"""
    )
    _seed(conn)
    return conn


def _seed(conn: sqlite3.Connection) -> None:
    """播种虚构订单（幂等）：覆盖状态矩阵 × 改地址边界（可改/不可改/已改过）。"""
    rows = [
        # (order_id, status, items, amount, address, addr_updated, logistics_no, logistics_events)
        ("XH202609010001", "shipped", [{"name": "星环 X1 Pro 旗舰手机", "qty": 1, "price": 4999}],
         4999.0, "浙江省杭州市西湖区文一西路 100 号", 0, "SF1234567890",
         [["2026-09-14 20:05", "包裹已从杭州转运中心发出"],
          ["2026-09-15 08:30", "到达苏州转运中心"],
          ["2026-09-16 07:12", "到达无锡转运中心，运输中"]]),
        ("XH202609010002", "pending", [{"name": "星环 X1 标准版手机", "qty": 1, "price": 3999}],
         3999.0, "江苏省南京市玄武区中山东路 200 号", 0, None, None),
        ("XH202609010003", "delivered", [{"name": "星环 X1 Lite 入门手机", "qty": 2, "price": 1999}],
         3998.0, "上海市浦东新区张江路 300 号", 0, "YT9876543210",
         [["2026-09-10 10:00", "已揽收"],
          ["2026-09-12 14:20", "派送中"],
          ["2026-09-12 18:45", "已签收"]]),
        ("XH202609010004", "cancelled", [{"name": "星环 X1 Pro 旗舰手机", "qty": 1, "price": 4999}],
         4999.0, "广东省深圳市南山区科技园 8 号", 0, None, None),
        # 改地址边界：已用掉 1 次修改额度（KB：每单限 1 次）
        ("XH202609010005", "pending", [{"name": "星环 X1 标准版手机", "qty": 1, "price": 3999}],
         3799.0, "北京市朝阳区科技园路 8 号", 1, None, None),
        ("XH202609010006", "shipped", [{"name": "星环 X1 Lite 入门手机", "qty": 1, "price": 1999}],
         1999.0, "四川省成都市高新区天府大道 500 号", 0, "JD5550001111",
         [["2026-09-15 09:00", "已揽收"],
          ["2026-09-16 06:30", "到达成都高新网点，派送中"]]),
        ("XH202609010007", "delivered", [
            {"name": "星环 X1 Pro 旗舰手机", "qty": 1, "price": 4999},
            {"name": "星环 X1 标准版手机", "qty": 1, "price": 3999}],
         8998.0, "湖北省武汉市洪山区珞瑜路 100 号", 0, "ZT0001112223",
         [["2026-09-08 11:00", "已揽收"],
          ["2026-09-09 16:10", "已签收"]]),
        ("XH202609010008", "pending", [{"name": "星环 X1 Lite 入门手机", "qty": 1, "price": 1999}],
         1899.0, "福建省厦门市思明区软件园 2 期", 0, None, None),
    ]
    created = "2026-09-13 10:00:00"
    conn.executemany(
        """INSERT OR IGNORE INTO orders
           (order_id, status, items, amount, address, address_updated,
            logistics_no, logistics_events, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [
            (oid, st, json.dumps(items, ensure_ascii=False), amt, addr, au,
             lno, json.dumps(ev, ensure_ascii=False) if ev else None, created, created)
            for (oid, st, items, amt, addr, au, lno, ev) in rows
        ],
    )
    conn.commit()


def _validate_order_id(order_id: str) -> str:
    """订单号白名单校验（不可信输入）：格式不符直接返回 not_found 语义。"""
    oid = str(order_id or "").strip().upper()
    if not _ORDER_ID_RE.match(oid):
        return ""
    return oid


def _get_order(conn: sqlite3.Connection, order_id: str):
    return conn.execute(
        "SELECT * FROM orders WHERE order_id = ?", (order_id,)
    ).fetchone()


# =============================================================================
# 工具实现（返回 str = tool 消息 content，模型可转述；异常由 execute_tool_call 兜底）
# =============================================================================

def query_order(order_id: str) -> str:
    """查询订单概要：状态/商品/金额/收货地址/物流单号摘要。"""
    oid = _validate_order_id(order_id)
    if not oid:
        return "订单号格式不正确（应为 XH + 12 位数字，如 XH202609010001），请与用户确认订单号。"
    with _lock:
        conn = _connect()
        try:
            row = _get_order(conn, oid)
            if not row:
                return f"未找到订单 {oid}，请与用户确认订单号是否正确。"
            items = json.loads(row["items"])
            item_desc = "、".join(f"{i['name']}×{i['qty']}" for i in items)
            logistics = row["logistics_no"] or "暂无物流信息（订单尚未发货）"
            return (
                f"订单号：{oid}\n状态：{STATUS_LABEL.get(row['status'], row['status'])}\n"
                f"商品：{item_desc}\n实付金额：{row['amount']:.2f} 元\n"
                f"收货地址：{row['address']}\n物流单号：{logistics}"
            )
        finally:
            conn.close()


def query_logistics(order_id: str) -> str:
    """查询物流轨迹：物流单号 + 事件时间线。"""
    oid = _validate_order_id(order_id)
    if not oid:
        return "订单号格式不正确（应为 XH + 12 位数字，如 XH202609010001），请与用户确认订单号。"
    with _lock:
        conn = _connect()
        try:
            row = _get_order(conn, oid)
            if not row:
                return f"未找到订单 {oid}，请与用户确认订单号是否正确。"
            if not row["logistics_no"]:
                return f"订单 {oid} 当前状态为「{STATUS_LABEL.get(row['status'], row['status'])}」，暂无物流信息，发货后 2 小时内同步单号。"
            events = json.loads(row["logistics_events"] or "[]")
            timeline = "\n".join(f"[{t}] {d}" for t, d in events) or "暂无轨迹更新"
            return (
                f"订单号：{oid}\n物流单号：{row['logistics_no']}\n"
                f"当前状态：{STATUS_LABEL.get(row['status'], row['status'])}\n轨迹：\n{timeline}"
            )
        finally:
            conn.close()


def update_order_address(order_id: str, new_address: str) -> str:
    """修改收货地址（政策代码化：仅待发货可改 + 每单限 1 次）。"""
    oid = _validate_order_id(order_id)
    if not oid:
        return "订单号格式不正确（应为 XH + 12 位数字，如 XH202609010001），请与用户确认订单号。"
    addr = str(new_address or "").strip()[:_MAX_ADDRESS_LEN]
    if len(addr) < 5:
        return "新地址内容无效（过短），请与用户确认完整地址。"
    with _lock:
        conn = _connect()
        try:
            row = _get_order(conn, oid)
            if not row:
                return f"未找到订单 {oid}，请与用户确认订单号是否正确。"
            if row["status"] != "pending":
                return (
                    f"修改失败：订单 {oid} 当前状态为「{STATUS_LABEL.get(row['status'], row['status'])}」，"
                    "地址修改仅支持「待发货」状态；已发货订单请走发货后修改地址流程（联系人工客服处理）。"
                )
            if row["address_updated"]:
                return (
                    f"修改失败：订单 {oid} 已使用过 1 次地址修改额度（每单限 1 次）。"
                    "如仍需变更，建议联系人工客服核实处理。"
                )
            conn.execute(
                "UPDATE orders SET address = ?, address_updated = 1, updated_at = ? WHERE order_id = ?",
                (addr, _now(), oid),
            )
            conn.commit()
            return (
                f"修改成功：订单 {oid} 的收货地址已更新为「{addr}」，"
                "仓库将按新地址打单发出。（提示：每个订单仅支持修改 1 次地址）"
            )
        finally:
            conn.close()


# =============================================================================
# OpenAI function schema + 分派器
# =============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": (
                "查询订单信息：状态、商品明细、实付金额、收货地址、物流单号。"
                "用户提供订单号并询问订单情况（到哪了/买了什么/多少钱）时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，格式 XH+12位数字，如 XH202609010001"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_logistics",
            "description": (
                "查询物流轨迹：物流单号与配送事件时间线。"
                "用户询问快递/物流进度、快递单号时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，格式 XH+12位数字"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_order_address",
            "description": (
                "修改订单收货地址（仅「待发货」状态可改，每单限 1 次）。"
                "用户明确要求改地址且订单号、新地址齐备时立即调用——"
                "新地址含省/市/区与路名门牌即视为完整（如\"北京市海淀区中关村大街1号\"），"
                "无需再向用户确认；仅缺订单号或地址时才先询问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单号，格式 XH+12位数字"},
                    "new_address": {"type": "string", "description": "新收货地址（完整地址）"},
                },
                "required": ["order_id", "new_address"],
            },
        },
    },
]

_TOOL_FUNCS = {
    "query_order": query_order,
    "query_logistics": query_logistics,
    "update_order_address": update_order_address,
}


def execute_tool_call(name: str, arguments_json: str) -> str:
    """执行单个 tool_call（分派 + 参数解析 + 异常兜底）。

    返回 str 作为 tool 消息 content 回填给模型；任何异常都转为可转述的错误
    描述（C5 纪律延伸：工具失败不硬答，由模型如实告知用户）。
    未知工具名返回明确错误（防 schema 漂移后模型幻觉调用）。
    """
    func = _TOOL_FUNCS.get(name)
    if func is None:
        return f"未知工具：{name}（可用工具：{', '.join(_TOOL_FUNCS)}）"
    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            return f"工具 {name} 参数格式错误（应为 JSON 对象），请重试。"
        return func(**args)
    except json.JSONDecodeError:
        return f"工具 {name} 参数格式错误（非法 JSON），请重试。"
    except TypeError as e:
        return f"工具 {name} 参数不匹配：{e}"
    except Exception as e:  # noqa: BLE001  工具故障兜底：不硬答，转述给模型
        logger.warning("工具 %s 执行异常: %s", name, e)
        return f"工具 {name} 执行出错，请如实告知用户系统暂时无法处理该请求，建议稍后重试。"
