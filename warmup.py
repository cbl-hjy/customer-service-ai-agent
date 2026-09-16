"""启动预热（第二波工程化，2026-09-15）：消除冷启动首 token 惩罚。

冷启动根因（verify_e2e_smoke Case A 实测：冷启动首 token 24.22s vs 稳态 4.76s）：
reranker/bge-m3/jieba/KB 索引全部懒加载，成本转嫁给了第一个真实用户。

前沿参照：
- Triton Inference Server model warmup：启动时对模型跑虚拟推理，完成才上线
- K8s readiness probe：慢启动服务就绪前不接流量（单机等价物 = /api/health 的 warmup 字段）
共同原则：把懒加载变成启动成本，就绪状态可观测，预热失败不阻断服务启动。

纪律（具名风险逐条对应）：
- R1 预热不阻塞启动：后台 daemon 线程执行
- R2 额度纪律：本地模型链（检索/图/客户端）零 API 消耗；LLM 通道仅一次
  ~2 token 探活调用（Triton/vLLM warmup 语义，成本 <¥0.0001，用于消除通道冷启动）
- R3 预热失败不阻断：全程 try/except，失败仅告警，服务回退懒加载行为
- R4 并发竞态安全：检索器单例均有 double-checked locking
  （kb_retriever._retriever_lock / hybrid_retriever._lock / dense_retriever._lock），
  预热线程与首个请求并发触发懒加载不会重复加载
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# 状态机：not_started → warming → ready / failed（failed 后回退懒加载，状态保留供观测）
_warmup_state = {"status": "not_started"}
_state_lock = threading.Lock()
_thread = None


def get_warmup_status() -> str:
    """环外只读：当前预热状态（/api/health 消费）。"""
    with _state_lock:
        return _warmup_state["status"]


def _do_warmup() -> None:
    """预热主体（同步执行，start_warmup 的线程目标；测试可直接调用避免线程时序）。"""
    try:
        with _state_lock:
            _warmup_state["status"] = "warming"
        t0 = time.time()

        # 1) LLM 客户端（纯对象创建，无网络调用）
        from llm import get_llm
        get_llm()

        # 2a) 检索器与 KB 索引/jieba（一次真实检索调用，走 exact 预检路径即可）
        from kb_retriever import retrieve_titles
        retrieve_titles("product_info", "手机多少钱")

        # 2b) 混合检索模型显式预热（dense/bge-m3 + reranker）——不能依赖查询走通
        # 全链：exact 命中的查询会被精确预检短路，rank() 永远不被触发（冷启动
        # 22s 复现根因），必须显式调 warm_up()
        from kb_retriever import _KB_PATH, _get_hybrid_retriever
        _get_hybrid_retriever(_KB_PATH).warm_up()

        # 3) 图编译 + A4/A5 工厂注入（毫秒级）
        from chat_web_service import get_app
        get_app()

        # 4) LLM 通道探活（一次 ~2 token 真实调用，Triton/vLLM warmup 语义）：
        # 建立 HTTPS 连接 + 触发服务端路由，消除首请求的通道冷启动（实测首 token
        # 8.6s 的残余差额即此因）。成本 <¥0.0001，远低于把 4s 冷启动转嫁给首个用户。
        from llm import get_llm
        get_llm().invoke([{"role": "user", "content": "请只回复数字1"}])

        with _state_lock:
            _warmup_state["status"] = "ready"
        logger.info("启动预热完成（%.1fs），检索链/图/LLM 通道已就绪", time.time() - t0)
    except Exception as e:  # noqa: BLE001  R3：预热失败不阻断，回退懒加载
        with _state_lock:
            _warmup_state["status"] = "failed"
        logger.warning("启动预热失败（回退懒加载，首个请求将承担加载成本）: %s", e)


def start_warmup() -> None:
    """启动后台预热线程（幂等：已启动过则直接返回，不起第二个线程）。"""
    global _thread
    with _state_lock:
        if _warmup_state["status"] != "not_started":
            return
        _warmup_state["status"] = "warming"
    _thread = threading.Thread(target=_do_warmup, name="warmup", daemon=True)
    _thread.start()
    logger.info("启动预热线程已发起（本地模型链加载，不消耗 LLM 额度）")


def reset_warmup_for_tests() -> None:
    """测试隔离原语（配对清理纪律）：复位状态机，允许重复驱动预热。"""
    global _thread
    with _state_lock:
        _warmup_state["status"] = "not_started"
    _thread = None
