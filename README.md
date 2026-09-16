# 多智能体电商客服升级系统（Multi-Agent Customer Support Escalation System）

<p align="center">
  <em>基于 LangGraph 的多智能体客服系统 · 意图分类 + 升级决策硬边界 + 混合 RAG 检索 + 企业级工程底座</em>
</p>

> 基于 [handsomestWei/customer-service-ai-agent](https://github.com/handsomestWei/customer-service-ai-agent) fork 二次开发。

## 设计理念（本项目最核心的立场）

> **"不是让 agent 知道做什么，而是让它知道不能做什么"**

- **约束层最小**：prompt 只引导输出格式（JSON），不堆行为规则——agent 在红线内自由思考
- **负向约束由代码硬边界执行**：`routing.should_escalate()` 在编排层做升级决策，不靠 prompt 劝
- **fail-safe 兜底**：agent 犯错的最低代价 = 升级人工（人是最后防线），不造成不可逆损失
- **harness 做厚**：checkpointer 持久化 / 熔断重试 / 全链路追踪 / 分层测试——工程底座扎实，agent 行为零侵入

## 系统架构

```
工单 (customer_query)
  │
  ▼
┌─────────────┐  结构化 JSON（label / confidence / complexity）
│ classify    │── LLM 强制 JSON + Pydantic 校验 + 上轮 context_signal 指代消解
└─────────────┘  不可解析 → confidence 强制 0（fail-safe）
  │
  ├─ out_of_scope ────────────► 护栏拒绝（不调用业务 agent）
  │
  ├─ should_escalate（代码硬边界）
  │    ├─ 低置信(<0.6) / 复杂(complex) / 投诉 ──► escalate → 人工客服 + 结构化摘要
  │    └─ 否则
  ▼
┌─────────────┐
│ 5 业务 agent │── product / tech / billing / complaint / general
└─────────────┘   混合 RAG：BM25 + bge-m3 dense → RRF 融合 → bge-reranker 精排
  │               KB 无匹配 → _no_answer → 升级（不编造）；回答附引用溯源
  ▼
final_response（SSE 流式：meta → stage → token → done）
```

## 核心能力

| 能力 | 说明 |
|---|---|
| 结构化意图分类 | 置信度 + 复杂度 + 上轮消息注入（指代词消解）；四路降级解析（JSON/字符串/垃圾/空） |
| 升级决策硬边界 | 低置信 / 复杂 / 投诉 / OOD 域外 / KB 无命中 → 升级人工 + 结构化摘要；升级态追问按原升级域检索补全，不重复升级 |
| 混合 RAG 检索 | BM25 + dense（bge-m3）双路召回 → 加权 RRF(k=20, 稠密 1.5) 融合 → bge-reranker-v2-m3 精排 → top-k + 置信门槛；217 条金标消融实验定参，本地模型缺失时优雅降级 BM25-only |
| 引用溯源 | 回答附知识库条目引用，可核查（`citations.py`） |
| SSE 流式输出 | 帧协议 meta→stage→token→done→[DONE]；节点阶段推送（前端可见处理进度）；断连后 worker 完成落库 |
| 全链路追踪 | `trace.py` 节点包装器零侵入：节点耗时 / token 增量 / 决策字段写 SQLite（`trace_store.py`） |
| 冷启动预热 | `warmup.py` 启动时 daemon 线程预热检索组件，`/api/health` 暴露 warming/ready；首 token 24.2s → 4.6s ≈ 稳态 |
| 运行时可观测 | `GET /api/metrics`：四大黄金信号聚合（P50/P95 延迟、决策分布、错误率、熔断/并发饱和度）+ 升级率与原因分布 + 窗口 token/成本 + 最慢节点 Top5；超阈（P50>6s / 错误率>5% / 熔断 OPEN）产出结构化 WARN 告警 |
| 弹性设计 | 熔断器（半开恢复） / 分层重试契约（4xx 永久失败不重试） / token 计量 / 并发信号量 |

## 快速开始

```bash
# 1. 克隆与环境
git clone https://github.com/cbl-hjy/customer-service-ai-agent.git
cd customer-service-ai-agent
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux

# 2. 配置（DeepSeek，OpenAI 兼容端点）
cp env_example.txt .env    # 编辑 .env：OPENAI_API_KEY=sk-xxx
# 可选：BGE_M3_PATH / BGE_RERANKER_PATH 指向本地模型快照；
# 不配置则自动降级 BM25-only 检索，功能完整、质量略降。

# 3. 启动 Web 服务（SSE 流式 + 预热）
.venv/Scripts/python web_app.py
# 浏览器打开 http://127.0.0.1:5000

# 4. CLI 演示（单条工单 → 完整协作过程）
.venv/Scripts/python demo_cli.py "物流太慢等了一周我要投诉"

# 5. 测试（密封批：mock + 轻量 KB，零 API 成本，~45s）
.venv/Scripts/python -m pytest -m "not api and not model" -q
```

## 测试与评估

**测试分层**（`pyproject.toml` markers）：默认密封批（mock + 轻量真实 KB）/ `api`（真实 LLM 调用）/ `model`（真实本地大模型加载）。全量 253 项（239 功能 + 14 故障注入）。

**W4 金标评估**（28 多轮 cases，deepseek-flash，2026-09-16 正式基线）：

| 指标 | 结果 |
|---|---|
| 决策正确率 | 100%（56/56） |
| 升级精确率 / 召回率 | 100% / 100%（误升级=0 漏升级=0） |
| 答案通过率 | 97.8% |
| judge 事实一致性 | 97.7% |
| 检索命中率 / 引用溯源 | 100% / 100% |
| 延迟 P50 | 5.5s（分类 LLM + 本地 GPU 检索精排 + 生成，trace 分解见 `eval/`） |

**其他实测**：

| 项目 | 结果 |
|---|---|
| OOD 域外升级触发率（banking77 前 50 条） | **100%** |
| 冷启动首 token（预热前 → 预热后） | 24.2s → 4.6s（稳态 4.4s） |
| 并发容量曲线（5→32 并发扫描 + soak + GPU 串行门控修复复测） | 全档 0 错误 0 限流；GPU 段互斥锁修复后 8-32 并发吞吐平稳 0.75-0.80 QPS（修复前 32 并发显存叠加崩塌至 0.13 QPS / P95 5min），过载退化为线性排队；soak 稳态无漂移 |
| 检索组件消融（217 金标，加权 RRF k=20/稠密 1.5） | R@1：BM25 0.357 → +混合 RRF 0.495 → +rerank 0.486；HIT@1 0.433 → 0.599 → 0.594；R@5 0.590 → 0.783 → 0.809 |

**诚实声明**：banking77 为英文银行意图数据集，相对本系统为**域外样本**，用于验证"未知意图 → 升级兜底"的 fail-safe 行为，非领域分类基准。

## 工程实践

- **本地 CI 门禁**（`scripts/gate.py`）：三层——`fast`（密封测试批，pre-push 钩子自动拦截）/ `eval`（W4 金标评估 + 基线噪声带对比：确定性指标 ±2%、生成类 ±3%、延迟成本 +30%，模型/KB/prompt 变更后必跑）/ `full`（全量测试 + E2E 冒烟）
- **可观测性**（`metrics.py`，调研定标 Google SRE 四大黄金信号 + RED 方法）：只读聚合 trace.db，零依赖不引 Prometheus；10k runs 窗口聚合 186ms；告警带最小样本量豁免（防小样本误报）
- **模型行为评估**：模型切换必须过 W4 金标（决策分/答案分/检索命中率对比基线），评估数据与脚本在 `eval/`
- **开发规范**：`AGENTS.md`（约束分级：硬约束 + 惯例，每条约束指认具体风险出处）
- **故障注入**：14 项故障注入测试（断连/熔断/超时/污染等），验证 fail-safe 真实性

## 项目结构

```
├── web_app.py              # Flask 入口（SSE 流式 + 预热 + health）
├── routing.py              # 升级决策硬边界
├── kb_retriever.py         # 检索入口（查询扩展/精确预检/混合精排/BM25兜底）
├── hybrid_retriever.py     # RRF 融合 + reranker 精排
├── nodes/                  # LangGraph 节点（分类/升级/复合查询/最终回复）
├── multi_agents/           # 5 业务 agent 基类与实现
├── graph/                  # 图构建
├── llm/                    # 客户端（流式/熔断/重试/token计量）
├── trace.py / trace_store.py  # 全链路追踪
├── warmup.py               # 冷启动预热
├── eval/                   # W4 金标评估 + 检索组件评估 + KB 数据
├── tests/                  # 分层测试（253 项）
└── scripts/gate.py         # 本地 CI 门禁
```

## 依赖

- Python 3.10+（开发环境 3.13）
- LangGraph + langgraph-checkpoint-sqlite / langchain-core / Flask / requests / pydantic
- 检索（可选，缺失自动降级）：sentence-transformers + transformers（bge-m3 / bge-reranker-v2-m3）
- pytest（测试）

## 数据来源与许可

- 模拟数据：fork 自 handsomestWei（电子品类产品库/投诉库）
- 评估数据：banking77（MTEB 镜像，域外样本）；检索金标种子部分提取自 [cooelf/DeepUtteranceAggregation](https://github.com/cooelf/DeepUtteranceAggregation)（ECD 语料）
- 原始仓库 License：Apache 2.0
