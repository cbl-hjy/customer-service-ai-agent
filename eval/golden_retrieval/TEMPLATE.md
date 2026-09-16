# Golden Retrieval Set 标注模板说明（P0 评估基建）

## 用途
为检索层消融实验（BM25-only / dense-only / 混合+RRF / +重排）提供"查询 → 正确条目"的金标，
可计算 Recall@K / MRR / nDCG。这是所有"深 RAG"优化的基准，没有它，任何消融都自嗨。

## 字段定义
| 字段 | 必填 | 说明 |
|---|---|---|
| id | 是 | 唯一编号 gr-001 起 |
| query | 是 | 用户原话（口语、简写、带错别字都行——越真实越好） |
| domain | 是 | 期望检索域，5 选 1：product / tech / billing / general / complaint（与 kb_retriever 域 key 一致，避免映射损耗） |
| expected_titles | 是 | 正确条目标题，分号分隔（1~3 个）。必须是 knowledge_base.json 中该域真实存在的 title，否则评估无法对齐 |
| difficulty | 是 | easy / hard / ambiguous（定义见下） |
| source | 是 | ECD train idx=NN（真实语料改写）/ 人工构造 / 复用 golden_multi_turn 轮次 |
| note | 否 | 备注（为什么难、歧义点、改写来源等） |

## difficulty 定义（决定消融能否拉开差距的关键）
- easy：query 含条目标题或 keywords 字面词 → BM25 应能命中（作为基线对照，占比 ~30%）
- hard：同义改写 / 口语化 / 新实体词 / 症状描述，与标题及 keywords 无字面重叠 → BM25 大概率漏，
        稠密向量才有机会中（这是证明"混合检索必要性"的核心样本，占比 ~40%）
- ambiguous：2 个以上条目都相关，需要重排分出主次（验证重排价值，占比 ~30%）

## 配额目标（总量 200+ 条）
- 每域 40+ 条（product 域现有 KB 仅 4 条，扩库后补标）
- easy : hard : ambiguous ≈ 3 : 4 : 3
- 来源构成：≥60% 来自 ECD 真实语料模式改写（保真实分布），≤40% 人工构造（补边界/对抗）

## 填写规范
1. expected_titles 先填现有 62 条 KB 内的；扩库新增条目后，可回头补标指向新条目的查询
2. **title 在域内不唯一**（如 general 域"在线客服"在 营业时间/联系方式 两个 category 下各有一条）：
   评估脚本将用「域内条目序号」定位，标注时若遇同名条目，请在 note 注明 category，如"在线客服（联系方式）"
3. hard 样本优先从 ECD 语料里找"绕弯问法"（指代、省略、口语），不要自己造生硬同义词
4. ambiguous 样本必须真实多义（如"能开发票吗"→电子发票+发票抬头都沾边），不要硬凑
5. 每条查询只标"期望检索到的条目"，不标决策（决策评估走 golden_multi_turn）

## 交付物
- 本 CSV 填完（200+ 行）
- 我这边生成：加载脚本 + Recall@K/MRR/nDCG 评估器 + 三基线对比报告
