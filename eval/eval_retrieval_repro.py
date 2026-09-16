"""检索层专项复现：对 v3 真实语料评估的 8 个失败轮，做确定性 BM25 检索测试。

不走 LLM（检索层是确定性逻辑），直接调用 kb_retriever，区分三种根因：
 ① keywords 覆盖不足（精确 _exact_hits 与模糊 _bm25_hits 都 miss）
 ② BM25 阈值/权重问题（有 BM25 命中但分数 <_SCORE_THRESHOLD 被过滤）
 ③ 分类器错域（检索本身有命中，但分类到错误 domain 导致查错域）

用法：python eval/eval_retrieval_repro.py
"""
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

from kb_retriever import get_retriever, SCORE_THRESHOLD, TOP_K  # noqa: E402

# (case_id, 失败轮用户输入, 评估实测分类域, 金标期望域)
FAILURES = [
    ("mt-202", "什么时候发货啊", "general_inquiry", "general_inquiry"),
    ("mt-203", "能发顺丰吗？我上次等太久了", "general_inquiry", "general_inquiry"),
    ("mt-204", "顺便问下现在有什么优惠券吗", "product_info", "general_inquiry"),
    ("mt-207", "那现在下单能明天发货吗", "general_inquiry", "general_inquiry"),
    ("mt-209", "算了还是留着吧，就是屏幕有点花", "complaint", "technical_support"),
    ("mt-214", "朋友买的那款降噪耳机好用吗", "general_inquiry", "product_info"),
    ("mt-215", "分期手续费怎么算的", "product_info", "billing"),
    ("mt-216", "那退的这个能开票吗", "billing", "billing"),
]

# 候选域：评估实测域 + 期望域 + 可能合理域
def candidate_domains(actual, expect):
    cands = [actual, expect]
    for d in ["product_info", "technical_support", "billing", "complaint", "general_inquiry"]:
        if d not in cands:
            cands.append(d)
    return cands

# domain 名 → kb_retriever 的 domain key
DOMAIN_MAP = {
    "product_info": "product",
    "technical_support": "tech",
    "billing": "billing",
    "complaint": "complaint",
    "general_inquiry": "general",
    "out_of_scope": "general",
}

retriever = get_retriever()

print(f"=== 检索层专项复现：{len(FAILURES)} 个失败轮 ===")
print(f"阈值 SCORE_THRESHOLD={SCORE_THRESHOLD}, TOP_K={TOP_K}\n")

for cid, user, actual, expect in FAILURES:
    print(f"--- {cid}「{user}」 评估分类={actual} 金标期望={expect} ---")
    for dom in candidate_domains(actual, expect):
        key = DOMAIN_MAP[dom]
        # 精确预检
        exact = retriever._exact_hits(key, user)
        # BM25
        bm25 = retriever._bm25_hits(key, user)
        # 原始分数
        index = retriever._indexes.get(key)
        q_toks = retriever._tokenize(user)
        scores = index.get_scores(q_toks) if (index and q_toks) else []
        entries = retriever._data.get(key, [])
        ranked = sorted(
            ((scores[i], entries[i].get("title", "")) for i in range(len(entries))),
            key=lambda x: x[0], reverse=True,
        )[:3]
        exact_titles = [e.get("title", "") for e in exact]
        bm25_titles = [e.get("title", "") for e in bm25]
        print(f"  [域 {key}]")
        print(f"    精确命中: {exact_titles if exact else '无'}")
        print(f"    模糊命中: {bm25_titles if bm25 else '无'}")
        if ranked:
            print(f"    原始分数Top3: {[(t, round(s,3)) for s,t in ranked]}")
        else:
            print(f"    原始分数: 无（tokenize 后为空）")
    print()

print("=== 决策判定 ===")
for cid, user, actual, expect in FAILURES:
    key = DOMAIN_MAP.get(expect, expect)
    exact = retriever._exact_hits(key, user)
    bm25 = retriever._bm25_hits(key, user)
    if exact:
        if actual == expect:
            verdict = "检索已解决(期望域精确命中, 评估分类域正确 → 应能 answer)"
        else:
            verdict = f"分类器错域(期望域有精确命中, 但评估分类到 {actual} 查错域)"
    elif bm25:
        verdict = "BM25 在期望域有命中(评估时分类错域牵连)"
    else:
        # 期望域都 miss，但要区分是阈值过滤还是真无覆盖
        index = retriever._indexes.get(key)
        q_toks = retriever._tokenize(user)
        scores = index.get_scores(q_toks) if (index and q_toks) else []
        entries = retriever._data.get(key, [])
        ranked = sorted(((scores[i], entries[i].get("title","")) for i in range(len(entries))), key=lambda x:x[0], reverse=True)
        top_score = ranked[0][0] if ranked else 0
        if ranked and top_score > 0:
            verdict = f"BM25 有候选但最高分 {round(top_score,3)} < 阈值 {SCORE_THRESHOLD}(阈值过滤)"
        else:
            verdict = "keywords 覆盖不足(期望域无任何命中)"
    print(f"{cid}: {verdict}")