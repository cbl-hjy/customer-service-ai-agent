#!/usr/bin/env python3
"""golden set 预填草稿生成：合并 ECD 种子 + golden_multi_turn 真实问法，逐条给建议。

对每条 query：
- 用 KB v1 跨 5 域 BM25 检索，输出 top-3 候选（含分数）供标注参考
- 预判 difficulty：有字面命中→easy；有语义候选（0.05~阈值）→hard；无候选→hard（待 v2 补条目）
- 预判保留建议：KB 有对应→保留；无对应→DROP 标记（用户确认）

用法：python eval/golden_retrieval/prefill_golden.py
输出：eval/golden_retrieval/golden_retrieval_prefill.csv（UTF-8 BOM，Excel 友好）
"""
import csv
import json
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from kb_retriever import get_retriever, SCORE_THRESHOLD  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, "ecd_seeds.csv")
GOLDEN_MT = os.path.join(PACKAGE_ROOT, "eval", "golden_multi_turn.json")
OUT = os.path.join(HERE, "golden_retrieval_prefill.csv")

DOMAINS = ["product", "tech", "billing", "general", "complaint"]


def search_all(retriever, q: str):
    """跨 5 域检索，返回 (score, domain, title) 排序列表（前 6）。"""
    best = []
    for dom in DOMAINS:
        index = retriever._indexes.get(dom)
        if index is None:
            continue
        q_toks = retriever._tokenize(q)
        if not q_toks:
            continue
        scores = index.get_scores(q_toks)
        entries = retriever._data.get(dom, [])
        ranked = sorted(
            ((scores[i], entries[i].get("title", "")) for i in range(len(entries))),
            key=lambda x: x[0], reverse=True,
        )[:3]
        for s, t in ranked:
            best.append((round(s, 3), dom, t))
    best.sort(key=lambda x: x[0], reverse=True)
    return best[:6]


def classify_suggest(top_score: float):
    """单条 query 的难度预判 + 保留建议（按跨域 top-1 分数）。

    >= SCORE_THRESHOLD(0.30) → easy + KEEP（有字面命中候选）
    0.05 ~ 0.30 → hard + KEEP（有语义候选，需人工从候选中选条目）
    < 0.05 → hard + DROP?（KB v1 无对应，待 v2 补条目后重评）
    """
    if top_score >= SCORE_THRESHOLD:
        return "easy", "KEEP"
    if top_score >= 0.05:
        return "hard", "KEEP"
    return "hard", "DROP?"


def main() -> None:
    retriever = get_retriever()
    queries = {}  # query -> {"source": [...], "hint": set}

    # 1) ECD 种子
    for r in csv.DictReader(open(SEEDS, encoding="utf-8")):
        q = r["query"]
        queries.setdefault(q, {"source": set(), "hint": set()})
        queries[q]["source"].add("ecd")
        queries[q]["hint"].add(r["domain_hint"])

    # 2) golden_multi_turn 真实问法
    if os.path.exists(GOLDEN_MT):
        gmt = json.load(open(GOLDEN_MT, encoding="utf-8"))
        for case in gmt.get("cases", []):
            for turn in case.get("turns", []):
                q = turn.get("user", "")
                if not q or q in queries:
                    continue
                queries.setdefault(q, {"source": set(), "hint": set()})
                queries[q]["source"].add("golden_mt")
                hint = turn.get("label_hint", "")
                queries[q]["hint"].add(hint)

    # 3) 生成预填行
    rows = []
    for q, meta in sorted(queries.items()):
        best = search_all(retriever, q)
        top_score = best[0][0] if best else 0.0
        diff, keep = classify_suggest(top_score)

        # 候选条目（格式：域:标题(分数)）
        cand = " | ".join(f"{d}:{t}({s})" for s, d, t in best) if best else "无候选"

        source = "+".join(sorted(meta["source"]))
        hint = ",".join(sorted(meta["hint"]))
        rows.append({
            "query": q,
            "domain": hint,           # 待用户确认
            "expected_titles": "",    # 待用户填（从候选里选）
            "difficulty": diff,       # 预判，可改
            "source": source,
            "suggest": keep,
            "candidates": cand,
        })

    # 4) 输出（UTF-8 BOM，Excel 打开不乱码）
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_keep = sum(1 for r in rows if r["suggest"] == "KEEP")
    n_drop = sum(1 for r in rows if r["suggest"] == "DROP?")
    n_easy = sum(1 for r in rows if r["difficulty"] == "easy")
    n_hard = sum(1 for r in rows if r["difficulty"] == "hard")
    print(f"预填完成：{len(rows)} 条 → {OUT}")
    print(f"  建议保留 {n_keep}，建议删除 {n_drop}（KB v1 无对应，待 v2 补条目后重评）")
    print(f"  难度预判：easy {n_easy} / hard {n_hard}")


if __name__ == "__main__":
    main()
