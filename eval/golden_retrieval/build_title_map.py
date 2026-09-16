#!/usr/bin/env python3
"""v1 → v2 条目标题映射表生成（金标集迁移前置）。

对 v1 每个标题，在 v2 同域内找最相似标题（词重叠 + keywords 重叠），
输出候选映射供人工确认。匹配不上的一并列出（需人工指定）。

用法：python eval/golden_retrieval/build_title_map.py
输出：eval/golden_retrieval/title_map_v1_to_v2.json
"""
import json
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(PACKAGE_ROOT, "data", "knowledge_base.json")
V2 = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")  # 单源化：canonical=data/
OUT = os.path.join(HERE, "title_map_v1_to_v2.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize_cn(text: str) -> set:
    """中文粗切：按 2-gram 集合近似（标题短，够用）。"""
    s = text.replace(" ", "")
    if not s:
        return set()
    if len(s) <= 2:
        return {s}
    return {s[i:i + 2] for i in range(len(s) - 1)}


def similarity(a: str, b: str) -> float:
    ta, tb = tokenize_cn(a), tokenize_cn(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(len(ta), len(tb))


def main() -> None:
    v1 = json.load(open(V1, encoding="utf-8"))
    v2 = json.load(open(V2, encoding="utf-8"))
    mapping = {}
    unmatched = []
    for dom, entries in v1.items():
        if dom.startswith("_"):
            continue
        v2_entries = v2.get(dom, [])
        v2_titles = [e["title"] for e in v2_entries]
        v2_keywords = {e["title"]: set(e.get("keywords", [])) for e in v2_entries}
        for e in entries:
            t = e["title"]
            # 1) 标题完全一致
            if t in v2_titles:
                mapping[t] = {"v2_title": t, "method": "exact", "domain": dom}
                continue
            # 2) 标题相似度
            best, best_score = None, 0.0
            for t2 in v2_titles:
                s = similarity(t, t2)
                if s > best_score:
                    best, best_score = t2, s
            # 3) keywords 重叠补充
            kw_overlap_best = None
            v1_kws = set(e.get("keywords", []))
            if v1_kws:
                best_kw, best_kw_score = None, 0.0
                for t2, kws in v2_keywords.items():
                    inter = len(v1_kws & kws)
                    if inter > best_kw_score:
                        best_kw, best_kw_score = t2, inter
                if best_kw_score >= 2:
                    kw_overlap_best = best_kw
            if best_score >= 0.4 or kw_overlap_best:
                chosen = kw_overlap_best if kw_overlap_best else best
                mapping[t] = {"v2_title": chosen, "method": "fuzzy", "domain": dom,
                              "score": round(best_score, 2), "kw_match": bool(kw_overlap_best)}
            else:
                unmatched.append({"v1_title": t, "domain": dom, "best_fuzzy": best,
                                  "best_score": round(best_score, 2)})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping, "unmatched": unmatched}, f, ensure_ascii=False, indent=1)
    print(f"映射完成：{len(mapping)} 条匹配（exact {sum(1 for m in mapping.values() if m['method']=='exact')} + fuzzy {sum(1 for m in mapping.values() if m['method']=='fuzzy')}）")
    print(f"未匹配 {len(unmatched)} 条（需人工指定）：")
    for u in unmatched:
        print(f"  [{u['domain']}] {u['v1_title']} → 最近 {u['best_fuzzy']} ({u['best_score']})")


if __name__ == "__main__":
    main()
