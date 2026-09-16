#!/usr/bin/env python3
"""确定性验证：补的以旧换新/自提柜两条目在 general 域能否命中，OOV 不误命中。"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from kb_retriever import KBRetriever

kr = KBRetriever()
cases = [
    # (查询, 期望命中标题, 说明)
    ("旧耳机能以旧换新吗，怎么操作", "以旧换新", "ho4-05 修复"),
    ("可以送到楼下自提柜吗，还是必须本人签收", "自提柜与签收", "ho4-06 修复"),
    ("旧平板能抵钱换新的吗", "以旧换新", "以旧换新变体"),
    ("快递放快递柜还是签收", "自提柜与签收", "自提柜变体"),
    # OOV 负例：不应误命中这两个新条目（但可能命中其他条目，只检查不强绑）
    ("推荐一部好看的科幻电影吧", None, "越界负例"),
]
for q, expect_title, note in cases:
    text = kr.retrieve("general", q)
    hit = [t for t in ("以旧换新", "自提柜与签收") if t in text]
    if expect_title is None:
        print(f"[{note}] query={q!r} -> 命中={hit}")
        continue
    ok = expect_title in text
    print(f"[{'PASS' if ok else 'FAIL'}] [{note}] query={q!r} -> 命中={hit}")