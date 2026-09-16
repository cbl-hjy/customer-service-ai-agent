#!/usr/bin/env python3
"""修复 KB v2 生成事故：重新生成异常条目（content 截断/keywords 空/category 缺）。

策略：对每条异常条目，用 LLM 单条重新生成（小 prompt，专治截断），
替换批次文件中的原条目，再重新合并 v2。

用法：python eval/kb_v2/repair_bad_entries.py
"""
import json
import os
import sys
import time

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import OPENAI_API_KEY, OPENAI_BASE_URL  # noqa: E402
import requests  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join(HERE, "batches")
OUT_KB = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")  # 单源化：直接写生产 KB（canonical=data/）
MODEL = "qwen3.7-max-2026-05-20"

REPAIR_PROMPT = """你是资深企业客服知识库工程师，为电商企业『星环数码』撰写一条客服知识库条目。

条目标题：{title}
所属域：{domain}，类别：{category}

要求：
1. content 必须包含可核对的原子事实（症状/原因/解决步骤），150-300 字
2. keywords 是检索关键词列表：4-8 个用户可能使用的说法
3. 输出严格 JSON 对象：{{"title": "...", "content": "...", "keywords": [...], "category": "..."}}
4. title 必须与给定标题完全一致
不要输出任何 JSON 之外的文字。"""


def call_llm(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你输出严格 JSON，不输出其他文字。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(5):
        try:
            resp = requests.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=300)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"  重试 {attempt + 1}: {e}")
            time.sleep(5 * (2 ** attempt))
    raise RuntimeError("LLM 调用失败")


def parse_obj(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    return data


def is_bad_entry(e: dict) -> bool:
    """异常判定：content 过短 / keywords 空 / category 缺 → 需修复。"""
    kws = e.get("keywords") or []
    cat = e.get("category") or ""
    cl = len(e.get("content", ""))
    return not (kws and cat and cl >= 20)


def main() -> None:
    repaired = 0
    for fname in sorted(os.listdir(BATCH_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(BATCH_DIR, fname)
        data = json.load(open(path, encoding="utf-8"))
        domain = data.get("domain", "")
        category = data.get("category", "")
        entries = data.get("entries", [])
        changed = False
        for i, e in enumerate(entries):
            if not is_bad_entry(e):
                continue  # 正常条目跳过
            title = e.get("title", "")
            print(f"[repair] {domain}/{category} #{i} {title}（content={len(e.get('content', ''))}, kws={len(e.get('keywords') or [])}）")
            raw = call_llm(REPAIR_PROMPT.format(title=title, domain=domain, category=category))
            new = parse_obj(raw)
            # 校验
            if new.get("title") != title:
                print(f"  WARN: 修复后 title 变了: {new.get('title')}")
            if not new.get("content") or not new.get("keywords"):
                print(f"  FAIL: 修复仍缺字段: {json.dumps(new, ensure_ascii=False)[:200]}")
                continue
            entries[i] = new
            changed = True
            repaired += 1
            time.sleep(1)
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"domain": domain, "category": category, "entries": entries}, f, ensure_ascii=False, indent=1)
            print(f"  ✓ 已写回 {fname}")

    # 重新合并 v2
    skeleton = json.load(open(os.path.join(HERE, "skeleton.json"), encoding="utf-8"))
    merged = {}
    for dom in skeleton:
        if dom.startswith("_"):
            continue
        merged[dom] = []
        for cat in skeleton[dom]:
            bf = os.path.join(BATCH_DIR, f"{dom}__{cat}.json")
            if os.path.exists(bf):
                merged[dom].extend(json.load(open(bf, encoding="utf-8"))["entries"])
    with open(OUT_KB, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    total = sum(len(v) for v in merged.values())
    print(f"\n修复 {repaired} 条，重新合并 → {OUT_KB}，共 {total} 条")


if __name__ == "__main__":
    main()
