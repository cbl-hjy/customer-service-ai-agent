#!/usr/bin/env python3
"""KB v2 生成管线：按骨架锚点 + 一致性事实表，用 qwen3.7-max 批量生成企业客服知识库。

设计：
- 锚点由人工定义（skeleton.json），LLM 只填充 content/keywords/models → 覆盖度可控
- 一致性事实表（policy_facts.json）注入每批 prompt → 政策数字跨批次不漂移
- response_format=json_object 强约束结构化输出
- 断点续跑：已完成的批次文件存在则跳过（幂等）
- 难度梯度：tech/产品故障类 keywords 只写字面词（不写口语同义改写）→ 为稠密检索留区分度

用法：python eval/kb_v2/generate_kb.py [--batch domain/category] [--dry-run]
"""
import argparse
import concurrent.futures
import json
import os
import random
import sys
import time

import requests

# Windows 控制台 GBK 兜底：强制 UTF-8 输出（中文 + 符号安全）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)

from config import OPENAI_API_KEY, OPENAI_BASE_URL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SKELETON = os.path.join(HERE, "skeleton.json")
FACTS = os.path.join(HERE, "policy_facts.json")
BATCH_DIR = os.path.join(HERE, "batches")
OUT_KB = os.path.join(HERE, "knowledge_base_v2.json")

MODEL = "qwen3.7-max-2026-05-17"
MAX_RETRIES = 5
# 并发批次数：批次间无依赖（独立生成/落盘/合并），并发不改变单批内容与成本。
# 429 限流由 call_llm 重试+退避兜底。2026-08-16 扩容提速 4→8：
# 每批为独立 LLM 调用（质量与并发无关），限流时退避自动节流不丢任务。
CONCURRENCY = 8

# keywords 只写字面词（不留口语同义改写）的域 → 制造检索难度梯度
HARD_KEYWORDS_DOMAINS = {"tech", "complaint"}

SYSTEM_PROMPT = """你是一名资深企业客服知识库工程师，为电商企业『星环数码』撰写客服知识库条目。
要求：
1. content 必须包含可核对的原子事实（数字/时限/条件/例外），面向真实客服对话场景
2. 结构：规则 → 例外/边界 → 时限（政策类条目）；症状 → 原因 → 解决步骤（故障类条目）
3. keywords 是检索关键词列表：必须是用户可能使用的说法；每条 4-8 个
4. 严格遵守下面给出的一致性事实表，数字不得与事实表冲突
5. 输出严格 JSON 数组，每个元素含 title/content/keywords/category 四个字段
6. 不要编造事实表中未定义的优惠政策数字；价格区间用合理模拟值"""


def build_user_prompt(domain: str, category: str, titles: list, facts: dict) -> str:
    """构造单批生成 prompt。"""
    # 事实表只注入相关部分：政策/物流/实体通用，产品/故障域的数值规则
    facts_json = json.dumps(facts, ensure_ascii=False, indent=1)
    kw_note = ""
    if domain in HARD_KEYWORDS_DOMAINS:
        kw_note = (
            "\n\n【本批 keywords 要求】只写与标题字面相关的词，不要写口语同义改写"
            "（例如标题『耳机单侧无声』的 keywords 写 [单侧无声, 一边没声音] 即可，"
            "不要写 [左耳不响, 右耳听不见] 这类未提供的说法）——保持检索的自然难度梯度。"
        )
    return f"""为以下 {len(titles)} 个条目标题各撰写一条知识库条目。
所属域：{domain}，类别：{category}。

【一致性事实表（必须遵守，数字不得冲突）】
{facts_json}

【条目标题】
{json.dumps(titles, ensure_ascii=False, indent=1)}

输出 JSON 数组，每个元素：
{{"title": "条目标题（必须与给定标题完全一致）", "content": "条目内容（含可核对事实）", "keywords": ["检索词1", "检索词2", ...], "category": "{category}"}}
{kw_note}
不要输出任何 JSON 之外的文字。"""


def call_llm(prompt: str) -> str:
    """调用 dashscope compatible-mode，重试 + 指数退避。"""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=600)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  429 限流，等待 {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 5 * (2 ** attempt) + random.uniform(0, 2)
            print(f"  第 {attempt + 1} 次失败: {e}，等待 {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM 调用失败（重试耗尽）: {last_err}")


def parse_entries(raw: str) -> list:
    """解析 LLM 输出：容忍 markdown 代码块包裹。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取最外层数组/对象
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            raise
        data = json.loads(text[start:end + 1])
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    # 容错：json_object 模式下 LLM 偶发输出单个对象（含 title 字段）而非数组 → 包成数组
    if isinstance(data, dict) and "title" in data:
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"LLM 输出非数组: {str(data)[:200]}")
    return data


def generate_batch(domain: str, category: str, titles: list, facts: dict) -> list:
    """生成一批条目并校验标题对齐。"""
    prompt = build_user_prompt(domain, category, titles, facts)
    raw = call_llm(prompt)
    entries = parse_entries(raw)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"解析失败: {raw[:200]}")

    # 标题对齐校验：LLM 必须按锚点生成
    title_map = {t: None for t in titles}
    for e in entries:
        t = e.get("title", "")
        if t in title_map and title_map[t] is None:
            title_map[t] = e
    missing = [t for t, e in title_map.items() if e is None]
    if missing:
        # 容错：若 LLM 微调了标题（加序号等），做包含匹配
        for m in missing:
            for e in entries:
                if m in e.get("title", "") or e.get("title", "") in m:
                    title_map[m] = e
                    break
        missing = [t for t, e in title_map.items() if e is None]
    if missing:
        raise ValueError(f"标题缺失 {len(missing)} 个: {missing[:5]}")
    return [title_map[t] for t in titles]


def batch_file(domain: str, category: str, part: int) -> str:
    """part 0 沿用旧命名（兼容已生成并评估过的批次文件），part≥1 加序号。"""
    base = f"{domain}__{category.replace('/', '_')}"
    name = base if part == 0 else f"{base}__part{part}"
    return os.path.join(BATCH_DIR, f"{name}.json")


def covered_anchors(batch_dir: str, domain: str, category: str, titles: list) -> int:
    """已有 part0（基础名）文件覆盖的锚点前缀长度。

    仅当文件条目标题与 skeleton 锚点前缀完全一致（顺序含）时视为已覆盖，
    否则返回 0（该类别全部重新生成）。扩容场景的锚点只能追加到类别末尾，
    不得插入中间——否则前缀校验失败触发全量重生成。
    """
    f = os.path.join(batch_dir, f"{domain}__{category.replace('/', '_')}.json")
    if not os.path.exists(f):
        return 0
    entries = json.load(open(f, encoding="utf-8"))["entries"]
    file_titles = [e.get("title", "") for e in entries]
    n = len(file_titles)
    return n if file_titles == titles[:n] else 0


def build_plan(skeleton: dict, batch_dir: str) -> list:
    """批次计划：对已有类别从「已覆盖前缀之后」切片（part≥1），新类别从 part 0 开始。

    2026-08-16 修复（扩容配套）：此前固定按 15 条切片，旧类别锚点 <15 时
    新增锚点会落入 part0 范围但旧文件不会重生成 → 合并时静默丢失。
    """
    plan = []  # (domain, category, titles_chunk, part)
    for domain, cats in skeleton.items():
        if domain.startswith("_"):
            continue
        for category, titles in cats.items():
            covered = covered_anchors(batch_dir, domain, category, titles)
            if covered == len(titles):
                continue  # 全部已生成
            if covered:
                rest = titles[covered:]
                for i in range(0, len(rest), 15):
                    plan.append((domain, category, rest[i:i + 15], i // 15 + 1))
            else:
                for i in range(0, len(titles), 15):
                    plan.append((domain, category, titles[i:i + 15], i // 15))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="只跑指定批次，格式 domain/category")
    ap.add_argument("--dry-run", action="store_true", help="只打印批次计划不调用")
    args = ap.parse_args()

    if not OPENAI_API_KEY:
        print("错误：OPENAI_API_KEY 未配置")
        sys.exit(1)

    skeleton = json.load(open(SKELETON, encoding="utf-8"))
    facts = json.load(open(FACTS, encoding="utf-8"))
    os.makedirs(BATCH_DIR, exist_ok=True)

    plan = build_plan(skeleton, BATCH_DIR)

    if args.dry_run:
        n_anchors = sum(len(t) for _, _, t, _ in plan)
        n_covered = sum(
            covered_anchors(BATCH_DIR, d, c, skeleton[d][c])
            for d in skeleton if not d.startswith("_") for c in skeleton[d]
        )
        print(f"批次计划：共 {len(plan)} 批 / {n_anchors} 条待生成锚点；已有前缀 {n_covered} 条复用")
        for domain, category, titles, part in plan:
            print(f"  {domain}/{category}#part{part}: {len(titles)} 条")
        return

    results = {}
    # 并发执行批次（每批独立文件，线程安全）
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {}
        for domain, category, titles, part in plan:
            batch_id = f"{domain}/{category}"
            if args.batch and args.batch != batch_id:
                continue
            out_file = batch_file(domain, category, part)
            if os.path.exists(out_file):
                print(f"[skip] {batch_id}#part{part} 已存在")
                results[(batch_id, part)] = json.load(open(out_file, encoding="utf-8"))
                continue
            futures[pool.submit(generate_batch, domain, category, titles, facts)] = (batch_id, part, out_file)

        failed = []
        for fut in concurrent.futures.as_completed(futures):
            batch_id, part, out_file = futures[fut]
            try:
                entries = fut.result()
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump({"domain": batch_id.split("/")[0], "category": batch_id.split("/")[1],
                               "entries": entries}, f, ensure_ascii=False, indent=1)
                results[(batch_id, part)] = entries
                print(f"[ok  ] {batch_id}#part{part}（{len(entries)} 条）", flush=True)
            except Exception as e:  # noqa: BLE001
                # 2026-08-16 修复：单批失败不再 raise（会丢弃其余 in-flight 批次结果）。
                # 记录后继续，末尾汇总失败批次并退出；重跑幂等补生成。
                failed.append(f"{batch_id}#part{part}: {e}")
                print(f"[FAIL] {batch_id}#part{part}: {e}", flush=True)
        if failed:
            print(f"\n{len(failed)} 个批次失败，未执行合并。重跑同一命令即可幂等补生成：")
            for msg in failed:
                print(f"  - {msg}")
            sys.exit(1)

    # 合并 v2：基础文件（已验证前缀）+ part1..N 顺序聚合；严格校验锚点全覆盖
    merged = {}
    problems = []
    for domain in skeleton:
        if domain.startswith("_"):
            continue
        merged[domain] = []
        for category, titles in skeleton[domain].items():
            entries = []
            if covered_anchors(BATCH_DIR, domain, category, titles):
                entries.extend(json.load(open(batch_file(domain, category, 0), encoding="utf-8"))["entries"])
            part = 1
            while os.path.exists(batch_file(domain, category, part)):
                entries.extend(json.load(open(batch_file(domain, category, part), encoding="utf-8"))["entries"])
                part += 1
            got = [e.get("title", "") for e in entries]
            if got != titles:
                missing = [t for t in titles if t not in got]
                extra = [t for t in got if t not in titles]
                problems.append(
                    f"{domain}/{category}: 锚点 {len(titles)} vs 合并 {len(got)}"
                    f"（缺 {len(missing)}: {missing[:3]}...；多 {len(extra)}）"
                )
            merged[domain].extend(entries)
    if problems:
        for p in problems:
            print(f"[MERGE-FAIL] {p}")
        sys.exit(1)
    with open(OUT_KB, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    total = sum(len(v) for v in merged.values())
    print(f"\n合并完成：{OUT_KB}，共 {total} 条（锚点校验全部通过）")


if __name__ == "__main__":
    main()
