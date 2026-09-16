"""分类层专项复现：对 v3 真实语料评估暴露的 4 个"分类器错域"case，量化分类稳定性。

背景：检索层专项复现已证明 mt-204/209/214/215 在期望域本有精确命中，失败根因是
分类器把 query 路由到错误 domain（查错域 → 检索 miss → 升级兜底）。本脚本只测
classify_query 的分类结果，不走检索/生成，量化每个 case 的失败率，区分稳定缺陷 vs 抖动。

用法：python eval/eval_classifier_repro.py [--runs N]
"""
import os
import sys
import argparse
from collections import Counter

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

# 进程隔离：评估用独立 DB，不影响线上
EVAL_DB = os.path.join(PACKAGE_ROOT, "eval", "eval_data", "checkpoints_eval.db")
os.environ["CHECKPOINT_DB_PATH"] = EVAL_DB

from tools import classify_query  # noqa: E402
from multi_agent_customer_service import get_llm  # noqa: E402

# (case_id, 失败轮用户输入, 金标期望域, 评估实测域[复现脚本记录])
CASES = [
    ("mt-204", "顺便问下现在有什么优惠券吗", "general_inquiry", "product_info"),
    ("mt-209", "算了还是留着吧，就是屏幕有点花", "technical_support", "complaint"),
    ("mt-214", "朋友买的那款降噪耳机好用吗", "product_info", "general_inquiry"),
    ("mt-215", "分期手续费怎么算的", "billing", "product_info"),
]

# 每个 case 期望域对应可接受的 label 集合（容忍合理等价）
ACCEPT = {
    "mt-204": {"general_inquiry"},          # 优惠券 → 会员服务(general)
    "mt-209": {"technical_support"},        # 屏幕花 → 屏幕问题(tech)
    "mt-214": {"product_info"},             # 降噪耳机 → 耳机(product)
    "mt-215": {"billing"},                  # 分期手续费 → 分期付款(billing)
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="每个 case 重复次数")
    args = parser.parse_args()
    runs = max(1, args.runs)

    llm = get_llm()
    if llm is None:
        print("❌ 无 LLM，无法复现（需 API key）")
        sys.exit(1)

    print(f"=== 分类层专项复现：{len(CASES)} case × {runs} 次 ===")
    print(f"LLM: {getattr(llm, 'model_name', 'qwen3.7-plus-2026-05-26')}\n")

    for cid, query, expect, last_eval in CASES:
        dist = Counter()
        for _ in range(runs):
            try:
                res = classify_query.invoke({"query": query, "llm": llm})
                import json as _json
                label = _json.loads(res).get("label")
            except Exception as e:
                print(f"  [调用失败] {e}")
                label = "ERROR"
            dist[label] += 1
        ok = sum(dist[l] for l in ACCEPT[cid])
        rate = ok / runs
        status = "✅ 归零" if ok == runs else ("⚠️ 部分失败" if ok > 0 else "❌ 稳定错域")
        print(f"--- {cid}「{query}」 期望={expect} 上次评估={last_eval} ---")
        for label, cnt in dist.most_common():
            print(f"    {label}: {cnt}/{runs}")
        print(f"    期望域命中率: {rate:.0%}  →  {status}\n")


if __name__ == "__main__":
    main()