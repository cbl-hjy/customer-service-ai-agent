#!/usr/bin/env python3
"""从 ECD sample test 集抽取真实用户问法种子，按本项目 5 域归类。

ECD 格式：label \t u1 \t r1 \t u2 \t r2 ...（tab 分隔，已分词空格隔开）
客服口语特征词（'亲'/'客官'/'小店'/以'呢'/'哦'结尾）区分角色。
输出：eval/golden_retrieval/ecd_seeds.csv —— 真实问法种子，供 golden set 标注与扩库生成复用。

用法：python eval/golden_retrieval/extract_ecd_seeds.py
"""
import csv
import os
import re
import sys

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "ecd-corpus", "ECD_sample", "test")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecd_seeds.csv")

# 客服口语特征：含称呼词 / 敬语（仅强信号，见 is_agent_utterance 注释）
AGENT_MARKERS = ["亲", "客官", "小店", "宝贝", "掌柜", "本店"]
TONE_END = ()  # 保留占位：语气词结尾不判客服（会误杀用户问句）

def is_agent_utterance(u: str) -> bool:
    """强信号判客服：称呼词 / 敬语。语气词结尾不再判客服——'呢'也是用户问句高频词，
    弱信号判客服会系统性丢问法（recall 优先，边缘话术混入种子由人工标注兜底）。"""
    if any(m in u for m in AGENT_MARKERS):
        return True
    # 客服常见话术特征（u 已去空格，模式同步去空格）
    if re.search(r"帮您|为您|给亲|给您|您的", u):
        return True
    return False

def _is_greeting(u: str) -> bool:
    """寒暄/无信息量话语。"""
    if len(u) < 4:
        return True
    if u in {"嗯嗯", "亲", "在", "好的", "好", "嗯", "哦", "谢谢", "嗯嗯嗯", "没事", "您好",
             "你好", "在吗", "在的", "嗯好的", "好的好的", "哦哦", "好吧", "行", "可以",
             "可以的", "知道了", "明白了", "嗯嗯嗯嗯", "哦哦哦", "亲在", "亲在吗", "哈喽"}:
        return True
    return False

QUESTION_MARKS = ("吗", "呢", "怎么", "什么", "哪", "为什么", "能", "可以", "多少", "几", "？", "?")

def domain_of(q: str) -> str:
    """按本项目 KB 域关键词粗分（种子清单用，标注时人工校正）。"""
    billing = ["退", "款", "邮费", "运费", "发票", "开票", "税", "分期", "支付", "改价", "价格", "钱", "优惠券", "满减"]
    tech = ["坏", "碎", "漏", "断", "故障", "开不了机", "卡", "充不进", "没声音", "黑屏", "响", "维修", "保修", "电池", "充电"]
    complaint = ["投诉", "差评", "态度", "慢", "气愤", "失望", "不理", "没人理", "气死", "骗", "假"]
    product = ["耳机", "手机", "电脑", "平板", "配件", "型号", "颜色", "配置", "推荐", "怎么样", "好用", "适合"]
    general = ["发货", "快递", "物流", "签收", "地址", "改地址", "到货", "时效", "营业", "几点", "电话", "收藏", "活动", "赠", "送", "满", "会员", "以旧换新", "补发", "少件", "缺货"]
    for d, ws in [("billing", billing), ("tech", tech), ("complaint", complaint), ("product", product), ("general", general)]:
        if any(w in q for w in ws):
            return d
    return "general"

def _pick_best_query(user_qs: list) -> str:
    """从用户问法里选最有信息量的一条：含疑问词且长度>=4 优先，其次长度>=4，最后保底取第一条。"""
    candidates = [u for u in user_qs if len(u) >= 4]
    if not candidates:
        return user_qs[0] if user_qs else ""
    for u in candidates:
        if any(m in u for m in QUESTION_MARKS):
            return u
    return max(candidates, key=len)

def main() -> None:
    if not os.path.exists(CORPUS):
        print(f"语料不存在: {CORPUS}")
        sys.exit(1)
    seeds = []
    with open(CORPUS, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            utts = parts[1:-1]  # 去 label 和 response
            # 用户问法 = 非客服 utterance（去掉空格还原）
            user_qs = []
            for u in utts:
                u_clean = u.replace(" ", "")
                if not u_clean:
                    continue
                if not is_agent_utterance(u_clean):
                    user_qs.append(u_clean)
            if not user_qs:
                continue
            q = _pick_best_query(user_qs)
            if not q or _is_greeting(q):
                continue
            dom = domain_of(q)
            seeds.append((f"ecd-test-{idx}", q, dom))

    # 去重（同问法保留第一个）
    seen, dedup = set(), []
    for sid, q, dom in seeds:
        if q in seen:
            continue
        seen.add(q)
        dedup.append((sid, q, dom))

    # 按域分组统计
    from collections import Counter
    cnt = Counter(d for _, _, d in dedup)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "domain_hint", "note"])
        for sid, q, dom in dedup:
            w.writerow([sid, q, dom, "ECD test 真实问法（原域为母婴/食品，需重写实体到电子品类）"])
    print(f"提取 {len(dedup)} 条去重问法 → {OUT}")
    for d, n in cnt.most_common():
        print(f"  {d}: {n}")

if __name__ == "__main__":
    main()
