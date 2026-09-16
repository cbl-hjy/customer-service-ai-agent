#!/usr/bin/env python3
"""金标集 v1 → v2 迁移：按 query 语义将 expected_titles 映射到 v2 具体条目。

v1 是类别级标题（如"声音问题"），v2 是品类×故障级（如"耳机单侧无声"），
必须按每条 query 的实际语义选 v2 条目，不能机械映射。

映射表 MIGRATE：query → (domain, [v2_titles], difficulty)
- 未列出的 query：保持原样（v1 标题在 v2 中同名存在，或无需迁移）
用法：python eval/golden_retrieval/migrate_golden_to_v2.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_retrieval_golden.csv")
V2_KB = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "knowledge_base_v2.json")  # 单源化：canonical=data/
OUT = os.path.join(HERE, "golden_retrieval_golden_v2.csv")

# query → (domain, v2_titles 分号分隔, difficulty)
MIGRATE = {
    # ---- general：类别 → 具体 ----
    "什么时候发货啊": ("general", "发货时效", "easy"),
    "2拍下什么时候发货": ("general", "发货时效", "easy"),
    "可以78号在发货吗": ("general", "发货时效", "easy"),
    "能早点发吗，我下周就要用": ("general", "发货时效", "hard"),
    "那现在下单能明天发货吗": ("general", "发货时效", "hard"),
    "几天可以到": ("general", "物流单号查询", "hard"),
    "三号可以到吗大概": ("general", "物流单号查询", "hard"),
    "后天能到吗": ("general", "物流单号查询", "hard"),
    "好的石家庄几天到": ("general", "物流单号查询", "hard"),
    "那几天能到呢": ("general", "物流单号查询", "hard"),
    "那大概最快能多久啊因为我比较着急": ("general", "物流单号查询", "hard"),
    "那到底是发什么快递？": ("general", "快递选择", "easy"),
    "上次发的快递太慢了，这次能换个快递吗": ("general", "快递选择", "easy"),
    "我不想发邮政能换别的么": ("general", "快递选择", "hard"),
    "我去不早说发韵达能到我家那儿我就能拿到": ("general", "快递选择", "hard"),
    "今天不到退了什么鬼快递": ("general", "物流信息更新延迟", "hard"),
    "怎么还没物流信息你这是要拖到快递放假再来给我发吗": ("general", "物流信息更新延迟", "ambiguous"),
    "真不知道这样快递搞什么小区有自取箱不给我放非给放私人那": ("general", "自提柜取件", "hard"),
    "发前一个地址": ("general", "发货后修改地址", "easy"),
    "地址我可以改的您把地址给我下吧": ("general", "修改订单信息", "easy"),
    "我刚拍的耳机地址选错了，能改吗": ("general", "未发货改地址", "easy"),
    "订单地址填错了能改吗": ("general", "未发货改地址", "easy"),
    "活动期间不能改吗": ("general", "发货后修改地址", "hard"),
    "拍十份可以吗不同地址": ("general", "修改订单信息", "easy"),
    "怎么说第一次有漏发吗": ("general", "少件处理", "hard"),
    "收到的耳机少了一个配件，能补发吗": ("general", "补发流程", "easy"),
    "我买了两份送一份怎么就给我发了一盒什么情况": ("general", "少件处理;满赠活动规则", "hard"),
    "补发的运费谁出？": ("general", "补发流程", "hard"),
    "有货了能通知我吗": ("general", "缺货通知", "easy"),
    "耳机什么时候能补货": ("general", "缺货通知", "easy"),
    "收藏你们店铺有送什么吗": ("general", "收藏送礼活动", "easy"),
    "我已经关注了，赠品需要备注吗": ("general", "收藏送礼活动", "hard"),
    "会送东西吗": ("general", "满赠活动规则;收藏送礼活动", "ambiguous"),
    "有送什么吗": ("general", "满赠活动规则;收藏送礼活动", "ambiguous"),
    "买耳机送什么？是拍一个还是拍两个？": ("general", "满赠活动规则", "easy"),
    "然后你送一盒": ("general", "满赠活动规则", "hard"),
    "还另外送湿巾吗": ("general", "满赠活动规则", "hard"),
    "送的那罐一起拍下系统自动减价的哦": ("general", "满赠活动规则", "hard"),
    "满199可以减十元是吧还送啥": ("general", "满赠活动规则", "easy"),
    "现在买手机有满赠活动吗，送什么": ("general", "满赠活动规则;节日大促规则", "ambiguous"),
    "优惠耳机也参与满赠活动吗": ("general", "满赠活动规则;节日大促规则", "ambiguous"),
    "耳机不参与买二送一活动吗": ("general", "满赠活动规则;节日大促规则", "ambiguous"),
    "哦哦今天消费69不能送了是么": ("general", "满赠活动规则", "hard"),
    "今天做活动的是这种吗": ("general", "节日大促规则", "easy"),
    "你们家在聚划算还什么时候做活动呢": ("general", "节日大促规则", "easy"),
    "那哪些产品参与呢": ("general", "节日大促规则", "hard"),
    "534597270770拍了很多东西": ("general", "满赠活动规则", "hard"),
    "那老客户还是原价": ("general", "老客回馈", "hard"),
    "别啰嗦了，直接给我转人工客服！": ("general", "在线客服入口", "hard"),
    "帮我看看这个订单耳机一共订了几个": ("general", "订单查询", "easy"),
    "那换货一般要多久能到？": ("general", "退换货总流程", "hard"),
    "为什么有运费": ("general", "运费计算规则", "easy"),
    "买多少斤包邮": ("general", "包邮条件", "easy"),
    "发中通圆通吗": ("general", "快递选择", "easy"),
    "能发顺丰吗？我上次等太久了": ("general", "快递选择;顺丰加急", "hard"),
    "取消重新加双袜子重新拍你说不用": ("general", "订单取消", "hard"),
    "我前几天加入购物车了今天看失效了": ("general", "订单查询", "hard"),
    # ---- billing ----
    "那退款一般几天到账？": ("billing", "退款到账时效", "easy"),
    "怎么还要运费": ("billing", "退货运费承担", "easy"),
    "我要退货，运费谁出": ("billing", "退货运费承担", "easy"),
    "我退回去的邮费你们也承担吗": ("billing", "退货运费承担", "easy"),
    "那它退货运费谁出": ("billing", "退货运费承担", "easy"),
    "那退货运费谁出？": ("billing", "退货运费承担", "easy"),
    "能改下运费吗": ("billing", "退货运费承担", "easy"),
    "因为是江西但是我要寄到江苏的江西的运费要贵点呢": ("billing", "退货运费承担", "easy"),
    "刚买的耳机有一边突然没声音了，能退吗": ("billing", "质量问题退款;7天无理由退款", "hard"),
    "这个平板我要退了": ("billing", "7天无理由退款", "hard"),
    "那款能七天无理由退货吗": ("billing", "7天无理由退款", "easy"),
    "今天不到退了什么鬼快递": ("general", "物流信息更新延迟", "hard"),
    "可以开发票吗？我要公司抬头": ("billing", "发票抬头修改;电子发票开具", "ambiguous"),
    "纸质发票邮寄费谁出？": ("billing", "纸质发票开具", "easy"),
    "能开纸质发票吗": ("billing", "纸质发票开具", "easy"),
    "那退的这个能开票吗": ("billing", "电子发票开具", "easy"),
    "支持分期付款吗": ("billing", "分期付款", "easy"),
    "分期手续费怎么算的": ("billing", "分期付款", "easy"),
    "还是一次付款吧可以领5元优惠券呢": ("billing", "在线支付方式", "hard"),
    "你们地址发我退还你们怎么": ("billing", "退货运费承担", "hard"),
    "行退多少": ("billing", "退款金额计算", "hard"),
    "价格可以有优惠吗": ("billing", "会员价规则", "hard"),
    "顺便问下现在有什么优惠券吗": ("billing", "优惠券使用规则", "hard"),
    "我买5个单套合一个三层套合有优惠吗": ("billing", "批量采购折扣", "hard"),
    "我买的有点多有没有什么优惠": ("billing", "批量采购折扣", "hard"),
    # ---- tech：品类×故障 ----
    "好吧，那耳机没声音了怎么办": ("tech", "耳机单侧无声", "easy"),
    "耳机左边没声音，能换货吗": ("tech", "耳机单侧无声", "hard"),
    "按你说的做了，还是黑屏": ("tech", "手机黑屏", "easy"),
    "电脑开不了机怎么回事": ("tech", "电脑无法开机", "easy"),
    "算了，还是留着吧，就是屏幕有点花": ("tech", "手机屏幕碎裂", "easy"),
    # ---- product ----
    "就平时打电话刷视频用，哪款合适？": ("product", "星环 X1 标准版手机规格", "hard"),
    "想给家里老人买台手机，帮我推荐下": ("product", "星环 X1 Lite 入门手机规格", "easy"),
    "朋友买的那款降噪耳机好用吗": ("product", "星环 AirBuds 3 Pro 降噪耳机规格", "hard"),
    "那个平板多少钱来着": ("product", "星环 Pad 11 平板规格", "easy"),
    # ---- complaint ----
    "你们客服太敷衍了，气死我了，我要投诉！": ("complaint", "客服态度差投诉", "easy"),
    "你们的补偿标准是什么": ("complaint", "补偿标准说明", "easy"),
    "收到的耳机有瑕疵，要求补偿": ("complaint", "外观瑕疵投诉;补偿标准说明", "ambiguous"),
    "我打算让淘宝来介入我是也卖家淘宝的规则我懂我看你怎么收场我做淘宝11年了只有十分钟等你答复妈的B不发飚当老娘好欺负吗": ("complaint", "处理流程繁琐投诉", "hard"),
    "淘宝为什么没有传照片功能": ("complaint", "功能缺陷投诉", "hard"),
}


def main() -> None:
    kb = json.load(open(V2_KB, encoding="utf-8"))
    real = {}
    for dom, entries in kb.items():
        if entries:
            real[dom] = {e["title"] for e in entries}

    rows = list(csv.DictReader(open(GOLDEN, encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    migrated, unmigrated = 0, 0
    for r in rows:
        q = r["query"]
        if q in MIGRATE:
            dom, titles, diff = MIGRATE[q]
            # 校验 v2 title 存在
            missing = [t for t in titles.split(";") if t not in real.get(dom, set())]
            if missing:
                print(f"[WARN] {q}: v2 标题缺失 {missing}，跳过迁移")
                unmigrated += 1
                continue
            r["domain"] = dom
            r["expected_titles"] = titles
            r["difficulty"] = diff
            migrated += 1
        else:
            unmigrated += 1

    # 校验全量 v2 title 存在
    bad = [(r["query"], r["domain"], t) for r in rows
           for t in r["expected_titles"].split(";") if t and t not in real.get(r["domain"], set())]
    if bad:
        print(f"[ERR] 迁移后仍有 {len(bad)} 个 title 不在 v2 域内:")
        for q, d, t in bad[:10]:
            print(f"  [{d}] {t} (query: {q})")

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"迁移完成：{migrated} 条已映射，{unmigrated} 条未在映射表（保持原样）")
    print(f"输出：{OUT}（共 {len(rows)} 条）")


if __name__ == "__main__":
    main()
