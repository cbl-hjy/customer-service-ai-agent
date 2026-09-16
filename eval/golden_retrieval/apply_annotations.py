#!/usr/bin/env python3
"""将人工判断的 golden set 标注应用回预填 CSV。

数据源：本文件内嵌的 ANNOTATIONS（人工逐条判断，2026-08-16），
格式：query -> (domain, expected_titles 分号分隔, difficulty, suggest)
- suggest=DROP? 表示 KB v1 无对应条目，跳过（v2 补条目后重评）
- expected_titles 必须是 kb_v1 真实标题（域内，可含分号多标）

用法：python eval/golden_retrieval/apply_annotations.py
输出：eval/golden_retrieval/golden_retrieval_golden.csv（正式金标集 v1）
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFILL = os.path.join(HERE, "golden_retrieval_prefill.csv")
OUT = os.path.join(HERE, "golden_retrieval_golden.csv")

# (domain, expected_titles(;分隔), difficulty, suggest)
ANNOTATIONS = {
    "299的湿巾和399的湿巾有什么区别": ("general", "", "hard", "DROP?"),
    "2拍下什么时候发货": ("general", "物流跟踪", "easy", "KEEP"),
    "528941799061这两个有啥区别不都是大尺寸的么": ("general", "", "hard", "DROP?"),
    "534597270770拍了很多东西": ("general", "满赠规则", "hard", "KEEP"),
    "三号可以到吗大概": ("general", "物流跟踪", "hard", "KEEP"),
    "上次发的快递太慢了，这次能换个快递吗": ("general", "物流跟踪", "easy", "KEEP"),
    "不下水我怎么知道会不会起球变形对不齐呢": ("general", "", "hard", "DROP?"),
    "为什么有运费": ("billing", "", "hard", "DROP?"),
    "买五袋多少": ("general", "", "hard", "DROP?"),
    "买多少斤包邮": ("general", "", "hard", "DROP?"),
    "买耳机送什么？是拍一个还是拍两个？": ("general", "满赠规则", "easy", "KEEP"),
    "五包是什么包": ("general", "", "hard", "DROP?"),
    "什么时候发货啊": ("general", "物流跟踪", "easy", "KEEP"),
    "今天不到退了什么鬼快递": ("general", "物流跟踪", "hard", "KEEP"),
    "今天做活动的是这种吗": ("general", "活动参与规则", "easy", "KEEP"),
    "价格可以有优惠吗": ("billing", "", "hard", "DROP?"),
    "优惠耳机也参与满赠活动吗": ("general", "活动参与规则;满赠规则", "ambiguous", "KEEP"),
    "会送东西吗": ("general", "满赠规则;收藏送礼", "ambiguous", "KEEP"),
    "你们地址发我退还你们怎么": ("billing", "所需材料", "hard", "KEEP"),
    "你们客服太敷衍了，气死我了，我要投诉！": ("complaint", "服务态度差", "easy", "KEEP"),
    "你们家在聚划算还什么时候做活动呢": ("general", "活动参与规则", "easy", "KEEP"),
    "你们的补偿标准是什么": ("complaint", "补偿标准", "easy", "KEEP"),
    "你会写诗吗，给我写一首": ("general", "", "hard", "DROP?"),
    "你好茶刀和茶针前面都是扁的吗": ("general", "", "hard", "DROP?"),
    "几天可以到": ("general", "物流跟踪", "hard", "KEEP"),
    "分期手续费怎么算的": ("billing", "分期付款", "easy", "KEEP"),
    "刚买的耳机有一边突然没声音了，能退吗": ("billing", "质量问题退款;7天无理由退款", "hard", "KEEP"),
    "别啰嗦了，直接给我转人工客服！": ("general", "在线客服", "hard", "KEEP"),
    "发前一个地址": ("general", "地址修改规则", "easy", "KEEP"),
    "取消重新加双袜子重新拍你说不用": ("general", "", "hard", "DROP?"),
    "可以78号在发货吗": ("general", "物流跟踪", "easy", "KEEP"),
    "可以开发票吗？我要公司抬头": ("billing", "发票抬头;电子发票", "ambiguous", "KEEP"),
    "可以的呢之前我们遇到过无理收费的如果出现这种情况您及时与我们联系": ("general", "", "hard", "DROP?"),
    "后天能到吗": ("general", "物流跟踪", "hard", "KEEP"),
    "哦哦今天消费69不能送了是么": ("general", "满赠规则", "hard", "KEEP"),
    "哪个是纯棉的": ("general", "", "hard", "DROP?"),
    "因为以前箱子有坏没有影响物品也就算了毕竟快递跟你们都不容易": ("general", "", "hard", "DROP?"),
    "因为是江西但是我要寄到江苏的江西的运费要贵点呢": ("billing", "退货运费", "easy", "KEEP"),
    "地址我可以改的您把地址给我下吧": ("general", "地址修改规则", "easy", "KEEP"),
    "好吧，那耳机没声音了怎么办": ("tech", "声音问题", "easy", "KEEP"),
    "好的盒子难包装吗": ("general", "", "hard", "DROP?"),
    "好的石家庄几天到": ("general", "物流跟踪", "hard", "KEEP"),
    "就平时打电话刷视频用，哪款合适？": ("product", "手机", "hard", "KEEP"),
    "帮我看看这个订单耳机一共订了几个": ("general", "订单查询", "easy", "KEEP"),
    "怎么说第一次有漏发吗": ("general", "补发/少件", "hard", "KEEP"),
    "怎么还没物流信息你这是要拖到快递放假再来给我发吗": ("general", "物流跟踪;节假日安排", "ambiguous", "KEEP"),
    "怎么还要运费": ("billing", "退货运费", "easy", "KEEP"),
    "想给家里老人买台手机，帮我推荐下": ("product", "手机", "easy", "KEEP"),
    "我不想发邮政能换别的么": ("general", "物流跟踪", "hard", "KEEP"),
    "我买5个单套合一个三层套合有优惠吗": ("general", "", "hard", "DROP?"),
    "我买了两份送一份怎么就给我发了一盒什么情况": ("general", "补发/少件;满赠规则", "hard", "KEEP"),
    "我买的有点多有没有什么优惠": ("general", "", "hard", "DROP?"),
    "我刚拍的耳机地址选错了，能改吗": ("general", "地址修改规则", "easy", "KEEP"),
    "我前几天加入购物车了今天看失效了": ("general", "", "hard", "DROP?"),
    "我去不早说发韵达能到我家那儿我就能拿到": ("general", "物流跟踪", "hard", "KEEP"),
    "我已经关注了，赠品需要备注吗": ("general", "收藏送礼", "hard", "KEEP"),
    "我打算让淘宝来介入我是也卖家淘宝的规则我懂我看你怎么收场我做淘宝11年了只有十分钟等你答复妈的B不发飚当老娘好欺负吗": ("complaint", "处理流程", "hard", "KEEP"),
    "我的核桃怎么还没寄呢": ("general", "", "hard", "DROP?"),
    "我要退货，运费谁出": ("billing", "退货运费", "easy", "KEEP"),
    "我退回去的邮费你们也承担吗": ("billing", "退货运费", "easy", "KEEP"),
    "拍十份可以吗不同地址": ("general", "地址修改规则", "easy", "KEEP"),
    "按你说的做了，还是黑屏": ("tech", "无法开机", "easy", "KEEP"),
    "支持分期付款吗": ("billing", "分期付款", "easy", "KEEP"),
    "收到的耳机少了一个配件，能补发吗": ("general", "补发/少件", "easy", "KEEP"),
    "收到的耳机有瑕疵，要求补偿": ("complaint", "外观瑕疵;补偿标准", "ambiguous", "KEEP"),
    "收藏你们店铺有送什么吗": ("general", "收藏送礼", "easy", "KEEP"),
    "是的呢拍下自动减价哦": ("general", "", "hard", "DROP?"),
    "有货了能通知我吗": ("general", "缺货补货通知", "easy", "KEEP"),
    "有送什么吗": ("general", "满赠规则;收藏送礼", "ambiguous", "KEEP"),
    "朋友买的那款降噪耳机好用吗": ("product", "耳机", "hard", "KEEP"),
    "核桃仁会有优惠吗": ("general", "", "hard", "DROP?"),
    "正常都可以到的呢": ("general", "", "hard", "DROP?"),
    "每次不差评快递也是顾虑商家怕对你们有影响所以就算是几经周折收到货后依然好评就把快递惯出毛病了": ("general", "", "hard", "DROP?"),
    "活动期间不能改吗": ("general", "地址修改规则", "hard", "KEEP"),
    "淘宝为什么没有传照片功能": ("complaint", "功能缺陷", "hard", "KEEP"),
    "满199可以减十元是吧还送啥": ("general", "满赠规则", "easy", "KEEP"),
    "然后你送一盒": ("general", "满赠规则", "hard", "KEEP"),
    "现在买手机有满赠活动吗，送什么": ("general", "满赠规则;活动参与规则", "ambiguous", "KEEP"),
    "电脑开不了机怎么回事": ("tech", "无法开机", "easy", "KEEP"),
    "真不知道这样快递搞什么小区有自取箱不给我放非给放私人那": ("general", "自提柜与签收", "hard", "KEEP"),
    "神马叫定金不退万一你们没货也不退吗": ("billing", "", "hard", "DROP?"),
    "算了，还是留着吧，就是屏幕有点花": ("tech", "屏幕问题", "easy", "KEEP"),
    "纸质发票邮寄费谁出？": ("billing", "纸质发票", "easy", "KEEP"),
    "耳机不参与买二送一活动吗": ("general", "活动参与规则;满赠规则", "ambiguous", "KEEP"),
    "耳机什么时候能补货": ("general", "缺货补货通知", "easy", "KEEP"),
    "耳机左边没声音，能换货吗": ("tech", "声音问题", "hard", "KEEP"),
    "能发顺丰吗？我上次等太久了": ("general", "", "hard", "DROP?"),
    "能开纸质发票吗": ("billing", "纸质发票", "easy", "KEEP"),
    "能改下运费吗": ("billing", "退货运费", "easy", "KEEP"),
    "能早点发吗，我下周就要用": ("general", "物流跟踪", "hard", "KEEP"),
    "茶针和茶刀用法区别云南普洱块用哪种": ("general", "", "hard", "DROP?"),
    "行退多少": ("billing", "", "hard", "DROP?"),
    "补发的运费谁出？": ("general", "补发/少件", "hard", "KEEP"),
    "订单地址填错了能改吗": ("general", "地址修改规则", "easy", "KEEP"),
    "还另外送湿巾吗": ("general", "满赠规则", "hard", "KEEP"),
    "还是一次付款吧可以领5元优惠券呢": ("billing", "在线支付", "hard", "KEEP"),
    "这个平板我要退了": ("billing", "7天无理由退款", "hard", "KEEP"),
    "这款现在有盖子吗": ("product", "", "hard", "DROP?"),
    "送的那罐一起拍下系统自动减价的哦": ("general", "满赠规则", "hard", "KEEP"),
    "送礼物哈不要红枣": ("general", "", "hard", "DROP?"),
    "那个平板多少钱来着": ("product", "平板", "easy", "KEEP"),
    "那几天能到呢": ("general", "物流跟踪", "hard", "KEEP"),
    "那到底是发什么快递？": ("general", "物流跟踪", "easy", "KEEP"),
    "那哪些产品参与呢": ("general", "活动参与规则", "hard", "KEEP"),
    "那大概最快能多久啊因为我比较着急": ("general", "物流跟踪", "hard", "KEEP"),
    "那它退货运费谁出": ("billing", "退货运费", "easy", "KEEP"),
    "那怎么办，重拍吗": ("general", "", "hard", "DROP?"),
    "那换货一般要多久能到？": ("general", "售后服务", "hard", "KEEP"),
    "那款能七天无理由退货吗": ("billing", "7天无理由退款", "easy", "KEEP"),
    "那现在下单能明天发货吗": ("general", "物流跟踪", "hard", "KEEP"),
    "那退款一般几天到账？": ("billing", "退款流程", "easy", "KEEP"),
    "那退的这个能开票吗": ("billing", "电子发票", "easy", "KEEP"),
    "那退货运费谁出？": ("billing", "退货运费", "easy", "KEEP"),
}


def main() -> None:
    rows = list(csv.DictReader(open(PREFILL, encoding="utf-8-sig")))
    out = []
    applied = 0
    for r in rows:
        q = r["query"]
        if q in ANNOTATIONS:
            dom, titles, diff, suggest = ANNOTATIONS[q]
            r["domain"] = dom
            r["expected_titles"] = titles
            r["difficulty"] = diff
            r["suggest"] = suggest
            applied += 1
        out.append(r)
    # 输出正式金标集（只保留 KEEP）
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            if r["suggest"] == "KEEP":
                w.writerow(r)

    keep = [r for r in out if r["suggest"] == "KEEP"]
    drop = [r for r in out if r["suggest"] == "DROP?"]
    from collections import Counter
    diff_cnt = Counter(r["difficulty"] for r in keep)
    dom_cnt = Counter(r["domain"] for r in keep)
    print(f"标注应用：{applied}/{len(rows)} 条")
    print(f"正式金标集：{len(keep)} 条 KEEP → {OUT}")
    print(f"难度分布：{dict(diff_cnt)}")
    print(f"域分布：{dict(dom_cnt)}")
    print(f"暂缓：{len(drop)} 条（v2 补条目后重评）")


if __name__ == "__main__":
    main()
