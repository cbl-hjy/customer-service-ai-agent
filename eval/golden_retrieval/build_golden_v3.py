"""A2-3 生成新增金标（122 条）并校验，追加到 golden_retrieval_golden_v2.csv。

设计依据：
- TEMPLATE.md 配额：每域 40+ 条；难度 easy:hard:ambiguous ≈ 3:4:3；≥60% ECD 来源
- KB 601 条 title 全集（kb_v2_titles.json）作 expected_titles 唯一合法取值
- 目标 217 条 = 现有 95 + 新增 122（product +36 / tech +35 / complaint +35 / billing +16，general 保持 57）
- 新增难度：easy 21 / hard 44 / ambiguous 57；来源：ecd 86 / 人工 36

安全：expected_titles 全部来自 KB title 全集（生成时强校验，缺标题即报错退出）。
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_retrieval_golden_v2.csv")
TITLES = os.path.join(HERE, "kb_v2_titles.json")
OUT = os.path.join(HERE, "golden_retrieval_golden_v2_add.csv")

kb_titles = json.load(open(TITLES, encoding="utf-8"))
all_titles = set()
for v in kb_titles.values():
    all_titles.update(v)

# (query, domain, expected_titles 分号分隔, difficulty, source, note)
ROWS = [
    # ===================== product +36（easy 6 / hard 14 / ambiguous 16） =====================
    ("X1 Pro 拍照怎么样", "product", "X1 Pro 摄像头配置", "easy", "ecd", "口语直问"),
    ("玩游戏选哪个手机好", "product", "游戏手机推荐", "easy", "人工构造", "字面命中"),
    ("AirBuds 3 和 Pro 有什么不一样", "product", "AirBuds 3 与 Pro 区别", "easy", "人工构造", "型号对比字面"),
    ("有没有适合学生用的笔记本", "product", "笔记本选购-学生用", "easy", "人工构造", "字面命中"),
    ("Pad 支持手写笔吗", "product", "Pad 手写笔兼容", "easy", "人工构造", "字面命中"),
    ("100W 移动电源是哪个", "product", "星环 100W 移动电源", "easy", "人工构造", "字面命中"),
    # hard
    ("我就想找个待机时间长的，别一天三充", "product", "长续航手机推荐", "hard", "ecd", "ECD 问法模式改写"),
    ("上班要经常带电脑跑客户，选哪个合适", "product", "笔记本选购-办公用", "hard", "ecd", "ECD 问法模式改写"),
    ("给爸妈买台省心的，字大声音大", "product", "老人机选购指南", "hard", "ecd", "ECD 问法模式改写"),
    ("想送女朋友一个惊喜，不知道送啥数码好", "product", "礼物推荐-送女友", "hard", "ecd", "ECD 问法模式改写"),
    ("戴耳机跑步会不会掉", "product", "运动耳机推荐", "hard", "ecd", "场景→品类（跑步→运动）"),
    ("坐地铁通勤想安静点，有没有隔音好的", "product", "降噪耳机推荐", "hard", "ecd", "口语诉求→功能（隔音→降噪）"),
    ("小孩上网课用，别伤眼睛", "product", "Pad 护眼认证", "hard", "ecd", "ECD 问法模式改写"),
    ("画插画用哪个平板合适", "product", "平板选购-绘画用", "hard", "ecd", "用途→选购"),
    ("做设计的电脑配置要高点", "product", "笔记本选购-设计师用", "hard", "人工构造", "角色→选购"),
    ("两台手机有啥差别，就屏幕大小吗", "product", "X1 与 X1 Pro 区别", "hard", "ecd", "口语指代对比"),
    ("这耳机戴久了耳朵疼，有没有舒服点的", "product", "头戴耳机推荐", "hard", "ecd", "舒适诉求→形态"),
    ("想看直播带货用的全套设备", "product", "直播开播装备推荐", "hard", "ecd", "场景→方案"),
    ("学生党预算不多，怎么配最划算", "product", "学生党性价比方案", "hard", "ecd", "人群+预算→方案"),
    ("白天开会晚上加班，买台电脑主要写报表做PPT", "product", "笔记本选购-办公用", "hard", "人工构造", "场景改写（语义不重叠）"),
    # ambiguous
    ("3000 块钱能买到啥好手机", "product", "3000元内手机推荐;星环 X1 Lite 入门手机规格", "ambiguous", "ecd", "预算→推荐与入门款皆沾边"),
    ("打电话多还要续航久，选哪个", "product", "长续航手机推荐;商务手机推荐", "ambiguous", "ecd", "续航与商务双意图"),
    ("X1 和 Fold1 怎么选，都想要", "product", "X1 Pro 与 Fold1 怎么选;X1 与 X1 Pro 区别", "ambiguous", "ecd", "折叠与直板对比纠缠"),
    ("想换台电脑，办公和游戏都想要", "product", "笔记本选购-办公用;星环 GameBook 15 游戏本规格", "ambiguous", "ecd", "办公游戏双需求"),
    ("屏幕好点的笔记本有推荐吗", "product", "笔记本屏幕分辨率;星环 Book Pro 14 笔记本规格", "ambiguous", "ecd", "屏幕素质与整机规格皆相关"),
    ("给小孩买平板，学习还是娱乐", "product", "平板选购-学习用;Pad 儿童模式", "ambiguous", "ecd", "ECD 问法模式改写"),
    ("AirBuds 和 HeadBuds 哪个好", "product", "AirBuds 3 Pro 与 HeadBuds 5 怎么选;头戴耳机推荐", "ambiguous", "ecd", "品牌线对比与形态推荐皆沾边"),
    ("耳机要不要买带降噪的", "product", "降噪耳机推荐;半入耳与入耳式区别", "ambiguous", "ecd", "ECD 问法模式改写"),
    ("充电器和充电宝都要，出门省事", "product", "星环 65W 氮化镓充电器;星环 磁吸充电宝", "ambiguous", "ecd", "ECD 问法模式改写"),
    ("手机电脑一起买有优惠吗", "product", "办公全家桶推荐;学生党性价比方案", "ambiguous", "ecd", "套装方案双条目"),
    ("安卓平板还是专用学习机好", "product", "Pad 适用人群;平板选购-学习用", "ambiguous", "人工构造", "人群与用途皆相关"),
    ("给老人买手机还是平板", "product", "老人机选购指南;Pad Mini 便携场景", "ambiguous", "ecd", "两类设备取舍"),
    ("拍照好看和打游戏爽能兼得吗", "product", "拍照手机推荐;游戏手机推荐", "ambiguous", "ecd", "双卖点对比"),
    ("轻薄本和游戏本到底买哪个", "product", "Book Air 与 Book Pro 区别;星环 GameBook 15 游戏本规格", "ambiguous", "ecd", "形态对比跨条目"),
    ("平板拿来画画和追剧选哪个尺寸", "product", "平板选购-绘画用;平板选购-追剧用", "ambiguous", "人工构造", "双用途选购"),
    ("无线充和有线充哪个快", "product", "星环 30W 无线充电器;星环 65W 氮化镓充电器", "ambiguous", "ecd", "两类充电对比"),

    # ===================== tech +35（easy 6 / hard 13 / ambiguous 16） =====================
    ("手机黑屏了怎么办", "tech", "手机黑屏", "easy", "人工构造", "字面命中"),
    ("电脑蓝屏怎么处理", "tech", "电脑蓝屏", "easy", "人工构造", "字面命中"),
    ("耳机一边没声音", "tech", "耳机单侧无声", "easy", "人工构造", "字面命中"),
    ("平板触屏没反应", "tech", "平板屏幕触控失灵", "easy", "人工构造", "字面命中"),
    ("手机怎么备份数据", "tech", "手机数据备份", "easy", "人工构造", "字面命中"),
    ("换新机怎么把旧手机数据弄过来", "tech", "数据迁移到新手机", "easy", "人工构造", "字面命中"),
    # hard
    ("手机摔了一下，屏幕花了", "tech", "手机屏幕碎裂", "hard", "ecd", "症状描述"),
    ("用着用着就卡死了，半天没反应", "tech", "手机死机无响应", "hard", "ecd", "ECD 问法模式改写"),
    ("充了一个小时还是百分之二", "tech", "手机充不进电", "hard", "ecd", "ECD 问法模式改写"),
    ("听筒有杂音，说话对方听不清", "tech", "手机麦克风失灵", "hard", "ecd", "症状描述"),
    ("电池半天就没电，一天两充", "tech", "手机电池续航差", "hard", "ecd", "症状描述"),
    ("开机一直转圈进不去桌面", "tech", "电脑系统崩溃", "hard", "ecd", "症状描述"),
    ("游戏打着打着就闪退", "tech", "电脑软件闪退", "hard", "ecd", "场景描述"),
    ("耳机连上手机一会就断了", "tech", "耳机连接频繁断开", "hard", "ecd", "症状描述"),
    ("耳机沙沙响有电流声", "tech", "耳机底噪大", "hard", "ecd", "症状描述"),
    ("平板画画笔没反应了", "tech", "平板手写笔没反应", "hard", "ecd", "症状描述"),
    ("手机昨天还能用今天打不开了", "tech", "手机无法开机", "hard", "ecd", "口语症状"),
    ("电脑触摸板点了没反应", "tech", "电脑触控板无响应", "hard", "ecd", "症状描述"),
    ("买的耳机能修吗，还在保修期", "tech", "保修期内维修流程", "hard", "人工构造", "保修场景"),
    # ambiguous
    ("手机老是发热，是不是坏了", "tech", "手机发热严重;X1 Pro 发热优化建议", "ambiguous", "ecd", "症状与优化建议皆沾边"),
    ("屏幕碎了能保修吗", "tech", "保修范围;不在保修范围的场景", "ambiguous", "ecd", "保修边界正反条目"),
    ("手机数据怎么备份到新机", "tech", "手机数据备份;数据迁移到新手机", "ambiguous", "ecd", "备份与迁移双条目"),
    ("怎么把手机里的照片导到电脑", "tech", "数据迁移服务;数据备份方式对比", "ambiguous", "人工构造", "服务与方式对比皆相关"),
    ("WiFi 老自动断，是路由器问题还是手机问题", "tech", "手机WiFi自动断开;手机无法连接WiFi", "ambiguous", "ecd", "断开与连接失败皆相关"),
    ("耳机丢了能单独配一只吗", "tech", "耳机单耳丢失购买;耳机丢失找回", "ambiguous", "ecd", "配单耳与找回双诉求"),
    ("系统更新以后还能退回去吗", "tech", "系统更新回退;手机系统更新失败", "ambiguous", "ecd", "回退与失败处理皆相关"),
    ("账号被盗了怎么办", "tech", "账号异地登录提醒;远程锁定与清除", "ambiguous", "ecd", "提醒与处置双条目"),
    ("重装系统会不会丢东西", "tech", "电脑系统重装;电脑文件误删恢复", "ambiguous", "ecd", "重装与文件恢复皆相关"),
    ("手机可以自己拆开修吗，还是寄回去", "tech", "寄修流程;官方维修与第三方对比", "ambiguous", "ecd", "寄修与自行维修对比"),
    ("保修期怎么算，延保值得买吗", "tech", "产品保修期;延保服务", "ambiguous", "人工构造", "保修与延保皆相关"),
    ("修手机要多久，能不能加急", "tech", "维修周期;维修进度短信通知", "ambiguous", "ecd", "周期与进度通知皆相关"),
    ("旧手机能抵钱吗", "tech", "以旧换新流程;数据迁移服务", "ambiguous", "ecd", "抵钱与迁移服务皆相关"),
    ("清理手机垃圾用什么办法", "tech", "手机存储空间不足;手机恢复出厂设置", "ambiguous", "人工构造", "清理与恢复皆相关"),
    ("指纹和面容哪个安全", "tech", "手机指纹解锁失效;手机面部识别失败", "ambiguous", "人工构造", "双生物识别故障"),
    ("平板总闪退是系统问题吗", "tech", "平板App闪退;平板存储不足", "ambiguous", "ecd", "闪退根因双条目"),

    # ===================== complaint +35（easy 5 / hard 14 / ambiguous 16） =====================
    ("客服半天不回消息，我要投诉", "complaint", "客服响应慢投诉", "easy", "人工构造", "字面命中"),
    ("收到的手机屏幕有划痕", "complaint", "外观瑕疵投诉", "easy", "人工构造", "字面命中"),
    ("赠品是坏的，我要投诉", "complaint", "赠品质量问题投诉", "easy", "人工构造", "字面命中"),
    ("快递给我送错地方了，气死", "complaint", "配送错误投诉", "easy", "人工构造", "字面命中"),
    ("你们赔偿标准是什么", "complaint", "补偿标准说明", "easy", "人工构造", "字面命中"),
    # hard
    ("态度恶劣，爱答不理的", "complaint", "客服态度差投诉", "hard", "ecd", "情绪化描述"),
    ("问啥啥不懂，换个客服又问一遍", "complaint", "客服专业不足投诉", "hard", "ecd", "行为描述"),
    ("就退个货来回折腾了三次还没搞定", "complaint", "处理流程繁琐投诉", "hard", "ecd", "经历描述"),
    ("打了三次电话每次都说稍等，结果就没下文", "complaint", "反复转接投诉", "hard", "ecd", "经历描述"),
    ("买的时候说七天到，现在十天了影子都没有", "complaint", "配送延迟投诉", "hard", "ecd", "承诺对比"),
    ("收到的盒子都瘪了，里面的东西还能好？", "complaint", "包装破损投诉", "hard", "ecd", "外观描述"),
    ("明明没收到说签收了，骗谁呢", "complaint", "物流信息造假投诉", "hard", "ecd", "失信描述"),
    ("不经同意就扔驿站，我腿脚不方便咋取", "complaint", "擅自放驿站投诉", "hard", "ecd", "服务不满"),
    ("宣传说 5000 万像素，拍出来糊得跟马赛克似的", "complaint", "宣传与实物不符投诉", "hard", "ecd", "对比描述"),
    ("换了一台还是有问题，是不是卖次品", "complaint", "质量问题重复出现投诉", "hard", "ecd", "经历描述"),
    ("说好今天回复我，结果又放鸽子", "complaint", "承诺未兑现投诉", "hard", "ecd", "失信描述"),
    ("你们的人工怎么老是排队等半天", "complaint", "客服电话难接通投诉", "hard", "ecd", "体验描述"),
    ("快递破损了你们不赔，就一句按流程", "complaint", "快递损坏拒赔投诉", "hard", "ecd", "维权描述"),
    ("延迟发货到底赔不赔，赔多少", "complaint", "延迟补偿标准", "hard", "ecd", "补偿咨询口语"),
    # ambiguous
    ("客服答应的事转头就不认，算谁的问题", "complaint", "客服承诺不认账投诉;承诺未兑现投诉", "ambiguous", "ecd", "两投诉条目皆沾边"),
    ("开箱发现是别人用过的，是不是翻新机", "complaint", "二手充新投诉;新机开机异常投诉", "ambiguous", "ecd", "翻新与开机异常皆相关"),
    ("东西有毛病，是想退货还是让赔钱", "complaint", "质量问题补偿标准;质量问题重复出现投诉", "ambiguous", "ecd", "补偿与投诉双意图"),
    ("这手机发热烫手，用起来卡，投诉有用吗", "complaint", "性能不达标投诉;功能缺陷投诉", "ambiguous", "人工构造", "性能与功能缺陷皆相关"),
    ("投诉了多久能给答复", "complaint", "投诉处理时限;投诉处理结果反馈", "ambiguous", "ecd", "时限与反馈皆相关"),
    ("处理结果我不满意还能再投诉吗", "complaint", "投诉处理结果不满意;投诉升级条件", "ambiguous", "ecd", "不满与升级皆相关"),
    ("快递员态度差还扔件，找谁投诉", "complaint", "快递员服务投诉;配送错误投诉", "ambiguous", "ecd", "服务与配送双投诉"),
    ("少发了一个赠品，是补发还是补偿", "complaint", "赠品缺失补偿标准;漏发补偿标准", "ambiguous", "人工构造", "两类补偿皆相关"),
    ("补偿是给钱还是给券", "complaint", "补偿发放方式;补偿金额计算", "ambiguous", "人工构造", "方式与金额皆相关"),
    ("这情况能赔多少，怎么算的", "complaint", "补偿金额计算;质量问题补偿标准", "ambiguous", "人工构造", "金额与标准皆相关"),
    ("要求人工介入怎么申请", "complaint", "人工介入申请;工单提交", "ambiguous", "ecd", "介入与工单皆相关"),
    ("能不能加急处理，真的很急", "complaint", "加急处理;升级响应时限", "ambiguous", "ecd", "加急与时限皆相关"),
    ("监管渠道能投诉你们吗", "complaint", "监管渠道投诉说明;投诉材料准备", "ambiguous", "ecd", "渠道与材料皆相关"),
    ("客服不理人，打客服电话也没人接", "complaint", "线上客服转接慢投诉;客服电话难接通投诉", "ambiguous", "人工构造", "双渠道投诉"),
    ("处理了半个月还没结果，我要找主管", "complaint", "投诉处理时限;主管处理通道", "ambiguous", "人工构造", "时限与主管皆相关"),
    ("外观瑕疵和功能问题能一起赔偿吗", "complaint", "外观瑕疵投诉;质量问题补偿标准", "ambiguous", "人工构造", "瑕疵与补偿皆相关"),

    # ===================== billing +16（easy 4 / hard 3 / ambiguous 9） =====================
    ("7 天无理由退款怎么申请", "billing", "7天无理由退款", "easy", "人工构造", "字面命中"),
    ("发票能开公司抬头吗", "billing", "发票抬头修改", "easy", "人工构造", "字面命中"),
    ("分期付款有什么要求", "billing", "分期付款", "easy", "人工构造", "字面命中"),
    ("满 199 减 30 怎么用", "billing", "满减活动规则", "easy", "ecd", "活动问法口语"),
    # hard
    ("昨天刚买今天就降价了，能退差价吗", "billing", "价格保护政策", "hard", "ecd", "场景描述"),
    ("退货的运费谁出", "billing", "退货运费承担", "hard", "ecd", "口语问法"),
    ("买东西送的券能变现吗", "billing", "红包提现说明", "hard", "ecd", "口语问法"),
    # ambiguous
    ("退款一般多久到账", "billing", "退款到账时效;超时自动退款", "ambiguous", "ecd", "时效与自动退款皆相关"),
    ("优惠券下单后退款，券还回来吗", "billing", "优惠券退款处理;优惠券有效期", "ambiguous", "ecd", "退款处理与有效期皆相关"),
    ("能开发票吗，要报销用", "billing", "电子发票开具;发票内容明细", "ambiguous", "ecd", "开票与明细皆相关"),
    ("买贵了能退吗，怎么退", "billing", "买贵退差;价格保护政策", "ambiguous", "ecd", "退差与保价皆相关"),
    ("分期退款的利息怎么算", "billing", "分期订单退款;分期付款", "ambiguous", "ecd", "退款与分期皆相关"),
    ("发票开错了能改吗", "billing", "发票抬头修改;发票重开", "ambiguous", "ecd", "抬头与重开皆相关"),
    ("退款被拒了怎么办", "billing", "退款申请被拒原因;退款被拒申诉", "ambiguous", "ecd", "原因与申诉皆相关"),
    ("用花呗买的退款退到哪", "billing", "退款原路返回;组合支付退款拆分", "ambiguous", "ecd", "原路与拆分皆相关"),
    ("换货和退货哪个划算", "billing", "换货与新购区别;7天无理由退款", "ambiguous", "人工构造", "换货与退货对比"),
]


def main() -> None:
    # 1) 校验 expected_titles 全部存在
    missing = []
    for r in ROWS:
        for t in r[2].split(";"):
            t = t.strip()
            if t not in all_titles:
                missing.append((r[1], r[0], t))
    if missing:
        for dom, q, t in missing:
            print(f"[MISSING] {dom} | {q} | {t}")
        print(f"缺失 title {len(missing)} 个，中止（防生成无效金标）")
        sys.exit(1)

    # 2) 校验难度/域枚举
    for r in ROWS:
        assert r[3] in ("easy", "hard", "ambiguous"), r
        assert r[1] in kb_titles, r

    # 3) query 查重（与现有 + 新行内部）
    existing = {r["query"] for r in csv.DictReader(open(GOLDEN, encoding="utf-8-sig"))}
    seen = set()
    for r in ROWS:
        if r[0] in existing or r[0] in seen:
            print(f"[DUP] {r[0]}")
            sys.exit(2)
        seen.add(r[0])

    # 4) 写入新 CSV（id 从 gr-096 起）
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "domain", "expected_titles", "difficulty", "source", "suggest", "candidates", "note"])
        for i, r in enumerate(ROWS, start=96):
            w.writerow([f"gr-{i:03d}", r[0], r[1], r[2], r[3], r[4], "KEEP", "", r[5]])

    # 5) 分布统计
    from collections import Counter
    dom = Counter(r[1] for r in ROWS)
    diff = Counter(r[3] for r in ROWS)
    src = Counter(r[4] for r in ROWS)
    print(f"新增 {len(ROWS)} 条已写入 {OUT}")
    print("新增 domain:", dict(dom))
    print("新增 difficulty:", dict(diff))
    print("新增 source:", dict(src))
    total_diff = Counter()
    total_src = Counter()
    for r in csv.DictReader(open(GOLDEN, encoding="utf-8-sig")):
        total_diff[r["difficulty"]] += 1
        total_src[r["source"]] += 1
    for r in ROWS:
        total_diff[r[3]] += 1
        total_src[r[4]] += 1
    total = sum(total_diff.values())
    print(f"合并后总 {total} 条")
    print("合并后 difficulty:", dict(total_diff))
    print("合并后 source:", dict(total_src), f"| ecd占比 {total_src.get('ecd',0)/total:.1%}")


if __name__ == "__main__":
    main()
