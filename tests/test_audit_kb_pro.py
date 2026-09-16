# -*- coding: utf-8 -*-
"""audit_kb_pro 价格矩阵检查（check_price_matrix）密封测试。

每个用例对应一个具名风险（2026-09-16 价格矩阵四次漏网 + 检查器首跑两只真矛盾的回归锁定）：
- P1 同商品点价跨条目不一致（AirBuds 价格矩阵四组拧形态）→ 必须报
- P2 同商品区间价不一致（Pad Mini 999-1299 vs 1299-1599 形态）→ 必须报
- P3 配件主题条目（手机壳/充电器选购）价格不绑主商品 → 不报
- P4 适配语境（配件词/兼容/支持夹在商品名与价格之间）→ 不报
- P5 合法配置档：同条目内多价格 → 不报
- P6 合法区间+双档：≥2 条目区间覆盖全部点价 → 不报
- P7 裸系列名分桶（X1 标准版 vs X1 Lite 不同商品不同价）→ 不报
- P8 「降噪耳机推荐」残留形态（399/499 同条目内两商品各自定价）→ 报错绑形态
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + "eval" + os.sep + "kb_v2")

from audit_kb_pro import check_price_matrix  # noqa: E402

import pytest  # noqa: E402


def _kb(*entries):
    return {"product": [{"title": t, "content": c, "keywords": [], "category": "test"} for t, c in entries]}


def test_p1_point_price_cross_entry_contradiction():
    """P1：同商品点价跨条目多值且无豁免 → 报（AirBuds 四组拧形态）。"""
    kb = _kb(
        ("星环 AirBuds Lite 入门耳机规格", "星环 AirBuds Lite：8mm 动圈，售价约 149 元。"),
        ("运动耳机推荐", "运动场景推荐星环 AirBuds Lite，售价约 129 元。"),
    )
    warns = check_price_matrix(kb)
    assert any("AirBudsLite" in w and "129" in w and "149" in w for w in warns), warns


def test_p2_range_price_inconsistency():
    """P2：同商品两个不同区间 → 报（Pad Mini 形态）。"""
    kb = _kb(
        ("Pad 存储配置", "星环 Pad Mini 8：4GB+64GB（约 999 元）/ 6GB+128GB（约 1299 元）。"),
        ("Pad 适用人群", "星环 Pad Mini 8 适合通勤，售价区间约 999-1299 元。"),
        ("Pad Mini 便携场景", "星环 Pad Mini 8 轻便，售价区间约 1299-1599 元。"),
    )
    warns = check_price_matrix(kb)
    assert any("PadMini8" in w and "区间价不一致" in w for w in warns), warns


def test_p3_accessory_topic_entry_skipped():
    """P3：配件主题条目整条跳过，主商品名不绑配件价（手机壳 149 案例）。"""
    kb = _kb(
        ("星环 X1 Pro 旗舰手机规格", "星环 X1 Pro 售价约 5499 元。"),
        ("星环 手机壳材质对比", "不影响星环 X1 Pro 120W 有线快充及 NFC 功能，售价约 149 元。"),
    )
    warns = check_price_matrix(kb)
    assert not any("X1Pro" in w for w in warns), warns


def test_p4_adapter_context_gap_filtered():
    """P4：商品名与价格之间夹配件词或兼容/支持 → 跳过（充电器 149 / 快充头 39 案例）。"""
    kb = _kb(
        ("会员专属价商品", "星环 Pad 11（原价约 1999 元）、星环 65W 氮化镓充电器（原价约 149 元）。"),
        ("充电配件说明", "适配星环 Pad Mini 8（18W 兼容），体积轻巧约 45g，售价约 39 元。"),
        ("Pad 存储配置", "星环 Pad 11：8GB+128GB（约 1599 元）。"),
    )
    warns = check_price_matrix(kb)
    # Pad11 只有 1999（会员条目）与 1599（存储条目）两点价，无区间覆盖 → 若 149 未被过滤会多一个值
    # 但 1599/1999 本身构成需人工确认的多值（配置档跨条目），149/39 不得出现在 Pad 桶告警里
    pad_warns = [w for w in warns if "Pad11" in w or "PadMini8" in w]
    assert all("149" not in w and "39 元" not in w.replace("539", "") for w in pad_warns), pad_warns


def test_p5_same_entry_multi_price_legal():
    """P5：所有点价仅出现在同一条目（配置档形态）→ 不报。"""
    kb = _kb(
        ("Pad 存储配置", "星环 Pad Pro 12.9：8GB+256GB（约 3999 元）/ 12GB+512GB（约 5299 元）。"),
    )
    warns = check_price_matrix(kb)
    assert not any("PadPro12.9" in w and "点价" in w for w in warns), warns


def test_p6_range_covered_multi_entry_legal():
    """P6：≥2 条目区间覆盖全部点价 → 不报（Pad Pro 双档+区间形态）。"""
    kb = _kb(
        ("Pad 存储配置", "星环 Pad Pro 12.9：8GB+256GB（约 3999 元）/ 12GB+512GB（约 5299 元）。"),
        ("Pad 适用人群", "星环 Pad Pro 12.9 适合专业创作，售价区间约 3999-5299 元。"),
        ("平板选购-绘画用", "进阶选星环 Pad Pro 12.9，售价区间约 3999-5299 元。"),
    )
    warns = check_price_matrix(kb)
    assert not any("PadPro12.9" in w for w in warns), warns


def test_p7_bare_series_name_bucketing():
    """P7：X1 标准版与 X1 Lite 分桶，各自单值 → 不报。"""
    kb = _kb(
        ("星环 X1 Lite 入门手机规格", "星环 X1 Lite 售价约 1599 元。"),
        ("星环 X1 标准版手机规格", "星环 X1 标准版售价约 2799 元。"),
        ("3000元内手机推荐", "推荐星环 X1 Lite（约 1599 元）和星环 X1 标准版（约 2799 元）。"),
    )
    warns = check_price_matrix(kb)
    assert not any("X1Lite" in w or w.startswith("[价格矩阵] X1:") for w in warns), warns


def test_p8_noise_cancel_entry_residual_form():
    """P8：「降噪耳机推荐」残留形态回归：同条目两商品各自定价，其中一商品价格
    与其他条目矛盾（399 vs 499）→ 必须报。"""
    kb = _kb(
        ("降噪耳机推荐", "星环 AirBuds 3 Pro：ANC 42dB，定价约 399 元；"
                     "星环 HeadBuds 5：40mm 大动圈，定价约 799 元。"),
        ("百元耳机推荐", "星环 AirBuds 3 Pro 定价约 499 元。"),
        ("AirBuds 3 Pro 与 HeadBuds 5 怎么选", "AirBuds 3 Pro 约 499 元，HeadBuds 5 约 799 元。"),
    )
    warns = check_price_matrix(kb)
    assert any("AirBuds3Pro" in w and "399" in w and "499" in w for w in warns), warns
    # HeadBuds 5 两处一致 799 → 不报
    assert not any("HeadBuds5" in w for w in warns), warns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
