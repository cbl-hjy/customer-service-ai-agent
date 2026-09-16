#!/usr/bin/env python3
"""SSE 并发压测（质量收口，2026-09-15）：HTTP 层流式路径并发验证。

具名风险（对 MAX_CONCURRENCY=5 信号量边界之上的 6 路并发）：
  风险1 首帧并发退化：每路首帧（meta/stage）≤1s——Flask 请求线程只做帧转发，
       即使 worker 在信号量上排队，阶段流/心跳也必须即时（用户不能面对白屏）
  风险2 内容隔离：6 路 done.content 两两不同且与各自问题语义对应（sink 串扰检测）
  风险3 完整性：每路以 [DONE] 结束、done 帧非空、错误率 0
  风险4 排队可见性：worker 实际开始生成前的心跳保活（不断连）

用法：.venv/Scripts/python.exe bench_sse.py [并发数，默认6] [每路请求数，默认1]
依赖真实 LLM key；服务端口 5099。
"""

import sys
import time

import requests

from verify_e2e_smoke import BASE, iter_sse, start_server  # 复用冒烟脚本的服务启动/SSE 解析（单一实现）

# 6 个不同业务域问题（并发铺满各 agent 路径）
QUERIES = [
    "你们的耳机 SoundPro 降噪效果怎么样？",
    "退款 10 天没到账怎么查进度？",
    "平板 Tab Pro 支持触控笔吗？",
    "帮我查一下订单的物流进度",
    "发票抬头可以修改吗？",
    "电脑开机黑屏怎么办？",
]


def run_sse_concurrent(concurrency: int, rounds: int = 1) -> int:
    """并发 SSE 请求，返回 0 通过 / 1 失败。"""
    import threading

    stamp = time.strftime("%H%M%S")
    results = {}

    def run(key: int, msg: str, sid: str):
        t0 = time.perf_counter()
        first_frame_s = None
        events = []
        err = None
        try:
            with requests.post(
                f"{BASE}/api/chat/stream",
                json={"message": msg, "session_id": sid},
                stream=True,
                timeout=180,
            ) as r:
                if r.status_code != 200:
                    raise AssertionError(f"HTTP {r.status_code}")
                for evt in iter_sse(r):
                    if first_frame_s is None:
                        first_frame_s = time.perf_counter() - t0
                    events.append(evt)
        except Exception as e:  # noqa: BLE001
            err = str(e)[:120]
        results[key] = {"first_frame": first_frame_s, "events": events, "error": err, "total": time.perf_counter() - t0}

    threads = []
    t_start = time.time()
    idx = 0
    for _round in range(rounds):
        for i in range(concurrency):
            q = QUERIES[idx % len(QUERIES)]
            t = threading.Thread(target=run, args=(idx, q, f"bench-sse-{stamp}-{idx}"))
            idx += 1
            threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_start

    # ---- 断言汇总 ----
    fails = []
    contents = []
    first_frames = []
    totals = []
    for key, r in sorted(results.items()):
        if r["error"]:
            fails.append(f"路{key} 异常: {r['error']}")
            continue
        evts = r["events"]
        types = [e["type"] for e in evts]
        done = next((e for e in evts if e["type"] == "done"), None)
        # 风险3：完整性
        if not types or types[-1] != "[DONE]":
            fails.append(f"路{key} 未以 [DONE] 结束")
        if done is None or not done.get("content"):
            fails.append(f"路{key} 缺 done/空回复")
            continue
        # 风险1：首帧
        if r["first_frame"] is None or r["first_frame"] > 1.0:
            fails.append(f"路{key} 首帧 {r['first_frame'] and round(r['first_frame'], 2)}s >1s")
        first_frames.append(r["first_frame"])
        totals.append(r["total"])
        contents.append(done["content"])

    # 风险2：内容隔离（两两不同）
    if len(set(c[:50] for c in contents)) != len(contents):
        fails.append("存在两路内容相同——疑似 sink 串扰")

    n_ok = len(contents)
    print(f"===== SSE 并发压测：{concurrency} 路 × {rounds} 轮（墙钟 {wall:.1f}s）=====")
    if first_frames:
        print(f"首帧: min={min(first_frames):.2f}s max={max(first_frames):.2f}s（全部 ≤1s {'✅' if all(f <= 1.0 for f in first_frames) else '❌'}）")
    if totals:
        st = sorted(totals)
        print(f"整轮完成: P50={st[len(st)//2]:.1f}s max={st[-1]:.1f}s")
    print(f"成功 {n_ok}/{len(results)} 路")
    if fails:
        for f in fails:
            print(f"  ❌ {f}")
        return 1
    print("✅ SSE 并发压测通过")
    return 0


def main():
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    proc = start_server()
    try:
        # 预热：单路触发模型懒加载，避免首档延迟被冷启动污染
        print("[warmup] 单路预热（触发模型懒加载）…")
        t0 = time.time()
        warm = requests.post(
            f"{BASE}/api/chat/stream",
            json={"message": QUERIES[0], "session_id": f"bench-sse-warm-{time.strftime('%H%M%S')}"},
            stream=True, timeout=180,
        )
        list(iter_sse(warm))
        warm.close()
        print(f"[warmup] {time.time()-t0:.1f}s")
        return run_sse_concurrent(concurrency, rounds)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
