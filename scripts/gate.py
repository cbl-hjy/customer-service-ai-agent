#!/usr/bin/env python3
"""本地 CI 门禁（第二波工程化，2026-09-16）：纯本地三层门禁，push 前拦截。

设计（用户确认形态：不依赖 GitHub Actions、API key 不出本机）：
  gate fast   密封测试批（pytest -m "not api and not model"，~45s，零 API 成本）
              ——pre-push 钩子默认挡位
  gate eval   全量金标评估 28 cases（JUDGE_ENABLED=0 省 judge 调用，成本≈¥0.04）
              + 与全量基线（eval/baselines.json 尾条）噪声带对比，越界即拦
              ——模型/KB/prompt 变更后必跑（AGENTS.md 硬约束"模型切换必须过 W4"的执行器）
  gate full   全量测试 259 项（含 api 真实调用 + model 真实加载）+ E2E 冒烟
              ——发布/大版本前手动跑

噪声带（2026-09-15 偶发应对策略定标）：确定性指标（决策/升级/检索）±2%，
生成类指标（答案/引用）±3%，延迟/成本 +30% 环境余量；judge 关不参与。

用法：python scripts/gate.py {fast|eval|full}
退出码：0=通过，1=拦截（门禁失败），2=用法错误。
"""

import json
import os
import subprocess
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

# 门禁基线读取源（全量基线的最新一条）；gate eval 自身的运行写入独立文件，
# 避免污染全量基线序列（judge 关的记录混入会导致下次全量对比口径错位）
_MAIN_BASELINES = os.path.join(PACKAGE_ROOT, "eval", "baselines.json")
_GATE_BASELINES = os.path.join(PACKAGE_ROOT, "eval", "baselines_gate.json")

# 噪声带：指标 → (方向, 容差)。None = 不检查
NOISE_BAND = {
    "decision_rate": ("gte", 0.02),
    "escalation_precision": ("gte", 0.02),
    "escalation_recall": ("gte", 0.02),
    "retrieval_hit_rate": ("gte", 0.02),
    "answer_rate": ("gte", 0.03),
    "faithfulness_rate": ("gte", 0.03),
    "latency_p50_s": ("lte_ratio", 1.30),
    "cost_yuan": ("lte_ratio", 1.30),
    # judge_rate 不检查：gate eval 关 judge，无同口径数据
}


def _run(cmd: list, env_extra: dict | None = None) -> int:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=PACKAGE_ROOT, env=env)


def _load_latest_baseline(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return entries[-1] if entries else None


def gate_fast() -> int:
    """密封测试批：mock + 轻量 KB，零 API 成本。"""
    rc = _run([sys.executable, "-m", "pytest", "-m", "not api and not model",
               "-q", "--no-header"])
    print("\n[gate fast] " + ("✅ 通过" if rc == 0 else "❌ 拦截：密封测试批失败"))
    return rc


def gate_eval() -> int:
    """全量评估（judge 关）+ 基线噪声带对比。"""
    baseline = _load_latest_baseline(_MAIN_BASELINES)
    if baseline is None:
        print("[gate eval] ❌ 拦截：eval/baselines.json 无基线（先跑一次全量评估建立基线）")
        return 1

    rc = _run([sys.executable, "eval/eval_multi_turn.py"],
              env_extra={"JUDGE_ENABLED": "0", "BASELINE_FILE": _GATE_BASELINES})
    if rc != 0:
        print("\n[gate eval] ❌ 拦截：评估进程异常退出")
        return 1

    # 读本次运行结果（eval 写入 _GATE_BASELINES 尾条）
    current = _load_latest_baseline(_GATE_BASELINES)
    if current is None:
        print("[gate eval] ❌ 拦截：评估完成但未沉淀基线记录")
        return 1

    print(f"\n[gate eval] 基线对比（基线={baseline['run_at']} vs 本次={current['run_at']}）")
    violations = []
    for metric, (op, tol) in NOISE_BAND.items():
        if metric not in baseline or metric not in current:
            continue
        b, c = baseline[metric], current[metric]
        if op == "gte":
            ok = c >= b - tol
            detail = f"{c:.4f} vs 基线{b:.4f}-容差{tol:.2f}"
        else:  # lte_ratio
            ok = c <= b * tol
            detail = f"{c} vs 基线{b}×{tol}"
        status = "✅" if ok else "❌越界"
        print(f"  {status} {metric}: {detail}")
        if not ok:
            violations.append(metric)

    if violations:
        print(f"[gate eval] ❌ 拦截：{len(violations)} 项指标越界 {violations}"
              f"（复跑确认，连续两次越界才定性为真回归——偶发应对策略）")
        return 1
    print("[gate eval] ✅ 通过：全指标在噪声带内")
    return 0


def gate_full() -> int:
    """全量测试（含 api/model 层）+ E2E 冒烟。"""
    rc = _run([sys.executable, "-m", "pytest", "-q", "--no-header"])
    if rc != 0:
        print("\n[gate full] ❌ 拦截：全量测试失败")
        return rc
    rc = _run([sys.executable, "verify_e2e_smoke.py"])
    if rc != 0:
        print("\n[gate full] ❌ 拦截：E2E 冒烟失败")
        return rc
    print("\n[gate full] ✅ 通过：全量测试 + E2E 冒烟")
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("fast", "eval", "full"):
        print(__doc__)
        return 2
    return {"fast": gate_fast, "eval": gate_eval, "full": gate_full}[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
