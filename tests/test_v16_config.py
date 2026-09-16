"""V16 测试：配置统一（默认模型与 .env/eval 真相源一致 + 无硅基流动残留）。

背景（审查清单 V16）：
- config.py 默认 qwen3.7-flash vs .env 实际 / eval 基线 / 报告全部 qwen3.7-plus-2026-05-26 → 漂移
- env_example.txt / README_LangGraph_CLI.md 残留 SiliconFlow（硅基流动）旧文案
  （base_url 是 api.siliconflow.cn，与新真相源阿里云百炼冲突）

修复：config 默认统一为 qwen3.7-plus-2026-05-26（跟随真相源）；env_example 重写；
README 示例更新；eval 复现脚本兜底字符串统一。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
    with open(p, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 默认模型一致性
# ---------------------------------------------------------------------------

def test_config_default_model_matches_truth():
    """config 默认模型 = deepseek-flash（与 .env/eval 真相源一致，2026-09-15 切换）。"""
    import config
    assert config.OPENAI_MODEL == "deepseek-flash", (
        f"config 默认模型漂移: {config.OPENAI_MODEL}"
    )


def test_config_default_base_url_is_deepseek():
    """config 默认端点 = DeepSeek 官方，非百炼/硅基流动。"""
    import config
    assert "api.deepseek.com" in config.OPENAI_BASE_URL
    assert "siliconflow" not in config.OPENAI_BASE_URL


def test_env_override_still_works(monkeypatch):
    """env 覆盖仍生效（统一的是默认值，不是写死）。"""
    monkeypatch.setenv("OPENAI_MODEL", "custom-test-model")
    import importlib
    import config
    importlib.reload(config)
    assert config.OPENAI_MODEL == "custom-test-model"
    monkeypatch.delenv("OPENAI_MODEL")
    importlib.reload(config)
    assert config.OPENAI_MODEL == "deepseek-flash"


# ---------------------------------------------------------------------------
# 文档残留
# ---------------------------------------------------------------------------

def test_env_example_matches_deepseek_truth():
    """env_example.txt 真相源 = DeepSeek 官方端点（2026-09-15 切换）。"""
    src = _read("env_example.txt")
    # 实际配置值（非注释）必须无硅基流动/百炼
    assert "api.siliconflow.cn" not in src, "env_example 残留硅基流动端点"
    assert "OPENAI_MODEL=Qwen" not in src, "env_example 残留硅基流动模型格式"
    # 真相源配置必须正确
    assert "api.deepseek.com" in src
    assert "OPENAI_MODEL=deepseek-flash" in src


def test_readme_cli_matches_deepseek_truth():
    """README_LangGraph_CLI.md 环境变量示例 = DeepSeek 官方端点。"""
    src = _read("README_LangGraph_CLI.md")
    # 示例里的 base_url 是 DeepSeek（迁移说明注释可提及历史，但示例本身必须正确）
    assert "api.siliconflow.cn" not in src, "README CLI 示例残留硅基流动端点"
    assert "api.deepseek.com" in src
    assert "deepseek-flash" in src


def test_eval_repro_fallback_model_unified():
    """eval 复现脚本的模型兜底字符串统一为 plus。"""
    src = _read("eval/eval_classifier_repro.py")
    assert "qwen3.7-flash" not in src, "eval 复现脚本残留 flash 兜底"


def test_config_default_line_matches_deepseek():
    """config 默认值行 = deepseek-flash（2026-09-15 真相源；旧 qwen-flash 防漂移断言随生态切换失效）。"""
    src = _read("config.py")
    assert 'OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-flash")' in src
    # 旧真相源不得作为默认值残留
    import re
    m = re.search(r'OPENAI_MODEL = os\.getenv\("OPENAI_MODEL", "([^"]+)"\)', src)
    assert m and m.group(1) == "deepseek-flash"
