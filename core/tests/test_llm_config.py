"""director_llm 的配置层与诊断层测试。

覆盖三件容易出错、且用户直接可见的事：
1. api_key 回退链的**优先级**与**本地服务例外**（环境变量污染真实发生过）
2. api_key_cleared 显式清空语义（否则用户点了清空、界面还显示已配置）
3. test_connection 的契约：**永不抛异常**，失败也要返回结构化诊断
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_clone import director_llm as llm  # noqa: E402
from voice_clone import llm_providers as LP  # noqa: E402


@pytest.fixture
def tmp_cfg(tmp_path, monkeypatch):
    """把 CONFIG_PATH / API_KEY_TXT 指向临时目录，避免污染真实配置。"""
    p = tmp_path / "llm_config.json"
    k = tmp_path / "api_key.txt"
    monkeypatch.setattr(llm, "CONFIG_PATH", str(p))
    monkeypatch.setattr(llm, "API_KEY_TXT", str(k))
    # 清掉可能存在的环境变量，避免宿主环境干扰断言
    for env in ("DEEPSEEK_API_KEY", "VOXCPM_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    return p, k


# --------------------------------------------------------------------- 默认值

def test_default_config_has_provider():
    assert "provider" in llm.DEFAULT_CONFIG
    assert "api_key_cleared" in llm.DEFAULT_CONFIG


def test_load_config_returns_defaults_when_no_file(tmp_cfg):
    c = llm.load_config()
    assert c["enabled"] is False
    assert c["provider"] == "deepseek"
    assert c["base_url"] == "https://api.deepseek.com/v1"
    assert c["model"] == "deepseek-chat"
    assert c["api_key"] == ""


def test_load_config_merges_user_file(tmp_cfg):
    p, _ = tmp_cfg
    p.write_text(json.dumps({"enabled": True, "model": "my-model"}), encoding="utf-8")
    c = llm.load_config()
    assert c["enabled"] is True
    assert c["model"] == "my-model"
    # 未指定的字段保留默认
    assert c["provider"] == "deepseek"


def test_load_config_survives_corrupt_json(tmp_cfg):
    p, _ = tmp_cfg
    p.write_text("{ 这不是 json", encoding="utf-8")
    c = llm.load_config()          # 不能抛
    assert c["provider"] == "deepseek"


# --------------------------------------------------------------------- 回退链

def test_api_key_falls_back_to_txt(tmp_cfg):
    p, k = tmp_cfg
    p.write_text(json.dumps({}), encoding="utf-8")
    k.write_text("sk-from-txt-file\n", encoding="utf-8")
    assert llm.load_config()["api_key"] == "sk-from-txt-file"


def test_api_key_falls_back_to_env(tmp_cfg, monkeypatch):
    p, _ = tmp_cfg
    p.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    assert llm.load_config()["api_key"] == "sk-from-env"


def test_config_file_wins_over_env(tmp_cfg, monkeypatch):
    p, _ = tmp_cfg
    p.write_text(json.dumps({"api_key": "sk-from-file"}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    assert llm.load_config()["api_key"] == "sk-from-file"


def test_env_priority_order(tmp_cfg, monkeypatch):
    p, _ = tmp_cfg
    p.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-first")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-third")
    assert llm.load_config()["api_key"] == "sk-first"


# ------------------------------------------------- 本地服务：环境变量不得污染

def test_local_provider_ignores_env_api_key(tmp_cfg, monkeypatch):
    """本地部署不该被环境里的云服务商密钥污染（这会把 key 发给本机地址）。"""
    p, _ = tmp_cfg
    p.write_text(json.dumps({"provider": "ollama", "api_key": ""}), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-cloud-secret")
    c = llm.load_config()
    assert c["api_key"] == "ollama", c["api_key"]
    assert c["api_key"] != "sk-cloud-secret"


def test_local_provider_gets_placeholder_when_key_empty(tmp_cfg):
    p, _ = tmp_cfg
    p.write_text(json.dumps({"provider": "ollama", "api_key": ""}), encoding="utf-8")
    c = llm.load_config()
    assert c["api_key"] == "ollama"


def test_local_provider_fills_url_and_model(tmp_cfg):
    p, _ = tmp_cfg
    p.write_text(json.dumps({"provider": "lmstudio", "base_url": "",
                             "model": ""}), encoding="utf-8")
    c = llm.load_config()
    assert c["base_url"] == "http://127.0.0.1:1234/v1"
    assert c["model"] == "local-model"


# ------------------------------------------------------------- 显式清空语义

def test_cleared_key_skips_env_fallback(tmp_cfg, monkeypatch):
    """用户显式清空 key 后，环境变量不得把它"变回来"。"""
    p, _ = tmp_cfg
    p.write_text(json.dumps({"api_key": "", "api_key_cleared": True}),
                 encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-be-ignored")
    c = llm.load_config()
    assert c["api_key"] == "", "清空被环境变量回填了"
    assert c["api_key_cleared"] is True


def test_cleared_key_skips_txt_fallback(tmp_cfg):
    p, k = tmp_cfg
    p.write_text(json.dumps({"api_key": "", "api_key_cleared": True}),
                 encoding="utf-8")
    k.write_text("sk-from-txt", encoding="utf-8")
    assert llm.load_config()["api_key"] == ""


def test_cleared_local_provider_still_gets_placeholder(tmp_cfg):
    p, _ = tmp_cfg
    p.write_text(json.dumps({"provider": "ollama", "api_key": "",
                             "api_key_cleared": True}), encoding="utf-8")
    assert llm.load_config()["api_key"] == "ollama"


# --------------------------------------------------------------------- 保存

def test_save_config_marks_cleared_when_key_empty(tmp_cfg):
    p, _ = tmp_cfg
    llm.save_config({"api_key": "sk-x", "model": "m"})
    assert json.loads(p.read_text(encoding="utf-8"))["api_key_cleared"] is False
    llm.save_config({"api_key": ""})
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["api_key"] == ""
    assert d["api_key_cleared"] is True


def test_save_config_clears_flag_on_new_key(tmp_cfg):
    p, _ = tmp_cfg
    llm.save_config({"api_key": ""})
    assert json.loads(p.read_text(encoding="utf-8"))["api_key_cleared"] is True
    llm.save_config({"api_key": "sk-new"})
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["api_key"] == "sk-new"
    assert d["api_key_cleared"] is False


def test_save_config_preserves_unknown_fields(tmp_cfg):
    p, _ = tmp_cfg
    p.write_text(json.dumps({"my_custom_field": 42}), encoding="utf-8")
    llm.save_config({"model": "m"})
    assert json.loads(p.read_text(encoding="utf-8"))["my_custom_field"] == 42


def test_save_config_roundtrip(tmp_cfg):
    p, _ = tmp_cfg
    llm.save_config({"provider": "moonshot", "model": "moonshot-v1-8k",
                     "api_key": "sk-roundtrip"})
    c = llm.load_config()
    assert c["provider"] == "moonshot"
    assert c["model"] == "moonshot-v1-8k"
    assert c["api_key"] == "sk-roundtrip"


def test_save_config_does_not_leak_placeholder_for_remote(tmp_cfg):
    """远端服务存空 key 后，load 不应凭空出现一个 key（除非环境变量真的给了）。"""
    p, _ = tmp_cfg
    llm.save_config({"provider": "deepseek", "api_key": ""})
    assert llm.load_config()["api_key"] == ""


# --------------------------------------------------------------------- 超时

def test_timeout_defaults_by_provider(tmp_cfg):
    """没显式设过 timeout → 按服务商给推荐值。"""
    p, _ = tmp_cfg
    p.write_text(json.dumps({"provider": "ollama"}), encoding="utf-8")
    assert llm.load_config()["timeout"] >= 300
    p.write_text(json.dumps({"provider": "deepseek"}), encoding="utf-8")
    assert llm.load_config()["timeout"] <= 120


def test_timeout_user_value_is_respected(tmp_cfg):
    """用户显式设过的 timeout 必须原样保留，不能被"智能纠偏"偷偷改掉。"""
    p, _ = tmp_cfg
    p.write_text(json.dumps({"provider": "ollama", "timeout": 45}), encoding="utf-8")
    assert llm.load_config()["timeout"] == 45


def test_timeout_none_removes_key(tmp_cfg):
    """传 timeout=None 表示"回到推荐值"，应把该键从文件删掉。"""
    p, _ = tmp_cfg
    llm.save_config({"provider": "deepseek", "timeout": 999})
    assert json.loads(p.read_text(encoding="utf-8"))["timeout"] == 999
    llm.save_config({"provider": "ollama", "timeout": None})
    d = json.loads(p.read_text(encoding="utf-8"))
    assert "timeout" not in d, "timeout=None 未能删除该键"
    assert llm.load_config()["timeout"] >= 300


def test_switching_provider_refreshes_timeout(tmp_cfg):
    """从云切到本地时，遗留的 60s 不该被继承（否则每次都超时）。"""
    p, _ = tmp_cfg
    llm.save_config({"provider": "deepseek", "timeout": 60})
    assert llm.load_config()["timeout"] == 60
    # 模拟界面切换服务商：先删掉遗留 timeout，再按新 provider 取值
    llm.save_config({"provider": "ollama", "base_url": "", "model": "",
                     "timeout": None})
    assert llm.load_config()["timeout"] >= 300


# --------------------------------------------------------------------- is_ready

def test_is_ready_local_provider_ok_without_key():
    ok, why = llm.is_ready({"enabled": True, "provider": "ollama",
                            "base_url": "http://127.0.0.1:11434/v1",
                            "model": "qwen2.5:7b", "api_key": ""})
    assert ok, why


# ------------------------------------------------------- list_models / test_connection

def test_list_models_returns_empty_on_bad_host():
    """不可达的地址返回空列表，不能抛异常。"""
    assert llm.list_models({"base_url": "http://127.0.0.1:9/v1",
                            "api_key": "x", "timeout": 1, "proxy": ""}) == []


def test_list_models_returns_empty_on_missing_base_url():
    assert llm.list_models({"base_url": "", "api_key": "x"}) == []


@pytest.mark.parametrize("cfg,expect_stage", [
    ({"provider": "custom", "base_url": "", "model": "x", "api_key": "sk-abcdef12"},
     "config"),
    # 用 custom：预设没有默认 model，空 model 才能真的触发"缺 model"分支
    # （deepseek 之类的预设会自动补 model，测不出这个分支）
    ({"provider": "custom", "base_url": "https://api.deepseek.com/v1",
      "model": "", "api_key": "sk-abcdef12"}, "config"),
    ({"provider": "deepseek", "base_url": "https://api.deepseek.com/v1",
      "model": "m", "api_key": ""}, "config"),
    ({"provider": "deepseek", "base_url": "https://api.deepseek.com/v1",
      "model": "m", "api_key": "中文key中文"}, "config"),
])
def test_test_connection_stops_at_config_stage(cfg, expect_stage):
    r = llm.test_connection(cfg, probe_chat=True)
    assert isinstance(r, dict)
    assert r["ok"] is False
    assert r["stage"] == expect_stage
    assert r["message"]


def test_test_connection_never_raises_on_dead_host():
    r = llm.test_connection({"provider": "ollama",
                             "base_url": "http://127.0.0.1:9/v1",
                             "model": "m", "api_key": "x", "timeout": 1},
                            probe_chat=True)
    assert isinstance(r, dict) and "ok" in r


def test_test_connection_result_shape():
    r = llm.test_connection({"provider": "custom", "base_url": "",
                             "model": "", "api_key": ""})
    for k in ("ok", "stage", "message", "provider", "base_url", "model",
              "models", "latency_ms", "chat_ok"):
        assert k in r, k


def test_test_connection_probe_chat_false_skips_chat_stage():
    """probe_chat=False 时不打 chat 端点，避免多花一次调用。"""
    r = llm.test_connection({"provider": "ollama",
                             "base_url": "http://127.0.0.1:9/v1",
                             "model": "m", "api_key": "ollama", "timeout": 1},
                            probe_chat=False)
    assert isinstance(r, dict)
    assert r.get("chat_ok") is False


def test_test_connection_preset_fills_missing_model():
    """预设会自动补 model，所以"空 model"对有预设的服务商不会停在 config 阶段。

    这是刻意的：用户只选了服务商、还没填 model 时，直接给他预设的默认模型
    体验更好。此用例把这个行为固化下来，避免以后误判为 bug。
    """
    r = llm.test_connection({"provider": "deepseek",
                             "base_url": "https://api.deepseek.com/v1",
                             "model": "", "api_key": "sk-abcdef123456",
                             "timeout": 2}, probe_chat=False)
    assert r["model"] == "deepseek-chat", r
    assert r["stage"] != "config", "预设应已补齐 model，不该停在 config"
