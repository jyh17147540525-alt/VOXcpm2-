"""llm_providers 单元测试：服务商预设表 + Key 清洗 + 软校验。

这层是"适配所有密钥格式"的地基，一旦漂移会同时影响 UI 下拉、后端校验和
文档，所以把不变量都钉死。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_clone import llm_providers as LP  # noqa: E402


# --------------------------------------------------------------------- 表结构

def test_providers_non_empty_and_unique():
    assert len(LP.PROVIDERS) >= 8
    ids = [p["id"] for p in LP.PROVIDERS]
    assert len(ids) == len(set(ids)), "provider id 必须唯一"


def test_every_provider_has_required_fields():
    for p in LP.PROVIDERS:
        assert p["id"], p
        assert p["name_zh"] and p["name_en"], p
        assert "base_url" in p and "models" in p, p
        assert isinstance(p["models"], list), p
        assert "local" in p, p
        # custom 之外都必须有 base_url，否则预设无意义
        if p["id"] != "custom":
            assert p["base_url"].startswith("http"), p


def test_every_provider_has_english_hints():
    """中英双语界面要求：key_hint / note 都必须有英文版本。

    否则英文界面下会出现「Key format: sk- 开头」这种半中半英的混排。
    """
    for p in LP.PROVIDERS:
        assert str(p.get("key_hint_en") or "").strip(), f"{p['id']} 缺 key_hint_en"
        assert str(p.get("note_en") or "").strip(), f"{p['id']} 缺 note_en"
        assert p["key_hint_en"] != p["key_hint"], f"{p['id']} 的英文提示没真正翻译"
        assert p["note_en"] != p["note"], f"{p['id']} 的英文说明没真正翻译"


def test_describe_projects_english_hints():
    """describe() 是预设表的投影，新字段也要一并透出。"""
    d = LP.describe("deepseek")
    assert str(d.get("key_hint_en") or "").strip()
    assert str(d.get("note_en") or "").strip()


def test_base_urls_do_not_end_with_slash():
    """统一不带尾斜杠，拼接 /chat/completions 时才不会出双斜杠。"""
    for p in LP.PROVIDERS:
        if p["base_url"]:
            assert not p["base_url"].endswith("/"), p["id"]


def test_base_urls_do_not_include_chat_completions():
    """预设里绝不该出现完整端点，否则 _chat 会拼出 .../chat/completions/chat/completions。"""
    for p in LP.PROVIDERS:
        assert "/chat/completions" not in p["base_url"], p["id"]


# --------------------------------------------------------------------- 查询

def test_get_provider_known():
    assert LP.get_provider("deepseek")["id"] == "deepseek"
    assert LP.get_provider("ollama")["local"] is True


@pytest.mark.parametrize("bad", [None, "", "  ", "not-exist", "DEEPSEEK"])
def test_get_provider_unknown_falls_back_to_custom(bad):
    assert LP.get_provider(bad)["id"] == "custom"


@pytest.mark.parametrize("pid", [p["id"] for p in LP.PROVIDERS])
def test_detect_provider_roundtrip(pid):
    p = LP.get_provider(pid)
    if not p["base_url"]:
        pytest.skip("custom 无 base_url")
    assert LP.detect_provider(p["base_url"]) == pid
    assert LP.detect_provider(p["base_url"] + "/") == pid
    assert LP.detect_provider(p["base_url"].upper()) == pid


def test_detect_provider_unknown_and_empty():
    assert LP.detect_provider("https://my-proxy.example.com/v1") == "custom"
    assert LP.detect_provider("") == "custom"
    assert LP.detect_provider(None) == "custom"


def test_detect_provider_host_level_fallback():
    """用户可能自行加了路径，域名相同也应认出来。"""
    assert LP.detect_provider("https://api.deepseek.com") == "deepseek"
    assert LP.detect_provider("https://api.deepseek.com/beta/v1") == "deepseek"


# --------------------------------------------------------------------- Key 清洗

@pytest.mark.parametrize("raw,expect", [
    ("  sk-abc  ", "sk-abc"),
    ("sk-abc\n", "sk-abc"),
    ("sk-a b c", "sk-abc"),
    ("sk-a\nb", "sk-ab"),
    ('"sk-x"', "sk-x"),
    ("'sk-x'", "sk-x"),
    ("Bearer sk-y", "sk-y"),
    ("bearer  sk-y", "sk-y"),
    ("BEARER sk-y", "sk-y"),
    ("", ""),
    (None, ""),
    ("   ", ""),
])
def test_normalize_key(raw, expect):
    assert LP.normalize_key(raw) == expect


def test_normalize_key_preserves_chinese():
    """不因为含中文就删字符，校验职责归 key_looks_valid。"""
    assert LP.normalize_key("sk-中文") == "sk-中文"


def test_normalize_key_does_not_strip_internal_dashes_dots():
    """智谱的 id.secret 格式必须原样保留。"""
    k = "abcdefg1234567.ABCDEFG1234567"
    assert LP.normalize_key(k) == k


# --------------------------------------------------------------------- 软校验

def test_key_valid_ok():
    ok, msg = LP.key_looks_valid("sk-abcdef123456", "deepseek")
    assert ok is True and msg == ""


def test_key_valid_empty():
    ok, msg = LP.key_looks_valid("", "deepseek")
    assert ok is False and "不能为空" in msg
    assert LP.key_looks_valid("   ", "deepseek")[0] is False


def test_key_valid_too_short():
    ok, msg = LP.key_looks_valid("short", "deepseek")
    assert ok is False and "太短" in msg


def test_key_valid_rejects_chinese():
    ok, msg = LP.key_looks_valid("sk-中文key12345", "deepseek")
    assert ok is False and "中文" in msg


def test_key_valid_rejects_url():
    ok, msg = LP.key_looks_valid("https://api.deepseek.com/v1", "deepseek")
    assert ok is False and "网址" in msg


def test_key_valid_rejects_absurdly_long():
    ok, msg = LP.key_looks_valid("sk-" + "x" * 600, "deepseek")
    assert ok is False and "长" in msg


def test_key_valid_local_provider_allows_placeholder():
    ok, msg = LP.key_looks_valid("ollama", "ollama")
    assert ok is True and "本地" in msg


def test_key_valid_local_provider_allows_empty():
    """本地服务无鉴权，空 key 也应放行（_chat 会发空 Bearer）。"""
    ok, _ = LP.key_looks_valid("", "ollama")
    assert ok is True


def test_key_valid_strips_before_judging():
    """带换行的合法 key 不应被拒。"""
    ok, _ = LP.key_looks_valid("  sk-abcdef123456\n", "deepseek")
    assert ok is True


# --------------------------------------------------------------------- describe

def test_describe_shape():
    d = LP.describe("moonshot")
    for k in ("provider", "name_zh", "name_en", "base_url", "models",
              "key_hint", "key_url", "local", "note",
              "detected_provider", "matches_detected", "chat_url"):
        assert k in d, k


def test_describe_without_base_url_has_empty_chat_url():
    assert LP.describe("deepseek")["chat_url"] == ""


def test_describe_chat_url_concatenation():
    d = LP.describe("deepseek", "https://api.deepseek.com/v1")
    assert d["chat_url"] == "https://api.deepseek.com/v1/chat/completions"
    assert d["matches_detected"] is True


def test_describe_chat_url_no_double_slash():
    """带尾斜杠的 base_url 不能拼出双斜杠。"""
    d = LP.describe("deepseek", "https://api.deepseek.com/v1/")
    assert d["chat_url"] == "https://api.deepseek.com/v1/chat/completions"


def test_describe_mismatch_flagged():
    d = LP.describe("deepseek", "https://api.moonshot.cn/v1")
    assert d["provider"] == "deepseek"
    assert d["detected_provider"] == "moonshot"
    assert d["matches_detected"] is False


# --------------------------------------------------------------------- apply_preset

def test_apply_preset_fills_blanks():
    c = LP.apply_preset({"provider": "zhipu"})
    assert c["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
    assert c["model"] == "glm-4-flash"


def test_apply_preset_never_overwrites_user_values():
    """用户手填的地址/模型必须原样保留——被静默冲掉是最难查的 bug。"""
    c = LP.apply_preset({
        "provider": "zhipu",
        "base_url": "https://my-gateway.internal/v1",
        "model": "my-custom-model",
    })
    assert c["base_url"] == "https://my-gateway.internal/v1"
    assert c["model"] == "my-custom-model"


def test_apply_preset_infers_provider_from_base_url():
    c = LP.apply_preset({"base_url": "https://api.moonshot.cn/v1"})
    assert c["model"] == "moonshot-v1-8k"


def test_apply_preset_custom_with_no_model_leaves_empty():
    c = LP.apply_preset({"provider": "custom"})
    assert c["base_url"] == ""
    assert c["model"] == ""


def test_apply_preset_does_not_mutate_input():
    src = {"provider": "ollama"}
    _ = LP.apply_preset(src)
    assert src == {"provider": "ollama"}, "不应改动调用方传入的 dict"


def test_apply_preset_empty_input():
    c = LP.apply_preset({})
    assert "base_url" in c and isinstance(c["base_url"], str)


def test_apply_preset_none_safe():
    c = LP.apply_preset(None)
    assert isinstance(c, dict)


# --------------------------------------------------------------- 超时推荐值

def test_recommended_timeout_local_is_longer():
    """本地模型首次调用要加载权重（实测 27B 冷启约 30s），超时必须给足。"""
    assert LP.recommended_timeout("ollama") >= 300
    assert LP.recommended_timeout("lmstudio") >= 300
    assert LP.recommended_timeout("vllm") >= 300


def test_recommended_timeout_cloud_is_moderate():
    for pid in ("deepseek", "openai", "moonshot", "dashscope", "zhipu"):
        t = LP.recommended_timeout(pid)
        assert 30 <= t <= 120, (pid, t)


def test_recommended_timeout_unknown_falls_back():
    assert LP.recommended_timeout("no-such-provider") == LP.recommended_timeout("custom")


def test_all_local_providers_are_flagged():
    """local 标记决定三件事：空 key 放行、跳过环境变量回退、长超时。
    标错会同时破坏这三处，所以逐个钉死。"""
    expected_local = {"ollama", "lmstudio", "vllm"}
    actual = {p["id"] for p in LP.PROVIDERS if p.get("local")}
    assert actual == expected_local, actual
    for pid in expected_local:
        assert "127.0.0.1" in LP.get_provider(pid)["base_url"]
