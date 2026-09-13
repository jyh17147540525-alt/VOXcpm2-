"""
AI 导演层回归测试（voice_clone.director / voice_clone.director_llm）
====================================================================
分三层，逐层加深依赖：

  L1 原文完整性（核心安全属性）
     锁死「只梳理、不改写」这个契约 —— 规则版与 LLM 版都必须逐字保留原文。
     这是整个模块存在的前提，优先级高于任何情绪判定。

  L2 判定逻辑（纯规则，只需 numpy）
     引号状态机 / 台词旁白归属 / 情绪惯性 / 停顿规划 / 显式情绪 / 接口同构。

  L3 LLM 适配层（完全离线）
     不联网，直接把模拟响应喂给 _extract_json / _merge，验证边界处理：
     数值夹回、漏标兑底、非法 role 纠正、**LLM 提供 text 时被忽略**。

导入方式沿用 test_mdx_separator 的做法：走文件路径 importlib，绕开
voice_clone/__init__.py 的链式导入（它会拉 librosa，裸 runner 不一定有）。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util as _ilu  # noqa: E402

_VOICE_CLONE = Path(__file__).resolve().parent.parent / "voice_clone"
_DIRECTOR_SRC = _VOICE_CLONE / "director.py"
_LLM_SRC = _VOICE_CLONE / "director_llm.py"

if not _DIRECTOR_SRC.exists() or not _LLM_SRC.exists():
    pytest.skip("director 源码缺失", allow_module_level=True)


def _load(mod_name: str, path: Path):
    """按文件路径加载模块并**手动注册进 sys.modules**。

    注册这一步不能省：@dataclass 在处理字符串注解（`from __future__ import
    annotations` 会把注解变成字符串）时会执行
    `sys.modules.get(cls.__module__).__dict__`，
    未注册的模块取到 None，直接抛
    `AttributeError: 'NoneType' object has no attribute '__dict__'`。

    用文件名（而非测试专用前缀）作模块名，这样 director_llm 内部的
    `from director import ...` 回退分支能直接命中已注册的那个实例，
    避免同源代码被加载成两份。
    """
    spec = _ilu.spec_from_file_location(mod_name, path)
    mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


try:
    director = _load("director", _DIRECTOR_SRC)
    llm = _load("director_llm", _LLM_SRC)
except Exception as _exc:  # pragma: no cover
    pytest.skip("director 不可导入: %s" % _exc, allow_module_level=True)


# ============================================================ L1 原文完整性

CORPUS = [
    ("empty", ""),
    ("spaces", "   \n\n  "),
    ("single", "他推开门。"),
    ("cn_quotes", "他推开门。“你来了。”阿秾的声音很轻。"),
    ("straight_quotes", '他说："好。"然后走了。'),
    ("nested_quotes", "她说：“他说‘好’。”然后走了。"),
    ("poem", "床前明月光，\n疑是地上霜。\n举头望明月，\n低头思故乡。"),
    ("no_punct", "他" * 200),
    ("mixed_lang", "他说 hello world，然后走了。"),
    ("ellipsis", "他张了张嘴……什么也没说。"),
    ("unclosed_quote", '她低声说："我等了三年。'),
    ("novel", "他推开门。\n“你来了。”阿秾的声音很轻。\n石头没说话。\n"
              "“我等了你三年！”她忽然吼道。\n窗外的雪，停了。\n"),
]


@pytest.mark.parametrize("text", [c[1] for c in CORPUS], ids=[c[0] for c in CORPUS])
def test_text_integrity(text):
    """梳理后逐字还原原文 —— 这是模块存在的底线。"""
    assert director.verify_text_intact(text, director.plan(text))


# ============================================================ L2 判定逻辑

def test_quote_direction_distinguishes_speech_from_narration():
    """引号开/闭方向决定角色：同一个 `"`，在句首可能是开也可能是闭。

    这正是必须用状态机、而不能静态判断的原因 —— 两者块首字符完全相同。
    """
    for quote_open, quote_close in (("\u201c", "\u201d"), ('"', '"')):
        t = "他推开门。%s你来了。%s阿秾的声音很轻。" % (quote_open, quote_close)
        roles = [s.role for s in director.plan(t).segments]
        assert roles[0] == "narration"
        assert roles[1] == "speech", "开引号所在块应判台词"
        assert roles[-1] == "narration", "闭引号之后应为旁白"


def test_emotion_inheritance_across_segments():
    """前句有情绪、后句无线索时应承接，否则同一场景的情绪会割裂成孤点。"""
    segs = director.plan("“我等了你三年！”她忽然吼道。").segments
    assert segs[-1].emotion == "exclamation"
    assert "承接" in segs[-1].reason


def test_pause_downgraded_when_quote_spans_chunks():
    """引号被分块切断时，块尾句号并非真正句末，应收敛为短停顿。"""
    segs = director.plan('他推开门。"你来了。"阿秾的声音很轻。').segments
    speech = [s for s in segs if s.role == "speech"]
    assert speech, "应至少有一段被判为台词"
    assert speech[0].pause_type == "comma"
    assert speech[0].pause_after < 0.30


def test_intensity_scales_with_signal_strength():
    """强度应随信号增强单调不减：单感叹 < 多重感叹。"""
    one = director.plan("“你来了！”").segments[0]
    many = director.plan("“你来了！！！”").segments[0]
    assert many.intensity >= one.intensity
    assert 0.0 <= one.intensity <= 1.0
    assert 0.0 <= many.intensity <= 1.0


def test_single_char_emotion_words_are_filtered_out():
    """单字情感词（悲/怒/喜）必须被剔除。

    「慈悲」被判成悲伤、「喜鹊」被判成喜悦、「息怒」被判成愤怒 —— 这类误判
    会给出方向完全相反的情绪，比漏判有害得多。历史实现只给单字降权（权重
    0.4），但降权不等于不命中，实测「慈悲」仍被判 sad 0.43。
    """
    for text, bad_label in (
        ("他心生慈悲，放过了那人。", "sad"),
        ("喜鹊落在枝头。", "happy"),
        ("他息怒之后，脸色平静。", "angry"),
    ):
        for s in director.plan(text).segments:
            assert s.emotion != bad_label, "%s 不应触发 %s" % (text, bad_label)


def test_single_char_only_expressions_fall_back_to_neutral():
    """已知代价（有意取舍）：只含单字词的情绪表达会漏判、回落中性。

    「勃然大怒」不含任何多字情感词，剔除单字后判为 neutral。
    这是刻意选择 —— 规则版宁可漏判也不误判，语义覆盖交给 LLM 后端。
    若哪天给词表补充了「大怒」「怒火」等词，本用例会失败，属预期变更。
    """
    segs = director.plan("他勃然大怒。").segments
    assert all(s.emotion == "neutral" for s in segs)


def test_emotion_words_contain_no_single_char_entries():
    """词表层面兜底：本模块使用的词表不得残留单字词。"""
    for label, words in director._EMOTION_WORDS.items():
        leftover = [w for w in words if len(w) < 2]
        assert not leftover, "%s 词表残留单字: %s" % (label, leftover)


def test_explicit_emotion_applies_to_all_segments():
    r = director.plan("他推开门。屋里没人。", emotion="悲伤")
    assert r.segments
    assert all(s.emotion == "悲伤" for s in r.segments)
    assert all(s.intensity == 1.0 for s in r.segments)


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t"])
def test_empty_input_yields_empty_plan(text):
    r = director.plan(text)
    assert r.segments == []
    assert r.emotions == []
    assert r.chunks_text == []


def test_interfaces_are_isomorphic_to_synthesize_stable():
    """三个属性必须与 synthesize_stable 的入参同构，接入时才可能只改一行。"""
    r = director.plan("他推开门。屋里没人。")
    n = len(r.segments)
    assert n > 0
    for seq in (r.chunks_text, r.chunks_with_pause, r.emotions, r.pause_plan):
        assert len(seq) == n
    assert all(isinstance(x, str) for x in r.chunks_text)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in r.chunks_with_pause)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in r.emotions)


def test_rule_backend_leaves_llm_fields_empty():
    """规则版不填韵律数值（None），保证走原有 _apply_emotion 路径、行为不变。"""
    for s in director.plan("他推开门。").segments:
        assert s.pace is None
        assert s.pitch_st is None
        assert s.cfg is None
        assert s.note == ""


def test_plan_result_defaults_to_rule_source():
    assert director.plan("他推开门。").source == "rule"


# ============================================================ L3 LLM 适配层

def _base(text="他推开门。屋里没人。"):
    return director.plan(text)


def test_extract_json_tolerates_markdown_and_prose():
    raw = '好的，以下是标注：\n```json\n{"tone": "克制", "segments": []}\n```\n完毕。'
    assert llm._extract_json(raw)["tone"] == "克制"


def test_extract_json_raises_on_garbage():
    with pytest.raises(Exception):
        llm._extract_json("这里根本没有 JSON")


def test_merge_ignores_text_supplied_by_llm():
    """核心安全属性：LLM 即使返回 text 字段也必须被丢弃，原文只来自分块。"""
    base = _base()
    fake = {"segments": [
        {"i": 0, "text": "恶意篡改：这句话原文里根本没有", "emotion": "平静的铺垫",
         "intensity": 0.2, "pace": 1.0, "pitch_st": 0.0, "cfg": 2.0,
         "pause_after": 0.3, "role": "narration", "note": "轻"},
        {"i": 1, "text": "同样篡改", "emotion": "空落", "intensity": 0.2,
         "pace": 1.0, "pitch_st": 0.0, "cfg": 2.0, "pause_after": 0.4,
         "role": "narration", "note": ""},
    ]}
    merged = llm._merge(base.segments, fake)
    assert [s.text for s in merged] == [s.text for s in base.segments]
    assert "篡改" not in "".join(s.text for s in merged)


def test_merge_clamps_out_of_range_numbers():
    """模型偶尔给出夸张值，必须夹回安全区间，否则会产生机械感或破音。"""
    base = _base()
    fake = {"segments": [{"i": 0, "emotion": "越界", "intensity": 99,
                          "pace": 99, "pitch_st": -99, "cfg": 99,
                          "pause_after": 99, "role": "narration", "note": ""}]}
    m = llm._merge(base.segments, fake)[0]
    assert 0.0 <= m.intensity <= 1.0
    assert 0.80 <= m.pace <= 1.20
    assert -2.0 <= m.pitch_st <= 2.0
    assert 1.8 <= m.cfg <= 3.2
    assert 0.0 <= m.pause_after <= 1.5


def test_merge_falls_back_to_rule_for_missing_segments():
    """LLM 漏标时沿用规则版结果，保证段数与顺序恒定。"""
    base = _base("他推开门。屋里没人。石头没说话。")
    fake = {"segments": [{"i": 0, "emotion": "只标了第一段"}]}
    merged = llm._merge(base.segments, fake)
    assert len(merged) == len(base.segments)
    assert merged[0].emotion == "只标了第一段"
    for i in range(1, len(base.segments)):
        assert merged[i].reason == base.segments[i].reason


def test_merge_corrects_invalid_role():
    base = _base()
    fake = {"segments": [{"i": 0, "emotion": "x", "role": "吼叫"},
                         {"i": 1, "emotion": "y", "role": None}]}
    merged = llm._merge(base.segments, fake)
    assert all(s.role in ("speech", "narration") for s in merged)


def test_merge_handles_non_dict_and_bad_index():
    base = _base()
    fake = {"segments": ["字符串", {"i": "不是数字"}, {"i": 0, "emotion": "有效"}]}
    merged = llm._merge(base.segments, fake)
    assert len(merged) == len(base.segments)
    assert merged[0].emotion == "有效"


def test_is_ready_rejects_incomplete_config():
    ok, why = llm.is_ready({"enabled": False})
    assert not ok and "enabled" in why

    ok, why = llm.is_ready({"enabled": True, "base_url": "u", "api_key": ""})
    assert not ok and "Key" in why

    ok, _ = llm.is_ready({"enabled": True, "base_url": "u", "api_key": "sk-x"})
    assert ok


def test_plan_llm_falls_back_to_rule_without_key():
    """没有 key 时必须静默降级到规则版，合成流程不能因此中断。"""
    cfg = {"enabled": True, "base_url": "https://example.invalid/v1",
           "model": "x", "api_key": ""}
    text = "他推开门。“你来了。”阿秾的声音很轻。"
    r = llm.plan_llm(text, config=cfg)
    assert r.source == "rule"
    assert r.error
    assert director.verify_text_intact(text, r)


def test_plan_llm_falls_back_on_network_error():
    """端点不可达时同样降级，且不抛异常。"""
    cfg = {"enabled": True, "base_url": "https://example.invalid/v1",
           "model": "x", "api_key": "sk-fake", "timeout": 5}
    r = llm.plan_llm("他推开门。", config=cfg)
    assert r.source == "rule"
    assert r.error


def test_plan_llm_raises_when_fallback_disabled():
    cfg = {"enabled": True, "base_url": "u", "model": "x", "api_key": ""}
    with pytest.raises(Exception):
        llm.plan_llm("他推开门。", config=cfg, fallback=False)


def test_load_config_fills_defaults(tmp_path):
    cfg = llm.load_config(str(tmp_path / "nonexistent.json"))
    assert cfg["base_url"]
    assert cfg["model"]
    assert "enabled" in cfg
