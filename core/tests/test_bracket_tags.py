# -*- coding: utf-8 -*-
"""括号标记解析的回归测试（防"模型把标签念出来"复发）。

背景（真实事故）
----------------
早期 `parse_multi_speaker_text` 只认 6 个情绪词（高兴/悲伤/严肃/温柔/愤怒/平静），
**其余括号内容一律"当普通文本保留"**。于是用户输入：

    （愤怒嘶吼）我说过多少次了！（压抑的愤怒）你从来不听。

中的两个标签被回填进朗读文本 → 模型真的把「压抑的愤怒」念了出来。

已用 ASR 实测确认（Paraformer）：
    · 修复前文本 → 转写 "我说过多少次了**压医的愤怒**你从来不听"   ← 标签被念
    · 修复后文本 → 转写 "我说过多少次了你从来不听"                 ← 干净

本测试把该行为固化：**情绪/语气标签必须从朗读文本里消失**，
而**音色/风格描述必须保留**（那是 Voice Design 用法，剥了会丢用户内容）。

这两条是本次修复的核心不变量，任何一条被破坏都应立刻失败。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import parse_multi_speaker_text  # noqa: E402


def joined(text: str) -> str:
    """把解析结果拼回"喂给模型的朗读文本"。"""
    return "".join(s["text"] for s in parse_multi_speaker_text(text))


# 情绪 / 语气标签：**必须剥离**（本 bug 的正面修复）
STRIP_CASES = [
    "（愤怒嘶吼）我说过多少次了！（压抑的愤怒）你从来不听。",
    "（大笑）哈哈哈哈",
    "（轻声耳语）别出声，他就在隔壁。",
    "（疲惫叹息）算了。",
    "（放声大哭）为什么……为什么你要走……",
    "（惊讶倒吸一口气）你……你怎么会在这里？",
    "（冷笑讽刺）你可真行啊。",
    "（咳嗽）不好意思，我失态了。",
    "（痛）嘶——好疼。",
    "（耳语）甲。（轻微）乙。",
    "（愤怒）我说过多少次了！",
    "（温柔）别出声。",
    "（高兴）太好了！",
    "（颤抖）我……我不敢。",
    "（喃喃自语）都过去了。",
    "（疯狂的大笑）哈哈哈哈哈！",
]

# 音色 / 风格描述：**必须保留**（Voice Design，剥了会丢内容）
KEEP_CASES = [
    "年轻女性，温柔甜美",
    "老年男性，沙哑",
    "上海口音",
    "东北口音",
    "台湾腔调",
    "沙哑嗓音",
    "磁性声音",
    "低沉男声",
    "清脆女声",
    "儿童声音",
    "清冷少女",
    "沉稳大叔",
    "播音腔",
    "娃娃音",
    "烟嗓",
    "少年音",
    "御姐音",
    "少女感",
    "温柔语气",
    "严肃语气",
    "English accent",
    "male, deep",
    "30岁男性",
]

# 更多语气标签（含语气库未收录、需靠补充层 + 结构兜底识别的）
EXTRA_STRIP_CASES = [
    "喃喃自语", "嘟囔", "嘀咕", "哽咽", "呜咽", "抽泣",
    "咆哮", "怒吼", "冷笑", "讥笑", "阴阳怪气",
    "深吸一口气", "清嗓子", "呻吟", "闷哼", "颤抖", "低声",
]


@pytest.mark.parametrize("text", STRIP_CASES)
def test_tone_tags_are_stripped(text):
    """情绪/语气标签不得出现在朗读文本里。"""
    out = joined(text)
    # 取出所有括号内容，逐个确认它们"不是被原样保留"
    import re
    for m in re.finditer(r"[（(]([^()（）]*)[)）]", text):
        tag = m.group(1).strip()
        if tag.startswith("@"):
            continue
        assert tag not in out, (
            f"标签「{tag}」泄漏进朗读文本！\n"
            f"  输入: {text}\n"
            f"  输出: {out!r}\n"
            f"  这正是「模型把括号内容念出来」那个 bug。"
        )


def test_reported_bug_specific():
    """用户报告的原始用例：逐字断言两个标签都不在朗读文本里。"""
    text = "（愤怒嘶吼）我说过多少次了！（压抑的愤怒）你从来不听。"
    out = joined(text)
    assert "压抑的愤怒" not in out
    assert "愤怒嘶吼" not in out
    assert "压抑" not in out
    assert "嘶吼" not in out
    # 台词本体必须完整保留
    assert out == "我说过多少次了！你从来不听。"


@pytest.mark.parametrize("text", KEEP_CASES)
def test_design_hints_are_kept(text):
    """音色/风格描述必须保留 —— 剥掉会丢用户内容，是反向事故。

    ⚠️ 这组用例是**真实事故的防线**：加结构性兜底时，曾把
    「低沉男声」「烟嗓」「少年音」「御姐音」「少女感」「播音腔」
    误判成语气标签剥掉 —— 修一个 bug 时引入了另一个 bug。
    该测试保证"剥离"不会吃掉 Voice Design 的音色描述。
    """
    tag = text
    out = joined(f"（{tag}）你好。")
    assert tag in out, (
        f"音色/风格描述「{tag}」被误剥！\n"
        f"  输出: {out!r}\n"
        f"  这类内容是 Voice Design 提示，必须留给模型。"
    )


@pytest.mark.parametrize("tag", EXTRA_STRIP_CASES)
def test_extra_tone_words_are_stripped(tag):
    """语气库未收录的常见语气词也必须剥离（补充层 + 结构兜底）。"""
    out = joined(f"（{tag}）你好。")
    assert tag not in out, (
        f"语气标签「{tag}」泄漏进朗读文本（模型会把它念出来）。输出: {out!r}"
    )


def test_emotion_tag_sets_segment_emotion():
    """情绪标签应影响该段的 emotion 字段（而非只是被丢掉）。"""
    segs = parse_multi_speaker_text("（愤怒）我说过多少次了！")
    assert segs, "应解析出至少一段"
    assert segs[0]["emotion"] == "愤怒"
    assert segs[0]["text"] == "我说过多少次了！"


def test_tone_tag_normalizes_to_engine_emotion():
    """细分语气应就近归一到一个引擎支持的规范情绪，而不是 neutral。"""
    from server import _BETA_ENGINE_EMOTIONS
    segs = parse_multi_speaker_text("（大笑）哈哈哈哈")
    assert segs[0]["emotion"] in _BETA_ENGINE_EMOTIONS
    assert segs[0]["emotion"] != "neutral", (
        "细分语气「大笑」应归一到「高兴」，落到 neutral 说明归一失效"
    )


def test_emotion_words_are_all_engine_supported():
    """词表里的每个规范情绪都必须是引擎真支持的键。

    否则 `_EMOTION_UNIFORM.get()` 落空 → 情绪静默不生效（写着的功能其实是死的）。
    """
    from server import _BETA_EMOTION_WORDS, _BETA_ENGINE_EMOTIONS
    for alias, emo in _BETA_EMOTION_WORDS.items():
        assert emo in _BETA_ENGINE_EMOTIONS, (
            f"别名「{alias}」映射到「{emo}」，但引擎不支持该键"
        )


def test_intensity_modifier_alone_does_not_leak():
    """裸强度修饰（如「（轻微）」）也必须剥离。"""
    out = joined("（耳语）甲。（轻微）乙。")
    assert "轻微" not in out
    assert out == "甲。乙。"


def test_no_voice_tag_leaks():
    """(@音色) 标记不得泄漏，且应切换 voice 字段。"""
    segs = parse_multi_speaker_text("(@小明)你好。(@小红)你好。")
    assert all("@小明" not in s["text"] for s in segs)
    assert all("@小红" not in s["text"] for s in segs)
    assert segs[0]["voice"] == "小明"
    assert segs[1]["voice"] == "小红"
