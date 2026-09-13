"""
台词对齐算法回归测试（voice_clone.transcriber）
=============================================
覆盖 v2 对齐链的纯逻辑部分，**不需要模型/音频/网络**。

对应历史事故：
  · 旧版按"语速均匀"把台词摊到各段 → 与真实语音节奏不符，长句漂移严重。
    v2 改为词级锚点 + 全局编辑对齐，这里锁死其行为。
  · 末段曾被台词尾部"硬塞"，污染最后一段文本 → 改为统计 dropped。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_clone import transcriber as T  # noqa: E402


# ============================== _fold / _text_atoms ==============================

class TestFold:
    def test_fullwidth_to_halfwidth(self):
        assert T._fold("Ａ") == "a"
        assert T._fold("１") == "1"

    def test_ascii_lowercased(self):
        assert T._fold("A") == "a"
        assert T._fold("Z") == "z"

    def test_cjk_unchanged(self):
        assert T._fold("中") == "中"
        assert T._fold("→") == "→"


class TestTextAtoms:
    def test_cjk_split_per_char(self):
        atoms = T._text_atoms("你好吗")
        assert [a["ch"] for a in atoms] == ["你", "好", "吗"]

    def test_ascii_word_kept_whole(self):
        atoms = T._text_atoms("hello world")
        assert [a["ch"] for a in atoms] == ["hello", "world"]

    def test_punctuation_is_glue_not_atom(self):
        atoms = T._text_atoms("你好，世界！")
        assert [a["ch"] for a in atoms] == ["你", "好", "世", "界"]

    def test_offset_points_into_original(self):
        text = "你好，world"
        atoms = T._text_atoms(text)
        for a in atoms:
            assert text[a["o"]:a["o"] + len(a["ch"])] == a["ch"]

    def test_mixed_cjk_ascii(self):
        atoms = T._text_atoms("AI 很好")
        assert [a["ch"] for a in atoms] == ["AI", "很", "好"]

    def test_empty_and_punct_only(self):
        assert T._text_atoms("") == []
        assert T._text_atoms("，。！ ") == []


# ============================== _add_word_anchors ==============================

class _W:
    """模拟 faster-whisper 的 word 对象。"""
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class TestAddWordAnchors:
    def test_cjk_expanded_per_char(self):
        tl = []
        T._add_word_anchors(tl, _W("你好", 0.0, 2.0))
        assert len(tl) == 2
        assert tl[0]["u"] == "你" and tl[1]["u"] == "好"
        assert tl[0]["s"] == pytest.approx(0.0)
        assert tl[1]["e"] == pytest.approx(2.0)

    def test_ascii_kept_whole(self):
        tl = []
        T._add_word_anchors(tl, _W("hello", 1.0, 2.0))
        assert len(tl) == 1 and tl[0]["u"] == "hello"

    def test_anchor_times_are_monotone(self):
        tl = []
        for w in [_W("a", 0.0, 1.0), _W("b", 1.0, 2.0), _W("c", 2.0, 3.0)]:
            T._add_word_anchors(tl, w)
        ts = [a["t"] for a in tl]
        assert ts == sorted(ts)

    def test_punctuation_produces_no_anchor(self):
        tl = []
        T._add_word_anchors(tl, _W("，", 0.0, 1.0))
        assert tl == []

    def test_zero_length_skipped(self):
        tl = []
        T._add_word_anchors(tl, _W("x", 1.0, 1.0))
        assert tl == []


# ============================== _match_atom_times ==============================

def _tl(pairs):
    """pairs: [(unit, t), ...] -> timeline。"""
    return [{"u": u, "s": t, "e": t, "t": t} for u, t in pairs]


class TestMatchAtomTimes:
    def test_perfect_match_uses_anchor_times(self):
        atoms = T._text_atoms("你好吗")
        timeline = _tl([("你", 1.0), ("好", 2.0), ("吗", 3.0)])
        times = T._match_atom_times(atoms, timeline)
        assert times is not None
        assert times == pytest.approx([1.0, 2.0, 3.0])

    def test_result_is_monotone(self):
        atoms = T._text_atoms("甲乙丙丁戊")
        timeline = _tl([("甲", 0.5), ("丙", 1.5), ("戊", 2.5)])
        times = T._match_atom_times(atoms, timeline)
        assert times is not None
        assert times == sorted(times)

    def test_interpolation_between_anchors(self):
        atoms = T._text_atoms("一二三四五")
        timeline = _tl([("一", 0.0), ("五", 4.0)])
        times = T._match_atom_times(atoms, timeline)
        assert times[0] == pytest.approx(0.0)
        assert times[4] == pytest.approx(4.0)
        assert times[2] == pytest.approx(2.0, abs=0.51)

    def test_tail_pushed_beyond_audio(self):
        """超出末尾锚点的原子应排到 base+20 之后，不污染末段。"""
        atoms = T._text_atoms("一二三")
        timeline = _tl([("一", 0.5), ("二", 1.0)])
        times = T._match_atom_times(atoms, timeline, tail_floor=10.0)
        assert times[2] > 10.0 + 19.0

    def test_tail_floor_respected(self):
        atoms = T._text_atoms("一二三")
        timeline = _tl([("一", 0.5)])
        t_low = T._match_atom_times(atoms, timeline, tail_floor=0.0)
        t_high = T._match_atom_times(atoms, timeline, tail_floor=100.0)
        assert t_high[2] > t_low[2]

    def test_returns_none_on_empty_inputs(self):
        assert T._match_atom_times([], _tl([("x", 0.0)])) is None
        assert T._match_atom_times(T._text_atoms("abc"), []) is None

    def test_returns_none_when_over_budget(self):
        atoms = T._text_atoms("你" * 50)
        timeline = _tl([("好" * 1, float(i)) for i in range(50)])
        assert T._match_atom_times(atoms, timeline, n_max_cells=10) is None

    def test_returns_none_when_no_equal_anchor(self):
        atoms = T._text_atoms("甲乙丙")
        timeline = _tl([("XYZ", 1.0), ("QQQ", 2.0)])
        assert T._match_atom_times(atoms, timeline) is None

    def test_garbage_interspersed_still_anchors(self):
        """识别结果里混入无关内容时，仍应锚住能对上的部分。

        注意锚点粒度必须与 _add_word_anchors 一致 —— CJK 是**逐字**锚点，
        不是整词锚点（整词锚无法与台词的逐字原子等值匹配）。
        """
        atoms = T._text_atoms("你好世界")
        timeline = _tl([("嗯", 0.1), ("你", 1.0), ("呃", 1.4),
                        ("好", 2.0), ("世", 3.0), ("界", 3.5)])
        times = T._match_atom_times(atoms, timeline)
        assert times is not None
        assert times[0] == pytest.approx(1.0)
        assert times[1] == pytest.approx(2.0)
        assert times[2] == pytest.approx(3.0)
        assert times[3] == pytest.approx(3.5)

    def test_coarse_anchor_falls_back_to_interpolation(self):
        """锚点粒度比台词粗时（如整词"世界" vs 逐字"世""界"），
        能锚的照锚，锚不上的在锚点间插值 —— 不应整体崩坏。"""
        atoms = T._text_atoms("你好世界")
        timeline = _tl([("你", 1.0), ("好", 2.0), ("世界", 3.0)])
        times = T._match_atom_times(atoms, timeline)
        assert times is not None
        assert times == sorted(times)
        # 前两个能精确锚住
        assert times[0] == pytest.approx(1.0)
        assert times[1] == pytest.approx(2.0)


# ============================== _merge_short_speech ==============================

class TestMergeShortSpeech:
    def test_short_fragment_merged_into_neighbour(self):
        speech = [{"start": 0.0, "end": 3.0, "text": "a"},
                  {"start": 3.1, "end": 3.3, "text": "b"}]
        out = T._merge_short_speech(speech, min_dur=1.0)
        assert len(out) == 1

    def test_all_segments_meet_min_duration(self):
        speech = [{"start": 0.0, "end": 3.0, "text": "a"},
                  {"start": 3.1, "end": 3.3, "text": "b"},
                  {"start": 5.0, "end": 7.0, "text": "c"}]
        out = T._merge_short_speech(speech, min_dur=1.0)
        for s in out:
            assert (s["end"] - s["start"]) >= 1.0 - 1e-9

    def test_does_not_mutate_input(self):
        speech = [{"start": 0.0, "end": 3.0, "text": "a"},
                  {"start": 3.1, "end": 3.3, "text": "b"}]
        snapshot = [dict(s) for s in speech]
        T._merge_short_speech(speech, min_dur=1.0)
        assert speech == snapshot

    def test_output_sorted_and_non_overlapping(self):
        speech = [{"start": 5.0, "end": 7.0, "text": "c"},
                  {"start": 0.0, "end": 3.0, "text": "a"},
                  {"start": 3.1, "end": 3.3, "text": "b"}]
        out = T._merge_short_speech(speech, min_dur=1.0)
        assert out == sorted(out, key=lambda x: x["start"])
        for a, b in zip(out, out[1:]):
            assert b["start"] >= a["end"] - 1e-9


# ============================== _align_proportional (兜底) ==============================

class TestAlignProportional:
    def test_no_tokens_returns_empty(self):
        out, note = T._align_proportional(
            [{"start": 0.0, "end": 1.0, "text": "x"}], "，。！")
        assert out == [] and "没有可朗读" in note

    def test_zero_duration_returns_empty(self):
        out, note = T._align_proportional(
            [{"start": 1.0, "end": 1.0, "text": "x"}], "你好")
        assert out == [] and note == ""

    def test_covers_all_segments(self):
        speech = [{"start": 0.0, "end": 2.0, "text": "a"},
                  {"start": 2.0, "end": 4.0, "text": "b"}]
        out, _ = T._align_proportional(speech, "今天天气很好我们去公园散步")
        assert len(out) == 2
        assert all(s["aligned"] for s in out)

    def test_text_is_partitioned_not_duplicated(self):
        speech = [{"start": 0.0, "end": 2.0, "text": "a"},
                  {"start": 2.0, "end": 4.0, "text": "b"},
                  {"start": 4.0, "end": 6.0, "text": "c"}]
        text = "今天天气很好我们去公园散步吧"
        out, _ = T._align_proportional(speech, text)
        joined = "".join(s["text"] for s in out).replace(" ", "")
        assert joined == text


# ============================== align_text_to_segments ==============================

class TestAlignTextToSegments:
    def _segs(self):
        return [{"start": 0.0, "end": 2.0, "text": "a", "duration": 2.0},
                {"start": 2.0, "end": 4.0, "text": "b", "duration": 2.0}]

    def test_empty_transcript_short_circuits(self):
        segs, note = T.align_text_to_segments(self._segs(), "   ")
        assert note == "台词为空"
        assert len(segs) == 2

    def test_punct_only_transcript(self):
        segs, note = T.align_text_to_segments(self._segs(), "，。！？")
        assert "没有可朗读" in note

    def test_oversized_transcript_rejected(self):
        segs, note = T.align_text_to_segments(self._segs(), "你" * 200001)
        assert "过长" in note

    def test_no_speech_segments(self):
        segs, note = T.align_text_to_segments([], "你好世界")
        assert "未识别到有效语音分段" in note

    def test_falls_back_without_timeline(self):
        segs, note = T.align_text_to_segments(
            self._segs(), "今天天气很好我们去公园")
        assert "退回按时长比例分配" in note

    def test_uses_timeline_when_given(self):
        segs, note = T.align_text_to_segments(
            self._segs(), "你好世界",
            timeline=_tl([("你", 0.5), ("好", 1.5), ("世", 2.5), ("界", 3.5)]),
            audio_total=4.0)
        assert "词级锚点" in note
        assert all(s["aligned"] for s in segs)

    def test_dropped_reported_for_overlong_transcript(self):
        """台词远超音频时应报 dropped，而不是把尾部塞进末段。"""
        segs, note = T.align_text_to_segments(
            self._segs(),
            "你好世界" + "额外的很长很长的台词内容" * 5,
            timeline=_tl([("你", 0.5), ("好", 1.5), ("世", 2.5), ("界", 3.5)]),
            audio_total=4.0)
        assert "超出音频时长" in note

    def test_aligned_text_matches_transcript_prefix(self):
        text = "你好世界"
        segs, _ = T.align_text_to_segments(
            self._segs(), text,
            timeline=_tl([("你", 0.5), ("好", 1.5), ("世", 2.5), ("界", 3.5)]),
            audio_total=4.0)
        joined = "".join(s["text"] for s in segs).replace(" ", "")
        assert joined.startswith("你好")

    def test_output_schema_stable(self):
        segs, _ = T.align_text_to_segments(
            self._segs(), "你好世界",
            timeline=_tl([("你", 0.5), ("好", 1.5), ("世", 2.5), ("界", 3.5)]),
            audio_total=4.0)
        for i, s in enumerate(segs):
            assert s["idx"] == i
            for k in ("text", "start", "end", "duration", "orig_text", "aligned"):
                assert k in s
            assert isinstance(s["aligned"], bool)

    def test_raw_tail_used_for_boundaries(self):
        """raw_tail 提供含 <1s 的原始边界，应被用于合并。"""
        segs, note = T.align_text_to_segments(
            self._segs(), "你好世界",
            timeline=_tl([("你", 0.5), ("好", 1.5), ("世", 2.5), ("界", 3.5)]),
            raw_tail=[{"start": 0.0, "end": 1.9, "text": "a"},
                      {"start": 1.9, "end": 4.0, "text": "b"}],
            audio_total=4.0)
        assert isinstance(segs, list) and len(segs) >= 1
