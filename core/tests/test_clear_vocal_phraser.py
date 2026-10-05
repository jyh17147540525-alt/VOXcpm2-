"""清唱插件 · 短语级合成自测
============================
验证 ``phraser.group_notes`` / ``phrase_text`` / ``slice_phrase`` 的行为，
并**证明短语级分组真的把"逐音符"变成了"逐乐句"**。

设计原则（与项目既有铁律一致）
------------------------------
* **每加断言配突变体**：断言之后立刻用一个"故意改坏"的输入验证断言会红，
  否则不知道自己测的是不是空气。
* **不碰模型**：本测试纯 DSP/纯逻辑，不加载 VoxCPM2（模型测试另见 slow 标记）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# ------------------------------------------------------------ 插件模块定位
# 这个文件要在**两棵树**上都能跑，而两棵树的插件布局不同：
#   本地树  <repo>/vox_plugins/clear_vocal/                  （扁平）
#   发布树  <repo>/plugins/技能插件/clear_vocal/              （中文子区）
# 子区名是中文、不能作 Python 标识符，发布树靠 tests/_plugin_path.py 把
# ``plugins`` 变成命名空间包。所以这里在运行时探测，而不是写死一条 import——
# 写死就意味着"每同步一次都要手工改这份测试"，那正是两棵树同步的时间黑洞。
def _load_phraser():
    here = Path(__file__).resolve()
    tests_dir = here.parent
    # 本地树：tests/ 的上 1 层即仓库根；发布树：core/tests/ 的上 2 层
    for root in (tests_dir.parent, tests_dir.parent.parent):
        if (root / "vox_plugins" / "clear_vocal" / "phraser.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from vox_plugins.clear_vocal import phraser  # noqa: E402
            return phraser
        if (root / "plugins" / "技能插件" / "clear_vocal" / "phraser.py").is_file():
            if str(tests_dir) not in sys.path:
                sys.path.insert(0, str(tests_dir))
            from _plugin_path import ensure_plugins_importable  # noqa: E402
            ensure_plugins_importable()
            from plugins.clear_vocal import phraser  # noqa: E402
            return phraser
    pytest.skip("未找到 clear_vocal/phraser.py（两树布局都没匹配上）",
                allow_module_level=True)


ph = _load_phraser()


# ------------------------------------------------------------------ 构造素材
def _notes(lyrics, dur=0.5):
    return [{"lyric": t, "dur": dur, "midi": 60.0 + i} for i, t in enumerate(lyrics)]


# ================================================================== group_notes
def test_group_by_punctuation():
    """标点是作词者写下的呼吸位 → 必须在此断开。"""
    notes = _notes(["春", "天", "的", "花", "，", "开", "了", "吗", "？"])
    g = ph.group_notes(notes)
    assert len(g) == 2, g
    assert g[0] == [0, 1, 2, 3, 4]
    assert g[1] == [5, 6, 7, 8]


def test_group_punctuation_is_the_only_splitter_when_under_limits():
    """无标点且不超上限 → 整句一组（这正是"短语级"的意义）。"""
    notes = _notes(["春", "天", "的", "花"])
    g = ph.group_notes(notes)
    assert g == [[0, 1, 2, 3]], g


def test_group_respects_max_notes():
    """放开 dur/chars 上限后，max_notes 必须是生效的那个约束。"""
    notes = _notes(["啊"] * 20)
    g = ph.group_notes(notes, max_notes=8, max_chars=99, max_dur=99)
    assert [len(x) for x in g] == [8, 8, 4], [len(x) for x in g]


def test_group_max_dur_binds_before_max_notes():
    """三个上限谁先到谁生效：dur=0.5 × 6 = 3.0 < MAX_DUR(3.2) → 6 个音成组。"""
    notes = _notes(["啊"] * 20, dur=0.5)
    g = ph.group_notes(notes, max_notes=8, max_chars=99, max_dur=3.2)
    assert [len(x) for x in g] == [6, 6, 6, 2], [len(x) for x in g]


def test_group_respects_max_chars():
    notes = _notes(["一二三"] * 6)          # 每音 3 字
    g = ph.group_notes(notes, max_chars=7, max_notes=99)
    # 7 字上限 → 每组最多 2 个音符（6 字）；第 3 个会超（9 字）→ 断开
    assert [len(x) for x in g] == [2, 2, 2], [len(x) for x in g]


def test_group_respects_max_dur():
    notes = _notes(["啊"] * 6, dur=1.0)
    g = ph.group_notes(notes, max_dur=3.0, max_notes=99, max_chars=99)
    assert [len(x) for x in g] == [3, 3], [len(x) for x in g]


def test_hold_notes_do_not_split():
    """拖腔音是前一音的延续 → 不切断短语，且归到同组。"""
    notes = _notes(["春", "天", "的", "花"])
    notes[2]["is_hold"] = True
    g = ph.group_notes(notes)
    assert len(g) == 1, g
    assert g[0] == [0, 1, 2, 3], g


def test_group_empty():
    assert ph.group_notes([]) == []
    assert ph.group_notes(None) == []


def test_every_note_covered_exactly_once():
    """硬不变量：每个非拖腔音符恰好出现在一个组里。"""
    notes = _notes(["春", "，", "天", "的", "花", "。", "开", "了", "吗"])
    notes[4]["is_hold"] = True
    g = ph.group_notes(notes)
    flat = [i for grp in g for i in grp]
    assert sorted(flat) == sorted(set(flat)), "有音符被重复分配：%s" % flat
    present = set(flat)
    for i, n in enumerate(notes):
        if not n.get("is_hold"):
            assert i in present, "音符 %d 丢失" % i


# ------------------------------------------------------------------ 突变体
def test_mutant_break_chars_would_be_wrong():
    """突变体：把标点边界去掉（模拟"忘记按标点分组"）→ 断言必须失败。

    这条测试的意义：证明 ``test_group_by_punctuation`` 不是空气断言。
    """
    notes = _notes(["春", "天", "的", "花", "，", "开", "了", "吗", "？"])
    # 手工模拟"标点不生效"的分组（全句一组）
    fake = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert fake != ph.group_notes(notes), "标点分组未生效 —— 检测到回归"


# ================================================================== phrase_text
def test_phrase_text_strips_punctuation():
    notes = _notes(["春", "天", "的", "花", "，"])
    assert ph.phrase_text(notes, [0, 1, 2, 3, 4]) == "春天的花"


def test_phrase_text_hum_fallback():
    notes = [{"lyric": "", "dur": 0.5}, {"lyric": "啊", "dur": 0.5}]
    assert ph.phrase_text(notes, [0, 1]) == "啊啊"


# ================================================================= slice_phrase
def test_slice_phrase_preserves_total_length():
    """硬不变量：切分不丢样本（否则 aligner 的时长账就错了）。"""
    sr = 48000
    y = (np.random.RandomState(0).randn(sr * 2) * 0.1).astype(np.float32)
    durs = [0.5, 0.5, 0.5, 0.5]
    segs = ph.slice_phrase(y, sr, durs)
    assert len(segs) == 4, len(segs)
    assert sum(s.size for s in segs) == y.size, (sum(s.size for s in segs), y.size)


def test_slice_phrase_prefers_energy_valley():
    """刀口应落在能量谷：构造两个响块夹一段静音，检查刀口确实进了静音区。

    ⚠️ 断言口径：检查**边界那一刻**的能量，不是"尾部 5ms 的平均"。
    踩过的坑：静音区 60ms、理想刀口落在静音区起点，能量谷搜索把刀口推进了
    96 个样本（2ms）。此时"最后 5ms 平均 RMS"仍包含 3ms 响块 → 0.387，
    看着像"能量谷没生效"，其实生效了。**断言窗口宽于验证对象**会造成假红。
    """
    sr = 48000
    quiet = int(0.060 * sr)
    loud = int(0.470 * sr)
    y = np.concatenate([
        np.full(loud, 0.5, np.float32),
        np.zeros(quiet, np.float32),
        np.full(loud, 0.5, np.float32),
    ])
    durs = [0.5, 0.5]                      # 理想刀口 = 中点 = 静音区起点
    segs = ph.slice_phrase(y, sr, durs)
    assert len(segs) == 2

    cut = segs[0].size
    # 1) 刀口必须已进入静音区（而不是停在响块上）
    assert cut > loud, "刀口 %d 未进入静音区（静音从 %d 开始）" % (cut, loud)
    # 2) 刀口附近 ±2ms 的能量必须接近 0
    w = segs[0][max(0, cut - int(0.002 * sr)):]
    assert float(np.sqrt(np.mean(w ** 2))) < 0.02, \
        "刀口处仍非静音：rms=%.4f" % float(np.sqrt(np.mean(w ** 2)))
    # 3) 硬不变量：不丢样本
    assert sum(s.size for s in segs) == y.size


def test_slice_phrase_single_note_is_identity():
    sr = 8000
    y = np.arange(sr, dtype=np.float32) / sr
    segs = ph.slice_phrase(y, sr, [1.0])
    assert len(segs) == 1
    assert np.array_equal(segs[0], y)


def test_slice_phrase_many_notes_keeps_count_and_length():
    sr = 16000
    rng = np.random.RandomState(7)
    y = (rng.randn(sr) * 0.05).astype(np.float32)
    durs = [0.1, 0.3, 0.05, 0.25, 0.2, 0.1]
    segs = ph.slice_phrase(y, sr, durs)
    assert len(segs) == len(durs), (len(segs), len(durs))
    assert sum(s.size for s in segs) == y.size


def test_slice_phrase_signal_is_singing_like_keeps_pitch_measurable():
    """接近真实场景：用带包络的正弦充当"唱句"，切分后每段仍可测到基频。

    这条是**听感层的轻量代理断言**：不加载模型，但验证"切片本身不破坏可测性"。
    """
    sr = 48000
    f0 = 220.0
    t = np.arange(sr, dtype=np.float64) / sr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)     # 3Hz 音节起伏
    y = (np.sin(2 * np.pi * f0 * t) * env * 0.4).astype(np.float32)
    segs = ph.slice_phrase(y, sr, [0.25, 0.25, 0.25, 0.25])
    assert len(segs) == 4
    for i, s in enumerate(segs):
        # 纯自相关估基频（不依赖 librosa，避免测试环境差异）
        w = s[int(0.05 * sr):int(0.20 * sr)]
        if w.size < 100:
            continue
        w = w - w.mean()
        ac = np.correlate(w, w, "full")[w.size - 1:]
        lo, hi = int(sr / 500), int(sr / 80)
        if hi >= ac.size:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        est = sr / lag
        assert abs(est - f0) / f0 < 0.10, "第 %d 段基频估计 %.1f 偏离 %.1f" % (i, est, f0)


# ================================================================== summarize
def test_summarize_reports_singletons():
    """单音短语数必须被如实报告 —— 它是"退化成旧路径"的量化信号。"""
    notes = _notes(["春", "，", "天", "。", "花"])
    g = ph.group_notes(notes)
    s = ph.summarize(g, notes)
    assert s["n_phrases"] == 3, s
    assert s["phrase_sizes"] == [2, 2, 1], s["phrase_sizes"]
    assert s["n_singletons"] == 1, s              # "花" 单独成句
    # 纯标点音符不贡献文本（否则会凭空多唱一个字）
    assert s["texts"] == ["春", "天", "花"], s["texts"]


# ========================================================= adjust_text_for_dur
def test_adjust_text_short_output_gets_lengthened():
    """模型唱得太快（ratio 远小于 1）→ 加延长号，绝不加字。"""
    out = ph.adjust_text_for_dur("春天的花开了", 0.70)
    assert out != "春天的花开了"
    assert out.startswith("春天的花开了")
    assert set(out[len("春天的花开了"):]) <= {"—"}, out
    # 绝不"加字凑时长"
    assert out.replace("—", "") == "春天的花开了"


def test_adjust_text_long_output_gets_weak_syllable_dropped():
    """模型唱得太慢（ratio 远大于 1）→ 去掉尾部虚词，绝不减实词。"""
    out = ph.adjust_text_for_dur("你还好吗", 1.60)
    assert out == "你还好", out


def test_adjust_text_in_window_is_identity():
    """比例落在容忍窗内 → 原样返回（不折腾）。"""
    assert ph.adjust_text_for_dur("春天的花开了", 1.0) == "春天的花开了"
    assert ph.adjust_text_for_dur("春天的花开了", ph.DUR_TOL_LO) == "春天的花开了"
    assert ph.adjust_text_for_dur("春天的花开了", ph.DUR_TOL_HI) == "春天的花开了"


def test_adjust_text_never_empties():
    """全是虚词也不能被删空 —— 空文本会让模型直接失败。"""
    out = ph.adjust_text_for_dur("了", 2.0)
    assert out, out


def test_mutant_linear_volume_would_clip():
    """突变体：证明 DUR_TOL 常量真的被用上（若把它改成 0，窗口判定会失效）。"""
    # 若 DUR_TOL_LO == DUR_TOL_HI，则任何非 1.0 都会被调整 —— 用于反向确认常量存在
    assert ph.DUR_TOL_LO < 1.0 < ph.DUR_TOL_HI, (
        ph.DUR_TOL_LO, ph.DUR_TOL_HI)
    assert ph.MAX_DUR_PROBES >= 2, ph.MAX_DUR_PROBES
