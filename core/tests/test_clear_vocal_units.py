"""清唱生成插件 · 文本 / 规划 / 合成器单元测试
==============================================
把开发期的一次性探针（``vox_plugins/clear_vocal/_t_*.py``）固化成 pytest 用例，
让这些行为进入常规回归（探针脚本不会被 ``pytest.ini`` 的 ``testpaths = tests`` 收集，
放在那里等于没有护栏）。

覆盖三块：
  * ``syllabify`` —— 中英歌词 → 音节单元（不用 whisper 词边界，见下）
  * ``planner``   —— 音符 × 歌词 → note_plan（含拖腔、校验、JSON 可读性）
  * ``singer``    —— 钩子重入守卫（最贵的一课，必须有回归）

为什么自己实现音节切分而不用 whisper
------------------------------------
实测 faster-whisper：朗读语音转写准确，但**歌声/哼唱**只会退化成 ``'嗯'``
（无语义），且**中文词边界不稳定** —— 同一句"你好啊"一次切成 ``['你','好','啊']``、
另一次切成 ``['你好','啊']``。用它当切分依据会让同一首歌的两次运行产生不同的
字音分配。所以：整句文本 + 自建音节切分 = 确定性。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _plugin_path import ensure_plugins_importable  # noqa: E402

ensure_plugins_importable()

from plugins.clear_vocal import analyzer as AN   # noqa: E402
from plugins.clear_vocal import planner as PL    # noqa: E402
from plugins.clear_vocal import singer as SG     # noqa: E402
from plugins.clear_vocal import syllabify as SY  # noqa: E402


# ===================================================================== analyzer
# 「大跳进被抹平」——本项目最难查的一个 bug，必须有回归。
#
# 现象：旋律 [72,57,58,59] 被读成 [60,57,58,59]（首音低一个八度）。
# 隐蔽性：**首音那段音频一个采样都没变**，只是后面多了一个音，读数就变了。
# 因此下面这些"合理的猜测"逐个实验后**全部无效**：
#   * 淡入淡出（去掉 fade 也错）
#   * fmax_note 边界（80→100 读数纹丝不动）
#   * 静音过渡帧（能量门控 -50~-20 dB 全无效）
#   * 初始相位（8 个相位全部误读）
# 真正原因：librosa.pyin 用 Viterbi 求全局最优路径，`max_transition_rate`
# 默认 35.92 → 换算成 **10 半音/帧**（hop=512, sr=22050）。超过 10 半音的跳进
# 会被"拉平"成八度内的小跳进，局部正确性被拿去换全局平滑性。
def _melody_signal(midis, sr: int = AN.DEFAULT_SR, dur: float = 0.28,
                   step: float = 0.35) -> np.ndarray:
    """合成一段确定性的音符旋律（带淡入淡出，模拟真实歌声包络）。"""
    n_total = int(round((len(midis) * step + 0.4) * sr))
    song = np.zeros(n_total, dtype=np.float32)
    for i, m in enumerate(midis):
        n = max(1, int(round(dur * sr)))
        t = np.arange(n, dtype=np.float64) / sr
        f = 440.0 * (2.0 ** ((float(m) - 69.0) / 12.0))
        y = 0.55 * np.sin(2.0 * np.pi * f * t)
        k = max(1, int(0.012 * sr))
        if n > 2 * k:
            env = np.ones(n)
            env[:k] = np.linspace(0.0, 1.0, k)
            env[-k:] = np.linspace(1.0, 0.0, k)
            y = y * env
        i0 = int(round(i * step * sr))
        song[i0:i0 + n] += y.astype(np.float32)
    return song


def _read_midis(midis) -> list:
    a = AN.analyze(_melody_signal(midis), AN.DEFAULT_SR)
    return [round(float(n["midi"]), 1) for n in a.get("notes", [])]


@pytest.mark.parametrize("melody", [
    [72.0, 57.0, 58.0, 59.0],   # 15 半音下行跳进（原始触发用例）
    [72.0, 56.0, 57.0, 58.0],   # 16 半音
    [55.0, 72.0, 57.0, 58.0],   # 17 半音上行跳进在中间
    [48.0, 63.0, 50.0, 51.0],   # 15 半音
    [60.0, 75.0, 62.0, 63.0],   # 15 半音上行
])
def test_large_leaps_are_not_flattened_into_octaves(melody):
    """大跳进必须被如实读出，不得被 Viterbi 抹平成一个八度内的小跳进。"""
    got = _read_midis(melody)
    assert len(got) == len(melody), \
        "音符数不符（期望 %d，实得 %s）" % (len(melody), got)
    for i, (g, w) in enumerate(zip(got, melody)):
        assert abs(g - w) < 0.6, (
            "第 %d 个音被读成 %.1f（应为 %.1f）—— 大跳进被抹平了。"
            "检查 analyzer.PYIN_MAX_TRANSITION_RATE 是否被改回默认值"
            % (i, g, w))


@pytest.mark.parametrize("melody", [
    [57.0, 58.0, 59.0, 60.0],   # 级进
    [60.0, 62.0, 64.0, 67.0],   # 小跳进（三度）
    [67.0, 65.0, 62.0, 60.0],   # 下行级进
    [72.0, 54.0, 55.0, 56.0],   # 18 半音（原本就正常，防退化）
])
def test_stepwise_melodies_stay_correct(melody):
    """提高转移率后，普通级进旋律不能被"带坏"（回归对照）。"""
    got = _read_midis(melody)
    assert len(got) == len(melody)
    for i, (g, w) in enumerate(zip(got, melody)):
        assert abs(g - w) < 0.6, "第 %d 个音读错：%.1f vs %.1f" % (i, g, w)


def test_transition_rate_constant_is_not_the_librosa_default():
    """护栏：常量必须与 librosa 默认值不同，否则大跳进会再次被抹平。

    这条断言的价值在于**锁住一个数值背后的理由**：光看 ``86.1`` 莫名其妙，
    但它对应 hop=512/sr=22050 下的 24 半音/帧（默认 35.92 → 10 半音/帧）。
    """
    import librosa
    import inspect
    default = inspect.signature(librosa.pyin).parameters[
        "max_transition_rate"].default
    assert AN.PYIN_MAX_TRANSITION_RATE > default * 1.5, (
        "PYIN_MAX_TRANSITION_RATE=%r 太接近 librosa 默认 %r，大跳进会被抹平"
        % (AN.PYIN_MAX_TRANSITION_RATE, default))

    hop, sr = AN.PYIN_HOP, AN.DEFAULT_SR
    semis = round(AN.PYIN_MAX_TRANSITION_RATE * 12 * hop / sr)
    assert semis >= 24, (
        "换算后只有 %d 半音/帧（hop=%d, sr=%d），不足以容纳八度跳进"
        % (semis, hop, sr))


def test_extract_f0_honours_custom_transition_rate():
    """``extract_f0`` 必须把 ``max_transition_rate`` 真正传下去（而非吞掉）。

    ⚠️ 断言用"**多数帧**正确"，不是"全部帧"。
    音符最后一个分析窗会跨到"信号→静音"边界，pyin 在该单帧上可能给出八度误值
    （实测 t=0.302 那一帧读 60，其余 13 帧都是 72）。这**不影响最终音符**，因为
    ``extract_notes`` 取段内中位数，单帧噪声会被吸收 —— 详见
    ``test_large_leaps_are_not_flattened_into_octaves``（走完整 analyze 路径，10/10）。
    若这里要求"每帧都对"，测的就不是转移率，而是那个已知的尾帧边界效应。
    """
    song = _melody_signal([72.0, 57.0, 58.0, 59.0])
    seg = slice(0, 14)          # 0.28 s 音符在 512 hop 下约 14 帧

    good = AN.extract_f0(song, AN.DEFAULT_SR)["midi"][seg]
    frac_good = float(np.mean(np.abs(good - 72.0) < 0.6))
    assert frac_good >= 0.85, (
        "默认（提高后的）转移率下首音多数帧仍被误读（仅 %.0f%% 正确）：%s"
        % (frac_good * 100, np.round(good, 1).tolist()))

    # 强行用 librosa 默认的紧约束 → 应整段塌成低八度（证明参数确实生效）
    tight = AN.extract_f0(song, AN.DEFAULT_SR,
                          max_transition_rate=35.92)["midi"][seg]
    frac_tight = float(np.mean(np.abs(tight - 60.0) < 0.6))
    assert frac_tight >= 0.85, (
        "传入紧约束后本应整段复现八度误判，实际仅 %.0f%% 落在 60：%s —— "
        "说明 max_transition_rate 没被透传"
        % (frac_tight * 100, np.round(tight, 1).tolist()))


# ===================================================================== syllabify
@pytest.mark.parametrize("word,expect", [
    ("beautiful", ["beau", "ti", "ful"]),
    ("music", ["mu", "sic"]),
    ("hello", ["hel", "lo"]),
    # 哑音尾 e：不该被当成音节核（strange/time/love 都是单音节）
    ("strange", ["strange"]),
    ("time", ["time"]),
    ("love", ["love"]),
    ("world", ["world"]),
])
def test_split_english_word(word, expect):
    assert SY.split_english_word(word) == expect


def test_silent_final_e_does_not_create_a_syllable():
    """静音尾 e 是这条切分规则最容易写错的地方（会把 time 切成 ti|me）。"""
    for w in ("time", "make", "love", "strange"):
        assert len(SY.split_english_word(w)) == 1, \
            "%s 被错误地切成多音节：%s" % (w, SY.split_english_word(w))


def test_cjk_is_one_char_per_unit():
    units = SY.split_lyric("你好世界啊")
    assert [u["text"] for u in units] == ["你", "好", "世", "界", "啊"]
    assert all(u["kind"] == "cjk" for u in units)


def test_mixed_text_keeps_order_and_src_word():
    units = SY.split_lyric("abc 你好 de")
    assert [u["text"] for u in units] == ["abc", "你", "好", "de"]
    kinds = [u["kind"] for u in units]
    assert kinds == ["en", "cjk", "cjk", "en"]
    # 英文单元要能回溯到原词，中文不需要
    en = [u for u in units if u["kind"] == "en"]
    assert all(u["src_word"] for u in en)


def test_count_units_matches_split():
    for t in ("你好世界啊", "hello world", "abc 你好 de", ""):
        assert SY.count_units(t) == len(SY.split_lyric(t)), t


def test_is_cjk_recognises_chinese_only():
    assert SY.is_cjk("你") is True
    assert SY.is_cjk("a") is False
    assert SY.is_cjk("1") is False


def test_empty_lyric_yields_no_units():
    assert SY.split_lyric("") == []
    assert SY.count_units("") == 0


# ===================================================================== planner
def mk_notes(specs) -> list[dict]:
    return [{"start": float(s), "end": float(s + d), "dur": float(d),
             "midi": float(m), "f0": 440.0 * 2 ** ((m - 69) / 12.0), "rms": 0.2}
            for s, d, m in specs]


def analysis(notes, bpm=120.0, dur=None):
    return {
        "sr": 22050,
        "duration": float(dur if dur is not None else (notes[-1]["end"] + 0.5)),
        "notes": notes,
        "rhythm": {"bpm": bpm, "beats": [i * 60.0 / bpm for i in range(16)]},
        "key": {"name": "C major"},
    }


def test_plan_assigns_one_syllable_per_note_when_counts_match():
    notes = mk_notes([(0.0, 0.5, 60), (0.5, 0.5, 62), (1.0, 0.5, 64)])
    pl = PL.plan(analysis(notes), lyric_text="你好啊", snap=False)
    assert [n["lyric"] for n in pl["notes"]] == ["你", "好", "啊"]
    assert pl["schema_version"] == PL.PLAN_SCHEMA_VERSION
    assert pl["lyric_text"] == "你好啊"


def test_melisma_repeats_lyric_instead_of_dropping_notes():
    """音多字少（拖腔）：必须**保留所有音符**，重复挂字，而不是丢音符。"""
    notes = mk_notes([(i * 0.3, 0.3, 60 + i) for i in range(6)])
    pl = PL.plan(analysis(notes), lyric_text="你好", snap=False)
    assert len(pl["notes"]) == 6, "拖腔时丢了音符"
    lyr = [n["lyric"] for n in pl["notes"]]
    assert lyr[0] == "你" and lyr[1] == "好"
    assert all(x for x in lyr), "有音符没分到字"
    assert any(n.get("melisma") for n in pl["notes"]), "拖腔标记缺失"


def test_plan_without_lyric_marks_all_as_hum():
    notes = mk_notes([(0.0, 0.4, 60), (0.4, 0.4, 62)])
    pl = PL.plan(analysis(notes), lyric_text="", snap=False)
    assert len(pl["notes"]) == 2
    assert all((n.get("lyric") or "") == "" for n in pl["notes"]), \
        "无歌词时不应硬塞字（由 singer 用哼鸣兜底）"


def test_plan_on_empty_analysis_warns_instead_of_crashing():
    pl = PL.plan({"sr": 22050, "notes": [], "rhythm": {"bpm": 120.0}},
                 lyric_text="你好", snap=False)
    assert pl["notes"] == []
    assert pl["warnings"], "无音符时必须给出可读的告警"


def test_verify_plan_detects_overlap():
    """校验器必须抓得住人工编辑引入的音符重叠。"""
    bad = [
        {"start": 0.0, "end": 0.6, "dur": 0.6, "midi": 60},
        {"start": 0.4, "end": 0.9, "dur": 0.5, "midi": 62},   # 与上一个重叠
    ]
    issues = PL.verify_plan(bad)
    assert issues, "重叠未被校验器发现"


def test_plan_saves_readable_chinese_json(tmp_path):
    """落盘必须是人类可直接编辑的 JSON（中文不转义、缩进 2）。"""
    notes = mk_notes([(0.0, 0.4, 60), (0.4, 0.4, 62), (0.8, 0.4, 64)])
    pl = PL.plan(analysis(notes), lyric_text="你好世界啊", snap=False)
    p = tmp_path / "note_plan.json"
    PL.save(pl, str(p))
    raw = p.read_text(encoding="utf-8")
    assert "\\u4f60" not in raw, "中文被转义成 \\uXXXX，人工没法改"
    assert '"你"' in raw or '"你好"' in raw or "你" in raw
    assert "\n  " in raw, "缺少缩进，人类不易读"
    back = PL.load(str(p))
    assert back["schema_version"] == PL.PLAN_SCHEMA_VERSION


def test_plan_roundtrip_via_save_and_load(tmp_path):
    notes = mk_notes([(i * 0.25, 0.25, 60 + i) for i in range(4)])
    pl = PL.plan(analysis(notes), lyric_text="一二三四", snap=False)
    p = tmp_path / "p.json"
    PL.save(pl, str(p))
    back = PL.load(str(p))
    assert [n["lyric"] for n in back["notes"]] == \
           [n["lyric"] for n in pl["notes"]]
    assert [n["midi"] for n in back["notes"]] == \
           [n["midi"] for n in pl["notes"]]


def test_relink_rebinds_plan_to_new_analysis():
    """改了歌词后重挂到同一份分析上（不重跑 pyin）时必须保持一致。"""
    notes = mk_notes([(0.0, 0.4, 60), (0.4, 0.4, 62), (0.8, 0.4, 64)])
    a = analysis(notes)
    pl = PL.plan(a, lyric_text="你好啊", snap=False)
    again = PL.relink(pl, a)
    assert len(again["notes"]) == len(pl["notes"])
    assert [n["midi"] for n in again["notes"]] == \
           [n["midi"] for n in pl["notes"]]


def test_plan_records_bpm_and_total_duration():
    notes = mk_notes([(0.0, 0.5, 60), (0.5, 0.5, 62)])
    pl = PL.plan(analysis(notes, bpm=90.0, dur=2.0), lyric_text="你好",
                 snap=True, total_dur=2.0)
    assert abs(float(pl["bpm"]) - 90.0) < 1e-6
    # 末音不得越过总长
    for n in pl["notes"]:
        assert float(n["start"]) + float(n["dur"]) <= 2.0 + 1e-6


# ===================================================================== singer 守卫
def test_singer_refuses_to_synthesize_inside_a_hook():
    """⚠️ 最贵的一课：钩子内回调合成 = 非可重入锁自锁 = 永久挂死且无报错。

    这里用真实的插件运行上下文（emit 进入钩子）来验证守卫会**立刻抛错**，
    而不是安静地挂住。若这条护栏失效，表现是服务整体卡死、日志一片空白。
    """
    from voice_clone import plugin_core as P

    P.reset_for_tests()
    import shutil, tempfile

    tmp = Path(tempfile.mkdtemp())
    pkg = tmp / P.PLUGIN_DIR_NAME / "技能插件"
    pkg.mkdir(parents=True)
    (pkg / "probe").mkdir()
    (pkg / "probe" / "plugin.json").write_text(json.dumps({
        "id": "probe", "api_version": "1.0",
        "hooks": {"report.enrich": "on_report_enrich"},
    }), encoding="utf-8")
    # 探针插件在**隔离的临时目录**里运行，拿不到测试进程的 plugins 命名空间
    # （那是 _plugin_path 现搭的），所以按绝对路径把 singer 载进来。
    from _plugin_path import zone_dir, ZONE_SKILL
    singer_file = zone_dir(ZONE_SKILL) / "clear_vocal" / "singer.py"
    assert singer_file.is_file(), f"找不到 singer.py: {singer_file}"
    (pkg / "probe" / "plugin.py").write_text(
        "import importlib.util as _iu\n"
        f"_P = r'{singer_file}'\n"
        "_spec = _iu.spec_from_file_location('probe_cv_singer', _P)\n"
        "S = _iu.module_from_spec(_spec); _spec.loader.exec_module(S)\n"
        "caught = []\n"
        "def on_report_enrich(p):\n"
        "    try:\n"
        "        S.Singer(None, 22050)._synth_text('啊')\n"
        "        caught.append('NOT_RAISED')\n"
        "    except S.HookContextError as e:\n"
        "        caught.append('HookContextError: ' + str(e))\n"
        "    except Exception as e:\n"
        "        caught.append(type(e).__name__ + ': ' + str(e))\n"
        "    return {'probe': caught[-1] if caught else 'none'}\n",
        encoding="utf-8")

    cfg = tmp / P.CONFIG_NAME
    cfg.write_text(json.dumps({"search_paths": [P.PLUGIN_DIR_NAME],
                               "autoload": True, "disabled": [],
                               "settings": {}}), encoding="utf-8")
    reg = P.init(str(tmp), str(cfg))
    try:
        out = P.emit("report.enrich", report={})
        assert "probe" in out
        msg = str(out["probe"])
        assert "NOT_RAISED" not in msg, "钩子内合成没有被拒绝（危险！）"
        assert "HookContextError" in msg, "抛的不是 HookContextError：%s" % msg
        # 错误信息必须说清原因，便于作者自救
        assert "_infer_lock" in msg or "死锁" in msg
    finally:
        reg.stop_all()
        P.reset_for_tests()


def test_singer_guard_allows_calls_outside_hooks():
    """守卫不能误伤：钩子之外正常调用必须**放行**（走到模型调用那一步）。"""
    s = SG.Singer(None, 22050)
    # 不在钩子里 → 守卫放行，因 model=None 而在模型调用处失败（证明没被守卫拦下）
    with pytest.raises(Exception) as ei:
        s._synth_text("啊")
    assert not isinstance(ei.value, SG.HookContextError), \
        "钩子之外竟然被守卫拦下了（误伤）"


# ===================================================================== 音高实测闸门
def _sine(midi: float, dur: float, sr: int = 22050, amp: float = 0.5) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    f = 440.0 * 2 ** ((midi - 69.0) / 12.0)
    y = amp * np.sin(2 * np.pi * f * t)
    y += 0.3 * amp * np.sin(2 * np.pi * 2 * f * t)
    return y.astype(np.float32)


def test_measure_rejects_boundary_pinned_reading():
    """⚠️ 真实踩到的 bug：pyin 在短促片段上会**饱和到搜索边界**（40.00 / 86.00），
    看起来像正常数值。这种读数必须被判废（→ 不搬移），否则音符会被贴着
    ±12 半音上限搬运，听感完全错。

    这里直接构造"钉在边界"的场景：给一个极低音（接近 fmin）的信号。
    """
    s = SG.Singer(None, 22050)
    # MIDI 40 = pyin 的 fmin 下界 → 中位数会贴边界
    clip = _sine(40.0, 0.6)
    got = s.median_midi_of(clip, 22050, default=None)
    assert got is None, "钉在 fmin 边界的读数竟然被接受：%r" % got


def test_measure_rejects_inconsistent_with_target():
    """与目标音高偏离过大（> tol 半音）的读数必须判废 —— 宁可不信测量。"""
    s = SG.Singer(None, 22050)
    clip = _sine(64.0, 0.6)          # 实际唱在 64
    # 目标说 60（差 4 半音，应通过）；目标说 48（差 16 半音，应判废）
    ok = s.median_midi_of(clip, 22050, default=None, target_midi=60.0)
    bad = s.median_midi_of(clip, 22050, default=None, target_midi=48.0)
    assert bad is None, "偏离目标 16 半音的读数竟然被接受：%r" % bad
    # ok 可能因 pyin 在纯音上表现好而给出接近 64 的值，也可能因闸门回落；
    # 关键是"离谱的那个必须被拒"，不强制 ok 一定非 None。
    assert ok is None or abs(float(ok) - 60.0) <= 7.0


def test_measure_returns_none_on_silence():
    """静音片段不得返回任何音高（否则会算出随机搬移量）。"""
    s = SG.Singer(None, 22050)
    assert s.median_midi_of(np.zeros(22050, dtype=np.float32), 22050,
                            default=None) is None


def test_measure_clips_length_matches_input():
    s = SG.Singer(None, 22050)
    clips = [_sine(64.0, 0.4), np.zeros(1000, dtype=np.float32)]
    out = s.measure_clips(clips, sr=22050, target_midis=[64.0, 60.0])
    assert len(out) == len(clips)
    assert out[1] is None, "静音片段应判废"


def test_measure_gates_are_configurable():
    """闸门阈值必须是类属性（便于按素材调），不是写死的魔法数。"""
    assert hasattr(SG.Singer, "F0_NOTE_RANGE")
    assert hasattr(SG.Singer, "BOUNDARY_GUARD")
    assert hasattr(SG.Singer, "MIN_VOICED_RATIO")
    lo, hi = SG.Singer.F0_NOTE_RANGE
    assert lo < hi and 0 < SG.Singer.MIN_VOICED_RATIO <= 1.0
