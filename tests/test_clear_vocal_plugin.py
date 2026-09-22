"""清唱生成插件 · 端到端与编排层测试
======================================
分三层，越往下越"真"：

  A. **纯 DSP 端到端**（无模型）：``planner.plan`` → ``aligner.align_all``，
     用合成出来的音调片段走完整链路，断言"时长误差 0 采样"这条硬指标。
     这一层是整个插件最核心的验收：节奏对齐精度。
  B. **编排层路径**：用假模型驱动 ``plugin.run()`` 全流程，验证文件产物齐全、
     中间产物可复现、诊断信息完整。
  C. **契约与安全**：``plugin.json`` 与 ``plugin.py`` 必须严格匹配插件 API；
     以及那条最贵的教训 —— 钩子内绝不许回调合成。

为什么 A 层不用真模型也有效
--------------------------
``aligner`` 是纯函数：输入（目标时刻/时长/音高 + 片段音频）→ 输出（音频）。
它的正确性与"片段是怎么来的"无关。所以只要片段是**真实音频**，
把 300 个音符串起来测漂移就是有效证据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vox_plugins.clear_vocal import aligner as AL       # noqa: E402
from vox_plugins.clear_vocal import analyzer as AN      # noqa: E402
from vox_plugins.clear_vocal import planner as PL       # noqa: E402
from vox_plugins.clear_vocal import plugin as PG        # noqa: E402
from vox_plugins.clear_vocal import singer as SG        # noqa: E402
from vox_plugins.clear_vocal import syllabify as SY     # noqa: E402

SR = 22050


# --------------------------------------------------------------------- 工具
def tone(midi: float, dur: float, sr: int = SR, amp: float = 0.5,
         fade: float = 0.01) -> np.ndarray:
    """生成一个带淡入淡出、基频对应 midi 的纯音片段（确定性，无随机）。"""
    n = max(1, int(round(dur * sr)))
    t = np.arange(n, dtype=np.float64) / float(sr)
    f = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    y = amp * np.sin(2.0 * np.pi * f * t)
    k = max(1, int(fade * sr))
    if n > 2 * k:
        env = np.ones(n)
        env[:k] = np.linspace(0.0, 1.0, k)
        env[-k:] = np.linspace(1.0, 0.0, k)
        y = y * env
    return y.astype(np.float32)


def mk_notes(specs) -> list[dict]:
    """specs: [(start, dur, midi), …] → analyzer 风格的 notes。"""
    out = []
    for s, d, m in specs:
        out.append({"start": float(s), "end": float(s + d), "dur": float(d),
                    "midi": float(m), "f0": 440.0 * 2 ** ((m - 69) / 12.0),
                    "rms": 0.2})
    return out


class _FakeModel:
    """最小 TTS 替身：返回**有音高**的音频，好让 measure_clips 有真实工作可做。

    刻意按调用序号换音高（60/64/67 循环），这样"实测 source_midi"这一步
    如果实现错了（比如永远返回同一个值），会被测试抓到。
    """

    class _TTS:
        sample_rate = 22050

    _PITCHES = (60.0, 64.0, 67.0)

    def __init__(self, dur: float = 0.30):
        self.tts_model = _FakeModel._TTS()
        self.n = 0
        self.dur = dur

    def generate(self, text=None, **kw):
        p = _FakeModel._PITCHES[self.n % len(_FakeModel._PITCHES)]
        self.n += 1
        return tone(p, self.dur, amp=0.6)


# ===================================================================== A. 纯 DSP 端到端
def _onset_sample(audio: np.ndarray, near: int, sr: int = SR,
                  win: float = 0.02, rise: float = 0.30) -> int | None:
    """从 ``near`` 起**向后**找真正起音的那个采样（真实落位的唯一可信证据）。

    为什么必须向后、且用"幅度上升沿"而不是"首次超过小阈值"：
    音符之间可能有较长的静音段，若向前搜索，窗口会落进前一个音的尾巴，
    5% 峰值阈值立刻命中 → 报出**前一个音**的位置。实测：300 音用例里
    这样会得到 -661 采样的假漂移，而音频其实完全正确。故：
      * 只在 [near, near+win] 内搜；
      * 以"局部峰值"为基准，找第一个超过 ``rise×峰值`` 的位置。
    """
    w = max(1, int(win * sr))
    lo = max(0, int(near))
    hi = min(audio.size, lo + w)
    if hi <= lo:
        return None
    seg = np.abs(audio[lo:hi])
    peak = float(seg.max())
    if peak <= 1e-6:
        return None
    above = np.nonzero(seg >= rise * peak)[0]
    if above.size == 0:
        return None
    return int(lo + above[0])


def test_pipeline_aligns_every_note_to_exact_sample():
    """核心验收：每个音符都被放到**精确**的采样位置，且整曲零漂移。

    断言分两层，缺一不可：
      * ``start_sample`` 精确等于目标采样号（计划层）
      * **音频里真实的音头**就落在那附近（实现层 —— 这层才能抓住整体偏移）
    为什么是"精确相等"而不是"误差 < 5ms"：误差会累积，
    5ms × 300 音符 = 1.5 秒的整体漂移，已经不能听了。
    """
    specs = [(i * 0.25, 0.20, 60 + (i % 5)) for i in range(12)]
    notes = mk_notes(specs)
    clips = [tone(m, 0.30) for _, _, m in specs]
    for i, n in enumerate(notes):
        n["source_midi"] = float(specs[i][2])   # 片段本身就唱在目标音高上

    audio, diags = AL.align_all(notes, clips, sr=SR, bpm=120.0,
                                snap=False, total_dur=3.2)

    assert len(diags) == len(specs)
    for i, (s, d, _m) in enumerate(specs):
        want = int(round(s * SR))
        got = int(diags[i]["start_sample"])
        assert got == want, "音符 %d 计划落点 %d != 目标 %d" % (i, got, want)
        # 时长也必须精确到采样
        assert int(diags[i]["n_samples_actual"]) == int(round(d * SR)), \
            "音符 %d 实际样本数 %d != 目标 %d" % (
                i, diags[i]["n_samples_actual"], int(round(d * SR)))
        # 实现层：真实音头必须落在目标附近。
        # 容差 8 ms 足够宽（淡入 10ms×30% 上升沿 ≈ 3ms），又足以抓住
        # "整体偏移 1 个采样以上"这类错误（1 采样 = 0.045ms，其实抓不住；
        # 但抓得住"错一个 20ms 帧"这种量级的真实故障）。
        got_onset = _onset_sample(audio, want)
        assert got_onset is not None, "音符 %d 在目标位置附近没有能量" % i
        assert abs(got_onset - want) <= int(0.008 * SR), \
            "音符 %d 真实音头 %d 偏离目标 %d（%d 采样）" % (
                i, got_onset, want, got_onset - want)

    assert audio.size == int(round(3.2 * SR)), \
        "整曲长度 %d != 目标 %d" % (audio.size, int(round(3.2 * SR)))


def test_no_accumulated_drift_over_300_notes():
    """300 个音符后整体漂移必须**恰好 0 采样**。

    同时检查计划层（``start_sample``）与实现层（真实音头）——
    仅看计划层的断言抓不到"整体偏移"（护栏自测实证）。
    """
    n = 300
    specs = [(i * 0.1, 0.08, 60 + (i % 7)) for i in range(n)]
    notes = mk_notes(specs)
    clips = [tone(m, 0.20) for _, _, m in specs]
    for i, x in enumerate(notes):
        x["source_midi"] = float(specs[i][2])

    audio, diags = AL.align_all(notes, clips, sr=SR, snap=False,
                                total_dur=n * 0.1)

    # 计划层：最后一个音符的落点必须精确
    last_want = int(round(specs[-1][0] * SR))
    assert int(diags[-1]["start_sample"]) == last_want, \
        "第 %d 个音符计划落点漂移 %d 采样" % (n, int(diags[-1]["start_sample"]) - last_want)

    # 实现层：最后一个音符的真实音头也必须还在原位（累计漂移为 0 的证据）
    onset = _onset_sample(audio, last_want, win=0.03)
    assert onset is not None, "末音符位置附近没有能量（可能被前一个音吞掉了）"
    assert abs(onset - last_want) <= int(0.008 * SR), \
        "末音符真实音头偏离 %d 采样 → 存在累计漂移" % (onset - last_want)

    assert audio.size == int(round(n * 0.1 * SR))


def test_pitch_is_shifted_toward_target_not_left_alone():
    """片段音高与目标差 5 半音时，必须真的被搬移（而不是原样放置）。

    用 pyin 实测输出音频的音高来验证 —— 这是"听感"层面的证据。
    """
    notes = mk_notes([(0.0, 0.6, 67.0)])       # 目标 G4
    clips = [tone(62.0, 0.45)]                  # 片段唱在 D4（差 +5）
    notes[0]["source_midi"] = 62.0

    audio, diags = AL.align_all(notes, clips, sr=SR, snap=False, total_dur=0.8)
    assert abs(float(diags[0]["applied_semitones"]) - 5.0) < 0.6, diags[0]

    mid = AN.extract_f0(audio, SR)
    m = np.asarray(mid["midi"], dtype=np.float64)
    v = np.asarray(mid["voiced"], dtype=bool)
    got = float(np.median(m[v])) if bool(v.any()) else float("nan")
    assert abs(got - 67.0) < 1.0, "搬移后实测 %.2f 半音，期望 ~67" % got


def test_silent_clip_is_handled_without_nan():
    """降级为静音的音符（合成失败）不得把 NaN 传染进整曲。"""
    notes = mk_notes([(0.0, 0.3, 60.0), (0.4, 0.3, 64.0)])
    clips = [np.zeros(int(0.05 * SR), dtype=np.float32), tone(64.0, 0.30)]
    for i, x in enumerate(notes):
        x["source_midi"] = float(x["midi"])
    audio, diags = AL.align_all(notes, clips, sr=SR, snap=False, total_dur=1.0)
    assert np.isfinite(audio).all(), "出现 NaN/Inf"
    assert float(np.abs(audio).max()) <= 1.0


def test_full_pipeline_reaches_dsp_with_a_synthetic_song(tmp_path):
    """不用模型的"整首歌"测：分析 → 规划 → 对齐，全链路打通。

    用一串不同音高、不同时长的确定音调当作"人声 stem"，这样既能真实地走
    pyin 分析，又完全可复现（不依赖任何模型权重）。
    """
    # 造一段"旋律"：8 个音，音高各不相同，时长交错
    specs = [(0.0, 0.35, 60.0), (0.4, 0.25, 62.0), (0.7, 0.30, 64.0),
             (1.05, 0.40, 67.0), (1.5, 0.25, 65.0), (1.8, 0.35, 62.0),
             (2.2, 0.30, 60.0), (2.55, 0.45, 59.0)]
    total = 3.1
    song = np.zeros(int(total * SR), dtype=np.float32)
    for s, d, m in specs:
        seg = tone(m, d - 0.02, amp=0.55)
        i0 = int(round(s * SR))
        song[i0:i0 + seg.size] += seg
    song = np.clip(song, -1.0, 1.0)

    analysis = AN.analyze(song, SR)
    got = analysis["notes"]
    assert len(got) >= 6, "pyin 至少应该认出 6 个音，实际 %d" % len(got)

    pl = PL.plan(analysis, lyric_text="一二三四五六七八",
                 snap=False, total_dur=total)
    assert len(pl["notes"]) == len(got)
    # 歌词被分配到音符上，且不是空字符串
    assert all(n["lyric"] for n in pl["notes"]), "有音符没分到歌词"

    clips = [tone(float(n["midi"]), 0.30) for n in pl["notes"]]
    for i, n in enumerate(pl["notes"]):
        n["source_midi"] = float(n["midi"])
    audio, diags = AL.align_all(pl["notes"], clips, sr=SR, snap=False,
                                total_dur=total)
    assert audio.size == int(round(total * SR))
    assert len(diags) == len(pl["notes"])
    # 每个音符的计划落点与实际样本数都必须自洽
    for i, d in enumerate(diags):
        assert int(d["n_samples_actual"]) == int(round(float(d["dur"]) * SR)), \
            "音符 %d 实际样本数 %d 与目标时长 %s 不符" % (
                i, d["n_samples_actual"], d["dur"])
    sm = AL.summary(diags)
    assert sm["n_notes"] == len(pl["notes"])
    # 片段与目标同音高 → 不应有搬移
    assert sm["semitone_max_abs"] < 0.6, "同音高素材不应产生搬移：%.3f" % sm["semitone_max_abs"]


# ===================================================================== B. 编排层
def test_run_produces_all_artifacts(tmp_path):
    """``plugin.run()`` 必须把每一步的中间产物都落到 out_dir（可调试性验收）。"""
    specs = [(0.0, 0.30, 60.0), (0.35, 0.30, 64.0), (0.70, 0.30, 67.0),
             (1.05, 0.30, 64.0)]
    total = 1.5
    song = np.zeros(int(total * SR), dtype=np.float32)
    for s, d, m in specs:
        seg = tone(m, d - 0.02, amp=0.55)
        i0 = int(round(s * SR))
        song[i0:i0 + seg.size] += seg

    out = tmp_path / "cv"
    res = PG.run(_FakeModel(), SR, reference_wav=None,
                 vocal_audio=song, vocal_sr=SR, lyric_text="一二三四",
                 out_dir=str(out), snap=False, keep_clips=True,
                 calibrate=False, logger=lambda m: None)

    assert res["ok"] is True
    for key in ("out_wav", "analysis_path", "plan_path", "diag_path"):
        assert Path(res[key]).is_file(), "缺少产物 %s -> %s" % (key, res[key])
    assert Path(res["clips_dir"]).is_dir()
    assert res["n_notes"] >= 3
    assert Path(res["out_wav"]).stat().st_size > 1000

    # 计划文件是可读 JSON，且中文没有被转义成 \uXXXX
    raw = Path(res["plan_path"]).read_text(encoding="utf-8")
    assert "\\u4e00" not in raw, "plan JSON 里的中文被转义了"
    pj = json.loads(raw)
    assert pj["schema_version"] == PL.PLAN_SCHEMA_VERSION
    # 无参考音色时 source_midi 被显式写为"等于目标"（即不搬移），而非缺字段：
    # 缺字段会让下游 aligner 拿不到信号，进而静默不搬移 —— 那是假绿。
    assert all("source_midi" in n for n in pj["notes"]), \
        "source_midi 未写回 note_plan（下游会静默不搬移）"
    for n in pj["notes"]:
        assert float(n["source_midi"]) == float(n["midi"]), \
            "无参考音色时应 source==target：%r vs %r" % (n["source_midi"], n["midi"])
    assert res["n_pitch_clamped"] == 0, "无参考音色时不该有任何钳制"

    # 诊断文件结构完整
    dj = json.loads(Path(res["diag_path"]).read_text(encoding="utf-8"))
    assert dj["summary"]["n_notes"] == res["n_notes"]
    assert len(dj["per_note"]) == res["n_notes"]
    # summary 必须带上实际生效的钳制上限（否则"是否被钳制"的判定会失真）
    assert "semitone_limit" in dj["summary"], \
        "summary 缺少 semitone_limit，无法判断钳制判定用的是哪个上限"


def test_run_shifts_relative_to_reference_anchor(tmp_path):
    """搬移基准必须来自**参考音色**，而不是逐片段测 pyin。

    ⚠️ 这是实跑纠正的设计：逐片段测 pyin 在短促 TTS 片段上会**饱和到搜索边界**
    （实测读数 ``[40.00, 40.00, 40.15, 47.95, 无, 84.50, 49.60, 85.40]``），
    其中 4 个音符被搬到 ±12 半音上限。改为：只在参考音频（长、稳态）上测一次，
    所有音符共用该基准 —— 整曲相对"这个嗓音的自然音高"整体移调。
    """
    specs = [(i * 0.35, 0.30, 60 + i) for i in range(4)]
    song = np.zeros(int(1.6 * SR), dtype=np.float32)
    for s, d, m in specs:
        seg = tone(m, d - 0.02, amp=0.55)
        song[int(round(s * SR)):int(round(s * SR)) + seg.size] += seg

    ref = tmp_path / "ref.wav"
    PG._write_wav(str(ref), tone(62.0, 1.5, amp=0.6), SR)

    out = tmp_path / "cv3"
    res = PG.run(_FakeModel(), SR, reference_wav=str(ref),
                 vocal_audio=song, vocal_sr=SR,
                 out_dir=str(out), snap=False, calibrate=False,
                 keep_clips=False, logger=lambda m: None)
    pj = json.loads(Path(res["plan_path"]).read_text(encoding="utf-8"))
    srcs = [float(n["source_midi"]) for n in pj["notes"]]
    # 所有音符共用同一基准（参考音色的实测音高）
    assert len(set(round(s, 3) for s in srcs)) == 1, \
        "音符的 source_midi 不一致，说明没有锚定参考音色：%s" % srcs
    assert 55.0 < srcs[0] < 70.0, \
        "参考音色实测音高离谱（应≈62）：%.2f" % srcs[0]


def test_run_without_reference_does_not_shift_at_all(tmp_path):
    """没有参考音色就无法确定基准 → 整曲**不搬移**（绝不猜一个八度）。"""
    specs = [(i * 0.35, 0.30, 60 + i) for i in range(3)]
    song = np.zeros(int(1.3 * SR), dtype=np.float32)
    for s, d, m in specs:
        seg = tone(m, d - 0.02, amp=0.55)
        song[int(round(s * SR)):int(round(s * SR)) + seg.size] += seg

    out = tmp_path / "cv4"
    res = PG.run(_FakeModel(), SR, reference_wav=None,
                 vocal_audio=song, vocal_sr=SR,
                 out_dir=str(out), snap=False, calibrate=False,
                 keep_clips=False, logger=lambda m: None)
    sm = res["summary"]
    assert sm["semitone_max_abs"] == 0.0, \
        "无参考音色时必须不搬移，实际 %.3f" % sm["semitone_max_abs"]
    pj = json.loads(Path(res["plan_path"]).read_text(encoding="utf-8"))
    for n in pj["notes"]:
        assert float(n["source_midi"]) == float(n["midi"]), \
            "source_midi 应等于目标（不搬移）：%r vs %r" % (n["source_midi"], n["midi"])


def test_run_never_shifts_notes_by_an_octave_silently(tmp_path):
    """回归：整曲不得出现"贴着 ±12 半音上限"的搬移。

    这是实跑发现的真 bug —— pyin 在合成片段上失效（读数钉在 40/86），
    4 个音符被搬到 -19/+24 半音（远超歌声合理音程）。锚定参考后必须消失。
    """
    specs = [(i * 0.35, 0.30, 60 + i) for i in range(4)]
    song = np.zeros(int(1.6 * SR), dtype=np.float32)
    for s, d, m in specs:
        seg = tone(m, d - 0.02, amp=0.55)
        song[int(round(s * SR)):int(round(s * SR)) + seg.size] += seg

    ref = tmp_path / "ref2.wav"
    PG._write_wav(str(ref), tone(60.0, 1.5, amp=0.6), SR)

    out = tmp_path / "cv5"
    res = PG.run(_FakeModel(), SR, reference_wav=str(ref),
                 vocal_audio=song, vocal_sr=SR,
                 out_dir=str(out), snap=False, calibrate=False,
                 keep_clips=False, logger=lambda m: None)
    sm = res["summary"]
    assert sm["semitone_max_abs"] < 12.0, \
        "仍有音符被搬到钳制上限（失效读数没被拦住）：%r" % sm
    assert sm["semitone_clamped"] == 0, \
        "有音符触及 ±12 半音钳制，说明基准音高不可信：%r" % sm


def test_octave_fold_makes_shift_octave_invariant(tmp_path):
    """**核心不变式**：整体八度折叠后，残留搬移量与"乐谱写在哪个人度"无关。

    这是实跑纠正的设计。原始现象：乐谱在 C4-G4、参考音色唱 D3 时，逐个音符要搬
    +9 ~ +16 半音，其中 3 个被钳到 12.00 → **旋律被静默改掉**，而日志只是显示
    "最大搬移 12.00 半音"，看起来很像正常。

    整数八度平移不改变任何音程，所以旋律逐音符保真，只改整体定调。正确实现应
    使"同一旋律在不同八度"收敛到同一个残移量 —— 若折叠写错（例如只折叠首音、
    或用了 2**n 而非 12n），这三个用例就会散开。

    ⚠️ 为什么三个测试八度选 45/57/69（A2/A3/A4）而**不是** 36、48、60
    ---------------------------------------------------------------
    分析层 pyin 的搜索范围是 ``fmin_note=40 .. fmax_note=88``（E2..E6）。
    实测：base=36(C2) 的旋律**整段塌成单个 40.0 的音符** —— 恰好钉在下边界，
    是 pyin 失效的典型签名（见 ``singer.median_midi_of`` 的边界闸门）。
    那是**分析器看不见**该八度，与折叠算法无关；拿它去断言折叠不变式，
    会把分析器的上限误报成折叠的 bug（第一版测试就是这么错的）。
    """
    anchor_tone = 60.0        # 参考音色 ≈ C4
    ref = tmp_path / "ref_fold.wav"
    PG._write_wav(str(ref), tone(anchor_tone, 1.5, amp=0.6), SR)

    results = []
    for label, base in [("low", 45), ("mid", 57), ("high", 69)]:
        specs = [(i * 0.35, 0.30, base + i) for i in range(4)]
        song = np.zeros(int(1.6 * SR), dtype=np.float32)
        for s, d, m in specs:
            seg = tone(m, d - 0.02, amp=0.55)
            song[int(round(s * SR)):int(round(s * SR)) + seg.size] += seg

        out = tmp_path / ("fold_" + label)
        res = PG.run(_FakeModel(), SR, reference_wav=str(ref),
                     vocal_audio=song, vocal_sr=SR,
                     out_dir=str(out), snap=False, calibrate=False,
                     keep_clips=False, logger=lambda m: None)
        pj = json.loads(Path(res["plan_path"]).read_text(encoding="utf-8"))
        results.append((label, res, pj))

    for label, res, pj in results:
        sm = res["summary"]
        assert res["n_notes"] == 4, \
            "乐谱在 %s 八度只解析出 %d 个音符（分析器在该八度不可靠）" \
            % (label, res["n_notes"])
        assert sm["semitone_clamped"] == 0, \
            "乐谱在 %s 八度时仍有音符被钳制（折叠没生效）：%r" % (label, sm)
        # 折叠后残留搬移应很小（折叠的目标就是把它压到 ±7 以内）
        assert sm["semitone_max_abs"] <= 7.0, \
            "乐谱在 %s 八度时残留搬移过大（%.2f），折叠未收敛到音域附近" \
            % (label, sm["semitone_max_abs"])

    means = [r[1]["summary"]["semitone_mean_abs"] for r in results]
    spread = max(means) - min(means)
    assert spread < 0.75, (
        "三个八度的残留搬移不一致（%s，极差 %.3f）—— 折叠不是八度不变的，"
        "说明整数八度平移算错了" % ([round(m, 3) for m in means], spread))

    # 音程（旋律轮廓）必须逐音符保真：折叠只改整体定调
    for label, res, pj in results:
        midis = [float(n["midi"]) for n in pj["notes"]]
        folds = [int(n.get("midi_octave_fold", 0)) for n in pj["notes"]]
        assert len(set(folds)) == 1, \
            "同一首曲子里各音符的八度折叠量不一致：%s" % folds
        planned = [float(n.get("midi_planned", n["midi"])) for n in pj["notes"]]
        for a, b in zip(midis, planned):
            assert abs((a - b) - 12.0 * folds[0]) < 1e-6, \
                "折叠量与 midi 不自洽：%r vs %r (fold=%d)" % (a, b, folds[0])
        deltas = [b - a for a, b in zip(midis, midis[1:])]
        planned_deltas = [b - a for a, b in zip(planned, planned[1:])]
        assert np.allclose(deltas, planned_deltas), \
            "折叠改变了音程（旋律被改动）：%s" % deltas


def test_octave_fold_uses_whole_melody_not_just_the_first_note(tmp_path):
    """**回归（突变自测逼出来的）**：折叠量必须由**整首旋律**最小化决定。

    背景：第一版八度不变式测试用的是 ``base+0, base+1, base+2, base+3`` 这种
    单调密集旋律 —— 它的**首音恰好落在整段中间**，于是"按首音折叠"和"按全曲折叠"
    给出完全相同的答案。把实现改成"只看第一个音"（突变 B6）时，20 个用例全绿，
    漏网。

    真实场景里首音完全可能是**高亢的起句**（比后面高很多）。此时若按首音折叠，
    整句会被压到偏高/偏低的音区，残留搬移显著变大 —— 旋律虽在（八度平移不改音程）
    但音色会失真，且更容易贴到钳制上限。

    本用例故意让首音高出其余音 12 半音，则：
    * 按全曲折叠 → 落在锚点附近的那个八度（残移小）
    * 按首音折叠 → 跟首音走，其余音被留在偏离锚点的八度（残移大）
    """
    ref = tmp_path / "ref_fold2.wav"
    PG._write_wav(str(ref), tone(60.0, 1.5, amp=0.6), SR)

    # 首音是"高亢起句"：72，其余落在 57~59（相差 13~15 半音）
    midis = [72.0, 57.0, 58.0, 59.0]
    specs = [(i * 0.35, 0.30, m) for i, m in enumerate(midis)]
    song = np.zeros(int(1.6 * SR), dtype=np.float32)
    for s, d, m in specs:
        seg = tone(m, d - 0.02, amp=0.55)
        song[int(round(s * SR)):int(round(s * SR)) + seg.size] += seg

    out = tmp_path / "cv_fold_outlier"
    res = PG.run(_FakeModel(), SR, reference_wav=str(ref),
                 vocal_audio=song, vocal_sr=SR,
                 out_dir=str(out), snap=False, calibrate=False,
                 keep_clips=False, logger=lambda m: None)
    sm = res["summary"]

    # ⚠️ 这里**不能**断言 ``semitone_clamped == 0``（曾经就是这么写的，然后它红了）。
    #
    # 八度折叠的契约是"让**全曲平均**搬移最小"，不是"让每个音都在 ±limit 内"。
    # 本用例的第一个音（72）是我**故意**做成的离群点：它与锚点所在八度恰好相距
    # 12 半音，正落在 ±12 的边界上。折叠器面向全曲做决策（主体 57~59 离锚点 3
    # 个半音，于是折叠 0），那个离群音就只能如实贴着上限 —— 这正是**正确行为**。
    #
    # 若为了让这一条断言变绿而去"顺带照顾离群音"，等于把折叠量交给首音决定，
    # 反而命中了本用例要防的 B6 突变（按首音折叠），自相矛盾。
    #
    # 所以验收改成核对折叠的**真实契约**：平均残移要小（说明主体音被照顾到了），
    # 而离群音**必须正好停在钳制上限**（说明它被如实记账、没有被静默改掉）。
    assert sm["semitone_mean_abs"] <= 7.0, (
        "平均残移 %.2f 过大 —— 折叠量似乎只看了首音（离群音）而非整首旋律"
        % sm["semitone_mean_abs"])
    limit = float(sm["semitone_limit"])
    assert sm["semitone_clamped"] == 1, (
        "离群首音应恰好有 1 个音被钳制在上限；实际 %d 个（离群音被静默改掉，"
        "或折叠把主体音也推进了钳制区）：%r" % (sm["semitone_clamped"], sm))
    assert abs(float(sm["semitone_max_abs"]) - limit) < 1e-6, (
        "最大残移 %.4f 没有停在钳制上限 %.4f —— 离群音要么被静默挪动，"
        "要么上限没生效：%r" % (sm["semitone_max_abs"], limit, sm))

    # 再直接核对：主体三个音的折叠后音高应贴近锚点所在的八度（57~59 附近）
    pj = json.loads(Path(res["plan_path"]).read_text(encoding="utf-8"))
    body = [float(n["midi"]) for n in pj["notes"][1:]]
    assert all(45.0 <= m <= 72.0 for m in body), \
        "主体音被折到离谱音区（说明折叠只跟了首音）：%s" % body
    near = sum(1 for m in body if abs(m - 60.0) <= 6.0)
    assert near >= 2, \
        "主体音多数不在锚点附近（折叠按首音而非全曲）：%s" % body


def test_run_without_vocal_raises_clear_error(tmp_path):
    """没给人声 stem 时必须给出**可操作**的错误，而不是空指针或静默产出静音。"""
    with pytest.raises(PG.ClearVocalError) as ei:
        PG.run(_FakeModel(), SR, None, vocal_audio=None,
               out_dir=str(tmp_path / "x"), logger=lambda m: None)
    msg = str(ei.value)
    assert "人声" in msg and "分离" in msg, "错误信息不够可操作：%s" % msg


def test_run_on_empty_audio_raises(tmp_path):
    with pytest.raises(PG.ClearVocalError):
        PG.run(_FakeModel(), SR, None,
               vocal_audio=np.zeros(0, dtype=np.float32), vocal_sr=SR,
               out_dir=str(tmp_path / "y"), logger=lambda m: None)


def test_resample_preserves_length_ratio():
    """分析用重采样必须保持时长（否则节拍/时刻全部错位）。"""
    y = tone(60.0, 1.0, sr=44100)
    out = PG._resample(y, 44100, 22050)
    assert abs(out.size - 22050) <= 1
    assert np.isfinite(out).all()
    # 升采样同样成立
    back = PG._resample(out, 22050, 44100)
    assert abs(back.size - 44100) <= 2


def test_resample_is_identity_when_rate_matches():
    y = tone(60.0, 0.25)
    out = PG._resample(y, SR, SR)
    assert out.size == y.size
    assert np.allclose(out, y)


# ===================================================================== C. 契约与安全
def _repo_root() -> Path:
    return _ROOT


def _registry_with_clear_vocal(tmp_path: Path):
    """把 vox_plugins 整个复制到 tmp，建配置并强制启用 clear_vocal。"""
    from voice_clone import plugins as P

    # ⚠️ 必须先清全局注册表：``plugins.init()`` 是**幂等**的 —— 已初始化时直接
    # 返回既有注册表，忽略新的 base_dir。不清就会拿到上一个用例的注册表，
    # 于是钩子根本没挂上、emit 静默返回原值，而测试还以为在测新插件。
    P.reset_for_tests()

    src = _repo_root() / P.PLUGIN_DIR_NAME
    assert src.is_dir(), "缺少插件目录 %s" % src
    dst = tmp_path / P.PLUGIN_DIR_NAME
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(src / "clear_vocal", dst / "clear_vocal")
    # example_gain 是仓库自带的示例，一并带上以验证共存
    if (src / "example_gain").is_dir():
        shutil.copytree(src / "example_gain", dst / "example_gain")

    cfg = tmp_path / P.CONFIG_NAME
    cfg.write_text(json.dumps({
        "search_paths": [P.PLUGIN_DIR_NAME],
        "autoload": True,
        "disabled": ["example_gain"],     # 只启用 clear_vocal，隔离验证
        "settings": {},
    }, ensure_ascii=False), encoding="utf-8")
    reg = P.init(str(tmp_path), str(cfg))
    # ⚠️ clear_vocal 随仓库发布时是 enabled=false（保证"零插件 = 行为不变"）。
    # 本文件要测的是它**被启用后**的行为，所以显式启用 —— 顺带把
    # set_enabled() 这条真实启用路径也覆盖到了。
    reg.set_enabled("clear_vocal", True)
    return reg


def test_manifest_matches_module_contract(tmp_path):
    """plugin.json 声明的每个钩子都必须有对应函数，且只接受 1 个参数。

    这条不是形式主义：``plugins.py`` 在**加载期**就要求钩子函数存在且签名合法，
    声明错一个函数名会让整个插件变成 failed 且只有一行日志。
    """
    from voice_clone import plugins as P
    reg = _registry_with_clear_vocal(tmp_path)
    try:
        lp = reg.plugins.get("clear_vocal")
        assert lp is not None, "clear_vocal 未被发现；已发现：%s" % list(reg.plugins)
        assert lp.state == "started", "插件启动失败：%s" % lp.error
        assert set(lp.handlers) == {"report.enrich", "api.routes"}
    finally:
        reg.stop_all()
        P.reset_for_tests()


def test_manifest_is_valid_json_with_expected_fields():
    p = _repo_root() / "vox_plugins" / "clear_vocal" / "plugin.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["id"] == "clear_vocal"
    assert d["api_version"] == "1.0"
    assert d["entry"] == "plugin.py"
    assert d["enabled"] is False, "清唱插件默认必须停用（保证零插件行为不变）"
    assert d["isolation"] == "inprocess"
    # 声明的每个钩子都必须真的存在
    src = (_repo_root() / "vox_plugins" / "clear_vocal" / "plugin.py").read_text(
        encoding="utf-8")
    for hook, fn in d["hooks"].items():
        assert ("def %s(" % fn) in src, "plugin.json 声明的 %s -> %s() 不存在" % (hook, fn)


def test_report_enrich_does_not_trigger_generation(tmp_path):
    """``report.enrich`` 必须是纯只读 —— 不得触碰模型。"""
    from voice_clone import plugins as P
    reg = _registry_with_clear_vocal(tmp_path)
    try:
        before = PG._STATE.get("n_runs", 0)
        out = P.emit("report.enrich", report={"n_chunks": 1})
        assert "clear_vocal" in out
        assert out["n_chunks"] == 1
        assert PG._STATE.get("n_runs", 0) == before, "report.enrich 触发了运行状态变更"
    finally:
        reg.stop_all()
        P.reset_for_tests()


def test_generation_is_refused_inside_a_hook(tmp_path):
    """最贵的一课：钩子内回调合成会**永久死锁且无报错**，必须被快速拒绝。"""
    from voice_clone import plugins as P
    reg = _registry_with_clear_vocal(tmp_path)
    caught: list[str] = []

    def _try(payload):
        try:
            SG.Singer(_FakeModel(), SR)._synth_text("啊")
        except SG.HookContextError as e:
            caught.append(str(e))
        except Exception as e:                     # 别的异常也算"没让它静默通过"
            caught.append("OTHER:" + type(e).__name__ + ": " + str(e))
        return None

    # 临时把处理器挂到 report.enrich 上跑一次（不改仓库文件）
    lp = reg.plugins["clear_vocal"]
    patched = dict(lp.handlers)
    lp.handlers["report.enrich"] = _try
    reg._reindex()
    try:
        P.emit("report.enrich", report={})
    finally:
        lp.handlers.clear()
        lp.handlers.update(patched)
        reg._reindex()
        reg.stop_all()
        P.reset_for_tests()

    assert caught, "钩子内调用 _synth_text 竟然没有被拦下（危险！）"
    assert not caught[0].startswith("OTHER"), \
        "抛的不是 HookContextError：%s" % caught[0]
    assert "_infer_lock" in caught[0] or "死锁" in caught[0]


def test_plugin_does_not_import_server_module():
    """编排层不得直接 import server —— 否则模型/锁的注入契约就被破坏了。"""
    src = (_repo_root() / "vox_plugins" / "clear_vocal" / "plugin.py").read_text(
        encoding="utf-8")
    for bad in ("import server", "from server import"):
        assert bad not in src, "plugin.py 不应依赖 server：发现 %r" % bad


def test_no_plugin_means_emit_is_unchanged():
    """零插件时 emit 必须原样返回（清唱插件的存在不得改变这条保证）。"""
    from voice_clone import plugins as P
    P.reset_for_tests()
    arr = np.zeros(8, dtype=np.float32)
    assert P.emit("output.post", audio=arr) is arr


# =================================================================== 自动识歌词
class _FakeTranscriber:
    """替身 transcriber：可控地"识别成功 / 失败 / 超时"，不加载真模型。"""

    def __init__(self, text="啊依呀", lang="zh", fail=None, hang=False):
        self.text = text
        self.lang = lang
        self.fail = fail
        self.hang = hang
        self.seen = []

    def start_transcribe(self, src_path, label="", transcript=""):
        self.seen.append(src_path)
        if self.fail == "start":
            raise RuntimeError("已有转写任务在运行")
        return {"job_id": "j1"}

    def get_job(self, jid):
        if self.hang:
            return {"status": "processing"}
        if self.fail == "error":
            return {"status": "error", "error": "whisper 崩了"}
        segs = [{"text": self.text, "start": 0.0, "end": 1.0}] if self.text else []
        return {"status": "done", "lang": self.lang, "segments": segs}


def _install_fake_transcriber(monkeypatch, fake):
    """把假 transcriber 注入 ``voice_clone.transcriber`` 的查找路径。

    ``plugin.transcribe_lyric`` 内部是 ``from voice_clone import transcriber as _tr``。

    ⚠️ 只改 ``sys.modules`` 是不够的（这里踩过坑）
    --------------------------------------------------
    ``from <包> import <子模块>`` 的解析顺序是：**先查包对象上的属性**，
    包属性不存在时才回落到 ``sys.modules``。

    所以只要**任何先前的测试**导入过真 transcriber，``voice_clone.transcriber``
    就会变成包对象上的一个真实属性；此后仅替换 ``sys.modules`` 会被包属性遮住，
    替身**静默失效**，真模块被调用（报 "音频文件不存在" 之类的环境错误）。

    这类 bug 特别阴 —— 单跑本文件时全绿（没有先序导入），
    只有和"导入过 server 的测试文件"一起跑才红，很容易被当成随机抖动。
    （实测：``tests/test_bracket_tags.py`` 导入 server 就足以触发。）

    因此这里**同时**替换包属性与 ``sys.modules``，两条路径都指向替身。
    """
    import sys as _sys
    import types as _types
    mod = _types.ModuleType("voice_clone.transcriber")
    mod.start_transcribe = fake.start_transcribe
    mod.get_job = fake.get_job
    _sys.modules["voice_clone.transcriber"] = mod

    # 关键补丁：把包属性也换掉，否则 from-import 会绕过 sys.modules。
    # monkeypatch.setattr 保证用例结束后自动还原。
    try:
        import voice_clone
        monkeypatch.setattr(voice_clone, "transcriber", mod, raising=False)
    except Exception:                            # pragma: no cover - 包不可导入
        pass


def test_transcribe_lyric_joins_segments_in_time_order(monkeypatch):
    """多段结果必须**按时间**拼回整句，而不是按 whisper 返回顺序。"""
    fake = _FakeTranscriber()

    def _jobs(jid):
        return {"status": "done", "lang": "zh", "segments": [
            {"text": "第二句", "start": 2.0, "end": 3.0},
            {"text": "第一句", "start": 0.0, "end": 1.0},
        ]}
    monkeypatch.setattr(fake, "get_job", _jobs)
    _install_fake_transcriber(monkeypatch, fake)

    r = PG.transcribe_lyric("x.wav", logger=lambda m: None)
    assert r["ok"] is True
    assert r["text"] == "第一句，第二句", r["text"]
    assert r["lang"] == "zh" and r["n_segments"] == 2


@pytest.mark.parametrize("fail,expect_kw", [
    ("start", "启动转写失败"),
    ("error", "转写失败"),
])
def test_transcribe_lyric_degrades_instead_of_raising(monkeypatch, fail,
                                                      expect_kw):
    """三种失败都必须**返回 ok=False 而不是抛异常**。

    理由：歌词缺失只是降级（退化成哼鸣），不该让整条流水线失败。
    若这里改成抛异常，``plugin.run`` 里那个 ``except`` 就成了死代码 ——
    而"自动识歌词失败导致整个任务崩掉"正是最容易被忽略的体验事故。
    """
    fake = _FakeTranscriber(fail=fail)
    _install_fake_transcriber(monkeypatch, fake)
    r = PG.transcribe_lyric("x.wav", logger=lambda m: None)
    assert r["ok"] is False, r
    assert r["text"] == "", "失败时 text 必须是空串（下游据此退化）"
    assert expect_kw in (r["error"] or ""), r


def test_transcribe_lyric_times_out_instead_of_hanging_forever(monkeypatch):
    """转写卡住时必须**超时返回**，不能把调用线程永久挂住。"""
    fake = _FakeTranscriber(hang=True)
    _install_fake_transcriber(monkeypatch, fake)
    r = PG.transcribe_lyric("x.wav", timeout=0.5, logger=lambda m: None)
    assert r["ok"] is False
    assert "超时" in (r["error"] or ""), r


def test_transcribe_lyric_missing_module_degrades(monkeypatch):
    """transcriber 整个不可用（缺依赖）时也必须优雅降级，而不是 ImportError。

    ⚠️ 只拦 ``__import__`` 是不够的：一旦 ``voice_clone.transcriber`` 已绑在包对象上，
    ``from voice_clone import transcriber`` 会走**包属性**而完全不调 ``__import__``，
    于是拦不住、真模块被执行。（与 ``_install_fake_transcriber`` 同一个坑。）

    所以这里同时**摘掉包属性 + 清掉 sys.modules**，才能真正模拟"模块不可用"。
    """
    import builtins
    import sys as _sys
    real_import = builtins.__import__

    # 确保解析真的会失败：清 sys.modules 且去掉包属性
    _sys.modules.pop("voice_clone.transcriber", None)
    try:
        import voice_clone
        monkeypatch.delattr(voice_clone, "transcriber", raising=False)
    except Exception:                            # pragma: no cover
        pass

    def _boom(name, *a, **kw):
        if name.startswith("voice_clone") and name.endswith("transcriber"):
            raise ImportError(" simulated missing dep ")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _boom)
    r = PG.transcribe_lyric("x.wav", logger=lambda m: None)
    assert r["ok"] is False
    assert "不可用" in (r["error"] or ""), r


def test_run_auto_lyric_off_by_default(tmp_path, monkeypatch):
    """默认 **不**开启自动识歌词：它会阻塞请求线程，必须由调用方显式承担。"""
    called = {"n": 0}

    def _spy(*a, **kw):
        called["n"] += 1
        return {"ok": True, "text": "啊", "lang": "zh",
                "n_segments": 1, "error": None}

    monkeypatch.setattr(PG, "transcribe_lyric", _spy)
    ref = tmp_path / "ref_auto.wav"
    PG._write_wav(str(ref), tone(60.0, 1.5, amp=0.6), SR)
    res = PG.run(_FakeModel(), SR, str(ref),
                 vocal_audio=tone(60.0, 1.0, amp=0.5), vocal_sr=SR,
                 out_dir=str(tmp_path / "auto_off"), snap=False,
                 calibrate=False, logger=lambda m: None)
    assert called["n"] == 0, "默认不该触发自动识歌词"
    assert res["lyric_source"] == "hum", res["lyric_source"]


def test_run_auto_lyric_feeds_text_into_plan(tmp_path, monkeypatch):
    """开启后：识别出的歌词必须**真的进入 note_plan**，否则等于没接。

    ⚠️ 只断言"函数被调用了"是假绿 —— 调用与接线是两件事。这里直接核对
    产物 JSON 里出现了识别文字的音节。
    """
    monkeypatch.setattr(PG, "transcribe_lyric", lambda *a, **kw: {
        "ok": True, "text": "春天来了", "lang": "zh",
        "n_segments": 1, "error": None})

    ref = tmp_path / "ref_auto2.wav"
    PG._write_wav(str(ref), tone(60.0, 1.5, amp=0.6), SR)
    out = tmp_path / "auto_on"
    res = PG.run(_FakeModel(), SR, str(ref),
                 vocal_audio=tone(60.0, 1.0, amp=0.5), vocal_sr=SR,
                 out_dir=str(out), snap=False, calibrate=False,
                 auto_lyric=True, logger=lambda m: None)

    assert res["lyric_source"] == "auto", res["lyric_source"]
    pj = json.loads(Path(res["plan_path"]).read_text(encoding="utf-8"))
    lyrics = "".join(str(n.get("lyric") or "") for n in pj["notes"])
    assert lyrics, "识别出了歌词但 note_plan 里一个音节都没有（接线断了）"
    assert set(lyrics) <= set("春天来了"), \
        "note_plan 里的歌词不是识别结果：%r" % lyrics


def test_run_auto_lyric_failure_still_completes_as_hum(tmp_path, monkeypatch):
    """识别失败 → 整条流水线**照常跑完**并退化为哼鸣，错误只进 errors 列表。"""
    monkeypatch.setattr(PG, "transcribe_lyric", lambda *a, **kw: {
        "ok": False, "text": "", "lang": None,
        "n_segments": 0, "error": "whisper 崩了"})

    ref = tmp_path / "ref_auto3.wav"
    PG._write_wav(str(ref), tone(60.0, 1.5, amp=0.6), SR)
    res = PG.run(_FakeModel(), SR, str(ref),
                 vocal_audio=tone(60.0, 1.0, amp=0.5), vocal_sr=SR,
                 out_dir=str(tmp_path / "auto_fail"), snap=False,
                 calibrate=False, auto_lyric=True, logger=lambda m: None)

    assert res["ok"] is True, "歌词识别失败不该让整条流水线失败"
    assert res["lyric_source"] == "hum"
    assert any("歌词" in e for e in res["errors"]), res["errors"]
    assert Path(res["out_wav"]).exists()


def test_run_explicit_lyric_wins_over_auto(tmp_path, monkeypatch):
    """用户手写的歌词优先级最高 —— 有歌词就不该再花时间去转写。"""
    called = {"n": 0}

    def _spy(*a, **kw):
        called["n"] += 1
        return {"ok": True, "text": "错的", "lang": "zh",
                "n_segments": 1, "error": None}

    monkeypatch.setattr(PG, "transcribe_lyric", _spy)
    ref = tmp_path / "ref_auto4.wav"
    PG._write_wav(str(ref), tone(60.0, 1.5, amp=0.6), SR)
    res = PG.run(_FakeModel(), SR, str(ref),
                 vocal_audio=tone(60.0, 1.0, amp=0.5), vocal_sr=SR,
                 lyric_text="啦啦啦", out_dir=str(tmp_path / "auto_skip"),
                 snap=False, calibrate=False, auto_lyric=True,
                 logger=lambda m: None)
    assert called["n"] == 0, "已有歌词时不该再跑转写"
    assert res["lyric_source"] == "text", res["lyric_source"]


# =================================================================== HTTP 路由
def test_api_routes_registers_status_route_via_testclient():
    """``api.routes`` 钩子必须真的注册出**可访问**的 `/api/plugins/clear_vocal/status`。

    ⚠️ 这个用例的写法是踩坑之后改的，别改回"扫 ``app.routes`` 找 path"。
    ---------------------------------------------------------------------
    我原先的探针是：

        paths = [r.path for r in app.routes if r.path.startswith("/api/plugins")]
        assert paths          # ← 这里永远是空的！

    FastAPI 0.141 把 ``app.include_router(router)`` 的产物包成一个
    ``_IncludedRouter`` 对象，**它没有 ``.path`` 属性**，于是列表推导式
    静默产出空列表 —— 看起来像"路由根本没注册"，实际 HTTP 200 完全正常。
    **一个错的探针会把正确的代码判成错的**，而且它的输出（空列表）
    看起来同样"合理"。

    正确做法：用 ``TestClient`` 发一次**真实请求**，断言状态码与响应体。
    这才是"路由通了"的唯一可信证据。
    """
    from voice_clone import plugins as P
    P.reset_for_tests()
    P.init(str(_ROOT))
    P.get_registry().set_enabled("clear_vocal", True)
    P.get_registry()._reindex()

    try:
        from fastapi import FastAPI
        try:
            from fastapi.testclient import TestClient
        except Exception:                                  # pragma: no cover
            pytest.skip("TestClient 不可用")
        app = FastAPI()
        P.emit("api.routes", app=app)
        client = TestClient(app)
        resp = client.get("/api/plugins/clear_vocal/status")
        assert resp.status_code == 200, (
            "插件状态路由没生效（HTTP %d）—— api.routes 钩子可能没注册成功"
            % resp.status_code)
        body = resp.json()
        for k in ("ready", "n_runs", "n_failed", "last_run"):
            assert k in body, "状态路由响应缺字段 %r：%r" % (k, body)
    finally:
        P.reset_for_tests()


def test_api_routes_hook_tolerates_bad_app():
    """``app`` 缺失/类型不对时钩子必须**安静返回**，不能让服务启动失败。"""
    assert PG.on_api_routes({}) is None
    assert PG.on_api_routes({"app": None}) is None
    assert PG.on_api_routes(None) is None
    assert PG.on_api_routes({"app": object()}) is None   # 非 FastAPI 对象

