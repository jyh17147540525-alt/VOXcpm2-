"""清唱生成插件 · 对齐器（aligner.py）与 DSP 层的正式回归测试

对应 ``vox_plugins/clear_vocal/_t_aligner.py`` 那套真值断言，收编进 pytest 套件。
侧重点与用户验收项一致：**时间对齐精度** 与 **音高准确性**。

为什么这些断言写成"精确 == 0 样本"而不是"误差 < 5ms"：
因为对齐误差会随音符数**累加**。单音差 5ms 不致命，300 个音各差 5ms
就是整曲跑偏 1.5 秒。所以量级必须卡在零 —— 见 ``test_no_drift_over_many_notes``。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 与其它测试文件一致的显式引导：把仓库根插到 sys.path 首位。
# 不依赖 pytest 的 rootdir 插入 —— 后者会随**收集顺序**变化。
# ⚠️ 包名是 vox_plugins，不是 plugins：仓库里已有 voice_clone/plugins.py，
#    而 pytest 会把 voice_clone/ 插到 sys.path，`import plugins` 会被它截胡。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vox_plugins.clear_vocal import aligner, dsp  # noqa: E402

SR = 22050


# ------------------------------------------------------------------ 夹具
def tone(midi: float, dur: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """带谐波的测试音（纯正弦会让 pyin 退化，故加 4 次谐波）。"""
    import librosa
    f = float(librosa.midi_to_hz(midi))
    t = np.arange(int(round(dur * sr)), dtype=np.float64) / sr
    y = np.zeros_like(t)
    for h, a in enumerate([1.0, 0.5, 0.3, 0.15], start=1):
        y += a * np.sin(2.0 * np.pi * f * h * t)
    k = min(int(0.01 * sr), len(y) // 4)
    if k > 1:
        r = np.linspace(0.0, 1.0, k)
        y[:k] *= r
        y[-k:] *= r[::-1]
    return (amp * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def measure_pitch(y: np.ndarray, sr: int = SR) -> float:
    """测中位 F0（Hz），取中段 60% 避开包络。"""
    import librosa
    n = len(y)
    if n < 2048:
        return float("nan")
    seg = y[int(0.2 * n):int(0.8 * n)]
    f0, _, _ = librosa.pyin(seg, fmin=65.0, fmax=1000.0, sr=sr, hop_length=512)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else float("nan")


def semitone_error(measured_hz: float, want_hz: float) -> float:
    if not np.isfinite(measured_hz) or measured_hz <= 0:
        return 99.0
    return abs(12.0 * np.log2(measured_hz / want_hz))


# ------------------------------------------------------------------ 长度精度
@pytest.mark.parametrize("n_in,n_out", [(1000, 1000), (1000, 940), (1000, 1063),
                                        (7, 3), (0, 128)])
def test_fit_length_is_exact(n_in: int, n_out: int) -> None:
    rng = np.random.default_rng(7)
    y = (rng.standard_normal(n_in).astype(np.float32) if n_in
         else np.zeros(0, np.float32))
    assert aligner._fit_length(y, n_out).size == n_out


def test_fit_length_pad_has_no_discontinuity() -> None:
    """补长必须用边界值延续；补零会在音符尾制造跳变（爆音）。"""
    y = np.ones(100, dtype=np.float32) * 0.3
    out = aligner._fit_length(y, 200)
    assert float(np.abs(np.diff(out)).max()) < 1e-6


@pytest.mark.parametrize("target", [0.20, 0.45, 0.80, 1.60])
def test_stretch_to_exact_length(target: float) -> None:
    out = aligner.stretch_to(tone(60.0, 0.40), SR, target)
    assert out.size == int(round(target * SR))


def test_stretch_beyond_algo_range_still_exact_length() -> None:
    """超出 time_stretch 有效倍率（内部钳制 4x）时，长度仍须精确。"""
    out = aligner.stretch_to(tone(60.0, 0.10), SR, 1.20)      # 12x
    assert out.size == int(round(1.20 * SR))


# ------------------------------------------------------------------ 音高
@pytest.mark.parametrize("semis", [-12.0, -7.0, -3.0, 3.0, 7.0, 12.0])
def test_shift_to_pitch_accuracy(semis: float) -> None:
    import librosa
    clip = tone(60.0, 0.50)
    out, applied = aligner.shift_to(clip, SR, 60.0 + semis, 60.0)
    assert abs(applied - semis) < 1e-6
    assert out.size == clip.size, "音高搬移不得改变时长"
    err = semitone_error(measure_pitch(out), float(librosa.midi_to_hz(60.0 + semis)))
    assert err < 0.5, "音高误差 %.3f 半音" % err


@pytest.mark.parametrize("target,source", [(84.0, 60.0), (36.0, 60.0)])
def test_shift_to_clamps_octave_jumps(target: float, source: float) -> None:
    """pyin 偶发八度误检时必须钳制，否则听感上是刺耳走音。"""
    _, applied = aligner.shift_to(tone(60.0, 0.40), SR, target, source,
                                  max_semitones=12.0)
    assert abs(abs(applied) - 12.0) < 1e-9


def test_shift_to_handles_nonfinite() -> None:
    clip = tone(60.0, 0.40)
    _, a = aligner.shift_to(clip, SR, float("nan"), 60.0)
    assert a == 0.0
    out, b = aligner.shift_to(clip.copy(), SR, None, None)
    assert b == 0.0 and out.size == clip.size


def test_stretch_preserves_pitch() -> None:
    clip = tone(60.0, 0.40)
    base = measure_pitch(clip)
    for target in (0.30, 0.60, 1.00):
        out = aligner.stretch_to(clip, SR, target)
        assert semitone_error(measure_pitch(out), base) < 0.5


# ------------------------------------------------------------------ 落点规划
def test_plan_timing_is_monotonic_and_bounded() -> None:
    notes = [
        {"start": 0.00, "end": 0.30, "dur": 0.30, "midi": 60.0},
        {"start": 0.20, "end": 0.50, "dur": 0.30, "midi": 62.0},    # 重叠
        {"start": 0.50, "end": 0.505, "dur": 0.005, "midi": 64.0},  # 过短
        {"start": 1.20, "end": 1.60, "dur": 0.40, "midi": 65.0},    # 越界
    ]
    timed = aligner.plan_note_timing(notes, bpm=0.0, snap=False,
                                     total_dur=1.0, min_dur=0.05)
    for a, b in zip(timed, timed[1:]):
        assert a["end"] <= b["start"] + 1e-9, "音符不得重叠"
    assert all(t["dur"] >= 0.05 - 1e-9 for t in timed), "过短音应被延长"
    assert all(t["end"] <= 1.0 + 1e-9 for t in timed), "末音不得越界"


def test_sample_grid_never_overlaps_after_rounding() -> None:
    notes = [{"start": 0.10002, "end": 0.10005, "dur": 3e-5, "midi": 60.0},
             {"start": 0.10006, "end": 0.20000, "dur": 0.09994, "midi": 61.0},
             {"start": 0.20001, "end": 0.30000, "dur": 0.09999, "midi": 62.0}]
    timed = aligner.to_sample_grid(
        aligner.plan_note_timing(notes, bpm=0.0, snap=False, min_dur=0.001), SR)
    for a, b in zip(timed, timed[1:]):
        assert a["start_sample"] + a["n_samples"] <= b["start_sample"]


def test_snap_times_120bpm_sixteenth_grid() -> None:
    t = np.array([0.0, 0.13, 0.249, 0.38])
    got = aligner.snap_times(t, 120.0, subdiv=4)
    assert np.allclose(got, [0.0, 0.125, 0.25, 0.375], atol=1e-9)


def test_snap_times_degrades_without_bpm() -> None:
    t = np.array([0.0, 0.13, 0.249])
    assert np.allclose(aligner.snap_times(t, 0.0), t)
    assert aligner.snap_times(np.zeros(0), 120.0).size == 0


# ------------------------------------------------------------------ 拼接
def test_assemble_places_and_leaves_gaps_silent() -> None:
    clip = tone(60.0, 0.20)
    out = aligner.assemble([({"start_sample": 0, "n_samples": clip.size}, clip),
                            ({"start_sample": int(0.5 * SR), "n_samples": clip.size}, clip)],
                           SR, total_dur=1.0)
    assert out.size == SR
    gap = out[int(0.30 * SR):int(0.45 * SR)]
    assert float(np.abs(gap).max()) < 1e-6, "音符间隙必须留白"
    assert float(np.abs(out[int(0.55 * SR):int(0.68 * SR)]).max()) > 0.05
    # 硬削波的标志是峰值顶格在 1.0，必须严格小于
    assert float(np.abs(out).max()) < 0.9999


def test_assemble_no_clipping_under_dense_overlap() -> None:
    """8 个满幅片段叠在同一落点 → 迫使软限幅器真正介入。

    没有这条用例，``peak_normalize`` 单独就足以把峰值压到 0.97，
    两个不重叠的片段永远碰不到 0.95 阈值 → 限幅那段代码形同未测。
    """
    clip = tone(60.0, 0.20, amp=0.95)
    dense = [({"start_sample": 0, "n_samples": clip.size}, clip) for _ in range(8)]
    out = aligner.assemble(dense, SR, total_dur=0.25)
    peak = float(np.abs(out).max())
    assert peak < 0.9999, "叠加溢出未被限幅处理（peak %.4f）" % peak
    assert peak > 0.80, "限幅过度导致响度塌陷（peak %.4f）" % peak


def test_no_drift_over_many_notes() -> None:
    """300 个音符后逐音长度误差必须仍为 0（不得累加）。"""
    n = 300
    notes = [{"start": i * 0.20, "end": i * 0.20 + 0.18, "dur": 0.18,
              "midi": 60.0 + (i % 5)} for i in range(n)]
    timed = aligner.to_sample_grid(
        aligner.plan_note_timing(notes, bpm=0.0, snap=False), SR)
    for t in timed:
        assert t["n_samples"] == int(round(t["dur"] * SR))
    last = timed[-1]
    assert last["start_sample"] + last["n_samples"] == int(round(last["end"] * SR))


# ------------------------------------------------------------------ 一体化
def test_align_note_hits_duration_and_pitch() -> None:
    import librosa
    out, diag = aligner.align_note(tone(60.0, 0.35), SR, target_dur=0.62,
                                   target_midi=67.0, source_midi=60.0)
    assert out.size == int(round(0.62 * SR))
    err = semitone_error(measure_pitch(out), float(librosa.midi_to_hz(67.0)))
    assert err < 0.5, "音高误差 %.3f 半音" % err
    assert abs(diag["stretch_ratio"] - 0.62 / 0.35) < 0.02
    assert abs(diag["applied_semitones"] - 7.0) < 1e-6


def test_align_all_truncates_to_shorter_side() -> None:
    notes = [{"start": 0.0, "end": 0.25, "dur": 0.25, "midi": 60.0, "lyric": "a"},
             {"start": 0.25, "end": 0.50, "dur": 0.25, "midi": 62.0, "lyric": "o"}]
    clips = [tone(60.0, 0.30), tone(62.0, 0.30), tone(64.0, 0.30)]   # 多一个
    audio, diags = aligner.align_all(notes, clips, SR, bpm=0.0, snap=False,
                                     total_dur=0.5)
    assert len(diags) == 2
    assert audio.size == int(round(0.5 * SR))
    assert aligner.summary(diags)["n_notes"] == 2


# ------------------------------------------------------------------ 诊断统计
def test_summary_counts_clamping_against_the_actual_limit() -> None:
    """**回归**：``summary`` 必须按调用方实际传的 ``max_semitones`` 判定钳制。

    曾经的写法是拿模块常量 ``DEFAULT_MAX_SEMITONES``(12.0) 去比。若调用方传 3.0，
    那些被钳到 3.00 的音符会被报成"没被钳制" —— 一个静默的假绿：旋律明明被改了，
    统计却说一切正常。所以这里用 3.0 上限造一个必然被钳的样本，断言它被数出来。
    """
    notes = [{"start": 0.0, "end": 0.25, "dur": 0.25, "midi": 60.0, "lyric": "a"},
             {"start": 0.25, "end": 0.50, "dur": 0.25, "midi": 62.0, "lyric": "o"}]
    # ⚠️ 必须显式给 source_midi：若省略，align_all 会回退到 ``midi`` 自身
    #    （``info.get("source_midi", info.get("midi"))``），于是 target==source、
    #    搬移恒为 0 —— 测试就变成了假绿，永远测不到钳制逻辑。
    clips = [tone(48.0, 0.35), tone(48.0, 0.35)]     # 比目标低 12 半音 → 必然越 3.0 上限
    for n, src in zip(notes, (48.0, 50.0)):
        n["source_midi"] = src
    audio, diags = aligner.align_all(notes, clips, SR, bpm=0.0, snap=False,
                                     total_dur=0.5, max_semitones=3.0)
    s = aligner.summary(diags, max_semitones=3.0)
    assert s["semitone_limit"] == 3.0, "summary 未反映实际生效的钳制上限"
    assert s["semitone_clamped"] == 2, (
        "按 3.0 上限应数出 2 个被钳制的音符，实际 %d（说明判定用的是别的上限）"
        % s["semitone_clamped"])
    # 同一批诊断，若误按默认 12.0 判定，就会漏报 —— 这正是被修掉的 bug
    assert aligner.summary(diags, max_semitones=12.0)["semitone_clamped"] == 0


def test_align_all_respects_a_custom_max_semitones() -> None:
    """``align_all`` 必须把自己收到的上限真正透传给 ``shift_to``（而非用默认值）。"""
    notes = [{"start": 0.0, "end": 0.25, "dur": 0.25, "midi": 72.0,
              "source_midi": 60.0, "lyric": "a"}]     # 需要 +12 半音
    clips = [tone(60.0, 0.35)]
    _, d_wide = aligner.align_all(notes, clips, SR, bpm=0.0, snap=False,
                                  total_dur=0.4, max_semitones=24.0)
    _, d_narrow = aligner.align_all(notes, clips, SR, bpm=0.0, snap=False,
                                    total_dur=0.4, max_semitones=5.0)
    assert abs(d_wide[0]["applied_semitones"] - 12.0) < 1e-6, \
        "宽上限下应完整搬移 12 半音，实际 %r" % d_wide[0]["applied_semitones"]
    assert abs(d_narrow[0]["applied_semitones"] - 5.0) < 1e-6, \
        "窄上限下应钳到 5 半音，实际 %r" % d_narrow[0]["applied_semitones"]


# ------------------------------------------------------------------ 退化输入
def test_degenerate_inputs_do_not_raise() -> None:
    assert aligner.stretch_to(np.zeros(0, np.float32), SR, 0.5).size == int(round(0.5 * SR))
    assert aligner.stretch_to(tone(60, 0.1), SR, 0.0).size == 0
    assert aligner.plan_note_timing([]) == []
    assert aligner.assemble([], SR, total_dur=0.5).size == int(round(0.5 * SR))
    audio, diags = aligner.align_all([], [], SR, total_dur=0.3)
    assert audio.size == int(round(0.3 * SR)) and diags == []
    assert aligner.summary([])["n_notes"] == 0


# ------------------------------------------------------------------ DSP 底层
def test_stft_istft_roundtrip() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_normal(SR).astype(np.float32) * 0.2
    err = float(np.abs(dsp.istft(dsp.stft(y), length=len(y)) - y).max())
    assert err < 1e-4, "STFT/iSTFT 往返误差 %.2e" % err


def test_peak_normalize_does_not_amplify_silence() -> None:
    quiet = np.full(100, 1e-8, dtype=np.float32)
    assert np.allclose(dsp.peak_normalize(quiet), quiet)
    assert dsp.peak_normalize(np.zeros(0, np.float32)).size == 0
