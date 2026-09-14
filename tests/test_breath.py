"""
呼吸声合成回归测试（audio_edit.generate_breath / BreathState）
================================================================
这些用例锁死的是**声学正确性**，不是"代码能跑"。每条断言都对应一个
可量化的物理事实，改参数时若破坏它就应当失败：

  L1 谱型
      摩擦音噪声的谱斜率必须落在 1/f^β 区间，而不是布朗噪声 1/f²。
      这是旧实现（np.cumsum 积分白噪声）最根本的错误，必须锁死。

  L2 共振峰
      声道塑形必须在设计频率附近形成谐振峰（源-滤波器模型）。

  L3 包络
      吸气的能量重心必须靠前、呼气靠后（生理起落方向不能反）。

  L4 响度与可复现性
      strength 必须是线性、单调、可复现的电平旋钮。

  L5 AR(1) 生理相关
      相邻换气必须相关（rho≈0.42），不能退化成 i.i.d.。

  L6 拼接契约
      _join_pieces 必须真的把双相换气插进句末停顿。

导入策略沿用 test_director.py：按文件路径加载，避开 voice_clone/__init__.py
的链式重依赖；音频依赖（scipy）缺失时整体 skip。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

pytest.importorskip("scipy", reason="呼吸声验证依赖 scipy（Welch PSD / 滤波器）")

import importlib.util as _ilu  # noqa: E402

_AUDIO_SRC = _ROOT / "audio_edit.py"
if not _AUDIO_SRC.exists():
    pytest.skip("audio_edit.py 缺失", allow_module_level=True)


def _load_audio_edit():
    spec = _ilu.spec_from_file_location("_ae_under_test", _AUDIO_SRC)
    mod = _ilu.module_from_spec(spec)
    sys.modules["_ae_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ae = _load_audio_edit()
SR = 24000


# --------------------------------------------------------------------- helpers
def _psd_slope(y, sr=SR, f_lo=200.0, f_hi=4000.0):
    """Welch PSD 在 [f_lo, f_hi] 上的 log-log 斜率。"""
    from scipy.signal import welch
    f, pxx = welch(y, fs=sr, nperseg=min(len(y), 2048))
    m = (f >= f_lo) & (f <= f_hi) & (pxx > 0)
    assert m.sum() >= 8, "有效频点太少，无法拟合斜率"
    return float(np.polyfit(np.log10(f[m]), np.log10(pxx[m]), 1)[0])


def _centroid(y):
    """能量包络的重心位置（0=最前，1=最后）。"""
    env = np.abs(y)
    k = max(1, len(env) // 200)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    t = np.linspace(0.0, 1.0, len(env))
    return float((env * t).sum() / (env.sum() + 1e-12))


def _acf1(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if x.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


SEEDS = tuple(range(16))


# ------------------------------------------------------------------- L1 谱型
def test_inhale_spectral_slope_is_pink_not_brown():
    """吸气噪声的谱斜率必须在 -0.6~-1.1（1/f^β 粉红/摩擦噪声区间）。

    布朗噪声（旧实现的 cumsum）斜率约 -2，会被这条断言拦下。
    """
    slopes = [_psd_slope(ae.generate_breath(SR, 0.6, seed=s, kind="inhale"))
              for s in SEEDS]
    mean = float(np.mean(slopes))
    assert -1.15 < mean < -0.55, \
        f"吸气谱斜率 {mean:+.3f} 超出粉红噪声区间（是否又退回 cumsum 了？）"


def test_exhale_spectral_slope_is_pink_not_brown():
    """呼气同样必须是 1/f^β，且比吸气略陡（气流减速、高频更快滚降）。"""
    slopes = [_psd_slope(ae.generate_breath(SR, 0.6, seed=s, kind="exhale"))
              for s in SEEDS]
    mean = float(np.mean(slopes))
    assert -1.35 < mean < -0.75, f"呼气谱斜率 {mean:+.3f} 超出粉红噪声区间"


def test_exhale_is_darker_than_inhale():
    """呼气的谱应比吸气更暗（斜率更陡 = 高频相对更少）。"""
    inh = np.mean([_psd_slope(ae.generate_breath(SR, 0.6, seed=s, kind="inhale"))
                   for s in SEEDS])
    exh = np.mean([_psd_slope(ae.generate_breath(SR, 0.6, seed=s, kind="exhale"))
                   for s in SEEDS])
    assert exh < inh, f"呼气({exh:+.3f})应比吸气({inh:+.3f})更暗（斜率更陡）"


def test_no_subsonic_energy():
    """换气声不该有次声/低频隆隆声：<80 Hz 能量占比必须极低。"""
    from scipy.signal import welch
    for kind in ("inhale", "exhale"):
        worst = 0.0
        for s in SEEDS:
            y = ae.generate_breath(SR, 0.6, seed=s, kind=kind)
            f, p = welch(y, fs=SR, nperseg=min(len(y), 4096))
            m = f < 80.0
            worst = max(worst, float(p[m].sum() / (p.sum() + 1e-20)))
        assert worst < 0.05, f"{kind} 的 <80Hz 能量占比 {worst:.1%} 过高"


def test_no_dc_offset():
    """换气声没有直流分量（零均值）。"""
    for kind in ("inhale", "exhale"):
        dc = [abs(float(np.mean(ae.generate_breath(SR, 0.6, seed=s, kind=kind))))
              for s in SEEDS]
        assert max(dc) < 1e-3, f"{kind} 存在直流偏置: max|DC|={max(dc):.2e}"


# ----------------------------------------------------------------- L2 共振峰
def test_formant_table_has_expected_shape():
    """共振峰表必须存在、按增益衰减排序、且吸气整体比呼气更亮。"""
    for kind in ("inhale", "exhale"):
        fms = ae._BREATH_FORMANTS[kind]
        assert len(fms) >= 3, f"{kind} 的共振峰少于 3 个，腔体感会不足"
        fcs = [fc for fc, _, _ in fms]
        assert fcs == sorted(fcs), f"{kind} 共振峰未按频率升序"
        gains = [g for _, g, _ in fms]
        assert gains[0] == 0.0, f"{kind} 第一个共振峰应为参考增益 0dB"
        assert all(g <= 0.01 for g in gains[1:]), f"{kind} 后续共振峰增益应递减"
        assert all(bw > 0 for _, _, bw in fms), f"{kind} 存在非正带宽"
    # 吸气共振峰应整体高于呼气（气流快 -> 腔体等效更短 -> 谐振更高）
    assert ae._BREATH_FORMANTS["inhale"][0][0] > ae._BREATH_FORMANTS["exhale"][0][0]


def test_resonators_actually_peak_at_design_frequency():
    """滤波器响应必须在设计共振峰附近出现峰（源-滤波器模型的核心）。"""
    from scipy.signal import iirpeak, freqz
    for kind in ("inhale", "exhale"):
        # 直接对**滤波器**取响应，而不是对输出取 PSD：输出的 1/f 源谱会
        # 把峰位往低频方向拉，是测量陷阱（第一版验证脚本就栽在这里）。
        w = np.geomspace(200.0, SR / 2 * 0.9, 8000)
        H = np.zeros_like(w)
        for fc, gain_db, bw in ae._BREATH_FORMANTS[kind]:
            if fc >= SR / 2.0 * 0.95:
                continue
            b, a = iirpeak(fc / (SR / 2.0), max(fc / bw, 0.5))
            _, h = freqz(b, a, worN=w, fs=SR)
            H += (10.0 ** (gain_db / 20.0)) * np.abs(h)
        peaks = [w[i] for i in range(1, len(H) - 1)
                 if H[i] > H[i - 1] and H[i] >= H[i + 1]]
        assert peaks, f"{kind} 的滤波器响应没有任何峰"
        # 至少一个峰落在最低设计共振峰的 ±15% 内
        fc0 = ae._BREATH_FORMANTS[kind][0][0]
        assert any(abs(p - fc0) / fc0 < 0.15 for p in peaks), \
            f"{kind} 最低共振峰 {fc0:.0f}Hz 附近无谐振峰，实测峰: {peaks[:5]}"


# ------------------------------------------------------------------- L3 包络
def test_inhale_envelope_centroid_is_front_loaded():
    """吸气必须「快起」：能量重心明确靠前。"""
    c = [_centroid(ae.generate_breath(SR, 0.6, seed=s, kind="inhale"))
         for s in SEEDS]
    mean = float(np.mean(c))
    assert mean < 0.50, f"吸气重心 {mean:.3f} 未靠前（应 <0.50）"


def test_exhale_envelope_centroid_is_back_loaded():
    """呼气必须「缓起」：能量重心明确靠后。"""
    c = [_centroid(ae.generate_breath(SR, 0.6, seed=s, kind="exhale"))
         for s in SEEDS]
    mean = float(np.mean(c))
    assert mean > 0.50, f"呼气重心 {mean:.3f} 未靠后（应 >0.50）"


def test_inhale_peak_precedes_exhale_peak():
    """吸气峰值位置必须早于呼气——这是双相结构的判别特征。"""
    def peak_pos(y):
        env = np.abs(y)
        k = max(1, len(env) // 200)
        env = np.convolve(env, np.ones(k) / k, mode="same")
        return float(np.argmax(env) / max(len(env) - 1, 1))

    inh = np.mean([peak_pos(ae.generate_breath(SR, 0.6, seed=s, kind="inhale"))
                   for s in SEEDS])
    exh = np.mean([peak_pos(ae.generate_breath(SR, 0.6, seed=s, kind="exhale"))
                   for s in SEEDS])
    assert inh < exh, f"吸气峰值{inh:.3f}应早于呼气峰值{exh:.3f}"


def test_envelope_has_no_onset_discontinuity():
    """起音不能是硬阶跃（否则插进静音处会"啪"一声）。"""
    for kind in ("inhale", "exhale"):
        y = ae.generate_breath(SR, 0.8, seed=3, kind=kind)
        # 前 2ms 的样本必须平滑启动
        head = np.max(np.abs(y[: int(SR * 0.002)]))
        peak = np.max(np.abs(y)) + 1e-12
        assert head < peak * 0.35, f"{kind} 起音过陡，可能产生咔哒声"


def test_ends_fade_to_near_silence():
    """尾部必须淡出到接近静音。"""
    for kind in ("inhale", "exhale"):
        y = ae.generate_breath(SR, 0.8, seed=3, kind=kind)
        tail = np.max(np.abs(y[-int(SR * 0.004):]))
        peak = np.max(np.abs(y)) + 1e-12
        assert tail < peak * 0.30, f"{kind} 尾部未充分淡出"


# ------------------------------------------------- L4 响度 / 可复现性
def test_strength_is_linear_in_rms():
    """strength 必须线性映射到 RMS（等响度归一化的直接后果）。"""
    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    rms = []
    for st in levels:
        r = [float(np.sqrt(np.mean(ae.generate_breath(SR, st, seed=s) ** 2)))
             for s in range(6)]
        rms.append(float(np.mean(r)))
    # 相邻档的比值应等于档位之比（0.4/0.2=2.0, 0.6/0.4=1.5, ...）
    for i in range(len(levels) - 1):
        expect = levels[i + 1] / levels[i]
        got = rms[i + 1] / rms[i]
        assert abs(got - expect) < 0.05 * expect, \
            f"strength {levels[i]}->{levels[i+1]} 的 RMS 比值 {got:.3f} != {expect:.3f}"


def test_strength_is_monotonic():
    rms = [float(np.sqrt(np.mean(ae.generate_breath(SR, s / 20, seed=5) ** 2)))
           for s in range(1, 21)]
    assert all(b > a for a, b in zip(rms, rms[1:])), "RMS 未随 strength 单调递增"


def test_zero_strength_returns_empty():
    assert len(ae.generate_breath(SR, 0.0)) == 0
    assert len(ae.generate_breath(SR, 0.005)) == 0


def test_same_seed_is_bit_identical():
    a = ae.generate_breath(SR, 0.5, seed=123, kind="inhale")
    b = ae.generate_breath(SR, 0.5, seed=123, kind="inhale")
    assert np.array_equal(a, b), "同 seed 必须完全可复现"


def test_different_seed_differs():
    a = ae.generate_breath(SR, 0.5, seed=1)
    b = ae.generate_breath(SR, 0.5, seed=2)
    assert not np.array_equal(a, b), "不同 seed 应产生不同波形"


def test_auto_seed_is_nondeterministic():
    a = ae.generate_breath(SR, 0.5, seed=None)
    b = ae.generate_breath(SR, 0.5, seed=None)
    assert not np.array_equal(a, b), "seed=None 时应每次不同"


def test_duration_hint_is_respected():
    """显式 duration 应大致决定输出长度。"""
    for want in (0.15, 0.30, 0.45):
        y = ae.generate_breath(SR, 0.5, duration=want, seed=9)
        got = len(y) / SR
        assert abs(got - want) < 0.02, f"duration={want} 实测 {got:.3f}s"


def test_output_dtype_and_range():
    for kind in ("inhale", "exhale"):
        y = ae.generate_breath(SR, 1.0, seed=4, kind=kind)
        assert y.dtype == np.float32, f"{kind} 输出应为 float32"
        assert np.max(np.abs(y)) < 1.0, f"{kind} 输出削波"


def test_unknown_kind_defaults_to_inhale():
    a = ae.generate_breath(SR, 0.5, seed=77, kind="whatever")
    b = ae.generate_breath(SR, 0.5, seed=77, kind="inhale")
    assert np.array_equal(a, b), "未知 kind 应回退到 inhale"


# ------------------------------------------------------------------ L5 AR(1)
def test_breath_state_autocorrelation_matches_rho():
    for rho in (0.0, 0.3, 0.42, 0.7):
        st = ae.BreathState(rho=rho, seed=5)
        amps = [st.next_params(0.5)[0] for _ in range(600)]
        got = _acf1(amps)
        assert abs(got - rho) < 0.12, \
            f"rho={rho} 实测一阶自相关 {got:+.3f}，AR(1) 建模未生效"


def test_breath_state_duration_also_correlates():
    st = ae.BreathState(rho=0.42, seed=8)
    durs = [st.next_params(0.5)[1] for _ in range(600)]
    assert abs(_acf1(durs) - 0.42) < 0.12, "时长通道的惯性未生效"


def test_explicit_duration_still_gets_inertia():
    """外部显式给 duration 时，惯性也必须生效。

    第一版这里会退化：duration 非 None 就直接原样返回，
    导致拼接路径里所有换气时长完全相同（被验证脚本抓出）。
    """
    st = ae.BreathState(rho=0.42, seed=11)
    durs = [st.next_params(0.5, duration=0.30)[1] for _ in range(300)]
    assert len(set(round(d, 6) for d in durs)) > 50, \
        "显式 duration 下时长无抖动，AR(1) 失效"
    assert abs(_acf1(durs) - 0.42) < 0.12, "显式 duration 下惯性未生效"


def test_state_reset_restores_sequence():
    st = ae.BreathState(rho=0.42, seed=99)
    first = [st.next_params(0.5)[:2] for _ in range(10)]
    st.reset()
    second = [st.next_params(0.5)[:2] for _ in range(10)]
    assert first == second, "reset() 后应重放同一序列"


def test_iid_mode_differs_from_correlated_mode():
    """rho=0 与 rho=0.42 应给出统计上可区分的序列。"""
    a = ae.BreathState(rho=0.0, seed=3)
    b = ae.BreathState(rho=0.8, seed=3)
    sa = [a.next_params(0.5)[0] for _ in range(400)]
    sb = [b.next_params(0.5)[0] for _ in range(400)]
    assert abs(_acf1(sa)) < 0.15, "rho=0 应接近 i.i.d."
    assert _acf1(sb) > 0.55, "rho=0.8 应表现出强记忆"


def test_breath_state_amplitude_stays_in_range():
    st = ae.BreathState(rho=0.9, seed=13)
    amps = [st.next_params(0.5)[0] for _ in range(1000)]
    assert all(0.0 <= a <= 1.5 for a in amps), "幅度越界"


# -------------------------------------------------------------- L6 拼接契约
def test_join_with_breath_inserts_double_phase():
    """join_with_breath 必须插入「吸气+呼气」两相，而不只是单段。"""
    tone = np.zeros(int(SR * 0.5), dtype=np.float32)
    tone[int(SR * 0.1):int(SR * 0.4)] = 0.3
    out = ae.join_with_breath([tone, tone], SR, pause=0.15, breath=0.5)
    silent_ref = ae.join_with_breath([tone, tone], SR, pause=0.15, breath=0.0)
    assert len(out) > len(silent_ref) + int(SR * 0.2), \
        "开启 breath 后长度应明显增加（双相换气）"


def test_join_pieces_wires_dual_phase_breath():
    """synthesis_stab._join_pieces 的句末停顿必须带上双相换气。"""
    stab = _ROOT / "voice_clone" / "synthesis_stab.py"
    if not stab.exists():
        pytest.skip("synthesis_stab.py 缺失")
    spec = _ilu.spec_from_file_location("_stab_under_test", stab)
    mod = _ilu.module_from_spec(spec)
    sys.modules["_stab_under_test"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # 重依赖缺失（librosa 等）
        pytest.skip(f"synthesis_stab 无法独立加载: {exc}")
    if not hasattr(mod, "_join_pieces"):
        pytest.skip("_join_pieces 不存在")

    tone = (0.25 * np.sin(2 * np.pi * 220 *
                          np.linspace(0, 0.8, int(SR * 0.8)))).astype(np.float32)
    pieces = [tone, tone, tone]
    chunks = [("一。", "end"), ("二。", "end"), ("三", "hard")]
    with_breath = mod._join_pieces(pieces, chunks, SR, 0.15, 0.4)
    without = mod._join_pieces(pieces, chunks, SR, 0.15, 0.0)
    assert len(with_breath) > len(without), "开启 breath 后应插入换气声"
    assert np.max(np.abs(with_breath)) < 1.0, "拼接结果削波"


def test_join_pieces_breath_is_not_mechanical():
    """连续多句的换气不能每次一模一样（AR(1) 的直接收益）。"""
    stab = _ROOT / "voice_clone" / "synthesis_stab.py"
    if not stab.exists():
        pytest.skip("synthesis_stab.py 缺失")
    spec = _ilu.spec_from_file_location("_stab_mech", stab)
    mod = _ilu.module_from_spec(spec)
    sys.modules["_stab_mech"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"synthesis_stab 无法独立加载: {exc}")
    if not hasattr(mod, "_join_pieces"):
        pytest.skip("_join_pieces 不存在")

    tone = np.zeros(int(SR * 0.6), dtype=np.float32)
    tone[int(SR * 0.1):int(SR * 0.5)] = 0.3
    pieces = [tone] * 6
    chunks = [("句。", "end")] * 5 + [("末", "hard")]
    out = mod._join_pieces(pieces, chunks, SR, 0.15, 0.5)
    # 用包络自相关衡量"重复度"：若每处换气完全一致，自相关会出现尖峰
    env = np.abs(out)
    k = max(1, int(SR * 0.005))
    env = np.convolve(env, np.ones(k) / k, mode="same")[::k]
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac /= (ac[0] + 1e-12)
    # 排除 lag=0 附近的包络自身宽度，找 0.4~1.5s 区间的次峰
    lo, hi = int(0.4 / 0.005), int(1.5 / 0.005)
    assert hi < len(ac)
    assert np.max(ac[lo:hi]) < 0.5, \
        "换气序列呈现强周期性 -> 机械重复（AR(1) 可能未生效）"
