"""
测试夹具 (test fixtures for VoxCPM2)
====================================
两层夹具，各自解决不同问题：

**A. 纯逻辑夹具（无需模型/音频/网络）**
   用于 transcriber 的对齐算法、mdx_separator 的数学约束。
   只需 numpy，CI 可以全量跑。

**B. 音频夹具（需要模型权重 / 真实素材）**
   用于分离质量回归。**关键教训**：合成人声（纯谐波堆）对 MDX-NET 属于
   分布外输入 —— 模型把它判成乐器而不是人声，会整体分离掉。
   因此质量测试**必须用真实语音**作为人声源：
     · 优先用 `VOXCPM2_FIXTURE_WAV` 环境变量指定的音频；
     · 否则在 `<repo>/outputs/` 里找第一个可用 wav；
     · 都没有则 **skip**（不是 fail）—— CI 无权重/无素材时不该红。
   伴奏仍用合成（可控、可复现），叠在真实人声上构成"已知答案"的混音。

指标（对干净人声参考计算）：
  corr              —— 皮尔逊相关，时间域
  sdr_like          —— 10*log10(||ref||^2 / ||est-ref||^2)
  spectral_centroid —— 谱质心，"闷不闷"最灵敏
  hf_keep           —— >2kHz 能量保留率
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_SR = 44100


# ============================== A. 纯逻辑夹具 ==============================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_voice(duration: float = 4.0, sr: int = DEFAULT_SR,
               f0: float = 180.0, seed: int = 1234) -> np.ndarray:
    """合成谐波堆（基频 + 共振峰包络 + 音节起伏）。

    ⚠️ 分布外：这是"像乐器"的纯谐波信号，MDX-NET 会把它当伴奏分离掉。
    仅用于**逻辑/形状/数值**类测试，不要用来断言分离质量。
    质量测试请用 `make_real_voice_mix()`。
    """
    rng = _rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr

    vib = 1.0 + 0.012 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * f0 * np.cumsum(vib) / sr

    sig = np.zeros(n)
    for k in range(1, 26):
        amp = 1.0 / (k ** 1.25)
        fk = f0 * k
        for fc, bw, gain in ((700.0, 350.0, 2.2), (1200.0, 400.0, 1.6),
                             (2600.0, 700.0, 0.9)):
            amp *= 1.0 + gain * np.exp(-((fk - fc) ** 2) / (2 * bw ** 2))
        sig += amp * np.sin(k * phase)

    sig *= 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 4.0 * t + 0.7))
    sig += rng.normal(0, 0.004, n)
    peak = np.max(np.abs(sig))
    return (sig / peak * 0.85).astype(np.float32) if peak > 0 else sig.astype(np.float32)


def make_accompaniment(duration: float = 4.0, sr: int = DEFAULT_SR,
                       seed: int = 4321) -> np.ndarray:
    """合成伴奏：A 小调和弦进行 + 底鼓 + 宽带噪声。确定性、可复现。"""
    rng = _rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    acc = np.zeros(n)

    chords = [(220.0, 261.63, 329.63), (196.0, 246.94, 293.66),
              (220.0, 277.18, 329.63), (174.61, 220.0, 261.63)]
    seg = max(1, n // len(chords))
    for ci, chord in enumerate(chords):
        a, b = ci * seg, min(n, (ci + 1) * seg)
        if a >= b:
            break
        tt = t[a:b]
        for f in chord:
            for k in range(1, 9):
                acc[a:b] += (0.30 / k) * np.sin(2 * np.pi * f * k * tt + k * 0.3)

    for b0 in np.arange(0, duration, 0.5):
        i0 = int(b0 * sr)
        ln = min(int(0.18 * sr), n - i0)
        if ln <= 0:
            continue
        tt = np.arange(ln) / sr
        acc[i0:i0 + ln] += 0.55 * np.exp(-tt * 22.0) * np.sin(2 * np.pi * 62.0 * tt)

    acc += rng.normal(0, 0.006, n)
    peak = np.max(np.abs(acc))
    return (acc / peak * 0.5).astype(np.float32) if peak > 0 else acc.astype(np.float32)


def make_vocal_mix(duration: float = 4.0, sr: int = DEFAULT_SR,
                   vocal_gain: float = 1.0, acc_gain: float = 0.55):
    """合成人声 + 合成伴奏。返回 (mix, clean_vocal, accompaniment)。

    仅用于数值/形状测试（见 make_voice 的分布外警告）。
    """
    v = make_voice(duration, sr)
    a = make_accompaniment(duration, sr)
    mix = vocal_gain * v + acc_gain * a
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        s = 0.98 / peak
        mix, v, a = mix * s, (v * s).astype(np.float32), (a * s).astype(np.float32)
    return mix.astype(np.float32), v, a


# ============================== B. 真实音频夹具 ==============================

def find_real_voice_wav(min_seconds: float = 2.0, max_seconds: float = 10.0):
    """找一个真实语音 wav 用作分离质量测试的人声源。

    查找顺序：
      1. 环境变量 VOXCPM2_FIXTURE_WAV
      2. <repo>/outputs/*.wav（取第一个时长合适的）
    返回 Path 或 None（None -> 调用方 skip）。
    """
    import soundfile as sf

    cands = []
    env = os.environ.get("VOXCPM2_FIXTURE_WAV")
    if env:
        cands.append(Path(env))

    here = Path(__file__).resolve().parent
    for base in (here.parent, here.parent.parent):
        out = base / "outputs"
        if out.is_dir():
            # 目录里可能有很多文件，全部列出来但设个上限，
            # 避免在超大 outputs/ 上无限遍历（早期版本只看前 60 个，
            # 恰好都不满足时长就误判"无素材"）。
            cands.extend(sorted(out.glob("*.wav"))[:400])
            break

    for p in cands:
        try:
            if not p.is_file():
                continue
            info = sf.info(str(p))
            if min_seconds <= info.duration <= max_seconds:
                return p
        except Exception:
            continue
    return None


def make_real_voice_mix(seconds: float = 6.0, sr: int = DEFAULT_SR,
                        acc_gain: float = 0.5, seed: int = 4321):
    """真实语音 + 合成伴奏 → (mix, clean_vocal, accompaniment)。

    真实语音才是 MDX 的分布内输入，这样测出的质量指标才有意义。
    找不到素材时返回 (None, None, None) → 调用方 skip。
    """
    import librosa

    p = find_real_voice_wav(min_seconds=max(1.0, seconds * 0.5),
                            max_seconds=max(20.0, seconds * 3))
    if p is None:
        return None, None, None

    v, _ = librosa.load(str(p), sr=sr, mono=True)
    v = np.asarray(v, dtype=np.float32)
    want = int(seconds * sr)
    if len(v) < want:
        v = np.pad(v, (0, want - len(v)))
    v = v[:want]
    peak = np.max(np.abs(v))
    if peak > 0:
        v = (v / peak * 0.7).astype(np.float32)

    a = make_accompaniment(len(v) / sr, sr, seed=seed)
    mix = v + acc_gain * a
    pk = np.max(np.abs(mix))
    if pk > 0.99:
        mix = mix / pk * 0.95
    return mix.astype(np.float32), v, a


def to_stereo(y: np.ndarray) -> np.ndarray:
    """单声道 → (2, N)。MDX 输入需 4 通道 = 2 声道 × (实, 虚)。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return np.repeat(y[None, :], 2, axis=0)
    return y


# ============================== 质量指标 ==============================

def corr(est: np.ndarray, ref: np.ndarray) -> float:
    e = np.asarray(est, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(e), len(r))
    if n < 2:
        return 0.0
    e, r = e[:n], r[:n]
    e, r = e - e.mean(), r - r.mean()
    de, dr = np.sqrt((e ** 2).sum()), np.sqrt((r ** 2).sum())
    if de < 1e-12 or dr < 1e-12:
        return 0.0
    return float((e * r).sum() / (de * dr))


def sdr_like(est: np.ndarray, ref: np.ndarray) -> float:
    e = np.asarray(est, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(e), len(r))
    if n < 2:
        return float("-inf")
    e, r = e[:n], r[:n]
    num = float((r ** 2).sum())
    den = float(((e - r) ** 2).sum())
    if den < 1e-12:
        return 120.0
    if num < 1e-12:
        return float("-inf")
    return 10.0 * np.log10(num / den)


def _mag_spec(y: np.ndarray, sr: int, n_fft: int = 2048):
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    win = np.hanning(n_fft)
    hop = n_fft // 4
    frames = [np.abs(np.fft.rfft(y[i:i + n_fft] * win))
              for i in range(0, len(y) - n_fft + 1, hop)]
    if not frames:
        frames = [np.abs(np.fft.rfft(np.pad(y, (0, n_fft - len(y))) * win))]
    return np.array(frames), np.fft.rfftfreq(n_fft, 1.0 / sr)


def spectral_centroid(y: np.ndarray, sr: int) -> float:
    S, f = _mag_spec(y, sr)
    tot = S.sum(axis=1)
    tot[tot < 1e-12] = 1e-12
    return float(np.mean((S * f).sum(axis=1) / tot))


def hf_keep(est: np.ndarray, ref: np.ndarray, sr: int, cutoff: float = 2000.0) -> float:
    Se, f = _mag_spec(est, sr)
    Sr, _ = _mag_spec(ref, sr)
    m = f >= cutoff
    if not m.any():
        return 1.0
    a = float((Se[:, m] ** 2).sum())
    b = float((Sr[:, m] ** 2).sum())
    return a / b if b > 1e-12 else 1.0


def rms_ratio(est: np.ndarray, ref: np.ndarray) -> float:
    """幅度保真度：est_rms / ref_rms。接近 1.0 表示没被整体压小。"""
    e = np.asarray(est, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(e), len(r))
    if n < 1:
        return 0.0
    er = np.sqrt((e[:n] ** 2).mean())
    rr = np.sqrt((r[:n] ** 2).mean())
    return float(er / rr) if rr > 1e-12 else 0.0
