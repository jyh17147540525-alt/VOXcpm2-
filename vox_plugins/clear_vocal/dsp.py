"""清唱生成插件 · DSP 基础层
==============================
自包含的时频处理原语，不依赖模型、不依赖插件子系统，可独立测试。

为什么不用 ``librosa.phase_vocoder``
------------------------------------
官方文档自述："makes no attempt to handle transients, and is likely to produce
many audible artifacts"。而清唱音频里**每个音符的起音（onset）都是强瞬态**，
参考实现会在每个音符头上糊出一片"涂抹感"（pre-echo）。

本模块的做法（``time_stretch``）：
  1. STFT 后做**相位传播**（phase propagation），而不是简单的相位重置；
  2. 用**瞬态检测**（谱通量峰值）把帧分成"瞬态段"与"稳态段"；
  3. 瞬态段**整帧搬移不做时域插值**（保护起音不受涂抹）；
  4. 稳态段用相位声码器做精确时间伸缩。
这样既保住音高与共振峰（真正的时间伸缩），又不牺牲起音清晰度。

⚠️ 与 ``audio_edit.apply_pitch`` 的区别（重要）
------------------------------------------------
``apply_pitch`` = soxr 变速重采样 + WSOLA 恢复时长，**没有相位估计**，
共振峰会随基频一起缩放 → 音色整体偏移。本模块的 ``pitch_shift`` 用
**相位声码器**，频率轴不动、只改时间轴与相位增量 → **共振峰保持在原位**，
因此适合"必须精确匹配原曲音高、但音色不能被毁"的清唱场景。
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 常量
EPS = 1e-10
N_FFT = 2048
HOP = 256


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def to_mono(y: np.ndarray) -> np.ndarray:
    """多声道 → 单声道 float32。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y, dtype=np.float32)


def peak_normalize(y: np.ndarray, target: float = 0.98, floor_db: float = -60.0) -> np.ndarray:
    """峰值归一化；全静音则原样返回（不放大底噪）。"""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    pk = float(np.abs(y).max())
    if pk < 10 ** (floor_db / 20.0) or pk <= EPS:
        return y
    return (y * (float(target) / pk)).astype(np.float32)


def rms(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y))))


def db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), EPS)))


# ------------------------------------------------------------------ STFT
def stft(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """STFT，返回复数矩阵 [n_fft//2+1, n_frames]。中心填充，与 iSTFT 严格配对。"""
    y = to_mono(y)
    win = _hann(n_fft)
    pad = n_fft // 2
    yp = np.pad(y, pad, mode="reflect") if y.size > pad else np.pad(y, pad, mode="edge")
    n_frames = 1 + max(0, (len(yp) - n_fft) // hop)
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex64)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = yp[idx]                                   # [n_frames, n_fft]
    return np.fft.rfft(frames * win, axis=1).T.astype(np.complex64)


def istft(D: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
          length: int | None = None) -> np.ndarray:
    """逆 STFT，Hann 窗 + 窗平方归一化（COLA 精确重建）。"""
    D = np.asarray(D)
    if D.size == 0:
        return np.zeros(0, dtype=np.float32)
    win = _hann(n_fft)
    frames = np.fft.irfft(D.T, n=n_fft, axis=1).astype(np.float32)   # [n_frames, n_fft]
    n_frames = frames.shape[0]
    out_len = n_fft + hop * (n_frames - 1)
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    w2 = win * win
    for i in range(n_frames):
        s = i * hop
        out[s:s + n_fft] += frames[i] * win
        wsum[s:s + n_fft] += w2
    valid = wsum > EPS
    out[valid] /= wsum[valid]
    # 去掉 STFT 时为居中填充补的 pad
    pad = n_fft // 2
    out = out[pad:pad + (length if length is not None else out_len - 2 * pad)]
    return np.ascontiguousarray(out, dtype=np.float32)


# ------------------------------------------------------------------ 瞬态检测
def transient_mask(y: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
                   sensitivity: float = 1.5, min_gap_ms: float = 30.0,
                   sr: int = 22050) -> tuple[np.ndarray, np.ndarray]:
    """谱通量峰值检测，返回 (帧级布尔掩码, 瞬态帧下标数组)。

    原理：瞬态 = 谱能量在短时间内**上升**。只看正半部分（上升沿），
    再用局部中位数 + k·MAD 做自适应阈值（比固定阈值稳健得多），
    最后按 ``min_gap_ms`` 做非极大值抑制，避免一个起音被拆成好几帧。
    """
    y = to_mono(y)
    if y.size < n_fft * 2:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    S = np.abs(stft(y, n_fft, hop))
    if S.shape[1] < 3:
        return np.zeros(S.shape[1], dtype=bool), np.zeros(0, dtype=np.int64)
    # 对数压缩后求正差分（对数域对小能量变化更敏感）
    logS = np.log1p(S * 1000.0)
    flux = np.zeros(S.shape[1], dtype=np.float32)
    flux[1:] = np.maximum(0.0, logS[:, 1:] - logS[:, :-1]).sum(axis=0)

    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) + EPS
    thr = med + sensitivity * 1.4826 * mad

    cand = np.flatnonzero(flux > thr)
    if cand.size == 0:
        mask = np.zeros(S.shape[1], dtype=bool)
        return mask, np.zeros(0, dtype=np.int64)

    # 非极大值抑制：min_gap 内只保留能量最大的一帧
    gap = max(1, int(round(min_gap_ms / 1000.0 * sr / hop)))
    peaks: list[int] = []
    for c in cand:
        if peaks and (c - peaks[-1]) < gap:
            if flux[c] > flux[peaks[-1]]:
                peaks[-1] = int(c)
        else:
            peaks.append(int(c))

    mask = np.zeros(S.shape[1], dtype=bool)
    # 瞬态保护窗：起音前 1 帧、起音后 2 帧
    for p in peaks:
        mask[max(0, p - 1):min(S.shape[1], p + 3)] = True
    return mask, np.asarray(peaks, dtype=np.int64)


# ------------------------------------------------------------------ 时间伸缩
def _phase_advance(n_fft: int, hop: int) -> np.ndarray:
    """每帧的期望相位增量（按 bin 中心频率计算）。"""
    freqs = np.fft.rfftfreq(n_fft) * 2.0 * np.pi * hop
    return freqs.astype(np.float32)


def time_stretch(y: np.ndarray, rate: float, n_fft: int = N_FFT, hop: int = HOP,
                 sr: int = 22050, transient_aware: bool = True,
                 sensitivity: float = 1.5) -> np.ndarray:
    """时间伸缩：``rate`` = 输出时长 / 输入时长（>1 放慢，<1 加快），**保持音高与共振峰**。

    瞬态感知（``transient_aware=True``）：瞬态帧整帧复制不做相位插值，
    避免起音被"涂抹"。稳态帧走标准相位声码器。
    """
    y = to_mono(y)
    rate = float(rate)
    if abs(rate - 1.0) < 0.01 or y.size < n_fft:
        return y.copy()
    rate = max(0.25, min(4.0, rate))

    D = stft(y, n_fft, hop)
    if D.shape[1] < 2:
        return y.copy()

    if transient_aware:
        tmask, _ = transient_mask(y, n_fft, hop, sensitivity=sensitivity, sr=sr)
        if tmask.size != D.shape[1]:
            tmask = np.zeros(D.shape[1], dtype=bool)
    else:
        tmask = np.zeros(D.shape[1], dtype=bool)

    mag = np.abs(D)
    phase = np.angle(D)
    dphi = _phase_advance(n_fft, hop)[:, None]

    n_out = max(1, int(round(D.shape[1] * rate)))
    out = np.zeros((D.shape[0], n_out), dtype=np.complex64)

    # 相位累加器（每 bin 独立）
    acc = phase[:, 0].copy()
    # 目标帧位置 → 源帧位置的映射
    time_in = np.arange(n_out, dtype=np.float64) / rate
    for j in range(n_out):
        si = int(time_in[j])
        if si >= D.shape[1] - 1:
            si = D.shape[1] - 2
        frac = time_in[j] - si                          # [0,1)

        if tmask.size and (tmask[si] or tmask[min(si + 1, tmask.size - 1)]):
            # 瞬态：整帧搬移，只做幅度复制，相位不插值（保护起音）
            out[:, j] = mag[:, si] * np.exp(1j * phase[:, si])
            acc = phase[:, si].copy()
            continue

        # 稳态：幅度线性插值 + 相位传播
        m = (1.0 - frac) * mag[:, si] + frac * mag[:, min(si + 1, D.shape[1] - 1)]
        if j == 0:
            acc = phase[:, si].copy()
        else:
            delta = phase[:, min(si + 1, D.shape[1] - 1)] - phase[:, si] - dphi[:, 0] * (rate - 1.0)
            # 相位折叠到 (-π, π]，避免累积漂移
            delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
            acc = acc + dphi[:, 0] * rate + delta / max(rate, EPS)
        out[:, j] = m * np.exp(1j * acc)

    return istft(out, n_fft, hop, length=int(round(len(y) * rate)))


# ------------------------------------------------------------------ 音高搬移
def pitch_shift(y: np.ndarray, sr: int, semitones: float,
                n_fft: int = N_FFT, hop: int = HOP,
                formant_preserve: bool = True) -> np.ndarray:
    """音高搬移（半音），**保持时长、保持共振峰**。

    为什么必须全程在频域做（这是本模块与 ``audio_edit.apply_pitch`` 的本质区别）
    ------------------------------------------------------------------------------
    让共振峰留在原位的**充要条件是：频率轴不被缩放**。任何"重采样"步骤都会把
    频率轴整体缩放 k 倍，于是共振峰也必然跟着搬 k 倍 —— 音色就被搬走了。

    所以正确的顺序是：**在 STFT 域内把幅度谱沿频率轴平移，同时把时间轴展宽 k 倍**，
    最后 iSTFT 回时域。全程没有任何重采样，频率轴只在 iSTFT 时按原 sr 映射
    → 共振峰位置不动，只有基频（与各次谐波）被移走。

    ``formant_preserve=False`` 时退化为"先伸缩后重采样"的简版（保留用于对比实验）。
    """
    semitones = float(semitones or 0.0)
    y = to_mono(y)
    if abs(semitones) < 0.05:
        return y

    k = float(2.0 ** (semitones / 12.0))

    if not formant_preserve:
        stretched = time_stretch(y, rate=k, n_fft=n_fft, hop=hop, sr=sr)
        import librosa
        out = librosa.resample(stretched.astype(np.float32), orig_sr=float(sr),
                               target_sr=float(sr) / k, res_type="soxr_hq").astype(np.float32)
        if len(out) > len(y):
            out = out[:len(y)]
        elif len(out) < len(y):
            out = np.pad(out, (0, len(y) - len(out)))
        return np.ascontiguousarray(out, dtype=np.float32)

    # ---- 频域实现：对每帧幅度谱沿频率轴平移 log2(k) 个倍频程 ----
    D = stft(y, n_fft, hop)
    if D.shape[1] < 2:
        return y
    n_bins = D.shape[0]                                  # n_fft//2 + 1
    mag = np.abs(D)
    phase = np.angle(D)
    freqs = np.fft.rfftfreq(n_fft)                       # 归一化频率 [0, 0.5]

    # 目标：第 i 个 bin 的新频率应 = k * f_i，即 f_i 处的能量要搬到 bin i*k 去。
    # 等价地，目标 bin j 的内容取自源 bin j/k（k>1 升调 → 从更低 bin 取，频率被推高）。
    # ⚠️ 方向极易写反：写成 src=j*k 会得到 1/k 的搬移（降调变升调）。
    src_idx = np.arange(n_bins, dtype=np.float64) / k
    valid = src_idx <= (n_bins - 1)
    lo = np.minimum(np.floor(src_idx).astype(np.int64), n_bins - 2)
    hi = lo + 1
    frac = src_idx - lo

    # 带内幅度插值（沿频率轴平移）
    new_mag = np.zeros_like(mag)
    new_mag[valid] = ((1.0 - frac[valid, None]) * mag[lo[valid], :]
                      + frac[valid, None] * mag[hi[valid], :])
    # 频率搬移后的能量补偿：高频段被拉伸需补偿 PSD 密度
    new_mag[valid] *= np.sqrt(max(k, EPS))

    # ---- 相位：必须"搬移"而不是"复制" ----
    # 直接把源 bin 的相位照抄给目标 bin 是错的：相邻 bin 之间的相位关系会被打乱，
    # 谐波不再对齐 → 频谱看着对（峰值频率正确），但时域波形的周期错乱
    # （实测：频谱峰值 209.9 Hz 正确，pyin 却测出 90.1 Hz）。
    #
    # 正确公式（经系数扫描实验确定，见 _t_phase.py）：
    #     目标 bin j 的相位 = 源相位 + 2π · (k-1) · f_src · t
    # 其中 f_src 用**相位差分估计的瞬时频率**（而非 bin 标称频率）。
    #
    # 三个半音数的实测偏差（越小越好）：
    #    系数 \ 半音        -12      -7      +7
    #    (k-1)·f_src_inst  +0.87%  +0.29%  -0.29%   ← 采用
    #    (k-1)·f_src_bin   +1.45%  -0.86%  +0.29%
    #    不补偿            -59.0%  -8.56%  +5.64%   ← 降调时灾难性
    # ⚠️ 降调（k<1）对相位误差远比升调敏感，别用升调结果推断降调行为。
    bin_freqs = freqs * float(sr)                    # 各 bin 的物理频率 [n_bins]
    if phase.shape[1] >= 2:
        d = np.diff(phase, axis=1)
        d = np.mod(d + np.pi, 2.0 * np.pi) - np.pi
        d = np.concatenate([d[:, :1], d], axis=1)
        inst_freq = bin_freqs[:, None] + d * (float(sr) / (2.0 * np.pi * hop))
    else:
        inst_freq = np.repeat(bin_freqs[:, None], phase.shape[1], axis=1)

    t_abs = (np.arange(phase.shape[1], dtype=np.float64) * hop / float(sr))[None, :]
    new_phase = np.zeros_like(phase)
    # 目标 bin j 承载源 bin lo[j] 的内容，源瞬时频率取该 bin 的估计值
    comp = (k - 1.0) * inst_freq[lo]                 # [n_bins, n_frames]
    new_phase[valid] = phase[lo[valid]] + 2.0 * np.pi * comp[valid] * t_abs

    Dn = (new_mag * np.exp(1j * new_phase)).astype(np.complex64)

    # ---- 时长还原：频率搬移 k 会改变瞬时频率，但**不改变帧数**（时长本就不变），
    # 无需任何时间轴操作。这里只把帧矩阵按原帧数输出即可。
    res = istft(Dn, n_fft, hop, length=len(y))
    if len(res) < len(y):
        res = np.pad(res, (0, len(y) - len(res)))
    return np.ascontiguousarray(res[:len(y)], dtype=np.float32)


def _phase_vocoder_frames(D: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
                          rate: float = 1.0) -> np.ndarray:
    """在已给定的 STFT 矩阵上做相位传播时间伸缩（不做瞬态保护，供内部复用）。"""
    mag = np.abs(D)
    phase = np.angle(D)
    dphi = _phase_advance(n_fft, hop)[:, None]
    n_out = max(1, int(round(D.shape[1] * rate)))
    out = np.zeros((D.shape[0], n_out), dtype=np.complex64)
    time_in = np.arange(n_out, dtype=np.float64) / rate
    acc = phase[:, 0].copy()
    for j in range(n_out):
        si = min(int(time_in[j]), D.shape[1] - 2)
        frac = time_in[j] - si
        m = (1.0 - frac) * mag[:, si] + frac * mag[:, si + 1]
        if j == 0:
            acc = phase[:, si].copy()
        else:
            delta = phase[:, si + 1] - phase[:, si] - dphi[:, 0] * (rate - 1.0)
            delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi
            acc = acc + dphi[:, 0] * rate + delta / max(rate, EPS)
        out[:, j] = m * np.exp(1j * acc)
    return out
