"""
模块 1：音频预处理 (Audio Preprocessing)
========================================
对外提供四个原子能力：
  - denoise(y, sr)              : v2 谱域降噪引擎（动态噪声跟踪 + 过减除谱减 +
                                  Decision-Directed Wiener 滤波），去除稳态/非稳态噪声
  - remove_background(y, sr)    : 人声/背景音(音乐)分离（demucs 模型级 -> REPET-lite 软掩码
                                  -> HPSS 三级回退）
  - isolate_vocals(y, sr)       : 「只保留纯净人声」入口（更强 BGM 抑制）
  - standardize(y, sr, target)  : 采样率统一 + 首尾静音裁剪 + 响度归一化

全部基于 librosa / numpy / scipy，无需联网下载权重，可在离线环境运行。
denoise / remove_background 的公开签名与 v1 完全兼容，v1 旧关键字参数仍被接受。
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d, minimum_filter1d
from scipy.signal import lfilter

TARGET_SR = 24000  # VoxCPM2 AudioVAE 编码采样率附近，统一到此避免二次重采样歧义


# ----------------------------------------------------------------------------- 加载
def load_audio(path: str, sr: int | None = None):
    """加载音频为单声道 float32 数组。sr=None 时保留原始采样率。"""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), int(orig_sr if sr is None else sr)


# =============================================================================
# v2 降噪引擎
# -----------------------------------------------------------------------------
# 设计（对标经典 SOTA 单通道降噪链，纯 numpy/scipy 向量化，离线可用）：
#   1. 动态噪声跟踪 —— 对功率谱做滑动窗口最小值统计(min-statistics)，逐帧估计
#      噪声基底 N(t,f)，可跟踪缓变/突发噪声，比 v1 的全局分位数精确；
#      配固定偏差补偿(bias≈1.4)抵消最小值统计的系统性低估。
#   2. 过减除谱减(Over-Subtraction) —— γ = P / (η·N)，η=1+0.9·strength，
#      在低 SNR 频段更激进地压低噪声，抑制残留“音乐噪声”。
#   3. Decision-Directed(DD) 先验 SNR —— 经典 Ephraim-Malah 式递归：
#      ξ(t) = α·G(t-1)²·γ(t-1) + (1-α)·max(γ(t)-1, 0)，再用 Wiener 增益
#      G = ξ/(1+ξ)。比直接谱减更少语音失真、几乎无“水声”。
#   4. 谱底限制 + 时间/频率增益平滑 —— 保留自然氛围音，消除掩码抖动。
# -----------------------------------------------------------------------------
def _ema_axis(x: np.ndarray, alpha: float, axis: int = 1) -> np.ndarray:
    """沿时间轴指数平滑: y[t] = alpha*y[t-1] + (1-alpha)*x[t]（向量化，axis 需>0）。"""
    b, a = [1.0 - alpha], [1.0, -alpha]
    return lfilter(b, a, x, axis=axis)


def _estimate_noise_floor(power: np.ndarray, sr: int, hop_length: int,
                          window_s: float = 1.5, bias: float = 1.25) -> np.ndarray:
    """动态噪声跟踪（min-statistics，时间相关噪声友好）。

    关键：先做时间指数平滑(α=0.85)再取滑动窗口最小值。若不先平滑，白噪等
    iid 噪声在窗口内的最小值远低于其均值，导致噪声基底系统性低估、静音段
    得不到衰减；平滑后可把最小统计的采样偏差压到可预测范围，再由 bias 补偿。
    """
    ps = _ema_axis(np.asarray(power, dtype=np.float64), alpha=0.85)
    win = max(5, int(round(window_s * sr / hop_length)))
    win = min(win, ps.shape[1])
    nmin = minimum_filter1d(ps, size=win, axis=1, mode="nearest")
    nmin = uniform_filter1d(nmin, size=max(3, win // 4), axis=1, mode="nearest")
    nmin = uniform_filter1d(nmin, size=5, axis=0, mode="nearest")  # 频域平滑
    return np.maximum(nmin * bias, 1e-12)


def _dd_wiener_gain(gamma: np.ndarray, alpha: float, floor_gain: float) -> np.ndarray:
    """Decision-Directed 先验 SNR -> Wiener 增益（沿时间轴递归，逐帧向量化于频域）。"""
    F = gamma.shape[1]
    G = np.empty_like(gamma, dtype=np.float64)
    xi_hat = np.maximum(gamma[:, 0] - 1.0, 0.0)          # ξ_hat 初值
    for t in range(F):
        gamma_t = gamma[:, t]
        xi = alpha * xi_hat + (1.0 - alpha) * np.maximum(gamma_t - 1.0, 0.0)
        G_t = xi / (xi + 1.0)
        G[:, t] = G_t
        xi_hat = (G_t * G_t) * gamma_t                   # 用当前帧更新先验
    return np.clip(G, floor_gain, 1.0)


def denoise(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256,
            strength: float = 1.0, dd_alpha: float = 0.92, window_s: float = 1.5,
            noise_bias: float = 1.25, noise_percentile: float | None = None,
            noise_mult: float | None = None) -> np.ndarray:
    """
    v2 谱域降噪（离线、无外部模型）。
    输入 y 为 float 单声道波形。strength∈[0,2]：越大去噪越强（默认 1.0）。
    兼容旧调用：noise_percentile 被忽略（新法为动态跟踪），noise_mult 映射到 strength。
    """
    if noise_mult is not None:  # v1 兼容：noise_mult=1.6 ↔ strength=1.0
        strength = float(np.clip(noise_mult / 1.6, 0.3, 2.0))
    if strength <= 0.02:
        return np.asarray(y, dtype=np.float32)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    power = mag ** 2
    N = _estimate_noise_floor(power, sr, hop_length,
                              window_s=window_s, bias=noise_bias)
    over = 1.0 + 0.9 * strength                     # 过减除因子
    gamma = power / (over * N)                      # 后验 SNR
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0)
    floor_gain = 10 ** (-(18.0 + 7.0 * strength) / 20.0)
    gain = _dd_wiener_gain(gamma, dd_alpha, floor_gain)
    # 掩码平滑：时间 5 帧 + 频率 3 点，抑制“水声/音乐噪声”
    gain = uniform_filter1d(gain, size=5, axis=1, mode="nearest")
    gain = uniform_filter1d(gain, size=3, axis=0, mode="nearest")
    S_clean = mag * gain * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean, hop_length=hop_length, length=len(y))
    return np.asarray(y_clean, dtype=np.float32)


# v1 保底参考实现（仅用于 A/B 对比，默认不再使用）
def denoise_v1(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256,
               noise_percentile: float = 8.0, noise_mult: float = 1.6) -> np.ndarray:
    """v1：全局低分位噪声基底 + Wiener 增益（保留供对比/回退）。"""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    noise = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    gain = (mag ** 2) / (mag ** 2 + (noise_mult * noise) ** 2 + 1e-10)
    gain = np.clip(gain, 0.0, 1.0)
    gain = uniform_filter1d(gain, size=3, axis=1, mode="nearest")
    S_clean = mag * gain * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean, hop_length=hop_length, length=len(y))
    return np.asarray(y_clean, dtype=np.float32)


# =============================================================================
# 背景音/音乐去除 v2
# -----------------------------------------------------------------------------
# 三级引擎：
#   ① demucs（若已安装）—— htdemucs 模型级分离，质量最佳；
#   ② REPET-lite —— 检测背景音乐节拍周期，把频谱按周期切成块，取中位数得到
#      “重复背景模型”，再用 过减除 + 软掩码 把非重复的人声保留下来
#      （对标经典 REPET 人声分离，纯 numpy，无需权重）；
#   ③ HPSS —— 谐波/冲击分离兜底。
# -----------------------------------------------------------------------------
def _estimate_music_period(power: np.ndarray, sr: int, hop_length: int,
                           lo_s: float = 0.30, hi_s: float = 3.0,
                           min_norm: float = 0.06) -> int | None:
    """用谱通量 novelty 的自相关估计背景音乐节拍/重复周期（帧数）。

    半波整流对数谱差分得到"响度变化事件"序列(对节拍与和弦切换都敏感)，
    其自相关峰给出节奏周期。比频谱帧相关性更稳：后者对稳态音乐处处饱和、
    无法区分真实重复滞后。仅在存在明显周期性(min_norm)时返回，否则 None。
    """
    F = power.shape[1]
    logP = np.log(power + 1e-12)
    flux = np.diff(logP, axis=1)
    flux = np.clip(flux, 0.0, None).mean(axis=0)
    flux = flux - flux.mean()
    std = flux.std()
    if std < 1e-9 or flux.size < 32:
        return None
    flux = flux / std
    n = flux.size
    n2 = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(flux, n2)
    ac = np.fft.irfft(spec * np.conj(spec), n2)[:n]
    ac /= (ac[0] + 1e-12)
    lo_f = max(2, int(round(lo_s * sr / hop_length)))
    hi_f = min(n - 1, int(round(hi_s * sr / hop_length)))
    if hi_f <= lo_f:
        return None
    seg = ac[lo_f:hi_f + 1]
    seg = seg / ((n - np.arange(lo_f, hi_f + 1)) / n)   # 无偏归一化（补偿重叠）
    k = int(np.argmax(seg))
    if seg[k] < min_norm:
        return None
    return lo_f + k


def _repet_vocals(y: np.ndarray, sr: int, strength: float,
                  n_fft: int = 2048, hop_length: int = 512) -> np.ndarray | None:
    """REPET-lite 软掩码人声分离。无法可靠估计音乐周期/样本不足时返回 None（由上层回退）。

    关键保护：逐帧计算"混合帧 vs 重复模型帧"的余弦相似度 gate ∈[0,1]——
    只有与重复背景模型相似的帧(音乐主导)才执行过减除；语音主导的帧 gate≈0，
    几乎不减除，避免把唱歌/说话声当背景削掉。这使算法对"节拍周期 ≠ 和声
    重复周期"的真实音乐也能优雅降级而非伤语音。
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    power = mag ** 2
    F = mag.shape[1]
    p = _estimate_music_period(power, sr, hop_length)
    if p is None or p < 8:                          # 周期不可靠
        return None
    nseg = F // p
    if nseg < 6:                                    # 重复样本太少，中位数模型不稳
        return None
    blocks = mag[:, :nseg * p].reshape(mag.shape[0], nseg, p)
    model = np.median(blocks, axis=1)               # 一个周期内的重复背景模型
    repeat = np.tile(model, (1, int(np.ceil(F / p))))[:, :F]
    # 逐帧余弦门控：混合帧与模型帧越像 -> 越可能是背景 -> gate→1
    fn = np.sqrt(np.sum(mag * mag, axis=0)) + 1e-12
    rn = np.sqrt(np.sum(repeat * repeat, axis=0)) + 1e-12
    cos = np.sum(mag * repeat, axis=0) / (fn * rn)
    gate = np.clip((cos - 0.35) / 0.40, 0.0, 1.0)   # 0.35~0.75 线性斜坡
    alpha = 0.6 + 1.0 * strength                    # 基础过减除强度
    beta = max(0.02, 0.10 - 0.06 * strength)        # 谱底（保留自然残响）
    y_mag = np.maximum(mag - alpha * repeat * gate[None, :], beta * mag)
    mask = np.clip(y_mag / (mag + 1e-9), 0.0, 1.0)
    mask = uniform_filter1d(mask, size=5, axis=1, mode="nearest")
    mask = uniform_filter1d(mask, size=3, axis=0, mode="nearest")
    y_voc = librosa.istft(mag * mask * np.exp(1j * phase),
                          hop_length=hop_length, length=len(y))
    return np.asarray(y_voc, dtype=np.float32)


def _hpss_vocals(y: np.ndarray, strength: float) -> np.ndarray:
    """HPSS 兜底：保留冲击成分（语音瞬态），按强度抑制谐波（背景音乐）。"""
    y_h, y_p = librosa.effects.hpss(y, margin=3.0)
    return np.asarray(y_p + (1.0 - strength) * y_h, dtype=np.float32)


def remove_background(y: np.ndarray, sr: int, strength: float = 0.75,
                      method: str = "auto") -> np.ndarray:
    """
    人声/背景音(音乐)分离（离线）。
    - method='auto'：demucs(已装) -> REPET-lite -> HPSS 三级自动回退；
    - method='repet'：仅 REPET-lite（音乐背景专用）；
    - method='hpss' ：仅 HPSS。
    strength∈[0,1]：越大去除越彻底（音乐残留越少、极端下语音略有损伤）。
    """
    y = np.asarray(y, dtype=np.float32)
    if method in ("auto", "demucs"):
        try:                                   # ① demucs 模型级分离（升级路径）
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
            import torch
            model = get_model(name="htdemucs")
            wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
            ref = model.sources.index("vocals")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.no_grad():
                est = apply_model(model, wav.to(dev), device=dev, progress=False)[0]
            return est[ref].cpu().numpy().astype(np.float32).squeeze()
        except Exception:
            pass                             # 未装 demucs / 加载失败 -> 继续回退
    if method in ("auto", "repet"):
        out = _repet_vocals(y, sr, strength)
        if out is not None:                  # ② REPET-lite（音乐检测成功）
            return out
    return _hpss_vocals(y, strength)         # ③ HPSS 兜底


def isolate_vocals(y: np.ndarray, sr: int, strength: float = 1.0,
                   method: str = "auto") -> np.ndarray:
    """「只保留纯净人声」：BGM 更强抑制 + 残留稳态噪声二次清理。"""
    y = remove_background(y, sr, strength=strength, method=method)
    y = denoise(y, sr, strength=min(strength + 0.3, 1.6))   # 清理分离引入的残余
    return np.asarray(y, dtype=np.float32)


# ----------------------------------------------------------------------------- 标准化
def _rms_normalize(y: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    return y * (target_rms / rms)


def _peak_limit(y: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > ceiling:
        y = y / peak * ceiling
    return y


def standardize(y: np.ndarray, sr: int, target_sr: int = TARGET_SR,
               top_db: float = 30.0, target_rms: float = 0.12) -> tuple[np.ndarray, int]:
    """
    标准化：重采样到 target_sr -> 裁剪首尾静音 -> 响度归一化(基于 RMS 对齐+峰值限幅)。
    返回 (y_std, target_sr)。
    """
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y, _ = librosa.effects.trim(y, top_db=top_db)
    y = _rms_normalize(y, target_rms)
    y = _peak_limit(y, 0.99)
    return np.asarray(y, dtype=np.float32), sr


def preprocess_file(path: str, denoise_on: bool = True, remove_bg_on: bool = False,
                    vocal_only: bool = False, target_sr: int = TARGET_SR,
                    denoise_strength: float = 1.0) -> tuple[np.ndarray, int, dict]:
    """一站式预处理：加载 -> (可选)只留人声/去背景 -> (可选)降噪 -> 标准化。
    返回 (y, sr, report)。"""
    report = {"denoise": denoise_on, "remove_bg": remove_bg_on,
              "vocal_only": vocal_only}
    y, sr = load_audio(path, sr=None)
    report["input_sr"] = sr
    report["input_duration"] = round(len(y) / sr, 2)
    if vocal_only:
        y = isolate_vocals(y, sr)
        report["engine"] = "isolate_vocals"
    elif remove_bg_on:
        y = remove_background(y, sr)
        report["engine"] = "remove_background"
    if denoise_on and not (vocal_only):     # isolate_vocals 内部已含二次降噪
        y = denoise(y, sr, strength=denoise_strength)
        report["denoise_strength"] = denoise_strength
    y, sr = standardize(y, sr, target_sr=target_sr)
    report["output_sr"] = sr
    report["output_duration"] = round(len(y) / sr, 2)
    return y, sr, report
