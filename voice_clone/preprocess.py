"""
模块 1：音频预处理 (Audio Preprocessing)
========================================
对外提供三个原子能力：
  - denoise(y, sr)              : 离线谱门控(Wiener 式)降噪，去除稳态/非稳态噪声
  - remove_background(y, sr)    : 基于 HPSS 的人声/背景音(音乐)分离（若安装 demucs 自动升级为模型级分离）
  - standardize(y, sr, target)  : 采样率统一 + 首尾静音裁剪 + 响度归一化

全部基于 librosa / numpy / scipy，无需联网下载权重，可在离线环境运行。
"""
from __future__ import annotations
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 24000  # VoxCPM2 AudioVAE 编码采样率附近，统一到此避免二次重采样歧义


# ----------------------------------------------------------------------------- 加载
def load_audio(path: str, sr: int | None = None):
    """加载音频为单声道 float32 数组。sr=None 时保留原始采样率。"""
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), int(orig_sr if sr is None else sr)


# ----------------------------------------------------------------------------- 降噪
def denoise(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256,
            noise_percentile: float = 8.0, noise_mult: float = 1.6) -> np.ndarray:
    """
    谱门控降噪（离线、无外部模型）。
    思路：在 STFT 频域估计每频段噪声基底（取时间维度低分位能量），
    用 Wiener 增益 G = |S|^2 / (|S|^2 + (k*N)^2) 衰减噪声，再逆变换。
    对稳态噪声(空调/电流声)与轻度非稳态噪声均有效，且不依赖任何下载权重。
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)
    # 每频段噪声基底：跨时间取低分位（假设噪声普遍存在、语音能量更高）
    noise = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    gain = (mag ** 2) / (mag ** 2 + (noise_mult * noise) ** 2 + 1e-10)
    gain = np.clip(gain, 0.0, 1.0)
    # 时间维轻微平滑，抑制“水声”起伏（沿时间轴做 1D 均值滤波）
    from scipy.ndimage import uniform_filter1d
    gain = uniform_filter1d(gain, size=3, axis=1, mode="nearest")
    S_clean = mag * gain * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean, hop_length=hop_length, length=len(y))
    return np.asarray(y_clean, dtype=np.float32)


# ----------------------------------------------------------------------------- 背景音/音乐去除
def remove_background(y: np.ndarray, sr: int, strength: float = 0.75) -> np.ndarray:
    """
    人声/背景音(音乐)分离（离线启发式）。
    - 优先：若已安装 demucs，使用 htdemucs 做“vocals / no_vocals”分离，质量最佳。
    - 回退：基于 HPSS（谐波-冲击分离）。背景音乐以谐波成分为主，语音瞬态在冲击成分中保留更多，
            故压制谐波分量(强度由 strength 控制)以突出人声。
    strength∈[0,1]：越大越激进去除谐波（背景音乐），过小则残留音乐、过大会损伤语音音色。
    """
    # 升级路径：demucs 模型级分离
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from demucs.audio import AudioLoader, AudioWriter
        import torch
        model = get_model(name="htdemucs")
        wav = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)  # (1,1,T)
        ref = model.sources.index("vocals")
        with torch.no_grad():
            estimates = apply_model(model, wav.cuda() if torch.cuda.is_available() else wav,
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                    progress=False)[0]
        return estimates[ref].cpu().numpy().astype(np.float32).squeeze()
    except Exception:
        pass  # 回退到 HPSS
    y_h, y_p = librosa.effects.hpss(y, margin=3.0)
    # 保留全部冲击成分，按强度抑制谐波（背景音乐）
    return np.asarray(y_p + (1.0 - strength) * y_h, dtype=np.float32)


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
                    target_sr: int = TARGET_SR) -> tuple[np.ndarray, int, dict]:
    """一站式预处理：加载 -> (可选)去背景 -> (可选)降噪 -> 标准化。
    返回 (y, sr, report)。"""
    report = {"denoise": denoise_on, "remove_bg": remove_bg_on}
    y, sr = load_audio(path, sr=None)
    report["input_sr"] = sr
    report["input_duration"] = round(len(y) / sr, 2)
    if remove_bg_on:
        y = remove_background(y, sr)
    if denoise_on:
        y = denoise(y, sr)
    y, sr = standardize(y, sr, target_sr=target_sr)
    report["output_sr"] = sr
    report["output_duration"] = round(len(y) / sr, 2)
    return y, sr, report
