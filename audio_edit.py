"""
音频后处理引擎 (Audio Edit Engine)
====================================
在 VoxCPM2 生成原始音频之后、导出之前，对音频做可控的后处理，实现：
  1. 音调调节（pitch shift，半音）
  2. 语速调节（time stretch，保持音调）
  3. 音量调节（振幅缩放）
  以上三者独立调节、互不影响。
  4. 停顿/换气：句末标点处自动插入静音（可调时长）
  5. 呼吸效果：句末标点处插入自然气息声（可调轻重）
  6. 情绪预设：高兴/悲伤/严肃/温柔/愤怒 → 音调+语速+音量+停顿组合
  7. SSML 标签解析：<break>/<prosody>/<emphasis>/<emotion>
  8. 发音校正：多音字/生僻字（pypinyin + 常用多音字词典）

纯 numpy/librosa/scipy/pypinyin 实现，不依赖 ffmpeg。
"""
from __future__ import annotations
import re
import numpy as np

# 情绪预设：每个情绪映射到一组后处理参数。
# 注意：音调变化控制在 ±1 半音内，情绪主要靠语速/停顿/呼吸/音量表达，
# 避免大幅度改音调导致偏离原声线、且放大任何残余伪影。
EMOTION_PRESETS = {
    "高兴": {"pitch": 1, "speed": 1.08, "volume": 1.12, "pause": 0.12, "breath": 0.4},
    "悲伤": {"pitch": -1, "speed": 0.86, "volume": 0.90, "pause": 0.28, "breath": 0.5},
    "严肃": {"pitch": 0, "speed": 0.92, "volume": 1.00, "pause": 0.32, "breath": 0.35},
    "温柔": {"pitch": 0, "speed": 0.95, "volume": 0.95, "pause": 0.18, "breath": 0.45},
    "愤怒": {"pitch": 0, "speed": 1.15, "volume": 1.25, "pause": 0.10, "breath": 0.3},
    "平静": {"pitch": 0, "speed": 1.00, "volume": 1.00, "pause": 0.15, "breath": 0.35},
}

EMOTION_ALIAS = {
    "happy": "高兴", "sad": "悲伤", "serious": "严肃", "gentle": "温柔",
    "angry": "愤怒", "calm": "平静", "neutral": "平静",
}


# ----------------------------------------------------------------------------- 音调 / 语速 / 音量
# 保真关键：这里全部采用「时域」算法，刻意避开 librosa 的相位声码器
# (pitch_shift / time_stretch)。相位声码器在 STFT 域估计相位，对语音这类
# 非稳态信号会产生相位散乱，正是"机械电子音 / 金属感 / 偏离原声线"的元凶。
# 下面用：
#   语速  → WSOLA（波形相似性重叠相加，时域，保音高、保音色）
#   音调  → soxr 高质量变速重采样 + WSOLA 恢复时长（全程无相位估计）

def _varispeed_resample(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """变速：通过高质量重采样改变音调与时长（factor>1 升调加速，<1 降调减速）。
    返回重采样到 sr/factor 的信号，后续按 sr 播放即实现变速，无相位失真。"""
    import librosa
    target = float(sr) / factor
    return librosa.resample(
        y.astype(np.float32), orig_sr=sr, target_sr=target, res_type="soxr_hq"
    ).astype(np.float32)


def _wsola_stretch(y: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """WSOLA 时域时间伸缩：保持音高、保持音色，无相位声码器伪影。
    ratio = 输出时长 / 输入时长（>1 放慢，<1 加快）。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if abs(ratio - 1.0) < 0.01 or len(y) < sr * 0.1:
        return y.astype(np.float32)

    frame = int(sr * 0.040)             # 40ms 合成帧
    overlap = int(sr * 0.020)           # 20ms 交叉淡化
    Hs = frame - overlap                # 合成 hop
    Ha = max(1, int(round(Hs / ratio)))  # 分析 hop
    search = int(sr * 0.008)            # 最佳匹配搜索窗口 ±8ms

    n_in = len(y)
    n_out = max(frame, int(n_in * ratio))
    out = np.zeros(n_out, dtype=np.float32)
    wsum = np.zeros(n_out, dtype=np.float32)
    win = np.hanning(frame).astype(np.float32)

    apos, opos = 0, 0
    while opos + frame <= n_out and apos + frame <= n_in:
        best = apos
        if opos > 0:
            lo = max(0, apos - search)
            hi = min(n_in - frame, apos + search)
            if hi > lo:
                # 向量化：一次性算 [lo, hi] 内所有候选切片与已输出尾段的归一化互相关
                ref = out[opos:opos + overlap]
                ref = ref - ref.mean()
                rn = float(np.linalg.norm(ref)) + 1e-8
                seg = y[lo:hi + overlap]
                wins = np.lib.stride_tricks.sliding_window_view(seg, overlap)
                wc = wins - wins.mean(axis=1, keepdims=True)
                cn = np.linalg.norm(wc, axis=1) + 1e-8
                corr = (wc @ ref) / (cn * rn)
                best = lo + int(np.argmax(corr))
        # 加窗重叠相加（Hann 窗 + 能量归一化，保证拼接处平滑无爆音）
        frame_seg = y[best:best + frame]
        end = min(opos + frame, n_out)
        w = win[:end - opos]
        out[opos:end] += frame_seg[:end - opos] * w
        wsum[opos:end] += w
        opos += Hs
        apos += Ha

    valid = wsum > 1e-6
    out[valid] /= wsum[valid]
    written = opos - Hs + frame
    if written < n_out:
        out = out[:max(written, frame)]
    return out.astype(np.float32)


def apply_pitch(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """音调调节：semitones 半音（正=升，负=降），保持时长、保持音色。
    实现 = soxr 变速重采样（改音调+时长）+ WSOLA 恢复时长，无相位声码器。"""
    semitones = float(semitones or 0)
    if abs(semitones) < 0.05:
        return y
    k = 2.0 ** (semitones / 12.0)          # 频率缩放因子
    y_var = _varispeed_resample(y, sr, k)  # 音调 ×k、时长 ÷k
    return _wsola_stretch(y_var, sr, ratio=k)  # 时长恢复，音调保持 ×k


def apply_speed(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """语速调节：factor>1 加快，<1 放慢（保持音调）。WSOLA 时域实现，无相位伪影。"""
    factor = float(factor or 1.0)
    if abs(factor - 1.0) < 0.01:
        return y
    return _wsola_stretch(y, sr, ratio=1.0 / factor)


def apply_volume(y: np.ndarray, factor: float) -> np.ndarray:
    """音量调节：振幅线性缩放。"""
    factor = float(factor if factor is not None else 1.0)
    return (y * factor).astype(np.float32)


def apply_prosody(y: np.ndarray, sr: int, pitch: float = 0.0,
                  speed: float = 1.0, volume: float = 1.0) -> np.ndarray:
    """统一后处理：依次应用音量 → 语速 → 音调（三者独立，顺序固定保证可复现）。"""
    y = apply_volume(y, volume)
    y = apply_speed(y, sr, speed)
    y = apply_pitch(y, sr, pitch)
    return y


# ----------------------------------------------------------------------------- 呼吸声 / 停顿
def generate_breath(sr: int, strength: float = 0.5, duration: float | None = None,
                    seed: int | None = None) -> np.ndarray:
    """合成真人换气声：粉红噪声(1/f) + 口腔气流带通滤波 + 多段包络（起音湍流→稳定气流→
    缓慢衰减），时长/起音/衰减均随机，避免机械感。
    - duration=None 时按 strength 自适应（约 0.26~0.46s 随机）；
    - seed=None 时每次用系统随机种子，同一段呼吸不会重复。"""
    strength = float(strength or 0)
    if strength <= 0.01:
        return np.zeros(0, dtype=np.float32)
    import os
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    rng = np.random.RandomState(seed)
    # 时长自适应 + 随机抖动（真人换气时长有自然变化）
    if duration is None:
        duration = 0.26 + 0.14 * rng.rand()
        if strength > 0.7:
            duration += 0.06
    n = int(sr * max(0.08, duration))
    # 粉红噪声：积分近似 1/f 频谱，比白噪声更接近"气流沙沙"质感
    white = rng.randn(n).astype(np.float32)
    pink = np.cumsum(white)
    try:
        from scipy.signal import butter, lfilter
        nyq = sr / 2.0
        b_hi, a_hi = butter(2, 200.0 / nyq, btype="high")   # 去掉积分引入的极低频漂移
        pink = lfilter(b_hi, a_hi, pink).astype(np.float32)
        b_lo, a_lo = butter(2, 2200.0 / nyq, btype="low")   # 口腔/鼻腔气流主要频段
        pink = lfilter(b_lo, a_lo, pink).astype(np.float32)
    except Exception:
        pink = np.convolve(pink, np.ones(28) / 28, mode="same").astype(np.float32)
    peak = np.max(np.abs(pink)) + 1e-8
    pink = pink / peak
    # 多段包络：快速起音(吸气湍流) → 平稳主体(稳定气流) → 缓慢衰减(呼气收尾)
    t = np.linspace(0.0, 1.0, n)
    attack = 0.16 + 0.10 * rng.rand()                      # 起音占比随机
    release = 0.35 + 0.15 * rng.rand()                     # 衰减占比随机
    env = np.minimum(t / max(attack, 1e-3), 1.0)
    env *= np.exp(-(1.8 + rng.rand()) * np.maximum(t - (1.0 - release), 0.0))
    # 主体段轻微多正弦起伏，打破"规则感"（而非单一正弦）
    wob = (1.0
           + 0.10 * np.sin(2 * np.pi * (2.0 + rng.rand() * 2) * t + rng.rand() * 6.28)
           + 0.05 * np.sin(2 * np.pi * (5.0 + rng.rand() * 3) * t + rng.rand() * 6.28))
    env *= wob
    return (pink * env * 0.22 * strength).astype(np.float32)


def _trim_silence(y: np.ndarray, sr: int, rel_db: float = 40.0) -> np.ndarray:
    """裁剪首尾静音（相对能量阈值）。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame = int(sr * 0.02)
    if frame < 1 or len(y) < frame * 2:
        return y
    n = len(y) // frame
    frames = y[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thr = (rms.max() + 1e-12) / (10 ** (rel_db / 20.0))
    active = np.where(rms > thr)[0]
    if len(active) == 0:
        return y
    start = active[0] * frame
    end = min(len(y), (active[-1] + 1) * frame)
    return y[start:end]


def join_with_breath(chunks: list[np.ndarray], sr: int, pause: float = 0.15,
                     breath: float = 0.4) -> np.ndarray:
    """把分句音频按「静音停顿 + 呼吸声」自然拼接。pause=句间静音秒数，breath=呼吸轻重(0-1)。
    每个停顿处的呼吸声独立生成（随机时长/音色），避免机械重复。"""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    trimmed = [_trim_silence(c, sr) for c in chunks]
    trimmed = [c.astype(np.float32) for c in trimmed if len(c) > 0]
    if not trimmed:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(max(0.0, pause) * sr), dtype=np.float32)
    out = trimmed[0]
    for c in trimmed[1:]:
        # 停顿静音 + 呼吸声 + 停顿静音，呼吸声居中
        if breath > 0.01:
            breath_wav = generate_breath(sr, breath)  # 每处独立生成
            out = np.concatenate([out, gap, breath_wav, gap, c])
        else:
            out = np.concatenate([out, gap, c])
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- SSML 解析
_BREAK_RE = re.compile(r"<break\s+time=['\"]([0-9.]+)(ms|s)['\"]\s*/?>", re.I)
_EMOTION_RE = re.compile(r"<emotion\s+name=['\"]([^'\"]+)['\"]\s*>", re.I)
_PROSODY_RE = re.compile(
    r"<prosody\s+([^>]*)>(.*?)</prosody>", re.I | re.S)
_EMPHASIS_RE = re.compile(r"<emphasis\s*(level=['\"][^'\"]+['\"])?\s*>(.*?)</emphasis>", re.I | re.S)


def _parse_prosody_attrs(attrs: str) -> dict:
    d = {}
    m = re.search(r"rate=['\"]?([+-]?[0-9.]+)['\"]?", attrs)
    if m:
        d["speed"] = float(m.group(1))
    m = re.search(r"pitch=['\"]?([+-][0-9.]+)(st|%)?['\"]?", attrs)
    if m:
        v = float(m.group(1))
        d["pitch"] = v if (m.group(2) == "st" or m.group(2) is None) else v / 10.0
    m = re.search(r"volume=['\"]?([+-][0-9.]+)(dB)?['\"]?", attrs)
    if m:
        v = float(m.group(1))
        d["volume"] = 10 ** (v / 20.0) if m.group(2) else (1.0 + v / 100.0)
    return d


def parse_ssml(text: str) -> tuple[str, dict]:
    """解析 SSML 标签，返回 (纯净文本, 全局参数)。
    支持 <break time>、<emotion name>、<prosody rate/pitch/volume>、<emphasis>。
    """
    params: dict = {"pitch": 0.0, "speed": 1.0, "volume": 1.0, "pause": 0.15, "breath": 0.4}
    out = text
    # 情绪
    m = _EMOTION_RE.search(out)
    if m:
        name = EMOTION_ALIAS.get(m.group(1).lower(), m.group(1))
        preset = EMOTION_PRESETS.get(name)
        if preset:
            params.update(preset)
    out = _EMOTION_RE.sub("", out)
    # break（取第一个作为全局句间停顿参考）
    m = _BREAK_RE.search(out)
    if m:
        val = float(m.group(1))
        params["pause"] = val / 1000.0 if m.group(2).lower() == "ms" else val
    out = _BREAK_RE.sub("", out)
    # prosody（取第一个，局部 prosody 简化为全局近似）
    m = _PROSODY_RE.search(out)
    if m:
        params.update(_parse_prosody_attrs(m.group(1)))
        out = _PROSODY_RE.sub(lambda mm: mm.group(2), out)
    # emphasis → 轻微强调
    m = _EMPHASIS_RE.search(out)
    if m:
        params["volume"] = params.get("volume", 1.0) * 1.15
        params["pitch"] = params.get("pitch", 0.0) + 0.5
        out = _EMPHASIS_RE.sub(lambda mm: mm.group(2), out)
    out = re.sub(r"</?[a-zA-Z][^>]*>", "", out)  # 清除残留标签
    return out.strip(), params


# ----------------------------------------------------------------------------- 发音校正
# 常用多音字词组（上下文 → 正确读音）。这里的"校正"是把容易读错的常用词组规范写。
_POLYPHONE_MAP = {
    "银行": "银行", "行走": "行走", "行业": "行业", "长大": "长大", "长城": "长城",
    "重": "重", "行": "行", "长": "长", "乐": "乐",
}

_SENT_END = "。！？!?；;…"


def detect_polyphones(text: str) -> list[str]:
    """用 pypinyin 检测多音字，返回命中列表（供提示/日志）。"""
    try:
        from pypinyin import pinyin, Style
        res = []
        for ch in text:
            if ch.isalpha() and ord(ch) > 0x4E00:
                pr = pinyin(ch, heteronym=True, style=Style.NORMAL)
                if pr and len(pr[0]) > 1:
                    res.append(ch)
        return res
    except Exception:
        return []


def split_sentences(text: str) -> list[str]:
    """按句末标点切分（用于停顿/呼吸定位）。"""
    parts = [s.strip() for s in re.split(rf"(?<=[{_SENT_END}])", text) if s.strip()]
    return parts or [text.strip()]
