"""
音频后处理引擎 (Audio Edit Engine)
====================================
在 VoxCPM2 生成原始音频之后、导出之前，对音频做可控的后处理，实现：
  1. 音调调节（pitch shift，半音）
  2. 语速调节（time stretch，保持音调）
  3. 音量调节（振幅缩放）
  以上三者独立调节、互不影响。
  4. 停顿/换气：句末标点处自动插入静音（可调时长）
  5. 呼吸效果：句末标点处插入自然换气声（可调轻重）
     —— 源-滤波器模型：精确 1/f^β 湍流源 + 声道共振峰塑形 + 生理包络，
        区分「吸气 / 呼气」两相，相邻换气按 AR(1) 生理相关
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
    """音调调节：semitones 半音（正=升，负=降），保持时长。

    ⚠️ 实现 = soxr 变速重采样（音调 ×k、时长 ÷k）+ WSOLA 恢复时长，**无相位声码器**。
    变速重采样会连共振峰（formant）一起缩放，WSOLA 只恢复时长、不恢复共振峰，
    因此本函数**不保持音色**：任何非零半音都会整体偏移音色（±1 半音 ≈ 共振峰移 ~6%）。
    对以「音色保真」为核心的声音克隆，默认应传 0；
    需要真正的保音色变调，须改用相位声码器 / PSOLA 实现。
    """
    semitones = float(semitones or 0)
    if abs(semitones) < 0.05:
        return y
    k = 2.0 ** (semitones / 12.0)          # 频率缩放因子
    y_var = _varispeed_resample(y, sr, k)  # 音调 ×k、时长 ÷k（共振峰同时 ×k）
    return _wsola_stretch(y_var, sr, ratio=k)  # 只恢复时长，共振峰停在 ×k


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
# 设计依据（换气声的声学事实，决定了下面每个参数为什么是这个值）：
#
#  1) 换气声 = 声门下气流通过声门/口腔狭窄处产生的**湍流摩擦噪声**，
#     不是乐音。因此它没有基频，只有**宽带噪声 + 声道共振峰塑形**。
#  2) 湍流噪声的源谱接近**粉红噪声 (1/f^β, β≈0.7~1.0)**——不是白噪声（太"嘶"、
#     太刺），更**不是布朗噪声 1/f²**（能量全砸低频，听感是闷响/风声，
#     而不是贴在嘴边的气流声）。这是旧实现最大的错：`np.cumsum` 积分白噪声
#     得到的是 1/f² 而非注释里写的 1/f。
#  3) 声道对噪声源的塑形 = **共振峰（formant）结构**，主要三个：
#     吸气 ~500/1500/2500 Hz；呼气偏低偏暗 ~350/900/1800 Hz。
#     旧实现用 200–2200 Hz 平顶带通，把共振结构抹平了 → "白噪音感 / 塑料感"。
#  4) 包络不是线性起音：吸气是**快起-缓落**的 Gamma 型（声门快速张开后缓慢回位），
#     呼气是**缓起-快落**。旧实现的 `np.minimum(t/attack, 1)` 线性斜坡
#     听感是"渐强"，完全不像"噗地吸一口气"。
#  5) 相邻换气的时长/强度存在**生理相关**（语速越快、吸气越短越浅），
#     纯 i.i.d. 随机反而暴露合成痕迹 → 用 AR(1) 马尔可夫过程建模。
#
# 算法链：精确谱塑形（Voss-McCartney 1/f^β）→ 三共振峰 IIR 滤波
#        → Gamma/双指数包络 + 湍流微调制 → 等响度归一化。

#: 湍流源谱指数 β（1/f^β），**已扣除共振峰滤波器的滚降贡献**。
#:
#: ⚠️ 这个值必须这么小，是实测反解出来的，不是拍脑袋：
#: 并联谐振器在每个共振峰之上各贡献约 -6 dB/oct 的额外滚降，4 个共振峰
#: 并联后，滤波链自身在 200–4000 Hz 就带来约 -0.46（吸气）/ -0.87（呼气）
#: 的**固定偏置斜率**。因此若想让**最终输出**的谱斜率落在听感正确的
#: -0.85 / -1.05（摩擦音噪声的实测范围），源谱只需 β≈0.39 / 0.18。
#:
#: 第一版曾直接照搬文献的 β≈0.85/1.05 当**源谱**，结果最终斜率跑到
#: -1.30/-1.92——能量过度集中低频，听感发闷，被客观验证（Welch 斜率）抓出。
#: 结论：**源谱指数 ≠ 输出谱指数**，中间隔着滤波器的传递函数。
_BREATH_BETA = {"inhale": 0.39, "exhale": 0.18}

#: 声道共振峰 (中心频率 Hz, 相对增益 dB, 带宽 Hz)。
#: 带宽决定共振的"尖锐度"：越窄越有金属腔体感，越宽越自然。
_BREATH_FORMANTS = {
    # 吸气：声门开度大、气流快、摩擦强 → 共振峰偏高偏亮
    "inhale": ((520.0, 0.0, 420.0),
               (1480.0, -5.0, 620.0),
               (2560.0, -9.5, 900.0),
               (3900.0, -15.0, 1400.0)),
    # 呼气：气流已减速、声门收拢 → 共振峰整体下移、高频更快滚降
    "exhale": ((360.0, 0.0, 380.0),
               (920.0, -6.0, 560.0),
               (1820.0, -11.0, 900.0),
               (3100.0, -18.0, 1400.0)),
}

#: 高斯白噪声经一阶 IIR 低通后的理论 RMS 增益 = 1/sqrt(1-a²)，a = exp(-2πfc/sr)。
#: 用于把滤波后的噪声重新归一到单位 RMS（**等响度**），而不是按峰值归一
#: ——噪声的峰值是极不稳定的离群量，按峰值归一会让 strength 严重非线性。


def _pink_noise(n: int, beta: float, rng) -> np.ndarray:
    """生成精确 1/f^β 噪声（频域幅度塑形）。

    比 Voss-McCartney 更可控：直接在频域给每个 bin 赋 1/f^(β/2) 的幅度、
    随机相位，再 IFFT。好处是谱指数**精确可验证**（Welch PSD 斜率应 ≈ -β），
    且同一 seed 完全可复现。

    注意 f=0（DC）必须置零：换气声没有直流分量，且 1/f 在 DC 处发散。
    """
    n = int(n)
    if n <= 1:
        return np.zeros(max(n, 0), dtype=np.float64)
    # rfft 长度取 n，保证 IFFT 回来正好 n 点
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    # 跳过 DC（freqs[0]==0），避免 1/0 与直流漂移
    scale[1:] = freqs[1:] ** (-beta / 2.0)
    # 归一化塑形曲线，避免整体能量随 n/beta 漂移
    scale /= np.sqrt(np.sum(scale ** 2) / max(len(scale), 1)) + 1e-12
    phase = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
    spec = scale * np.exp(1j * phase)
    spec[0] = 0.0                                    # 强制无 DC
    if n % 2 == 0:
        spec[-1] = spec[-1].real                     # Nyquist bin 必须为实数
    y = np.fft.irfft(spec, n=n)
    return y


def _nf_highpass(x: np.ndarray, sr: int, f_lo: float) -> np.ndarray:
    """高通：滤掉换气声里不应存在的次声/低频隆隆声。

    ⚠️ 必须用**单遍因果** `lfilter`（或 sosfilt），**不能用** `filtfilt`。
    `filtfilt` 是前向+后向各滤一遍，等效于把滤波器传递函数**平方**：
    一阶高通 6 dB/oct 会变成 12 dB/oct，把额外滚降叠加到后面
    Welch 拟合的 200–4000 Hz 区间里，**实测把谱斜率拉陡约 0.5**（-0.85 → -1.34）。
    换气声的起音瞬态本就该保留，不需要零相位。
    """
    try:
        from scipy.signal import butter, sosfilt, sosfilt_zi
        sos = butter(1, max(f_lo, 10.0) / (sr / 2.0), btype="high", output="sos")
        # 用稳态初值避免滤波起始瞬态（否则会在开头引入一个 DC 阶跃）
        zi = sosfilt_zi(sos) * x[0]
        y, _ = sosfilt(sos, x, zi=zi)
        return y.astype(np.float64)
    except Exception:
        # 无 scipy 的降级：一阶差分高通
        return (x - np.concatenate([[x[0]], x[:-1]])).astype(np.float64)


def _formant_shaping(x: np.ndarray, sr: int, formants) -> np.ndarray:
    """用并联二阶 IIR 共振器实现声道共振峰塑形。

    每个共振峰 = 一个 `iirpeak` 二阶带通（品质因数 Q = fc/BW），
    按相对增益加权后并联相加。这与语音合成里经典的**共振峰合成**
    （Klatt / Fant 源-滤波器模型）用的是同一套原理：
    噪声源 → 若干并联谐振器 → 输出。

    并联而非串联：串联会让共振峰互相抑制、整体过窄；
    并联保留每个共振峰的独立能量，听感更"有腔体"。
    """
    out = np.zeros_like(x)
    try:
        from scipy.signal import iirpeak, lfilter
        for fc, gain_db, bw in formants:
            if fc >= sr / 2.0 * 0.95:                # 超过 Nyquist 的共振峰丢弃
                continue
            q = max(fc / max(bw, 1e-6), 0.5)
            b, a = iirpeak(fc / (sr / 2.0), q)
            g = 10.0 ** (gain_db / 20.0)
            out += g * lfilter(b, a, x)
    except Exception:
        # 降级：用 FFT 频域高斯叠加做同样的谱形（无论有无 scipy 都能跑）
        freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
        shape = np.zeros_like(freqs)
        for fc, gain_db, bw in formants:
            g = 10.0 ** (gain_db / 20.0)
            shape += g * np.exp(-0.5 * ((freqs - fc) / max(bw / 2.355, 1e-6)) ** 2)
        spec = np.fft.rfft(x) * shape
        out = np.fft.irfft(spec, n=len(x))
    return out


def _breath_envelope(n: int, kind: str, rng) -> np.ndarray:
    """生理型换气包络。

    ⚠️ 关键数学事实：Gamma 型 ``t^α · e^(-βt)`` 的峰值（众数）在 ``t = α/β``，
    不是"看起来像靠前"。要峰值落在 25% 处就必须让 ``α/β ≈ 0.25``。
    （第一版给 α≈1.8 / β≈2.7 → 峰值跑到 t≈0.67，重心实测 0.596，
      与"吸气快起"的意图**正好相反**，被客观验证抓到。）

    吸气：Gamma 型，峰值约在 22~30% —— 声门快速张开、气流迅速达峰、尾部缓回。
    呼气：双指数，峰值约在 58~70% —— 气流缓起、达峰后较快收住。

    再叠加 3~9 Hz 的湍流微调制（幅度很小，只用来打破"光滑包络"的合成感）。
    """
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if kind == "exhale":
        # 呼气：气流缓起、达峰后收住，整体重心明确靠后。
        # ⚠️ 关键：rise 的时间常数必须**足够大**，否则指数上升早早饱和到 1、
        # 之后完全由 fall 主导，形状退化成近似对称（实测重心 0.458，与吸气
        # 几乎无差别）。实测：tau_r 取 0.16 时重心仅 0.46；取 0.42 才拉到 0.60+。
        tau_r = 0.38 + 0.10 * rng.rand()
        rise = 1.0 - np.exp(-t / tau_r)
        t_peak = 0.58 + 0.10 * rng.rand()
        # fall 从 t_peak 起按固定时间常数衰减（分母归一化，保证尾部真正落到 0）
        tau_f = 0.30 + 0.08 * rng.rand()
        fall = np.exp(-np.maximum(t - t_peak, 0.0) / tau_f)
        env = rise * fall
    else:
        # 吸气：Gamma 型，峰值靠前（快起缓落）。
        # 众数 = α/β，直接按目标峰值位置反解 α。
        t_peak = 0.20 + 0.07 * rng.rand()         # 峰值 20~27%
        beta = 3.0 + 0.8 * rng.rand()             # 尾部衰减速度
        alpha = max(t_peak * beta, 0.2)           # 保证众数 = α/β = t_peak
        env = (np.maximum(t, 1e-12) ** alpha) * np.exp(-beta * t)
    # 归一化到峰值 1，让后续等响度归一化只关心电平
    peak = float(np.max(env)) + 1e-12
    env = env / peak
    # 湍流微调制：3~9 Hz 的轻微起伏（比例很小，只破"规则感"）
    wob = (1.0
           + 0.055 * np.sin(2 * np.pi * (3.0 + 2.5 * rng.rand()) * t + rng.rand() * 6.28)
           + 0.030 * np.sin(2 * np.pi * (7.0 + 3.0 * rng.rand()) * t + rng.rand() * 6.28))
    return (env * wob).astype(np.float64)


#: 目标响度（RMS，满幅=1.0）。换气声整体应明显低于人声，但要在每一处
#: 保持一致的可闻度——否则有的地方听不见、有的地方"呼"一下很突兀。
_BREATH_RMS = 0.055


def generate_breath(sr: int, strength: float = 0.5, duration: float | None = None,
                    seed: int | None = None, kind: str = "inhale") -> np.ndarray:
    """合成真人换气声（源-滤波器模型 / 共振峰合成）。

    算法链（每一环都对应一个可验证的声学事实）：

      1. **精确 1/f^β 湍流源** —— 频域幅度塑形，β 由 ``_BREATH_BETA`` 给定。
         （旧实现用 `np.cumsum` 积分白噪声，得到的是 1/f² 布朗噪声，
          频谱斜率错了整整一档，听感闷响。）
      2. **声道共振峰塑形** —— 3~4 个并联二阶谐振器（``iirpeak``），
         吸气/呼气各有自己的共振峰表（``_BREATH_FORMANTS``）。
         （旧实现是 200–2200 Hz 平顶带通，没有共振结构。）
      3. **Gamma/双指数生理包络** —— 吸气快起缓落、呼气缓起快落，
         叠加 3~9 Hz 湍流微调制。零相位高通先滤掉次声。
         （旧实现是线性上升 + 尾部才生效的指数衰减。）
      4. **等响度归一化** —— 按 RMS 而非峰值归一，保证
         ``strength`` 是**单调、可复现**的电平旋钮。

    参数
    ----
    sr        : 采样率
    strength  : 0~1 换气轻重（乘在最终 RMS 上，线性）
    duration  : 秒；None 时按 strength 与 kind 自适应随机
    seed      : None 时用系统熵，同一段不重复；给定则完全可复现
    kind      : ``"inhale"``（吸气）/ ``"exhale"``（呼气）
    """
    strength = float(strength or 0)
    if strength <= 0.01:
        return np.zeros(0, dtype=np.float32)
    kind = "exhale" if str(kind).lower().startswith("ex") else "inhale"
    import os
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    rng = np.random.RandomState(seed)

    # --- 时长：按换气类型给不同的基线区间，并随 strength 轻微加长 ---
    if duration is None:
        if kind == "exhale":
            duration = 0.30 + 0.16 * rng.rand()     # 呼气通常比吸气长
        else:
            duration = 0.24 + 0.13 * rng.rand()
        duration += 0.05 * max(0.0, strength - 0.6)  # 深呼吸略长
    n = int(sr * max(0.08, duration))
    if n < 8:
        return np.zeros(0, dtype=np.float32)

    # --- 1) 湍流噪声源（精确谱指数）---
    src = _pink_noise(n, _BREATH_BETA[kind], rng)
    # 高通：去次声。换气声没有 100 Hz 以下的能量，留着只会糊。
    src = _nf_highpass(src, sr, 120.0 if kind == "inhale" else 90.0)

    # --- 2) 声道共振峰塑形 ---
    voiced = _formant_shaping(src, sr, _BREATH_FORMANTS[kind])

    # --- 3) 生理包络 ---
    env = _breath_envelope(n, kind, rng)
    y = voiced * env

    # --- 4) 等响度归一化（RMS，而非峰值）---
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    y = y / rms * _BREATH_RMS * strength
    # 输出前再做一次首尾微淡化，避免插进静音段时产生咔哒
    fade = min(int(sr * 0.006), n // 4)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y.astype(np.float32)


class BreathState:
    """换气序列的生理状态机（AR(1) 一阶自回归）。

    真人连续朗读时的换气不是独立同分布的：说得越急，相邻几次吸气
    **都**偏短偏浅；情绪平复下来又整体变长变深。这种"惯性"用 AR(1) 建模：

        x_i = rho * x_{i-1} + sqrt(1 - rho^2) * eps_i

    平稳态下方差恒为 1（系数开根号保证了这一点），rho 控制记忆强度：
    rho=0 退化为 i.i.d.（旧行为），rho→1 则几乎不变。
    取 0.35~0.5 之间最像真人——既有一致性，又不会呆板。

    ``next_params()`` 返回 (strength, duration, seed)，直接喂给
    :func:`generate_breath`。
    """

    def __init__(self, rho: float = 0.42, seed: int | None = None):
        import os
        self.rho = float(min(max(rho, 0.0), 0.95))
        self._seed = int(seed) if seed is not None else int.from_bytes(
            os.urandom(4), "little")
        self._rng = np.random.RandomState(self._seed)
        # 两个独立通道：时长惯性 与 强度惯性
        self._dur = 0.0
        self._amp = 0.0
        self._count = 0

    def _step(self, prev: float) -> float:
        k = np.sqrt(max(1.0 - self.rho ** 2, 0.0))
        return self.rho * prev + k * float(self._rng.randn())

    def next_params(self, strength: float, duration: float | None = None):
        """推进一次状态，返回 (strength, duration, seed)。

        ``strength`` 是本次换气的目标轻重（来自情绪/用户设定），
        AR(1) 在此之上叠加 ±12% 的惯性起伏。

        ``duration`` 为 None 时同样叠加 ±10% 惯性起伏并返回一个具体秒数；
        为具体值时也叠加惯性（否则外部一旦指定时长，惯性就完全失效——
        这是第一版被客观验证暴露出的设计缺陷）。
        返回的 seed 供 :func:`generate_breath` 生成噪声与包络随机细节。
        """
        self._count += 1
        self._dur = self._step(self._dur)
        self._amp = self._step(self._amp)
        # 幅度惯性：以传入 strength 为中心，波动 ±12%
        amp = float(strength) * (1.0 + 0.12 * self._amp)
        amp = float(min(max(amp, 0.0), 1.5))
        # 时长惯性：无论外部是否给出基准，都叠加 ±10% 起伏
        base = float(duration) if duration is not None else 0.30
        dur = base * (1.0 + 0.10 * self._dur)
        dur = float(max(dur, 0.08))
        return amp, dur, int(self._rng.randint(0, 2 ** 31 - 1))

    def reset(self):
        self._rng = np.random.RandomState(self._seed)
        self._dur = self._amp = 0.0
        self._count = 0


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
                     breath: float = 0.4, state: "BreathState | None" = None,
                     seed: int | None = None) -> np.ndarray:
    """把分句音频按「静音停顿 + 换气声」自然拼接。

    换气声用**吸气 → 呼气**双相结构（真人朗读一次换气的完整周期），
    两句之间：

        静音 → [吸气] → 极短间歇 → [呼气] → 静音 → 下一句

    吸气偏亮偏短（气流快）、呼气偏暗偏长（气流减速），这与真人一致；
    旧实现只在中间插一段单相呼吸声，听感是"每句之间糊一下"。

    ``state`` 传入 :class:`BreathState` 可让相邻换气按 AR(1) 生理相关；
    不传则内部自建一个（同一次调用内仍然相关）。
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    trimmed = [_trim_silence(c, sr) for c in chunks]
    trimmed = [c.astype(np.float32) for c in trimmed if len(c) > 0]
    if not trimmed:
        return np.zeros(0, dtype=np.float32)
    if state is None:
        state = BreathState(seed=seed)
    gap = np.zeros(int(max(0.0, pause) * sr), dtype=np.float32)
    out = trimmed[0]
    for c in trimmed[1:]:
        if breath > 0.01:
            # 一次完整换气：吸气在前，呼气在后
            amp_in, dur_in, s_in = state.next_params(breath * 0.90)
            amp_out, dur_out, s_out = state.next_params(breath * 0.65)
            inhale = generate_breath(sr, amp_in, duration=dur_in,
                                     seed=s_in, kind="inhale")
            exhale = generate_breath(sr, amp_out, duration=dur_out,
                                     seed=s_out, kind="exhale")
            # 吸/呼之间的极小间歇（气流换向），过长会显得犹豫
            inner = np.zeros(int(max(0.02, pause * 0.35) * sr), dtype=np.float32)
            breath_wav = np.concatenate([inhale, inner, exhale])
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
    params: dict = {"pitch": 0.0, "speed": 1.0, "volume": 1.0, "pause": 0.15, "breath": 0.0}
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


# ----------------------------------------------------------------------------- 清晰度增强
def enhance_clarity(y: np.ndarray, sr: int, amount: float = 0.94) -> np.ndarray:
    """温和的清晰度增强：一阶高通 pre-emphasis（衰减 6dB/oct 的低频倾斜），
    相对提升辅音/齿音/爆破音能量，改善长音频中个别词语咬字不清的问题。
    - amount 越接近 1 高频提升越强（语音常用 0.9~0.97）；
    - 处理后 RMS 对齐回原值，整体响度不变、音色不飘，并做防爆音限幅。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < 2:
        return y.astype(np.float32)
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[1:] - amount * y[:-1]
    rms_in = float(np.sqrt(np.mean(y ** 2)) + 1e-8)
    rms_out = float(np.sqrt(np.mean(out ** 2)) + 1e-8)
    out = out * (rms_in / rms_out)
    peak = float(np.max(np.abs(out))) + 1e-12
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)
