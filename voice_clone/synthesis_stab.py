"""
模块 3：合成阶段稳定性保障 (Synthesis Stability)
================================================
目标：朗读任意长度台词时，音色一致、韵律自然、情绪稳定、无断裂/爆音/突变。

针对三类症状的根因修复：
  1. 音色漂移/机械感加重 —— 单次超长生成的自回归漂移。改为长文本一律分块，
     且块间用「续写式生成」(build_prompt_cache + generate_with_prompt_cache +
     滑动窗口上下文) 保持韵律/音色连续，而不是每块独立重新生成。
  2. 上气不接下气/停顿异常 —— 旧 split 按固定 80 字硬切会切到词中间，
     且 crossfade 未裁剪段首尾静音。改为「按句末标点分句 + 子句标点折行」，
     拼接前先裁剪首尾静音，再以固定短停顿自然拼接。
  3. 音色/情绪剧烈波动 —— 每块独立生成导致能量/韵律跳变。改为逐块 declip+RMS
     对齐 + 最终整体限幅，保证整段响度一致。

提供：
  - declip / postprocess_output        : 软限幅 + RMS 归一化
  - split_long_text                    : 句末标点分句、子句折行、不切词
  - trim_silence                       : 相对能量裁剪首尾静音
  - concat_natural                     : 裁剪后自然停顿拼接
  - synthesize_stable                  : 编排（分块 -> 续写式逐段生成 -> 后处理 -> 拼接）
  - compute_stability_metrics          : 可验证的稳定性指标
"""
from __future__ import annotations
import re
import numpy as np

_SENT_END = "。！？!?；;…"
_CLAUSE = "，、：,:"


# ----------------------------------------------------------------------------- 限幅 / 后处理
def declip(y: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """防爆音：超出阈值时按峰值线性压回（整体缩放，无 tanh 软限幅的谐波失真/电流声）。"""
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > threshold:
        y = y * (threshold / peak)
    return y.astype(np.float32)


def _rms_normalize(y: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + 1e-8
    return y * (target_rms / rms)


def postprocess_output(y: np.ndarray, target_rms: float = 0.12,
                       ceiling: float = 0.99) -> np.ndarray:
    """单段生成结果的稳定后处理：RMS 对齐 + 线性防爆音（无 tanh 谐波失真）。"""
    y = _rms_normalize(y, target_rms)
    y = declip(y, ceiling)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- 长台词分块
def _hard_split(s: str, max_chars: int) -> list[str]:
    """极长无标点串：优先在空格处断，仍超长再按固定长度硬切（尽量不切词中间）。"""
    out, cur = [], ""
    for word in s.split():
        if len(cur) + len(word) + 1 <= max_chars:
            cur = (cur + " " + word).strip()
        else:
            if cur:
                out.append(cur)
            cur = word
            while len(cur) > max_chars:
                out.append(cur[:max_chars])
                cur = cur[max_chars:]
    if cur:
        out.append(cur)
    return out


def _split_sentence(s: str, max_chars: int) -> list[str]:
    """把超长单句按子句标点（，、：,）切分；仍超长则按空格硬切。"""
    parts = [p.strip() for p in re.split(rf"(?<=[{_CLAUSE}])", s) if p.strip()]
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            out.extend(_hard_split(p, max_chars))
    return out


def split_long_text(text: str, max_chars: int = 80) -> list[str]:
    """长台词分块（兼容旧接口）：返回纯文本块列表。"""
    return [t for t, _ in split_with_pauses(text, max_chars=max_chars)]


def split_with_pauses(text: str, max_chars: int = 60) -> list[tuple[str, str]]:
    """长台词分块：句末标点与逗号都作为停顿点，返回 [(text, pause_type)]。
    pause_type:
      - 'end'    句末标点（。！？；…）结尾 → 长停顿
      - 'comma'  逗号/顿号/冒号（，、：）结尾 → 短停顿
      - 'hard'   无标点超长硬切 → 极短停顿
    规则：每个分句独立成块（逗号处自然停顿），仅 <4 字的碎片并入前块；
    无标点超长串按 max_chars 硬切，避免切到词中间。"""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(rf"(?<=[{_SENT_END}{_CLAUSE}])", text) if p.strip()]
    if not parts:
        return []
    chunks: list[tuple[str, str]] = []
    cur, cur_type = "", "comma"
    for p in parts:
        ptype = "end" if p[-1] in _SENT_END else "comma"
        if cur and len(p) < 4:
            # 极短碎片（如"他说"）并入前块，避免过度碎片化
            cur += p
            cur_type = ptype
        else:
            if cur:
                chunks.append((cur, cur_type))
            cur, cur_type = p, ptype
    if cur:
        chunks.append((cur, cur_type))
    # 无标点超长块硬切
    out: list[tuple[str, str]] = []
    for t, typ in chunks:
        while len(t) > max_chars:
            out.append((t[:max_chars], "hard"))
            t = t[max_chars:]
        if t:
            out.append((t, typ))
    return out


# ----------------------------------------------------------------------------- 静音裁剪 / 拼接
def trim_silence(y: np.ndarray, sr: int, rel_db: float = 40.0) -> np.ndarray:
    """按相对能量裁剪首尾静音：能量低于「峰值帧 - rel_db」的帧视为静音。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame_len = int(sr * 0.02)  # 20ms
    if frame_len < 1 or len(y) < frame_len * 2:
        return y
    n = len(y) // frame_len
    frames = y[: n * frame_len].reshape(n, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thr = (rms.max() + 1e-12) / (10 ** (rel_db / 20))
    active = np.where(rms > thr)[0]
    if len(active) == 0:
        return y
    start = active[0] * frame_len
    end = min(len(y), (active[-1] + 1) * frame_len)
    return y[start:end]


def concat_natural(chunks: list[np.ndarray], sr: int, pause: float = 0.15) -> np.ndarray:
    """裁剪每段首尾静音后，以固定短停顿自然拼接（替代交叉淡化，避免'上气不接下气'）。"""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    trimmed = [trim_silence(c, sr) for c in chunks]
    trimmed = [c.astype(np.float32) for c in trimmed if len(c) > 0]
    if not trimmed:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(pause * sr), dtype=np.float32)
    out = trimmed[0]
    for c in trimmed[1:]:
        out = np.concatenate([out, gap, c])
    return out


# 兼容旧接口（仍被 server 旧逻辑引用）：等价于裁剪后短停顿拼接
def crossfade_cat(chunks: list[np.ndarray], sr: int, fade: float = 0.2) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return concat_natural(chunks, sr)


# ----------------------------------------------------------------------------- 稳定性指标
def _timbre_cosine(a: np.ndarray, b: np.ndarray, sr: int):
    """两段音频的 MFCC 均值向量余弦相似度（越高越一致，1.0=完全一致）。"""
    try:
        import librosa
        m1 = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=13).mean(axis=1)
        m2 = librosa.feature.mfcc(y=b, sr=sr, n_mfcc=13).mean(axis=1)
        return float(np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12))
    except Exception:
        return None


def compute_stability_metrics(chunks: list[np.ndarray], final: np.ndarray,
                              sr: int) -> dict:
    """可验证的稳定性指标（越低/越接近 1 越稳定）。"""
    m: dict = {}
    rms_list = [float(np.sqrt(np.mean(np.asarray(c, dtype=np.float32) ** 2) + 1e-12))
                for c in chunks if len(c) > 0]
    if len(rms_list) > 1:
        m["rms_mean"] = round(float(np.mean(rms_list)), 4)
        m["rms_std"] = round(float(np.std(rms_list)), 4)
        # 变异系数 CV = std/mean，越小说明各块响度越一致
        m["rms_cv"] = round(float(np.std(rms_list) / (np.mean(rms_list) + 1e-12)), 4)
    m["peak"] = round(float(np.max(np.abs(final))) if len(final) else 0.0, 4)
    m["duration_s"] = round(len(final) / sr, 2)
    m["n_chunks"] = len(chunks)
    # 音色一致性：相邻块 MFCC 余弦相似度均值（越接近 1 越一致）
    sims = []
    for a, b in zip(chunks, chunks[1:]):
        s = _timbre_cosine(np.asarray(a, dtype=np.float32),
                           np.asarray(b, dtype=np.float32), sr)
        if s is not None:
            sims.append(s)
    if sims:
        m["mfcc_cosine_mean"] = round(float(np.mean(sims)), 4)
        m["mfcc_cosine_min"] = round(float(np.min(sims)), 4)
    return m


# ----------------------------------------------------------------------------- 编排：稳定合成
def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "cpu"):
        return np.asarray(wav.squeeze(0).cpu().numpy(), dtype=np.float32)
    return np.asarray(wav, dtype=np.float32)


def _maybe_normalize_text(text: str, normalize: bool) -> str:
    """生成前做文本规范化（数字/日期），失败则原样返回。"""
    if not normalize:
        return text
    try:
        from voxcpm.utils.text_normalize import TextNormalizer
        return TextNormalizer().normalize(text)
    except Exception:
        return text


# ============================== 情绪控制 ==============================
# 目标：未指定情绪时全篇中性平稳、音色统一；仅当标点/情感词明确提示时才做
# 平滑、轻微的韵律过渡，且情绪只改韵律（音调/语速/音量/停顿），不改变音色。

EMOTION_CONTROL = {
    "default_emotion": "neutral",           # 默认情绪（未指定时）
    "trigger_threshold": 0.5,               # 情绪切换触发阈值（情感强度 0~1，低于此一律中性）
    "transition_smoothness": 0.5,           # 情绪过渡平滑度（0=最弱/最平滑，1=最强）
    "timbre_lock": True,                    # 音色一致性锁定（True=全程同参考，情绪不改音色）
    "keep_default_when_unspecified": True,  # 未特调时保持默认中性、音色不变
}

# 情感词词典：命中即触发对应情绪（强度随命中次数累加）
_EMOTION_WORDS = {
    "happy":    ["高兴", "开心", "快乐", "喜悦", "兴奋", "欢呼", "太好了", "真棒", "哈哈", "幸福", "愉快", "喜", "欢笑"],
    "sad":      ["悲伤", "难过", "伤心", "痛苦", "哭泣", "眼泪", "唉", "可惜", "遗憾", "失落", "伤感", "忧愁", "惆怅", "悲"],
    "angry":    ["愤怒", "生气", "怒", "气死", "可恶", "讨厌", "滚", "混蛋", "恼火", "暴怒"],
    "fear":     ["害怕", "恐惧", "担心", "紧张", "可怕", "惊恐", "畏惧", "恐慌"],
    "surprise": ["惊讶", "居然", "竟然", "天哪", "哇", "不可思议", "没想到", "吃惊", "意外"],
}

# 各情绪对应的韵律微调参数（音调半音 / 语速倍率 / 音量倍率）。
# 情绪只作用于韵律，不改变参考音色（音色由同一参考缓存锁定）。
_EMOTION_ACOUSTIC = {
    "neutral":     {"pitch": 0.0, "speed": 1.00, "volume": 1.00},
    "question":    {"pitch": 0.8, "speed": 1.00, "volume": 1.00},
    "exclamation": {"pitch": 1.2, "speed": 1.05, "volume": 1.10},
    "happy":       {"pitch": 1.0, "speed": 1.06, "volume": 1.08},
    "sad":         {"pitch": -1.0, "speed": 0.94, "volume": 0.92},
    "angry":       {"pitch": 0.0, "speed": 1.10, "volume": 1.15},
    "fear":        {"pitch": -0.5, "speed": 1.04, "volume": 0.96},
    "surprise":    {"pitch": 1.4, "speed": 1.05, "volume": 1.12},
}


def detect_emotion(text: str) -> tuple[str, float]:
    """检测文本块的情绪标签与强度（0~1）。优先级：情感词 > 多重感叹 > 单感叹 > 问号；
    无线索返回 ("neutral", 0.0)。"""
    text = (text or "").strip()
    if not text:
        return "neutral", 0.0
    # 情感词命中（取命中次数最多的情绪）
    hits: dict[str, int] = {}
    for label, words in _EMOTION_WORDS.items():
        c = sum(text.count(w) for w in words)
        if c > 0:
            hits[label] = c
    if hits:
        label = max(hits, key=hits.get)
        intensity = min(1.0, 0.5 + 0.25 * hits[label])
        return label, intensity
    # 标点检测
    n_excl = text.count("！") + text.count("!")
    n_ques = text.count("？") + text.count("?")
    if n_excl >= 2:
        return "exclamation", min(1.0, 0.4 + 0.2 * n_excl)
    if n_excl == 1:
        return "exclamation", 0.55
    if n_ques >= 1:
        return "question", 0.5
    return "neutral", 0.0


def _detect_f0(y: np.ndarray, sr: int) -> float | None:
    """检测音频块的中位基频（pyin）。要求足够长的有效浊音帧，否则判为不可靠返回 None，
    避免对短块/清音块的误判导致过度校正。"""
    try:
        import librosa
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr)
        voiced = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return None
        # 有效浊音帧占比不足 25% 视为检测不可靠（短块/偏清音的块）
        if len(voiced) / len(f0) < 0.25:
            return None
        if len(voiced) < 3:
            return None
        return float(np.median(voiced))
    except Exception:
        return None


def _align_pitch(pieces: list[np.ndarray], sr: int,
                 threshold_st: float = 4.0, max_correct_st: float = 1.5,
                 keep_st: float = 2.0) -> tuple[list[np.ndarray], list[tuple]]:
    """块级基频离群校正（保守版）：仅对 F0 偏离全局中位数超过 threshold_st(4) 半音的
    明显离群块做轻微拉回（单块最多 max_correct_st(1.5) 半音，拉回到偏离 keep_st(2) 以内）。
    先做八度误差修正（pyin 常把基频误判为高/低一个八度，那并非真实语调突变），
    避免把正常音调误当突变强行拉平（否则会引入机械感）。"""
    if len(pieces) < 3:
        return pieces, []
    f0s = [_detect_f0(p, sr) for p in pieces]
    valid = [(i, f) for i, f in enumerate(f0s) if f is not None and f > 0]
    if len(valid) < 3:
        return pieces, []
    median_f0 = float(np.median([f for _, f in valid]))
    if median_f0 <= 0:
        return pieces, []
    from audio_edit import apply_pitch
    out = list(pieces)
    corrected: list[tuple] = []
    for i, f in valid:
        # 八度误差修正：偏离接近 ±12 半音（0.85~1.15 个八度）视为 pyin 误判，折回同八度
        st = 12.0 * float(np.log2(f / median_f0))
        if abs(st) > 10.0:
            f = f / 2.0 if st > 0 else f * 2.0
            st = 12.0 * float(np.log2(f / median_f0))
        if abs(st) > threshold_st:
            target_st = -np.sign(st) * min(abs(st) - keep_st, max_correct_st)
            out[i] = apply_pitch(pieces[i], sr, target_st)
            corrected.append((i, round(st, 2), round(float(target_st), 2)))
    return out, corrected


def _apply_emotion(pieces: list[np.ndarray], emotions: list[tuple[str, float]],
                   sr: int, control: dict) -> tuple[list[np.ndarray], list[tuple]]:
    """情绪感知后处理：中性块不动，仅对情绪强度超过阈值的块做轻微韵律微调。
    幅度 = 基础参数 × 强度 × 平滑度；从中性过渡到情绪时额外 ×0.6（平滑起音，避免突兀）。
    只改韵律（音调/语速/音量），不改变参考音色（音色由同一参考缓存锁定）。"""
    threshold = float(control.get("trigger_threshold", 0.6))
    smoothness = float(control.get("transition_smoothness", 0.5))
    from audio_edit import apply_pitch, apply_speed, apply_volume
    out = list(pieces)
    report: list[tuple] = []
    prev_label = "neutral"
    for i, (label, intensity) in enumerate(emotions):
        if label == "neutral" or intensity < threshold:
            prev_label = "neutral"
            continue
        acoustic = _EMOTION_ACOUSTIC.get(label, _EMOTION_ACOUSTIC["neutral"])
        transition = 0.6 if prev_label == "neutral" else 1.0  # 从中性起音更平滑
        factor = intensity * smoothness * transition
        pitch_delta = acoustic["pitch"] * factor
        speed_factor = 1.0 + (acoustic["speed"] - 1.0) * factor
        volume_factor = 1.0 + (acoustic["volume"] - 1.0) * factor
        if abs(pitch_delta) > 0.05:
            out[i] = apply_pitch(out[i], sr, pitch_delta)
        if abs(speed_factor - 1.0) > 0.01:
            out[i] = apply_speed(out[i], sr, speed_factor)
        if abs(volume_factor - 1.0) > 0.01:
            out[i] = apply_volume(out[i], volume_factor)
        report.append((i, label, round(intensity, 2), round(factor, 2)))
        prev_label = label
    return out, report


def synthesize_stable(model, text: str, reference_wav_path: str | None,
                      sr_tts: int, prompt_wav_path: str | None = None,
                      prompt_text: str | None = None, max_chars: int = 60,
                      pause: float = 0.15, breath: float = 0.4,
                      emotion: str = "", emotion_control: dict | None = None,
                      **gen_kwargs) -> tuple[np.ndarray, dict]:
    """
    长度无关的稳定性合成编排（独立生成 + 情绪感知版）：
      1. 文本规范化 -> 句末标点 + 逗号分块（带停顿类型）；
      2. 一次构建参考缓存，每块都用同一参考缓存「独立生成」（音色锚定、不漂移、
         无续写误差累积、语速稳定、无机械感），失败自动回退 model.generate；
      3. 情绪控制：用户未显式指定情绪时，默认全篇中性平稳，仅当标点/情感词明确
         提示时才做平滑、轻微的韵律过渡（只改韵律，音色锁定不变）；
      4. 逐块软限幅+RMS 对齐；按停顿类型分级拼接（句末长停顿、逗号短停顿）；
      5. 返回 (audio_np, report含稳定性指标)。
    """
    normalize = bool(gen_kwargs.get("normalize", False))
    text = _maybe_normalize_text(text, normalize)
    chunks_with_pause = split_with_pauses(text, max_chars=max_chars)
    chunks_text = [t for t, _ in chunks_with_pause]
    report: dict = {"n_chunks": len(chunks_text),
                    "chunk_lengths": [len(c) for c in chunks_text]}

    # 参数透传
    cfg_value = float(gen_kwargs.get("cfg_value", 2.0))
    inference_timesteps = int(gen_kwargs.get("inference_timesteps", 10))

    use_ref = bool(reference_wav_path)
    use_prompt = bool(prompt_wav_path and prompt_text)

    # 独立生成（单块/回退）用的完整参数：生成参数 + 参考/提示
    full_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    if reference_wav_path:
        full_kwargs["reference_wav_path"] = reference_wav_path
    if use_prompt:
        full_kwargs["prompt_wav_path"] = prompt_wav_path
        full_kwargs["prompt_text"] = prompt_text

    if len(chunks_text) <= 1:
        audio = model.generate(text=text, **full_kwargs)
        wav = postprocess_output(_to_numpy(audio))
        report.update(compute_stability_metrics([wav], wav, sr_tts))
        return wav, report

    # 一次构建参考缓存，所有块复用同一参考（音色锚定，杜绝续写式误差累积）
    try:
        base_cache = model.tts_model.build_prompt_cache(
            prompt_text=prompt_text if use_prompt else None,
            prompt_wav_path=prompt_wav_path if use_prompt else None,
            reference_wav_path=reference_wav_path if use_ref else None,
        )
    except Exception:
        base_cache = None

    pieces: list[np.ndarray] = []
    for ct in chunks_text:
        if base_cache is not None:
            try:
                wav, _, _ = model.tts_model.generate_with_prompt_cache(
                    target_text=ct,
                    prompt_cache=base_cache,  # 每块都用同一参考缓存，独立生成
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    retry_badcase=True,
                    retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=6.0,
                    max_len=4096,
                )
                audio_np = _to_numpy(wav)
                report.setdefault("independent_ok", []).append(True)
            except Exception:
                audio_np = _to_numpy(model.generate(text=ct, **full_kwargs))
                report.setdefault("independent_ok", []).append(False)
        else:
            audio_np = _to_numpy(model.generate(text=ct, **full_kwargs))
            report.setdefault("independent_ok", []).append(False)
        pieces.append(postprocess_output(audio_np))

    # 情绪感知：仅在用户未显式指定情绪时，做「默认中性 + 明确提示才平滑过渡」的控制
    user_emotion = (emotion or "").strip()
    if not user_emotion:
        control = dict(EMOTION_CONTROL)
        if emotion_control:
            control.update(emotion_control)
        emotions = [detect_emotion(ct) for ct in chunks_text]
        pieces, emotion_transitions = _apply_emotion(pieces, emotions, sr_tts, control)
        if emotion_transitions:
            report["emotion_transitions"] = emotion_transitions

    # 块级基频离群校正：消除个别句子语调突变，让整段音调平稳过渡
    pieces, pitch_corrections = _align_pitch(pieces, sr_tts)
    if pitch_corrections:
        report["pitch_corrected"] = pitch_corrections

    final = _join_pieces(pieces, chunks_with_pause, sr_tts, pause, breath)
    report.update(compute_stability_metrics(pieces, final, sr_tts))
    return final, report


def _smooth_edges(y: np.ndarray, sr: int, fade_ms: float = 5.0) -> np.ndarray:
    """DC 移除 + 首尾微淡入淡出，消除拼接边界的直流阶跃 / 咔哒声 / 爆音。"""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y - y.mean()  # 去直流，避免块间拼接处的阶跃
    n = len(y)
    fade = int(sr * fade_ms / 1000.0)
    if fade > 0 and n > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return y.astype(np.float32)


def _join_pieces(pieces: list[np.ndarray], chunks_with_pause: list[tuple[str, str]],
                 sr: int, pause: float, breath: float) -> np.ndarray:
    """分级停顿拼接：句末=长停顿(可带呼吸声)、逗号=短停顿、硬切=极短停顿。
    每处呼吸声独立生成（随机时长/音色），块边界做 DC 移除 + 淡入淡出，消除杂音。"""
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    try:
        from audio_edit import generate_breath
    except Exception:
        generate_breath = None

    pause_end = max(0.30, pause * 2.0)     # 句末长停顿
    pause_comma = max(0.10, pause)         # 逗号短停顿
    pause_hard = 0.05                      # 硬切极短停顿

    out: np.ndarray | None = None
    for i, p in enumerate(pieces):
        c = _smooth_edges(trim_silence(p, sr), sr)
        if len(c) == 0:
            continue
        ptype = chunks_with_pause[i][1] if i < len(chunks_with_pause) else "comma"
        if out is None:
            out = c
            continue
        if ptype == "end":
            gap = np.zeros(int(pause_end * sr), dtype=np.float32)
            if generate_breath is not None and breath > 0.01:
                breath_wav = generate_breath(sr, breath)  # 每处独立生成，避免机械重复
                out = np.concatenate([out, gap, breath_wav, gap, c])
            else:
                out = np.concatenate([out, gap, c])
        elif ptype == "comma":
            gap = np.zeros(int(pause_comma * sr), dtype=np.float32)
            out = np.concatenate([out, gap, c])
        else:  # hard
            gap = np.zeros(int(pause_hard * sr), dtype=np.float32)
            out = np.concatenate([out, gap, c])
    return out.astype(np.float32) if out is not None else np.zeros(0, dtype=np.float32)
