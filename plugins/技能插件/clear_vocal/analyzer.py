"""清唱生成插件 · 音频分析器
==============================
从**分离后的音轨**提取音乐结构信息，产出后续规划所需的全部参数：

  节奏  → BPM、拍点时刻、节拍网格（含相位校正）
  音高  → F0 曲线（逐帧）、音符序列（midi + 起止时刻）
  调式  → 音阶/调性估计 + 音高量化到音阶

⚠️ 三条来自实测的硬约束（务必遵守，否则结果会明显不对）
--------------------------------------------------------
1. **必须分析分离后的 stems，不能直接吃原混音。**
   持续音（如铺底 Pad、持续弦乐）会污染 onset 包络。实测：一段精确 120 BPM 的
   信号叠加持续正弦后，librosa 测出的 BPM 从 117.45 崩到 92.29（偏差 -23%）。

2. **librosa 的 ``beat_track`` 有 ~2% 系统偏差**，且它自报的 tempo 与自算的
   beat interval 会互相矛盾（实测 120 BPM 报 117.45）。因此**不能裸信其 tempo**，
   必须以推导出的 beat 序列为准，再做**相位校正**（对齐到最强 onset 处）。

3. **不要在裸正弦上测 BPM**：librosa 对单频正弦的 onset 包络响应极差
   （曾出现 44.6 BPM 这类完全离谱的值）。节奏分析要喂真正的打击/起音信号。
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 常量
DEFAULT_SR = 22050              # 分析采样率（pyin 在 22.05k 上精度已足够且更快）
PYIN_HOP = 512
NOTE_MIN_SEMITONES = 0.08       # 音符最短时长（秒）；短于此的碎片并入邻音
                                # ⚠️ 别设太大：清唱里大量短音符（十六分、装饰音），
                                #    过大会把整个旋律吃空（曾设 0.55 导致 0 音符返回）
PITCH_MERGE_SEMITONES = 0.6     # 相邻同音高合并阈值（半音）
SILENCE_DB = -40.0              # 静音判定阈值（相对峰值）

#: pyin 的音高转移率上限（**倍频程/秒**，librosa 语义）。
#:
#: 为什么必须从默认 35.92 提高到 86.1
#: ---------------------------------
#: librosa 的 pyin 用 **Viterbi/HMM 求全局最优音高路径**，并用
#: ``max_transition_rate`` 限制帧间音高变化速率：
#:
#:     max_semitones_per_frame = round(max_transition_rate * 12 * hop / sr)
#:
#: 本模块参数（hop=512, sr=22050）下：默认 35.92 → **10 半音/帧**。
#: 后果：当旋律出现 **> 10 半音的大跳进**时，Viterbi 为了让全局路径代价最小，
#: 会**把跳进前的音符也拉低一个八度** —— 即把「72→57（15 半音）」改写为
#: 「60→57（3 半音）」这种"平缓"进行。**大跳进被系统性抹平。**
#:
#: 实测（10 条含大跳进的旋律）：默认 35.92 只对 5/10；提高后 10/10。
#: 该误判的隐蔽性极强：**首音那段音频一个采样都没变**，只是后面多了个音，
#: 读数就从 72 变 60 —— 因此改相位、改能量门控、改 fmax、改时长**全部无效**
#: （这些都被逐一实验否证过），只有改这个参数才有效。
#:
#: 为什么可以取这么松
#: ----------------
#: 本模块的输入按设计是**已分离的人声 stem**（见模块开头约束 1）。干净人声里的
#: 音高跳变都是真实存在的，不需要靠转移率去抑制 —— 松约束不会"放过噪声"。
#: 24 半音/帧已是"近似不约束"，收益在此饱和（再高不再提升）。
PYIN_MAX_TRANSITION_RATE = 86.1

# 调式估计：Krumhansl-Schmuckler 风格的音级权重
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ------------------------------------------------------------------ 节奏
def detect_tempo_and_beats(y: np.ndarray, sr: int = DEFAULT_SR,
                           start_bpm: float = 120.0,
                           tightness: float = 100.0,
                           correct_phase: bool = True,
                           auto_start_bpm: bool = True) -> dict:
    """检测 BPM 与拍点时刻，并做相位校正。

    返回: {bpm, beat_times, beat_strength, raw_bpm, hop}

    ⚠️ **``start_bpm`` 先验会主导结果**（实测坑）
    ------------------------------------------------
    librosa 的 ``beat_track`` 用 ``start_bpm`` 作为搜索先验，若信号特征不足，
    它会把结果收敛到该先验附近。实测：真值 90 BPM 与 128 BPM 的两种信号，
    **都**被报成 136.00 BPM、拍数都是 36 —— 与真值无关，纯粹是先验造成的。
    （开启/关闭重音也完全不影响，排除信号因素。）

    ``auto_start_bpm=True`` 时改为**扫描多个先验并选最优**：
    对每个候选先验跑一次 beat_track，用"拍点处 onset 平均强度"打分，取最高者。
    这消除了单一先验的偏置。
    """
    import librosa
    from librosa.beat import beat_track
    from librosa.onset import onset_strength

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < sr:
        return {"bpm": 0.0, "beat_times": np.zeros(0), "beat_strength": np.zeros(0),
                "raw_bpm": 0.0, "hop": PYIN_HOP}

    hop = PYIN_HOP
    oenv = onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)

    # onset 峰值（作为"应该被拍点命中的位置"的参考）
    peaks = librosa.util.peak_pick(
        oenv, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
        delta=0.02 * float(oenv.max() + 1e-9), wait=3)

    def score(beats) -> float:
        """F-measure 式打分：既要求拍点踩在 onset 上（精度），
        也要求覆盖住 onset 峰（召回）。

        ⚠️ 只用 "拍点处 onset 平均强度" 会**偏好拍点极少的结果**
        （只有 16 个拍点、恰好都踩在强 onset 上 → 平均强度最高）。
        实测：该朴素指标把 70~140 BPM 的信号一律判成 60 BPM。
        加入召回项后，漏拍会被扣分，指标不再可被"少而精"钻空子。
        """
        if len(beats) < 4:
            return -np.inf
        fr = np.clip(beats, 0, len(oenv) - 1)
        precision = float(np.mean(oenv[fr])) / (float(oenv.max()) + 1e-9)
        if peaks.size:
            # 每个 onset 峰到最近拍点的距离，<3 帧算命中
            d = np.abs(peaks[:, None] - beats[None, :]).min(axis=1)
            recall = float(np.mean(d <= 3))
        else:
            recall = 0.0
        if precision + recall <= 0:
            return -np.inf
        return 2 * precision * recall / (precision + recall)

    if auto_start_bpm:
        priors = [60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
        best = None
        for p in priors:
            t, b = beat_track(y=y, sr=sr, hop_length=hop,
                              start_bpm=float(p), tightness=float(tightness))
            s = score(b)
            if best is None or s > best[0]:
                best = (s, b, float(np.mean(np.atleast_1d(t))), p)
        _, beats, raw_bpm, used_prior = best
    else:
        raw_t, beats = beat_track(y=y, sr=sr, hop_length=hop,
                                  start_bpm=float(start_bpm), tightness=float(tightness))
        raw_bpm = float(np.mean(np.atleast_1d(raw_t)))
        used_prior = float(start_bpm)

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)

    # ⚠️ 以 beat 序列为准反推 BPM（librosa 自报 tempo 与 beat 间隔常不一致）
    if len(beat_times) >= 3:
        intervals = np.diff(beat_times)
        # 用中位数抗离群（漏拍的倍长间隔）
        med = float(np.median(intervals))
        # 收敛到合理区间：把异常大的间隔折半
        while med > 1.5 and np.median(intervals / 2.0) > 0.25:
            intervals = intervals[intervals < med * 1.6]
            if intervals.size < 2:
                break
            med = float(np.median(intervals))
        bpm = 60.0 / med if med > 1e-6 else raw_bpm
    else:
        bpm = raw_bpm

    # 倍频修正：先用 onset 对比度消歧（比单纯折进 [50,200) 可靠得多）
    bpm, octave_shift = _resolve_octave_ambiguity(bpm, oenv, sr, hop)

    if correct_phase and len(beat_times) >= 2:
        beat_times, strength = _correct_beat_phase(beat_times, oenv, sr, hop, bpm)
    else:
        strength = _beat_strength(beat_times, oenv, sr, hop)

    return {
        "bpm": float(bpm),
        "beat_times": np.asarray(beat_times, dtype=np.float64),
        "beat_strength": np.asarray(strength, dtype=np.float32),
        "raw_bpm": float(raw_bpm),
        "octave_shift": int(octave_shift),
        "hop": int(hop),
    }


def _fold_bpm(bpm: float, lo: float = 50.0, hi: float = 200.0) -> tuple[float, int]:
    """把 BPM 折进 [lo, hi)，返回 (bpm, 缩放次数)。"""
    if bpm <= 0:
        return bpm, 0
    shift = 0
    while bpm < lo and shift > -4:
        bpm *= 2.0
        shift -= 1
    while bpm >= hi and shift < 4:
        bpm /= 2.0
        shift += 1
    return float(bpm), int(shift)


def _resolve_octave_ambiguity(bpm: float, oenv: np.ndarray, sr: int, hop: int,
                               lo: float = 55.0, hi: float = 190.0) -> tuple[float, int]:
    """把 BPM 折叠进人类感知区间 [lo, hi)。

    ⚠️ BPM 存在**根本性的倍频歧义**，无法用 onset 信息消除：
       当信号每拍都有 onset 时，"120 BPM 每拍一音" 与 "240 BPM 每半拍一音"
       在 onset 包络上完全等价，数学上不可区分。
       我曾尝试用"拍点/半拍点 onset 强度对比度"来消歧，**实测方向是错的**
       （把正确的 120 判成了 58.7），因为该判据在等间隔脉冲串上必然退化为噪声。

    因此这里只做**保守的区间折叠**：把 BPM 用 2 的幂折进感知区间，
    并在返回时报告折叠次数（`octave_shift`）供上层知情。
    对下游对齐而言，真正被使用的是 **beat 间隔**（`beat_times` 的差分），
    它不受 BPM 数值倍频歧义的影响 —— 所以这里**不需要**也不应该做激进猜测。
    """
    if bpm <= 0:
        return bpm, 0
    return _fold_bpm(bpm, lo, hi)


def _beat_strength(beat_times, oenv, sr, hop) -> np.ndarray:
    """每个拍点处的 onset 强度（用于找重拍）。"""
    import librosa
    if len(beat_times) == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.clip(librosa.time_to_frames(beat_times, sr=sr, hop_length=hop),
                     0, len(oenv) - 1)
    return oenv[frames].astype(np.float32)


def _correct_beat_phase(beat_times: np.ndarray, oenv: np.ndarray,
                        sr: int, hop: int, bpm: float,
                        search_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """节拍**相位校正**：把整条网格平移，使其与最强 onset 对齐。

    librosa 的 beat 序列常有固定的时间偏移（整条网格"偏左/偏右"）。
    做法：在一个拍长范围内滑动相位，选使"拍点处 onset 总强度"最大的偏移量。
    """
    import librosa
    n = len(beat_times)
    if n < 2:
        return beat_times, _beat_strength(beat_times, oenv, sr, hop)

    period = 60.0 / bpm
    span = period * search_ratio
    offsets = np.linspace(-span, span, 41)

    best_off, best_score = 0.0, -np.inf
    for off in offsets:
        cand = beat_times + off
        cand = cand[(cand >= 0)]
        if cand.size < 2:
            continue
        fr = np.clip(librosa.time_to_frames(cand, sr=sr, hop_length=hop), 0, len(oenv) - 1)
        score = float(np.mean(oenv[fr]))
        if score > best_score:
            best_score, best_off = score, off

    corrected = (beat_times + best_off)
    corrected = corrected[corrected >= 0]
    return corrected, _beat_strength(corrected, oenv, sr, hop)


def snap_to_grid(times: np.ndarray, bpm: float, origin: float = 0.0,
                 subdiv: int = 4) -> np.ndarray:
    """把时刻吸附到节拍网格（``subdiv`` = 每拍细分格数，4 = 十六分音符）。"""
    times = np.asarray(times, dtype=np.float64)
    if bpm <= 0 or times.size == 0:
        return times.copy()
    step = 60.0 / bpm / max(subdiv, 1)
    k = np.round((times - origin) / step)
    return origin + k * step


# ------------------------------------------------------------------ 音高
def extract_f0(y: np.ndarray, sr: int = DEFAULT_SR,
               fmin_note: int = 40, fmax_note: int = 88,
               hop: int = PYIN_HOP,
               max_transition_rate: float = PYIN_MAX_TRANSITION_RATE) -> dict:
    """逐帧 F0（pyin），返回 {f0, voiced, midi, times, hop}。

    ``max_transition_rate`` 的取值理由见模块常量 ``PYIN_MAX_TRANSITION_RATE`` 的
    注释 —— **用默认值 35.92 会系统性抹平大跳进**（实测 5/10 正确率）。
    """
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    fmin, fmax = float(librosa.midi_to_hz(fmin_note)), float(librosa.midi_to_hz(fmax_note))
    if y.size < 2048:
        return {"f0": np.zeros(0), "voiced": np.zeros(0, dtype=bool),
                "midi": np.zeros(0), "times": np.zeros(0), "hop": hop}

    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop,
                                 max_transition_rate=float(max_transition_rate))
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    with np.errstate(invalid="ignore"):
        midi = librosa.hz_to_midi(f0)
    return {
        "f0": np.asarray(f0, dtype=np.float64),
        "voiced": np.asarray(voiced, dtype=bool),
        "midi": np.asarray(midi, dtype=np.float64),
        "times": np.asarray(times, dtype=np.float64),
        "hop": int(hop),
    }


def extract_notes(f0info: dict, y: np.ndarray, sr: int = DEFAULT_SR,
                  min_dur: float = NOTE_MIN_SEMITONES,
                  merge_semi: float = PITCH_MERGE_SEMITONES,
                  quantize_scale: list[int] | None = None) -> list[dict]:
    """F0 曲线 → 音符序列。

    步骤：① 取 voiced 帧并按 midi 四舍五入成"音高段"；
          ② 合并相邻同音段、丢弃过短碎片（碎片的时长并入前一个音）；
          ③ 可选：量化到给定音阶（调式内）。
    返回 [{start, end, dur, midi, f0, conf}]，按时间排序。
    """
    midi = np.asarray(f0info["midi"], dtype=np.float64)
    voiced = np.asarray(f0info["voiced"], dtype=bool)
    times = np.asarray(f0info["times"], dtype=np.float64)
    if midi.size == 0:
        return []
    dt = (times[1] - times[0]) if times.size > 1 else PYIN_HOP / float(sr)

    # ① 连续同音段
    if quantize_scale is not None and len(quantize_scale):
        q_midi = _quantize_to_scale(midi, quantize_scale)
    else:
        q_midi = np.round(midi)

    segs: list[dict] = []
    cur = None
    for i in range(len(q_midi)):
        if not voiced[i] or not np.isfinite(q_midi[i]):
            if cur is not None:
                segs.append(cur)
                cur = None
            continue
        m = float(q_midi[i])
        if cur is None:
            cur = {"start": times[i], "end": times[i], "midis": [m], "f0s": [midi[i]]}
        elif abs(m - cur["midis"][-1]) < 0.5:
            cur["end"] = times[i]
            cur["midis"].append(m)
            cur["f0s"].append(midi[i])
        else:
            segs.append(cur)
            cur = {"start": times[i], "end": times[i], "midis": [m], "f0s": [midi[i]]}
    if cur is not None:
        segs.append(cur)

    # ② 收尾时刻补一个帧长；丢弃过短碎片（并入邻音，首个碎片则保留给后一个音的起点）
    notes: list[dict] = []
    pending_start: float | None = None
    for s in segs:
        start, end = s["start"], s["end"] + dt
        if end - start < min_dur:
            if notes:
                # 碎片并进前一个音（延长其结尾）
                notes[-1]["end"] = max(notes[-1]["end"], end)
                notes[-1]["dur"] = notes[-1]["end"] - notes[-1]["start"]
            else:
                # ⚠️ 别直接丢掉：它是下一个音的真正起点（音头常被判成短碎片）
                pending_start = start if pending_start is None else pending_start
            continue
        if pending_start is not None:
            start = pending_start
            pending_start = None

        med_midi = float(np.median(s["midis"]))
        if notes and abs(notes[-1]["midi"] - med_midi) < merge_semi \
                and (start - notes[-1]["end"]) < min_dur:
            notes[-1]["end"] = end
            notes[-1]["dur"] = notes[-1]["end"] - notes[-1]["start"]
            notes[-1]["midis"].extend(s["midis"])
            notes[-1]["f0s"].extend(s["f0s"])
            notes[-1]["midi"] = float(np.median(notes[-1]["midis"]))
            notes[-1]["f0"] = float(np.median(notes[-1]["f0s"]))
            continue
        notes.append({
            "start": float(start), "end": float(end), "dur": float(end - start),
            "midi": med_midi, "f0": float(np.median(s["f0s"])),
            "midis": s["midis"], "f0s": s["f0s"],
        })

    # ②b 离群音高清理：pyin 在音头/音尾偶发八度跳变，把明显偏离前后邻音
    #     且极短的音并回邻音（避免生成出"忽然高八度"的音符）
    notes = _drop_pitch_outliers(notes, min_dur=min_dur, merge_semi=merge_semi)

    # ③ 逐音响度（给规划器做力度）
    for n in notes:
        seg = y[int(n["start"] * sr):int(n["end"] * sr)]
        n["rms"] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    return notes


def _drop_pitch_outliers(notes: list[dict], min_dur: float = NOTE_MIN_SEMITONES,
                         merge_semi: float = PITCH_MERGE_SEMITONES,
                         jump: float = 7.0) -> list[dict]:
    """把"孤立的大跳 + 极短"的音并回邻音。

    判据：时长 < 1.5*min_dur 且与前后邻音都相距 > jump 半音，而前后邻音彼此接近。
    这类音几乎都是 pyin 的八度误检，而非真实旋律。
    """
    if len(notes) < 3:
        return notes
    keep = [True] * len(notes)
    for i in range(1, len(notes) - 1):
        if not keep[i]:
            continue
        n, prev, nxt = notes[i], notes[i - 1], notes[i + 1]
        if not (keep[i - 1] and keep[i + 1]):
            continue
        short = n["dur"] < max(min_dur * 1.5, 0.12)
        far_prev = abs(n["midi"] - prev["midi"]) > jump
        far_next = abs(n["midi"] - nxt["midi"]) > jump
        near_neighbors = abs(prev["midi"] - nxt["midi"]) < merge_semi * 2
        if short and far_prev and far_next and near_neighbors:
            keep[i] = False
            # 邻音之间拉直（跨越被删音）
            prev["end"] = nxt["start"]
            prev["dur"] = prev["end"] - prev["start"]
    return [n for i, n in enumerate(notes) if keep[i]]


def _quantize_to_scale(midi: np.ndarray, scale_pcs: list[int]) -> np.ndarray:
    """把 midi 量化到指定音级集合（scale_pcs 是相对 C 的半音集合）。"""
    out = np.full_like(midi, np.nan, dtype=np.float64)
    ok = np.isfinite(midi)
    m = midi[ok]
    base = np.round(m)
    pc = np.mod(base, 12).astype(int)
    # 对每个音，在音阶内找最近的音级（含上下八度）
    cands = []
    for p in scale_pcs:
        for oct_shift in (-12, 0, 12):
            cands.append(p + oct_shift)
    cands = np.array(sorted(set(cands)), dtype=float)
    diff = np.abs(base[:, None] + 0.0 - (np.floor(base[:, None] / 12) * 12 + cands[None, :]))
    pick = np.argmin(diff, axis=1)
    out[ok] = np.floor(base / 12) * 12 + cands[pick]
    return out


# ------------------------------------------------------------------ 调式
def estimate_key(chroma: np.ndarray) -> dict:
    """由 chroma 估计调性（Krumhansl-Schmuckler 相关法）。"""
    c = np.asarray(chroma, dtype=np.float64).mean(axis=1)
    if c.sum() <= 0:
        return {"tonic": 0, "mode": "major", "name": "C major", "score": 0.0,
                "scale_pcs": list(_MAJOR_PROFILE.argsort()[-7:])}
    c = c / (c.sum() + 1e-12)
    best = {"score": -np.inf}
    for root in range(12):
        for mode, prof in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            p = np.roll(prof, root)
            p = p / p.sum()
            score = float(np.corrcoef(c, p)[0, 1])
            if score > best["score"]:
                best = {"score": score, "tonic": root, "mode": mode}
    root, mode = best["tonic"], best["mode"]
    if mode == "major":
        scale_pcs = [0, 2, 4, 5, 7, 9, 11]
    else:
        scale_pcs = [0, 2, 3, 5, 7, 8, 10]      # 自然小调
    scale_pcs = [int((root + p) % 12) for p in scale_pcs]
    return {
        "tonic": int(root),
        "mode": mode,
        "name": "%s %s" % (NOTE_NAMES[root], mode),
        "score": float(best["score"]),
        "scale_pcs": sorted(scale_pcs),
    }


def chroma_from_audio(y: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray:
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.feature.chroma_cqt(y=y, sr=sr)


# ------------------------------------------------------------------ 总入口
def analyze(y: np.ndarray, sr: int = DEFAULT_SR,
            do_tempo: bool = True, do_pitch: bool = True,
            do_key: bool = True) -> dict:
    """一次跑完分析，返回完整结构（可直接 JSON 化，除 numpy 数组需转换）。"""
    out: dict = {"sr": int(sr), "duration": float(len(y) / sr)}
    if do_tempo:
        out["rhythm"] = detect_tempo_and_beats(y, sr)
    if do_pitch:
        f0 = extract_f0(y, sr)
        out["f0"] = f0
        out["notes"] = extract_notes(f0, y, sr)
    if do_key:
        ch = chroma_from_audio(y, sr)
        out["key"] = estimate_key(ch)
        out["chroma"] = ch.mean(axis=1)
    return out
