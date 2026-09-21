"""清唱生成插件 · 对齐器（时间轴 + 音高）
============================================
把"合成出的一个个短句片段"精确摆回原曲的时间轴与音高上。

这是用户明确点名的验收重点 —— *"生成歌声与原曲节奏的时间对齐精度"* ——
所以本模块的接口设计原则是：**纯函数、可单独测、有精确契约**。

    stretch_to(clip, sr, target_dur)   → 时长精确等于 target_dur（误差 = 0 样本）
    shift_to(clip, sr, target_midi, source_midi) → 基频精确等于目标音高
    plan_note_timing(notes, bpm, ...)  → 落点/时长/网格吸附（纯几何，不产音频）
    assemble([(info, clip), ...], sr)  → 按整数样本落点拼接 + 限幅收尾
    align_all(notes, clips, sr, ...)   → 上面几步的一体化批量入口

为什么"精确 == 0 样本"是可达的，而不是"大致对齐"
----------------------------------------------------
朴素做法（``audio_edit.apply_pitch`` 那一套）有两处系统性误差：

1. **时长误差累积**：变速重采样后用 WSOLA 还原时长，WSOLA 的帧长/步长
   量化会让输出长度有一个几百样本级的残差。一个音符差 5 ms 不致命，
   但 300 个音符各差 5 ms 之后，整曲会跑偏 1.5 秒 —— 这是致命的。
   本模块的做法是：伸缩之后**用精确长度裁/补**（``_fit_length``），
   把残差强制归零。裁/补发生在音符两端，而两端在拼接时会被交叉淡化
   吃掉，所以听感上无副作用。

2. **音高误差**：重采样变速会让"实际音高"变成 ``目标 × (重采样比)``，
   依赖两次操作互相抵消。而 ``dsp.pitch_shift`` 是频域直接搬移，
   时长本来就不变，音高搬移量就是设定量 —— 没有二次误差来源。
   （实测偏差 ≤0.87% ≈ 0.15 半音，见 dsp.pitch_shift 的系数扫描表。）

时间对齐的两级策略
------------------
* **拍级**：音符起止时刻先吸附到节拍网格（复用 ``analyzer.snap_to_grid``）。
  原曲本身是量化的（人声有 rubato，但伴奏是网格），吸附能消除 pyin 的
  帧量化误差（``PYIN_HOP=512`` @22.05k → 每帧 23.2 ms，这是底层分辨率上限）。
* **样本级**：摆位用整数样本落点，绝不出现"浮点时刻 + 插值重采样"的
  二次失真。整曲时间轴由 ``assemble`` 用整数样本索引构建。

⚠️ 网格吸附是**可关闭**的（``snap=True/False``）
-------------------------------------------------
不是所有素材都在网格上。戏曲散板、自由速度的吟唱吸附后会明显"变死"。
所以吸附是默认开启的可选项，关掉后退化为"直接用原曲实测时刻"，
这条路径同样精确（因为原曲时刻就是真值）。

与 ``aligner`` 无关、但必须记住的一条
--------------------------------------
``singer.py`` 调用 VoxCPM2 生成时**必须在 ``_infer_lock`` 之外**进行
（``server.py`` 的锁非可重入，在钩子内回调生成路径会永久死锁）。
本模块不接触模型，是纯 DSP，故完全没有这个约束。
"""
from __future__ import annotations

import numpy as np

from . import dsp

# ------------------------------------------------------------------ 常量
FADE_MS = 8.0                   # 音符边界交叉淡化时长（毫秒）
MIN_FADE_SAMPLES = 8
SNAP_SUBDIV = 4                 # 网格细分：4 = 十六分音符
DEFAULT_MAX_SEMITONES = 12.0    # 单音符最大搬移量（超过则判为误检，钳制而非硬搬）


# ------------------------------------------------------------------ 长度精确化
def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """把信号裁/补到**精确** n 个样本。

    这是"零累积漂移"的关键。伸缩类算法的输出长度总有几个到几百样本的
    残差（相位声码器按帧量化、WSOLA 按步长量化），若放任不管，
    每个音符各偏几毫秒，几百个音符后整曲会跑偏到无法对拍。

    补的时候用**边界值延续**（而非补零）：补零会在音符尾制造一个
    人为的跳变 → 爆音。延续值不引入新频率成分，配合淡化几乎不可闻。
    """
    y = np.asarray(y, dtype=np.float32)
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if y.size == n:
        return y
    if y.size > n:
        return np.ascontiguousarray(y[:n], dtype=np.float32)
    pad = n - y.size
    tail = y[-1] if y.size else 0.0
    return np.ascontiguousarray(np.concatenate(
        [y, np.full(pad, tail, dtype=np.float32)]), dtype=np.float32)


# ------------------------------------------------------------------ 时长
def stretch_to(clip: np.ndarray, sr: int, target_dur: float,
               transient_aware: bool = True) -> np.ndarray:
    """把片段伸缩到**精确** ``target_dur`` 秒，保持音高与共振峰。

    内部调 ``dsp.time_stretch``（瞬态感知的相位声码器），再用 ``_fit_length``
    把残差强制归零。返回长度 = ``round(target_dur * sr)``，精确相等。

    ``target_dur`` 与输入时长之比超出 [0.25, 4.0] 时，``time_stretch`` 内部会钳制；
    此时 ``_fit_length`` 仍会把长度校到目标值，但**音高会因超出算法有效范围
    而出现相位不连续**。调用方（``planner``）应保证每个音符的伸缩比在 4 倍以内 ——
    这也是"先做短句 + 拉伸"这条路线的固有上限：长句需要 4 倍以上拉伸时，
    应该改走"多段拼接"而不是硬拉。
    """
    clip = dsp.to_mono(clip)
    n_target = int(round(float(target_dur) * int(sr)))
    if n_target <= 0:
        return np.zeros(0, dtype=np.float32)
    if clip.size == 0:
        return np.zeros(n_target, dtype=np.float32)

    rate = n_target / float(clip.size)
    if abs(rate - 1.0) < 0.01:
        return _fit_length(clip, n_target)

    out = dsp.time_stretch(clip, rate, sr=sr, transient_aware=transient_aware)
    return _fit_length(out, n_target)


# ------------------------------------------------------------------ 音高
def shift_to(clip: np.ndarray, sr: int, target_midi: float,
             source_midi: float,
             max_semitones: float = DEFAULT_MAX_SEMITONES) -> tuple[np.ndarray, float]:
    """把片段从 ``source_midi`` 搬移到 ``target_midi``，保持时长与共振峰。

    ``source_midi=None`` 或非有限值时按 0 半音处理（即不搬移）。

    返回 ``(音频, 实际搬移的半音数)`` —— 返回第二个值是为了让调用方能
    在日志/``note_plan.json`` 里核对"我要求的搬移"与"实际执行的搬移"一致。

    为什么要钳制 ``max_semitones``
    ------------------------------
    pyin 在音头/音尾偶发八度跳变（差 12 半音）。若不钳制，一个误检音符会被
    硬搬到差一个八度的位置，在听感上是极刺耳的"走音"。钳制把损害限制在
    "轻微走音"而非"崩坏"。

    ⚠️ **但钳制是最后一道防线，不是设计手段**（这一点曾经搞错过）
    ----------------------------------------------------------
    早期版本把 ±12 当成"正常工作的保护"，结果实测出现 3 个音符被钳到 12.00 ——
    旋律被**静默改掉**，而日志里看起来只是"最大搬移 12.00 半音"、很像正常。
    正确架构是：搬移量应由 ``plugin.run`` 的**整体八度折叠**压到 ±5 以内，
    使这里永远不该触发钳制。一旦 ``summary["semitone_clamped"] > 0``，就说明
    上游出了问题，应当报错而不是当成正常统计看过去。
    """
    clip = dsp.to_mono(clip)
    if clip.size == 0:
        return clip, 0.0

    if target_midi is None or source_midi is None:
        return clip, 0.0
    if not (np.isfinite(target_midi) and np.isfinite(source_midi)):
        return clip, 0.0

    semis = float(target_midi) - float(source_midi)
    semis = float(np.clip(semis, -abs(max_semitones), abs(max_semitones)))
    if abs(semis) < 0.05:
        return clip, 0.0

    # ⚠️ pitch_shift 时长不变（频域搬移，无重采样），故此处**不需要**任何
    #    长度修正。但为防御实现细节变动，仍做一次断言性对齐。
    out = dsp.pitch_shift(clip, int(sr), semis)
    out = _fit_length(out, clip.size)
    return out, semis


# ------------------------------------------------------------------ 网格吸附
def snap_times(times, bpm: float, origin: float = 0.0,
               subdiv: int = SNAP_SUBDIV) -> np.ndarray:
    """把时刻吸附到节拍网格（``subdiv=4`` → 十六分音符）。

    ``bpm <= 0`` 时原样返回 —— 分析失败的素材不该被强加一个假网格。
    """
    t = np.asarray(times, dtype=np.float64)
    if bpm is None or bpm <= 0 or t.size == 0:
        return t.copy()
    step = 60.0 / float(bpm) / max(int(subdiv), 1)
    return origin + np.round((t - origin) / step) * step


def plan_note_timing(notes: list[dict], bpm: float = 0.0, origin: float = 0.0,
                     snap: bool = True, subdiv: int = SNAP_SUBDIV,
                     total_dur: float | None = None,
                     min_dur: float = 0.05) -> list[dict]:
    """给音符序列算出**最终落点与时长**（不产生音频，纯几何计算，便于单测）。

    返回 ``[{start, dur, start_sample, n_samples, midi, ...}]``，其中：

    * ``start`` / ``dur`` —— 吸附后的时刻（秒）
    * ``start_sample`` / ``n_samples`` —— 整数样本落点与长度（时间轴的唯一真值）

    三条必须守住的规则
    ------------------
    1. **单调不重叠**：后一个音的起点不早于前一个音的终点。pyin 偶尔会给出
       重叠的音符（尤其滑音），重叠在拼接时会互相抵消成"打嗝"。
    2. **最小可听时长**：短于 ``min_dur`` 的音符会给不出足够周期做相位声码，
       而且听感上像爆音。低于阈值时**延长**而非丢弃 —— 丢音符会让旋律断掉。
    3. **末音不越界**：若给了 ``total_dur``，最后一个音不得越过它。
    """
    if not notes:
        return []

    starts = np.array([float(n["start"]) for n in notes], dtype=np.float64)
    ends = np.array([float(n.get("end", n["start"] + n.get("dur", 0.1)))
                     for n in notes], dtype=np.float64)

    if snap:
        starts = snap_times(starts, bpm, origin, subdiv)
        ends = snap_times(ends, bpm, origin, subdiv)

    out: list[dict] = []
    prev_end = -np.inf
    for i, n in enumerate(notes):
        s, e = float(starts[i]), float(ends[i])
        if total_dur is not None:
            s = min(s, float(total_dur))
            e = min(e, float(total_dur))
        # 规则1：单调推进，绝不与前一个音重叠
        if s < prev_end:
            s = prev_end
        # 规则2：保底时长
        if e - s < min_dur:
            e = s + min_dur
        if total_dur is not None and e > float(total_dur):
            e = float(total_dur)
            if e - s < min_dur:
                s = max(0.0, e - min_dur)
        if e <= s:
            continue

        item = dict(n)
        item.update({
            "start": s,
            "end": e,
            "dur": e - s,
            "start_sample": int(round(s * 1.0)),   # 由调用方用 sr 覆盖
            "n_samples": 0,
        })
        out.append(item)
        prev_end = e
    return out


def to_sample_grid(timed: list[dict], sr: int) -> list[dict]:
    """把秒级时刻转成整数样本落点，并保证**样本级**不重叠。

    ⚠️ 分两步做（先秒级、再样本级）而不是一步到位，是因为秒→样本的四舍五入
    可能让两个相邻音符的样本区间重新产生 1 个样本的重叠。
    """
    sr = int(sr)
    out: list[dict] = []
    prev_end_sample = 0
    for n in timed:
        s = int(round(float(n["start"]) * sr))
        e = int(round(float(n["end"]) * sr))
        s = max(s, prev_end_sample)
        if e - s < MIN_FADE_SAMPLES:
            e = s + MIN_FADE_SAMPLES
        item = dict(n)
        item["start_sample"] = s
        item["n_samples"] = e - s
        out.append(item)
        prev_end_sample = e
    return out


# ------------------------------------------------------------------ 位放/拼接
def _window_fade(n: int, sr: int, fade_ms: float = FADE_MS) -> np.ndarray:
    """生成一个两端带升余弦淡入淡出的窗（长度为 n）。"""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    w = np.ones(n, dtype=np.float32)
    k = min(int(round(fade_ms / 1000.0 * sr)), n // 2)
    k = max(k, 0)
    if k >= 2:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k, dtype=np.float32) / k)
        w[:k] = ramp
        w[-k:] = ramp[::-1]
    return w


def _place(canvas: np.ndarray, clip: np.ndarray, start: int, sr: int,
           fade_ms: float = FADE_MS) -> None:
    """把片段以淡入淡出方式叠加到画布上（原地修改）。"""
    if clip.size == 0:
        return
    n_canvas = canvas.size
    start = int(start)
    if start >= n_canvas:
        return
    if start < 0:
        clip = clip[-start:]
        start = 0
    if clip.size == 0:
        return
    n = min(clip.size, n_canvas - start)
    seg = clip[:n] * _window_fade(n, sr, fade_ms)
    canvas[start:start + n] += seg


def assemble(timed_clips: list[tuple[dict, np.ndarray]], sr: int,
             total_dur: float | None = None,
             limit: bool = True) -> np.ndarray:
    """把 ``[(音符信息, 片段音频), ...]`` 摆到同一时间轴，拼成整曲。

    * 音符之间的空隙**保持静音**（清唱本就该有换气与停顿，不填内容）
    * 片段边界交叉淡化（升余弦窗），避免拼接爆音
    * ``limit=True`` 时做峰值归一化 + 软限幅（防叠加溢出削波）

    返回单声道 float32。长度 = ``total_dur`` 或"最后一个音的终点"。
    """
    sr = int(sr)
    if not timed_clips:
        return np.zeros(int(round((total_dur or 0.0) * sr)), dtype=np.float32)

    last_end = 0
    for info, clip in timed_clips:
        last_end = max(last_end, int(info.get("start_sample", 0)) + int(clip.size))
    n_total = int(round(float(total_dur) * sr)) if total_dur else last_end
    n_total = max(n_total, last_end, 1)

    canvas = np.zeros(n_total, dtype=np.float32)
    for info, clip in timed_clips:
        _place(canvas, dsp.to_mono(clip), int(info.get("start_sample", 0)), sr)

    if limit:
        canvas = dsp.peak_normalize(canvas, target=0.97)
        # 软限幅：tanh 式压缩，只削掉超过 0.95 的尖峰，不影响整体动态
        over = np.abs(canvas) > 0.95
        if np.any(over):
            canvas = np.where(over,
                              np.sign(canvas) * (0.95 + 0.05 * np.tanh(
                                  (np.abs(canvas) - 0.95) / 0.05)),
                              canvas).astype(np.float32)
    return np.ascontiguousarray(canvas, dtype=np.float32)


# ------------------------------------------------------------------ 一体化入口
def align_note(clip: np.ndarray, sr: int, target_dur: float,
               target_midi: float | None = None,
               source_midi: float | None = None,
               max_semitones: float = DEFAULT_MAX_SEMITONES) -> tuple[np.ndarray, dict]:
    """单个音符的完整对齐：先拉伸时长，再搬移音高。

    顺序为什么是"先时长后音高"
    --------------------------
    ``dsp.pitch_shift`` 时长不变、``dsp.time_stretch`` 音高不变，两者交换序在
    理论上等价。但 ``time_stretch`` 的瞬态保护依赖 onset 检测，而音高搬移
    会改变谱包络；反过来做会让瞬态检测的输入被扰动。故**先做时长**，
    让瞬态检测工作在"最接近原始合成音"的信号上，起音保护最可靠。

    返回 ``(片段, 诊断信息)``。诊断信息直接写进 ``note_plan.json`` 供人工核对：
    例如"这个音要求拉伸 3.8 倍"是可疑信号（超出舒适区，听感会明显失真）。
    """
    clip = dsp.to_mono(clip)
    diag: dict = {
        "src_dur": float(clip.size) / float(sr) if sr else 0.0,
        "target_dur": float(target_dur),
        "stretch_ratio": 0.0,
        "applied_semitones": 0.0,
    }
    if clip.size == 0 or target_dur <= 0:
        return np.zeros(0, dtype=np.float32), diag

    diag["stretch_ratio"] = float(target_dur) / diag["src_dur"] if diag["src_dur"] else 0.0
    out = stretch_to(clip, sr, target_dur)

    if target_midi is not None and source_midi is not None:
        out, semis = shift_to(out, sr, target_midi, source_midi,
                              max_semitones=max_semitones)
        diag["applied_semitones"] = float(semis)
    return _fit_length(out, int(round(float(target_dur) * int(sr)))), diag


def align_all(notes: list[dict], clips: list[np.ndarray], sr: int,
              bpm: float = 0.0, origin: float = 0.0, snap: bool = True,
              subdiv: int = SNAP_SUBDIV, total_dur: float | None = None,
              max_semitones: float = DEFAULT_MAX_SEMITONES,
              progress=None) -> tuple[np.ndarray, list[dict]]:
    """批量对齐 + 拼接（``singer`` 产出片段后的调用入口）。

    ``notes`` 与 ``clips`` 按下标一一对应；``clips[i]`` 是"未对齐"的合成片段，
    ``notes[i]`` 提供目标时刻/时长/音高（其 ``source_midi`` 是片段自身的音高 ——
    由合成片段实测得到，通常取该片段的 pyin 中位数，或直接用 TTS 的基准音高）。

    返回 ``(整曲音频, 逐音符诊断列表)``。诊断列表是 ``note_plan.json`` 的
    人工可校正产物：用户能看到每个音"要求什么、实际做了什么"。
    """
    timed = plan_note_timing(notes, bpm=bpm, origin=origin, snap=snap,
                             subdiv=subdiv, total_dur=total_dur)
    timed = to_sample_grid(timed, sr)

    pairs: list[tuple[dict, np.ndarray]] = []
    diags: list[dict] = []
    n = min(len(timed), len(clips))
    for i in range(n):
        info, clip = timed[i], clips[i]
        src_midi = info.get("source_midi", info.get("midi"))
        aligned, diag = align_note(
            clip, sr, target_dur=info["dur"],
            target_midi=info.get("midi"), source_midi=src_midi,
            max_semitones=max_semitones)
        d = dict(diag)
        # ⚠️ ``start`` 是**计划**时刻（秒），而 ``start_sample``/``n_samples``
        #    是**实际**落位（整数采样，经过 to_sample_grid 的单调不重叠钳制）。
        #    两者都要暴露：只看 ``start`` 无法发现"整体偏了 1 个采样"这类错误 ——
        #    它恰好会在诊断里表现为"完全正确"，而音频其实已经错位。
        #    （这是护栏自测发现的：把落点 +1 采样注入后，所有断言仍然全绿。）
        d.update({
            "index": i,
            "start": info["start"],
            "dur": info["dur"],
            "start_sample": int(info.get("start_sample", 0)),
            "n_samples": int(info.get("n_samples", 0)),
            "n_samples_actual": int(aligned.size),
            "midi": info.get("midi"),
            "lyric": info.get("lyric"),
        })
        diags.append(d)
        pairs.append((info, aligned))
        if progress is not None:
            progress(i + 1, n)

    audio = assemble(pairs, sr, total_dur=total_dur)
    return audio, diags


def summary(diags: list[dict],
            max_semitones: float = DEFAULT_MAX_SEMITONES) -> dict:
    """把逐音符诊断汇总成人类可读的统计（用于日志/UI 展示）。

    ``max_semitones`` 必须与 ``align_all`` 实际传入的值一致，否则"被钳制"的判定
    会失真：若调用方传 3.0 而这里仍拿模块默认 12.0 去比，那些被钳到 3.0 的音符
    会被报成"没被钳制"—— 一个静默的假绿。故此参数显式暴露，且不设默认陷阱：
    默认值仅为无参调用时的兜底。
    """
    if not diags:
        return {"n_notes": 0}
    ratios = np.array([d.get("stretch_ratio", 0.0) for d in diags], dtype=np.float64)
    semis = np.array([abs(d.get("applied_semitones", 0.0)) for d in diags], dtype=np.float64)
    ratios = ratios[np.isfinite(ratios)]
    lim = abs(float(max_semitones))
    return {
        "n_notes": len(diags),
        "stretch_mean": float(ratios.mean()) if ratios.size else 0.0,
        "stretch_max": float(ratios.max()) if ratios.size else 0.0,
        # 拉伸超出 [0.5, 2.0] 的音符数：听感失真风险区
        "stretch_outliers": int(np.sum((ratios < 0.5) | (ratios > 2.0))) if ratios.size else 0,
        "semitone_mean_abs": float(semis.mean()) if semis.size else 0.0,
        "semitone_max_abs": float(semis.max()) if semis.size else 0.0,
        "semitone_clamped": int(np.sum(semis >= lim - 1e-6)),
        "semitone_limit": lim,
    }
