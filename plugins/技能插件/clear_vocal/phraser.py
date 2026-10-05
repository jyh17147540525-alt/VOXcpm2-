"""清唱生成插件 · 短语级合成（修复"单音节不是有效合成单位"）
==============================================================
本模块解决一个**实测归因出来的根本问题**，不是参数调优问题。

问题（2026-09-22 实测归因，详见 docs 与 align_diag）
-----------------------------------------------------
清唱插件原设计是「逐音符送模型」：8 个音符 → 8 次独立 TTS 调用，
每次只喂 1 个汉字（~0.5 秒的孤立音节）。逐帧频谱体检显示模型在这种输入上的输出是：

    前 180ms  宽带噪声（谱质心 2400~5300Hz，谱平坦度最高 0.167）
    0.18~0.30s 谱质心从 1868 **快速下滑**到 762Hz 的乐音段
    之后       衰减

也就是「**噪声爆发 + 下滑音 + 衰减**」，**没有稳定持续基频**。

量化证据
--------
    stem 输入有声占比      0.86
    逐音符合成输出有声占比  0.27        （8 个音符只有 2 个能测到基频）
    clips 有声占比          0.22~0.50
    clips peak              0.990（触限幅）

后果链：没有稳定基频 → ``pyin`` 锁不住（它不返回低置信度，而是**饱和到搜索边界**）
→ ``aligner`` 拿着一段噪声去搬音高 → 听感"跑调"。用户报的"存在一定的偏移"
**不是移调算错，是根本没有可搬的音**。

根因
----
**VoxCPM2 是 TTS 模型，不是歌手模型。** 孤立单音节对 TTS 而言不是合法输入：
模型看不到上下文，不知道该把韵律落在哪里，于是甩一个音头就滑走。

已否证的两个方向（别再重走）
----------------------------
1. **静音修剪**：反事实实验（不动插件代码，对已有 clips 单独试修剪）显示
   有声占比确实提升（0.28→0.55、0.31→0.77），但 **0/8 片段恢复可测**；
   有的片段修剪后有声率 0.62 仍无基频 → 问题不在"静音太多"，在"有声部分没有基频"。
2. **调 cfg_value / inference_timesteps**：参数救不了"输入单位本身不合法"。

正解：短语级合成
----------------
不再逐音符送模型，而是**按短语（呼吸/乐句）分组**送整句：

    ["春", "天", "的", "花"]  →  一次合成 "春天的花"  →  短语内按音符切分对齐

这样模型能看到完整语义上下文，输出是**连续、有稳定基频的歌唱性语流**，
再在短语内部按音符边界切分，交给 ``aligner`` 做时长/音高对齐。

模块职责（只做"分组"与"切分"，不碰模型，不碰 DSP）
--------------------------------------------------
    group_notes()      音符序列 → 短语分组（按歌词标点 / 呼吸位 / 时长上限）
    slice_phrase()     短语音频 → 按音符时长比例切分（能量谷优先，避免切在元音上）

设计约束
--------
* **零模型接触**：本模块是纯函数，模型调用仍在 ``singer`` 里，保持"唯一接触模型的模块"
  这一约束不被破坏。
* **可回退**：``group_notes`` 返回空或退化分组（每组 1 个音符）时，调用方应走原路径，
  保证行为与改造前一致。
* **不 import server**：模型与锁由调用方注入（与 ``singer`` 同一约定）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

#: 短语内音符数上限。太多会让模型"读不完"（TTS 对长文本会用朗读韵律，
#: 失去歌唱性），太少则退化成单音节问题。
MAX_NOTES_PER_PHRASE = 8

#: 短语文本字符数上限。与 ``singer.MAX_CHARS_PER_NOTE``(6) 是不同量纲：
#: 那是"每个音符"，这里是"整句"。实测 8~12 字是 TTS 保持连贯又不过长的甜点区。
MAX_CHARS_PER_PHRASE = 12

#: 短语时长上限（秒）。超过则强制断开 —— 长句会让 VoxCPM2 的韵律飘移。
MAX_PHRASE_DUR = 3.2

#: 短语合成音频与音符目标总时长的**可接受比例偏离**。
#:
#: 为什么需要这个阈值（实测踩到的坑）
#: --------------------------------
#: TTS 输出时长**不可指定**。实测：音符目标总时长 2.97s 的短语「春天的花开了」，
#: 模型只输出 **2.08s**（0.70 倍）→ 切分后每段都偏短 → ``aligner`` 必须把每段
#: 拉伸 1.43~2.25 倍才能落位，而大比例拉伸会明显损坏音质（拉伸中位 2.249、
#: 越界 5 个）。
#:
#: 解法：**按实测时长反推该说几个字**，迭代 1~2 轮把比例逼近 1.0。
#: 接受窗口取 [0.85, 1.25] —— 松于 1.0 是因为"略长于目标"可由 aligner
#: 轻微压缩处理，而"短于目标"要拉伸，对音质伤害更大（拉伸会引入周期断裂）。
DUR_TOL_LO = 0.85
DUR_TOL_HI = 1.25

#: 短语时长适配的最大尝试轮数（每轮一次模型调用，必须设上限）。
MAX_DUR_PROBES = 3

#: 短语之间是否插入呼吸位。True 时相邻短语间留 ``PHRASE_BREATH`` 秒静音。
INSERT_BREATH = True
#: 短语间呼吸时长（秒）。注意：这是**拼接时**插入的静音，不是模型生成的呼吸声
#: （后者会污染音符边界，见 ``singer._synth_text`` 的 ``breath=0.0``）。
PHRASE_BREATH = 0.06

#: 当作短语边界的标点。命中即在此音符后断开。
_BREAK_CHARS = set("，。！？；、,.!?;:：…—～~")
#: 拖腔标记：带这个属性的音符不切断短语（它是前一个音的延续）。
_HOLD_KEYS = ("is_hold", "hold", "tie", "slur")


def _note_is_break(lyric: str) -> bool:
    """该音符的歌词是否以短语边界标点结尾。"""
    t = (lyric or "").strip()
    return bool(t) and t[-1] in _BREAK_CHARS


def _note_is_hold(note: dict) -> bool:
    """该音符是否为"拖腔延续"（不构成独立音节，不切断短语）。"""
    for k in _HOLD_KEYS:
        if note.get(k):
            return True
    return False


def _strip_punct(s: str) -> str:
    """去掉短语文本尾部的标点（模型不需要我们告诉它句号，标点会变成停顿）。"""
    t = (s or "").strip()
    while t and t[-1] in _BREAK_CHARS:
        t = t[:-1].rstrip()
    return t


def group_notes(notes: list[dict],
                max_notes: int = MAX_NOTES_PER_PHRASE,
                max_chars: int = MAX_CHARS_PER_PHRASE,
                max_dur: float = MAX_PHRASE_DUR) -> list[list[int]]:
    """把音符序列按"短语"分组，返回**下标列表的列表**。

    分组规则（按优先级）
    -------------------
    1. **标点边界**：音符歌词以 ``，。！？；、`` 等结尾 → 在此断开（最强信号，
       这是作词者自己标的呼吸位）；
    2. **拖腔不断开**：``is_hold`` 标记的音符是前一音的延续，不构成边界；
    3. **硬上限**：任一组达到 ``max_notes`` / ``max_chars`` / ``max_dur`` → 强制断开。

    为什么以标点为主而不是平均分组
    ------------------------------
    标点是**作词者写下的呼吸位**，天然对应"一口气能唱完的乐句"。
    平均分组会把"我爱你"切成"我爱"+"你"，既破坏语义也破坏语流；
    按标点分组则恰好落在语义完整的位置。

    返回
    ----
    ``[[0,1,2], [3,4], ...]``。空输入返回 ``[]``。
    若全是拖腔/无标点且不超上限，会返回一个含全部音符的组（正常情况）。
    """
    idx = [i for i, n in enumerate(notes or []) if not _note_is_hold(n)]
    if not idx:
        return []

    groups: list[list[int]] = []
    cur: list[int] = []
    cur_chars = 0
    cur_dur = 0.0

    for i in idx:
        n = notes[i] or {}
        lyric = str(n.get("lyric") or "")
        try:
            dur = float(n.get("dur") or 0.0)
        except Exception:
            dur = 0.0

        # 加入当前组会超上限 → 先收口（但至少保证组内有 1 个音）
        overflow = (len(cur) >= int(max_notes)
                    or (cur and cur_chars + len(lyric) > int(max_chars))
                    or (cur and cur_dur + dur > float(max_dur)))
        if overflow and cur:
            groups.append(cur)
            cur, cur_chars, cur_dur = [], 0, 0.0

        cur.append(i)
        cur_chars += len(lyric)
        cur_dur += dur

        # 标点边界 → 在此收口
        if _note_is_break(lyric) and cur:
            groups.append(cur)
            cur, cur_chars, cur_dur = [], 0, 0.0

    if cur:
        groups.append(cur)

    # 拖腔音符（hold）跟在它所属组的末尾 —— 不单独成组，也不参与分组计数。
    #
    # ⚠️ 必须按"下标相邻"归属，不能按"组尾下标 < h"扫。
    # 反例（实测踩到）：notes = [春, 天, 的(hold), 花]，非 hold 下标 [0,1,3]
    # → 分组 [[0,1,3]]，而 hold 下标 2 用"组尾 3 < 2"判断为假 → **音符 2 直接丢失**。
    # 正确语义：hold 音属于**它前面最近的那个非 hold 音**所在的组。
    holds = [i for i, n in enumerate(notes or []) if _note_is_hold(n)]
    if holds:
        for h in holds:
            owner = None
            for g in groups:
                if any(i < h for i in g):
                    owner = g
            if owner is not None:
                owner.append(h)
            else:
                # 找不到前驱（hold 出现在首位）→ 自成一组，绝不丢音
                groups.append([h])
        for g in groups:
            g.sort()
    return groups


def phrase_text(notes: list[dict], group: list[int]) -> str:
    """拼出一个短语的合成文本（去掉标点与拖腔标记的裸字串）。

    关键规则：**纯标点音符不贡献文本**。
    作词者把标点单独占一个音符（如 ``["春","天","的","花","，"]``）时，
    那个"逗号音符"是**边界标记**，不是要唱的字节 —— 若给它兜底一个 ``啊``，
    合成文本会变成"春天的花啊"，凭空多唱一个字。空歌词（真·哼鸣）才用 ``啊``。
    """
    parts: list[str] = []
    for i in group:
        n = notes[i] or {}
        raw = str(n.get("lyric") or "").strip()
        stripped = _strip_punct(raw)
        if stripped:
            parts.append(stripped)
        elif not raw:
            # 真·无歌词 → 哼鸣兜底
            parts.append("啊")
        # 纯标点（raw 非空但 stripped 为空）→ 不贡献文本
    return "".join(parts)


def slice_phrase(phrase_audio: np.ndarray, sr: int, durs: list[float],
                 sr_out: int | None = None) -> list[np.ndarray]:
    """把一句连续音频按音符时长比例切成若干段。

    为什么不能简单按比例硬切
    ------------------------
    按比例切会把刀口落在**元音中间**，切出半个音节 → ``aligner`` 拿到的是
    "半个元音"，音高检测再次失败（这正是单音节问题的变体）。
    所以真实刀口选在**该比例附近的能量谷**：音节之间必然有能量凹陷
    （辅音阻塞或换字），落在谷底对听感与后续 F0 检测都最友好。

    参数
    ----
    phrase_audio : 短语音频（1-D）
    sr           : phrase_audio 的采样率
    durs         : 每个音符的目标时长（秒），长度 = 要切出的段数
    sr_out       : 输出目标采样率（None = 保持 sr，不重采样）

    返回
    ----
    长度 == ``len(durs)`` 的片段列表。**总长度严格等于输入长度**（不丢样本）。
    """
    y = np.asarray(phrase_audio, dtype=np.float32).reshape(-1)
    n = len(y)
    k = len(durs)
    if k <= 0 or n == 0:
        return []
    if k == 1:
        return [y]

    tot = float(sum(float(d) for d in durs)) or 1.0
    # 理想刀口位置（累计比例 × 总长）
    cuts: list[int] = []
    acc = 0.0
    for d in durs[:-1]:
        acc += float(d)
        cuts.append(int(round(n * acc / tot)))
    cuts = [min(max(c, 1), n - 1) for c in cuts]
    cuts = sorted(set(cuts))

    # 能量谷微调：在每个刀口邻域内找 RMS 最小点
    #
    # ⚠️ 搜索窗不能是固定 ±30ms。反例（实测踩到）：
    #   两个 470ms 响块夹一段 60ms 静音，理想刀口恰好落在静音区**起点**，
    #   ±30ms 窗只覆盖到静音区一半，argmin 按索引顺序取到窗内最小点后
    #   仍可能停在**响块边缘**（实测第一段尾部 RMS 0.387 = 完全没切进静音）。
    #   窗口必须足够大，才能"探到"整段静音；但也不能大到切进隔壁音节。
    #   取窗宽 = 相邻两音符中较短者的 40%，上限 80ms。
    env = _rms_envelope(y, sr)
    adjusted: list[int] = []
    prev_c = 0
    for ci, c in enumerate(cuts):
        seg_len = c - prev_c
        nxt = cuts[ci + 1] if ci + 1 < len(cuts) else n - c
        win_s = min(0.080, 0.40 * (min(seg_len, nxt) / float(sr)))
        win_frames = max(1, int(win_s * sr / _HOP))
        fi = min(len(env) - 1, max(0, c // _HOP))
        lo = max(0, fi - win_frames)
        hi = min(len(env), fi + win_frames + 1)
        if hi > lo:
            adjusted.append((lo + int(np.argmin(env[lo:hi]))) * _HOP)
        else:
            adjusted.append(c)
        prev_c = c
    if len(adjusted) == len(cuts):
        cuts = sorted(set(min(max(c, 1), n - 1) for c in adjusted))

    out: list[np.ndarray] = []
    prev = 0
    for c in cuts:
        if c > prev:
            out.append(y[prev:c])
            prev = c
    out.append(y[prev:])

    # 应切 k 段；数量偏差（刀口去重导致的）用均分补齐/合并
    if len(out) != k:
        out = _rebalance(out, k, n)
    if sr_out and int(sr_out) != int(sr):
        out = [_resample_linear(a, sr, int(sr_out)) for a in out]
    return out


# ---------------------------------------------------------------- 内部工具
_HOP = 128   # 能量包络的帧移（样本）


def _rms_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """逐帧 RMS 包络（用于找能量谷）。无 scipy 依赖，纯 numpy。"""
    if y.size < _HOP * 2:
        return np.asarray([float(np.sqrt(np.mean(y ** 2) + 1e-12))], dtype=np.float32)
    n_frames = y.size // _HOP
    trimmed = y[:n_frames * _HOP].reshape(n_frames, _HOP)
    return np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1) + 1e-12).astype(np.float32)


def _rebalance(segs: list[np.ndarray], k: int, n: int) -> list[np.ndarray]:
    """把 ``segs`` 调成恰好 ``k`` 段（合并最短的 / 均分最长的），总长保持 ``n``。"""
    segs = [s for s in segs if s.size]
    if not segs:
        return [np.zeros(n, dtype=np.float32)]
    while len(segs) > k:
        # 合并最短的两段
        lens = [s.size for s in segs]
        j = int(np.argmin(lens[:-1]))
        segs[j] = np.concatenate([segs[j], segs.pop(j + 1)])
    while len(segs) < k:
        # 劈开最长的一段
        lens = [s.size for s in segs]
        j = int(np.argmax(lens))
        s = segs[j]
        half = s.size // 2
        if half < 1:
            break
        segs[j:j + 1] = [s[:half], s[half:]]
    return segs


def _resample_linear(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """线性重采样（与 ``plugin._resample`` 同策略：仅用于长度适配，非音色路径）。"""
    if sr_in == sr_out or y.size == 0:
        return np.asarray(y, dtype=np.float32)
    m = max(1, int(round(y.size * float(sr_out) / float(sr_in))))
    xp = np.linspace(0.0, 1.0, y.size, endpoint=False)
    xq = np.linspace(0.0, 1.0, m, endpoint=False)
    return np.interp(xq, xp, y).astype(np.float32)


def adjust_text_for_dur(text: str, ratio: float) -> str:
    """当短语合成长度偏离目标时，调整文本以把时长推向目标。

    ``ratio`` = 实测时长 / 目标时长。返回值是**调整后的文本**。

    策略（保守、可逆、不引入无意义字符）
    ------------------------------------
    * ``ratio < 0.85``（合成太短）→ **拉长**：
      在最后一个字后加延长号「——」。长元音会直接增加合成时长，且这是
      演唱里自然的拖腔写法，不产生额外音节。
    * ``ratio > 1.35``（合成太长）→ **缩短**：
      去掉尾部的虚词（「了」「的」「吗」「吧」等）——它们对语义贡献最低，
      也是作词时最容易安全省略的部分。
    * 窗口内 → 原样返回（不做无谓改动）。

    ⚠️ 绝不能"加字凑时长"（如重复字符）：那会**多唱一个字**，
    改变歌词内容，属于不可接受的副作用。
    """
    t = text or ""
    if not t:
        return t
    r = float(ratio)
    if r < DUR_TOL_LO:
        # 越短加越多延长号；一轮最多 2 个
        need = min(2, max(1, int(round((DUR_TOL_LO / max(0.01, r) - 1.0) * 3))))
        return t + "——" * need
    if r > 1.35:
        # 去掉尾部虚词（最多 2 个）
        weak = set("了的吗吧呢啊呀嘛哦")
        for _ in range(2):
            if len(t) > 1 and t[-1] in weak:
                t = t[:-1]
        return t
    return t


def summarize(groups: list[list[int]], notes: list[dict]) -> dict:
    """给出分组诊断（落盘用），便于确认"短语级合成"真的生效了。"""
    sizes = [len(g) for g in groups]
    return {
        "n_phrases": len(groups),
        "phrase_sizes": sizes,
        "max_phrase": max(sizes) if sizes else 0,
        "min_phrase": min(sizes) if sizes else 0,
        "n_singletons": sum(1 for s in sizes if s == 1),
        "texts": [phrase_text(notes, g) for g in groups],
    }
