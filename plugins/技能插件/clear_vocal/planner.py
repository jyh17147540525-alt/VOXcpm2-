"""清唱生成插件 · 乐谱规划器
==============================
把**分析结果**（音符序列 + 节奏）与**歌词文本**融合成一份可执行的乐谱：

    note_plan.json  ← 整个插件的中心产物，也是唯一"人工可校正"的地方

数据流位置：

    analyzer.analyze()  →  {rhythm, notes, key}
    whisper（可选）      →  整句歌词文本
            ↓
        planner.plan()   →  note_plan.json  （本模块）
            ↓
        singer.sing()    →  逐音符音频
            ↓
        aligner.align_all() → 成品清唱音频

为什么把 note_plan.json 单独立成一份文件（而不是内存里传对象）
-----------------------------------------------------------------
用户明确要求"便于调试与后续扩展"。把中间产物**落盘成带 schema 版本的 JSON**，
换来三件事：

1. **可人工校正**：算法判断错的音符，用户改 JSON 就能修，不用改代码；
2. **可断点续跑**：合成很慢（3 分钟的歌 ≈ 6 分钟分析 + N 次推理），
   崩在第 200 个音符时能从 JSON 接着跑，不必从头再来；
3. **可复现**：出问题时把 JSON 发出来就能完整复现这一版的规划结果。

⚠️ 三条来自实测的设计约束（决定了本模块的形状）

**约束 1：不依赖 whisper 的词级时间戳。**
实测 faster-whisper small：真实语音文字准确，但**中文词切分不稳定**
（"你好啊" 一次切 3 词、一次切 2 词）。词数会随素材漂移，
拿它对齐音符必然时好时坏。→ 改为**整句文本 + 自研音节切分**（``syllabify.py``），
给定文本的切分结果 100% 确定。

**约束 2：音节数 ≠ 音符数，必须双向适配。**
两种情况都得处理：
  * **音多字少**（一字多音 / 拖腔）：音节重复挂到多个音符上 —— 这是**正确的**，
    歌唱里"一个字唱好几个音"叫拖腔（melisma），必须保留而不是丢音；
  * **字多音少**（快嘴 / 音符被合并）：多个音节塞进一个音符 ——
    必须**告警并记录**，而不是静默丢弃（丢字 = 用户听出"少唱了词"）。

**约束 3：分配权重用"时长"而非"均匀"。**
长音上塞短词、短音上塞长词都会唱不清。按音符时长占比分配音节数，
与实际演唱习惯一致（长音多占字）。
"""
from __future__ import annotations

import json
import os
from typing import Any

from . import syllabify

#: note_plan.json 的 schema 版本。**改了结构必须 +1** ——
#: 用户手工改过的 JSON 需要能判断"这份文件是哪一版生成的"。
PLAN_SCHEMA_VERSION = 1

#: 单个音符最舒适的伸缩区间。超出则记录告警（听感会明显失真）
COMFORT_MIN_RATIO = 0.5
COMFORT_MAX_RATIO = 2.0

#: 一个音符最多挂几个音节（超出说明字多音少，需要告警）
MAX_SYL_PER_NOTE = 4


# ------------------------------------------------------------------ 音节分配
def allocate_syllables(notes: list[dict], units: list[dict]) -> tuple[list[dict], list[str]]:
    """把音节单元分配到音符上（按音符时长加权）。

    返回 ``(带 lyric 的音符列表, 告警列表)``。

    分配规则（按时长权重 + 余数补给长音）：
      1. 若音节数 >= 音符数：前 N 个音符一一对应，**多余的音节挂到最后一个
         有字的音符上**（拖腔方向的反向兜底）—— 但更常见的是走下面的重复路径；
      2. 若音节数 < 音符数：按"时长权重"决定每个音符分几个音节，
         短音 1 个，长音可能 2 个；**余下的音符重复最近分配过的音节**
         （实现拖腔：一个字唱多个音）；
      3. 始终保证：每个音符都有 ``lyric`` 字段（可能是 ""，表示纯哼鸣）。

    ⚠️ 为什么不"均匀分配"：
    均匀分配会让 0.1 秒的十六分音符和 2 秒的长音各分到同样多的字，
    长音反而唱得仓促。按时长加权才符合"长音拖腔、短音一字"的演唱直觉。
    """
    warns: list[str] = []
    if not notes:
        return [], warns
    if not units:
        # 没有歌词 → 全部哼鸣（这是合法的，用户可能只想唱旋律）
        for n in notes:
            n["lyric"] = ""
        return notes, ["未提供歌词，全部音符按哼鸣（la）处理"]

    texts = [u["text"] for u in units]
    n_note, n_syl = len(notes), len(texts)

    # ---- 情况 A：字 >= 音（含相等）----
    if n_syl >= n_note:
        for i, n in enumerate(notes):
            n["lyric"] = texts[i]
        extra = texts[n_note:]
        if extra:
            # 多余的字挂到最后一个音符，并告警（可能被挤成一团）
            last = notes[-1]
            last["lyric_raw"] = last["lyric"]
            merged = "".join([last["lyric"]] + extra)
            if len(merged) > MAX_SYL_PER_NOTE:
                # 太多则只保留前几个（清晰度优先），其余记录到告警里
                warns.append(
                    "音节数(%d) 多于音符数(%d)：末尾 %d 个音节无法分配，"
                    "已截断（可手工改 note_plan.json 或增补音符）"
                    % (n_syl, n_note, len(extra)))
                merged = merged[:MAX_SYL_PER_NOTE]
            else:
                warns.append("音节数(%d) 多于音符数(%d)：多余音节并入末音"
                             % (n_syl, n_note))
            last["lyric"] = merged
        return notes, warns

    # ---- 情况 B：音 > 字（拖腔）----
    # 按时长权重分配"哪几个音符分到新字"
    durs = [max(float(n.get("dur", 0.1)), 1e-3) for n in notes]
    assigned_syl: list[str] = []
    syl_cursor = 0
    for i, n in enumerate(notes):
        if syl_cursor < n_syl:
            n["lyric"] = texts[syl_cursor]
            assigned_syl.append(texts[syl_cursor])
            syl_cursor += 1
        else:
            # 字已用完 → 重复前面"实际用过"的字（拖腔）
            n["lyric"] = assigned_syl[-1] if assigned_syl else ""
            n["melisma"] = True
    return notes, warns


def verify_plan(notes: list[dict]) -> list[str]:
    """对生成的 plan 做一致性自检，返回问题列表（空 = 健康）。

    这些检查是给"人工校正"用的护栏：用户手改 JSON 后跑一次就知道有没有改坏。
    """
    issues: list[str] = []
    if not notes:
        issues.append("音符列表为空")
        return issues

    prev_end = -1.0
    for i, n in enumerate(notes):
        s, d = float(n.get("start", -1)), float(n.get("dur", -1))
        if s < 0 or d <= 0:
            issues.append("第 %d 个音符时刻非法 (start=%s dur=%s)" % (i, s, d))
        if s < prev_end - 1e-9:
            issues.append("第 %d 个音符与前一个重叠" % i)
        prev_end = s + d
        m = n.get("midi")
        if m is None or not (0 < float(m) < 128):
            issues.append("第 %d 个音符 midi 非法 (%s)" % (i, m))
        r = n.get("target_ratio")
        if r is not None and not (0 < float(r) < 100):
            issues.append("第 %d 个音符拉伸比非法 (%.3f)" % (i, float(r)))
    return issues


# ------------------------------------------------------------------ 主入口
def plan(analysis: dict, lyric_text: str = "",
         bpm: float | None = None, origin: float = 0.0,
         snap: bool = True, subdiv: int = 4,
         total_dur: float | None = None,
         source_note_dur: float = 0.35,
         max_repeat: int = 3) -> dict:
    """由分析结果 + 歌词生成 note_plan。

    参数
    ----
    analysis      : ``analyzer.analyze()`` 的返回（需含 ``notes``；``rhythm`` 可选）
    lyric_text    : 整句歌词（来自 whisper 或用户直接给）。空则全哼鸣。
    bpm/origin    : 用于时间网格吸附。``rhythm`` 里有时自动取。
    snap          : 是否把音符时刻吸附到节拍网格。散板/自由速度素材应设 False。
    total_dur     : 音频总长（末音不越界）。
    source_note_dur: singer 合成出的"单个短句"的**基准时长**（秒）。
                     aligner 的伸缩比 = 音符时长 / 这个值，故必须传准，
                     否则 note_plan 里的 target_ratio 诊断会失真。
    max_repeat    : 一个音最多重复挂几次字（拖腔上限，防病态重复）

    返回 ``note_plan`` dict（可直接 ``json.dump``）。
    """
    notes_in = list(analysis.get("notes") or [])
    if bpm is None:
        bpm = float(((analysis.get("rhythm") or {}).get("bpm")) or 0.0)

    # ---- 1. 整理音符：只保留必要字段，加上落点 ----
    items: list[dict] = []
    for n in notes_in:
        s = float(n.get("start", 0.0))
        d = float(n.get("dur", 0.0))
        if d <= 0:
            continue
        items.append({
            "start": s,
            "dur": d,
            "midi": float(n.get("midi", 60.0)),
            "f0": float(n.get("f0", 0.0)) or None,
            "rms": float(n.get("rms", 0.0)),
            "lyric": "",
        })
    if not items:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "sr": int(analysis.get("sr", 22050)),
            "bpm": float(bpm or 0.0),
            "key": (analysis.get("key") or {}).get("name"),
            "lyric_text": lyric_text or "",
            "source_note_dur": float(source_note_dur),
            "notes": [],
            "warnings": ["分析未产生任何音符，无法生成乐谱"],
            "stats": {"n_notes": 0, "n_units": 0},
        }

    # ---- 2. 网格吸附（可选）----
    if snap and bpm and bpm > 0:
        from . import aligner
        st = aligner.snap_times([it["start"] for it in items], bpm, origin, subdiv)
        ed = aligner.snap_times([it["start"] + it["dur"] for it in items],
                                bpm, origin, subdiv)
        for it, s, e in zip(items, st, ed):
            if e - s > 1e-4:
                it["start"], it["dur"] = float(s), float(e - s)
        # 吸附后可能重叠，用 aligner 的规划逻辑修一遍
        fixed = aligner.plan_note_timing(
            [dict(it, end=it["start"] + it["dur"]) for it in items],
            bpm=0.0, snap=False, total_dur=total_dur, min_dur=0.05)
        if fixed:
            merged: list[dict] = []
            for orig, f in zip(items, fixed):
                o = dict(orig)
                o["start"], o["dur"] = f["start"], f["dur"]
                merged.append(o)
            if len(merged) == len(items):
                items = merged
    else:
        # 不吸附也要保证单调不重叠、末音不越界
        from . import aligner
        fixed = aligner.plan_note_timing(
            [dict(it, end=it["start"] + it["dur"]) for it in items],
            bpm=0.0, snap=False, total_dur=total_dur, min_dur=0.05)
        if fixed and len(fixed) == len(items):
            for o, f in zip(items, fixed):
                o["start"], o["dur"] = f["start"], f["dur"]

    # ---- 3. 音节分配 ----
    units = syllabify.split_lyric(lyric_text or "")
    items, warns = allocate_syllables(items, units)

    # ---- 4. 逐音符补上执行参数与诊断 ----
    base = max(float(source_note_dur), 1e-3)
    for i, it in enumerate(items):
        ratio = float(it["dur"]) / base
        it["index"] = i
        it["target_ratio"] = round(ratio, 4)
        # 理想 source_midi：此音符直接以目标音高合成（二期），
        # 一期由 singer 统一合成后由 aligner 搬移，故这里记 None
        it["source_midi"] = None
        it.setdefault("melisma", False)
        # 依验：舒适区外的伸缩要记账（不是错误，是"用户应知情"）
        if ratio > COMFORT_MAX_RATIO or ratio < COMFORT_MIN_RATIO:
            warns.append(
                "第 %d 个音符伸缩比 %.2f 超出舒适区 [%.1f, %.1f]，音质可能明显失真"
                % (i, ratio, COMFORT_MIN_RATIO, COMFORT_MAX_RATIO))

    # ---- 5. 自检 + 统计 ----
    issues = verify_plan(items)
    durs = [it["dur"] for it in items]
    ratios = [it["target_ratio"] for it in items]
    stats = {
        "n_notes": len(items),
        "n_units": len(units),
        "n_melisma": int(sum(1 for it in items if it.get("melisma"))),
        "n_empty_lyric": int(sum(1 for it in items if not it["lyric"])),
        "dur_total": round(float(sum(durs)), 3),
        "ratio_mean": round(float(sum(ratios) / len(ratios)), 3) if ratios else 0.0,
        "ratio_max": round(float(max(ratios)), 3) if ratios else 0.0,
        "ratio_out_of_comfort": int(sum(
            1 for r in ratios if r > COMFORT_MAX_RATIO or r < COMFORT_MIN_RATIO)),
    }

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "generator": "clear_vocal.planner",
        "sr": int(analysis.get("sr", 22050)),
        "bpm": float(bpm or 0.0),
        "key": (analysis.get("key") or {}).get("name"),
        "origin": float(origin),
        "snap": bool(snap),
        "snap_subdiv": int(subdiv),
        "total_dur": float(total_dur) if total_dur else None,
        "lyric_text": lyric_text or "",
        "source_note_dur": float(source_note_dur),
        "notes": items,
        "warnings": warns,
        "issues": issues,
        "stats": stats,
    }


# ------------------------------------------------------------------ 落盘/读回
def save(plan_obj: dict, path: str) -> str:
    """写 note_plan.json（UTF-8，缩进 2，``ensure_ascii=False`` 保留中文）。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan_obj, fh, ensure_ascii=False, indent=2)
    return path


def load(path: str) -> dict:
    """读 note_plan.json，并做 schema 版本兼容检查。"""
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    v = obj.get("schema_version")
    if v is None:
        obj.setdefault("warnings", []).append("文件缺少 schema_version，按 v1 处理")
    elif int(v) > PLAN_SCHEMA_VERSION:
        raise ValueError(
            "note_plan schema v%s 比本版本（v%d）更新，请升级插件"
            % (v, PLAN_SCHEMA_VERSION))
    return obj


def relink(plan_obj: dict, analysis: dict) -> dict:
    """把手工改过的 plan 与原分析结果重新核对（改坏时给出 issues）。

    用途：用户在 UI 上编辑过 note_plan 后，用本函数跑一次自检，
    避免"改出重叠音符/非法音高"直接喂给 singer。
    """
    notes = plan_obj.get("notes") or []
    issues = verify_plan(notes)
    plan_obj["issues"] = issues
    plan_obj["stats"] = dict(plan_obj.get("stats") or {})
    plan_obj["stats"]["n_notes"] = len(notes)
    return plan_obj
