"""清唱生成插件 · 编排层（插件入口）
======================================
把六个纯模块串成一条**可调试、可中断、可复现**的流水线：

    ┌── 输入 ────────────────────────────────────────────────┐
    │  原曲音频（含伴奏）  +  歌词文本  +  目标音色参考音频      │
    └───────────────────────┬────────────────────────────────┘
                            │
     ① 分离人声 stem  ← 由调用方（server/CLI）用既有的 MDX 分离器完成，
                        本插件**不重复实现**，只接收 stems 目录或人声数组
                            ▼
     ①b 没有歌词且 auto_lyric=True 时，用 whisper 识出**整句歌词**
        （transcribe_lyric）—— 只取文字，不要词级时间戳。失败一律降级
        为哼鸣，绝不让整条流水线崩掉。
                            ▼
     ② analyzer.analyze(人声 stem)
        → rhythm(bpm/beats) + f0 + notes(音高/时长/时刻) + key
                            ▼
     ③ planner.plan(analysis, lyric_text, ...)  +  syllabify.split_lyric(text)
        → note_plan.json（每个音符挂哪个字、目标时刻/时长/音高、拖腔标记）
                            ▼
     ④ singer.Singer.sing_plan(plan)          ⚠️ 必须在钩子外、在锁内串行
        → clips[i]（每音符一小段"唱出来的"音频，音色全程锚定同一参考）
                            ▼
     ⑤ singer.Singer.measure_clips(clips)
        → 实测每个片段自身的音高，回填 notes[i]['source_midi']
          （这一步是"整首歌别被整体移调一个八度"的保险丝）
                            ▼
     ⑤b **整体八度折叠**：把乐谱平移整数个八度，使平均搬移量最小。
        整数八度平移**不改变任何音程**（旋律保真），但能把残留搬移从
        ~12 半音压到 ~2.5 半音 —— 否则音色会明显失真、且会贴满钳制上限。
                            ▼
     ⑥ aligner.align_all(notes, clips, sr)
        → 精确对齐 + 拼接：时长误差 **0 采样**，音高按目标搬移
                            ▼
     ⑦ 输出 wav + 诊断 JSON（每音符"要求什么 / 实际做了什么"）

约定与边界
----------
* **只在钩子外调用**：``run()`` 会触发模型合成，绝不能从钩子里进来
  （``server._infer_lock`` 非可重入 → 静默死锁）。本模块的钩子处理器
  一律**只读状态 / 注册路由**，不做任何合成。
* **不碰全局状态**：不 import ``server``，模型与锁由调用方注入，
  因此可以被 CLI、单元测试、HTTP 路由以完全相同的方式驱动。
* **中间产物全部落盘**：``analysis.json`` / ``note_plan.json`` /
  ``clips/*.wav`` / ``align_diag.json``，出问题时能精确知道是哪一步坏了。

对应关系（谁负责什么）
----------------------
    dsp.py       底层 DSP（STFT / 变速 / 变调），无音乐语义
    analyzer.py  音频 → 音乐语义（节拍 / 音高 / 音符 / 调式）
    syllabify.py 歌词文本 → 音节单元（替代不可靠的 whisper 词边界）
    planner.py   语义 + 歌词 → note_plan（谁唱哪个字、多长、多高）
    singer.py    语义 → 音频（唯一接触模型的模块）
    aligner.py   音频 → 精确落位（零漂移对齐 + 拼接）
    plugin.py    编排 + 对外接口（本文件）
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import numpy as np

#: 输出文件名约定（保持稳定，便于 UI / 脚本消费）
ANALYSIS_NAME = "analysis.json"
PLAN_NAME = "note_plan.json"
DIAG_NAME = "align_diag.json"
SUMMARY_NAME = "run_summary.json"
CLIPS_DIRNAME = "clips"
OUTPUT_NAME = "clear_vocal.wav"

#: 模块级运行态：仅用于内省/UI 展示，不参与算法
_STATE: dict[str, Any] = {
    "last_run": None,        # 最近一次 run() 的摘要
    "n_runs": 0,
    "n_failed": 0,
    "ready": False,
    "tmp_dir": "",
}


class ClearVocalError(RuntimeError):
    """编排层可预期的失败（如没有人声、没有音符）。详细原因写在消息里。"""


# ============================================================ 小工具
def _dumps(obj: Any, path: str) -> str:
    """把可能含 numpy 的对象写成 JSON（数组转 list），保持中文可读。"""
    def _conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(v) for v in o]
        return o

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_conv(obj), fh, ensure_ascii=False, indent=2)
    return path


def _write_wav(path: str, y: np.ndarray, sr: int) -> str:
    """写 16-bit PCM wav（与 singer 内部一致，避免额外的 soundfile 依赖）。"""
    import wave as _wave

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    x = np.asarray(y, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with _wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def _progress_printer(tag: str) -> Callable[[int, int, Any], None]:
    """默认进度回调：每 10% 打一行（几百个音符时避免刷屏）。"""
    last = {"pct": -1}

    def cb(done: int, total: int, item: Any = None) -> None:
        total = max(1, int(total))
        pct = int(done * 100 / total)
        if pct >= last["pct"] + 10 or done >= total:
            last["pct"] = pct
            print("[clear_vocal] %s %d/%d (%d%%)" % (tag, done, total, pct), flush=True)

    return cb


#: 转写整首曲子的等待上限（秒）。whisper 本身对 10 分钟音频约需数十秒，
#: 这里给足余量；超时视为失败，避免把整个请求线程永久挂住。
TRANSCRIBE_TIMEOUT = 900.0


def transcribe_lyric(vocal_wav: str,
                     timeout: float = TRANSCRIBE_TIMEOUT,
                     logger: Callable[[str], None] | None = None) -> dict:
    """用既有的 whisper 转写拿到**整句歌词文本**（不取词级时间戳）。

    为什么只要文本、不要时间戳
    --------------------------
    ``voice_clone/transcriber`` 已实现 ``word_timestamps=True`` 的词级时间戳，
    但那是给**说话**用的。我在本机实测：**歌声**的词边界极不稳定 —— 逐字时间戳
    会漂到相邻音符上。因此本插件按 ``syllabify.split_lyric`` 把整句切音节、
    再按音符时长分配（见 ``syllabify.py`` 开头的实测结论）。

    所以这里只用 whisper 最强的那部分（**文字本身是准的**），
    把它的弱项（中文词切分/时间戳）留给 syllabify 处理。

    为什么要写这个同步包装
    ----------------------
    ``start_transcribe`` 是**后台线程 + 轮询 job** 的异步 API。本函数把它
    收敛成"等结果"的同步调用，供 CLI / 离线脚本使用。

    ⚠️ **不要在 HTTP 请求线程里直接调它** —— 会阻塞事件循环最长 ``timeout`` 秒。
    服务侧要暴露"自动识歌词"能力时，应当复用同一套 job 轮询语义，把
    轮询交给前端（前端本来就在轮询 ``/api/transcribe/status``）。

    返回
    ----
    dict: ``{ok, text, lang, n_segments, error}``。``ok=False`` 时 ``text`` 为空
    字符串，**调用方应当照常继续**（退化成整首哼鸣），而不是让整条流水线失败 ——
    歌词缺失是降级，不是错误。

    ``error`` 一律带上**失败类别**（``transcriber 不可用 / 启动转写失败 /
    转写失败 / 转写超时``）再跟具体原因。只留裸原因（如 "已有转写任务在运行"）
    会让排障时看不出究竟是"环境缺依赖"还是"并发撞车"还是"模型炸了" ——
    这三者的处置方式完全不同。
    """
    out = {"ok": False, "text": "", "lang": None, "n_segments": 0, "error": None}

    def _log(msg: str) -> None:
        if logger is not None:
            try:
                logger(msg)
            except Exception:
                pass

    try:
        from voice_clone import transcriber as _tr
        # 仅在**没有**替身模块时才碰真 transcriber 的全局状态：
        # 未初始化的真 transcriber 会在 start_transcribe 里抛 RuntimeError，
        # 但那属于环境问题，提前探明能让报错类别更准确。
        if getattr(_tr, "JOBS_ROOT", "sentinel") is None:
            out["error"] = "transcriber 不可用：未初始化（JOBS_ROOT 为空）"
            _log("[clear_vocal] 自动识歌词不可用（transcriber 未初始化），将退化为哼鸣")
            return out
    except Exception as e:                       # pragma: no cover - 环境缺依赖
        out["error"] = "transcriber 不可用：%s" % e
        _log("[clear_vocal] 自动识歌词不可用（%s），将退化为哼鸣" % e)
        return out

    try:
        job = _tr.start_transcribe(vocal_wav, label="clear_vocal_lyric")
    except Exception as e:
        out["error"] = "启动转写失败：%s" % e
        _log("[clear_vocal] 启动转写失败（%s），将退化为哼鸣" % e)
        return out

    jid = job.get("job_id")
    t0 = time.time()
    while True:
        j = _tr.get_job(jid) or {}
        st = j.get("status")
        if st == "done":
            break
        if st == "error":
            out["error"] = "转写失败：%s" % (j.get("error") or "未知原因")
            _log("[clear_vocal] 转写失败（%s），将退化为哼鸣" % out["error"])
            return out
        if time.time() - t0 > float(timeout):
            out["error"] = "转写超时（%.0fs）" % float(timeout)
            _log("[clear_vocal] %s，将退化为哼鸣" % out["error"])
            return out
        time.sleep(0.2)

    segs = j.get("segments") or []
    out["lang"] = j.get("lang")
    out["n_segments"] = len(segs)
    # 按时间顺序拼回整句；段间用逗号而不是空格，让 syllabify 知道这里有停顿
    text = "，".join((s.get("text") or "").strip()
                     for s in sorted(segs, key=lambda x: x.get("start", 0.0))
                     if (s.get("text") or "").strip())
    out["text"] = text
    out["ok"] = bool(text)
    if not text:
        out["error"] = "未识别到歌词"
        _log("[clear_vocal] 未识别到歌词，将退化为哼鸣")
    else:
        _log("[clear_vocal] 自动识歌词：%d 段 / %d 字（%s）"
             % (len(segs), len(text), out["lang"] or "?"))
    return out


# ============================================================ 主流程
def run(model, sr: int, reference_wav: str,
        vocal_audio: np.ndarray | str | None = None,
        lyric_text: str = "",
        out_dir: str | None = None,
        vocal_sr: int | None = None,
        lock=None,
        prompt_wav: str | None = None,
        prompt_text: str | None = None,
        emotion: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        bpm: float | None = None,
        snap: bool = True,
        calibrate: bool = True,
        keep_clips: bool = True,
        max_semitones: float = 12.0,
        auto_lyric: bool = False,
        progress: Callable[[str, int, int, Any], None] | None = None,
        logger: Callable[[str], None] | None = None) -> dict:
    """跑完整条清唱流水线，返回摘要 dict（含所有中间产物的路径）。

    参数
    ----
    model          : VoxCPM2 模型实例（调用方持有；本函数只在合成瞬间加锁）
    sr             : 合成采样率 ``sr_tts``
    reference_wav  : **目标音色**参考音频（音色锁定的关键，全程复用同一个）
    vocal_audio    : 原曲的**人声 stem**。可以是 ndarray，也可以是 wav 路径。
                     注意：不要传混音原曲 —— 伴奏会污染节拍/音高估计
                     （实测：纯净 120 BPM 信号叠加持续音后，librosa 给出 92.29 BPM）。
    lyric_text     : 歌词整句。空则整首用哼鸣（"啊"）。
    out_dir        : 输出目录；None 时用系统临时目录下带时间戳的子目录
    vocal_sr       : 人声 stem 的采样率（内部会重采样到分析用的 22050）
    lock           : ``server._infer_lock``；传了就只在推理瞬间加锁
    bpm/snap       : 手工指定 BPM / 是否吸附到节拍网格（散板素材应 snap=False）
    calibrate      : 是否实测"单次短句合成"的基准时长（多花几次合成，更准）
    max_semitones  : 单音符允许的搬移上限（超过即钳制并计入诊断）。
                     正常路径下由**整体八度折叠**把搬移压到 ±5 以内，
                     这里只是最后一道防线 —— 一旦触发就说明上游有问题。
    auto_lyric     : ``lyric_text`` 为空时，是否用 whisper 自动识别歌词。
                     默认 **False**：转写要几百毫秒到数十秒，且**必须在
                     HTTP 请求线程之外**跑（见 ``transcribe_lyric`` 的告警）。
                     CLI / 离线脚本可以打开它图省事。
    progress       : 可选 ``(stage, done, total, item)`` 回调，用于 UI 进度条
    logger         : 可选日志函数

    返回
    ----
    dict: ``{ok, out_wav, analysis_path, plan_path, diag_path, summary,
             n_notes, n_degraded, octave_fold, n_pitch_clamped, errors, elapsed}``
    """
    from . import aligner as _al
    from . import analyzer as _an
    from . import planner as _pl
    from . import singer as _sg

    log = logger or (lambda m: print("[clear_vocal] " + str(m), flush=True))
    t0 = time.time()

    def _prog(stage: str):
        if progress is None:
            return None

        def cb(done: int, total: int, item: Any = None) -> None:
            try:
                progress(stage, done, total, item)
            except Exception:
                pass

        return cb

    # ---------- 输出目录 ----------
    if not out_dir:
        out_dir = os.path.join(
            os.path.abspath(os.environ.get("TEMP") or os.getcwd()),
            "clear_vocal_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    clips_dir = os.path.join(out_dir, CLIPS_DIRNAME)

    errors: list[str] = []

    # ---------- ② 读人声 ----------
    if vocal_audio is None:
        raise ClearVocalError(
            "缺少人声 stem：请先用既有的分离器（MDX / 人声分离）从原曲取出人声轨，"
            "再交给本插件。整个的混音原曲会让节拍与音高估计严重失真。")
    if isinstance(vocal_audio, str):
        y, in_sr = _sg._read_wav(vocal_audio)
        vocal_sr = int(vocal_sr or in_sr)
    else:
        y = np.asarray(vocal_audio, dtype=np.float32).reshape(-1)
        vocal_sr = int(vocal_sr or sr)
    if y.size == 0:
        raise ClearVocalError("人声 stem 为空音频。")

    # 统一到分析采样率：分析精度足够且比 44.1k 快一倍
    y_an = y if vocal_sr == _an.DEFAULT_SR else _resample(y, vocal_sr, _an.DEFAULT_SR)

    # ---------- ③ 分析 ----------
    log("① 分析人声：节拍 / 音高 / 音符 / 调式 …")
    analysis = _an.analyze(y_an, _an.DEFAULT_SR)
    notes = list(analysis.get("notes") or [])
    rhythm = analysis.get("rhythm") or {}
    log("    BPM≈%.2f，音符 %d 个，调式 %s"
        % (float(rhythm.get("bpm") or 0.0), len(notes),
           (analysis.get("key") or {}).get("name")))
    if not notes:
        _dumps(analysis, os.path.join(out_dir, ANALYSIS_NAME))
        raise ClearVocalError(
            "分析没有产生任何音符。常见原因：传入的是伴奏轨而非人声；"
            "或人声过弱/过噪。中间结果已落盘：%s" % os.path.join(out_dir, ANALYSIS_NAME))
    _dumps(analysis, os.path.join(out_dir, ANALYSIS_NAME))

    # ---------- ③b 可选：自动识歌词 ----------
    # 放在分析之后、规划之前，因为转写需要一段 wav 落地（whisper 要读文件）。
    # ⚠️ 转写是**同步阻塞**的，见 transcribe_lyric 的告警 —— 只在离线场景打开。
    if auto_lyric and not (lyric_text or "").strip():
        log("①b 未提供歌词，尝试自动识别…")
        job_dir = os.path.join(out_dir, "lyric")
        os.makedirs(job_dir, exist_ok=True)
        vwav = os.path.join(job_dir, "vocal_for_lyric.wav")
        try:
            _write_wav(vwav, y, vocal_sr)
            tr = transcribe_lyric(vwav, logger=log)
            if tr["ok"]:
                lyric_text = tr["text"]
                analysis["auto_lyric"] = {
                    "lang": tr["lang"], "n_segments": tr["n_segments"]}
            else:
                errors.append("自动识歌词失败（%s），已退化为哼鸣" % tr["error"])
        except Exception as e:
            errors.append("自动识歌词异常（%s），已退化为哼鸣" % e)

    # ---------- ④ 规划 ----------
    log("② 规划乐谱：分配歌词音节…")
    src_note_dur = _pl.DEFAULT_SOURCE_NOTE_DUR if hasattr(_pl, "DEFAULT_SOURCE_NOTE_DUR") else 0.35
    singer = _sg.Singer(
        model, sr, lock=lock, reference_wav=reference_wav,
        prompt_wav=prompt_wav, prompt_text=prompt_text, emotion=emotion,
        cfg_value=cfg_value, inference_timesteps=inference_timesteps,
        tmp_dir=clips_dir if keep_clips else None, keep_tmp=keep_clips)

    # 校准：实测"一次短句合成"的典型时长。TTS 时长随音色而变，
    # 写死会让 note_plan 的 target_ratio 诊断失真。
    # ⚠️ 这一步会真的调用模型，所以必须同样遵循"钩子外"约束。
    if calibrate:
        try:
            src_note_dur = singer.calibrate_source_dur()
            log("    实测单句基准时长 %.3f s" % src_note_dur)
        except _sg.HookContextError:
            raise
        except Exception as e:
            errors.append("calibrate 失败（用默认 %.2fs）：%s" % (src_note_dur, e))

    total_dur = float(analysis.get("duration") or (len(y_an) / _an.DEFAULT_SR))
    plan_obj = _pl.plan(
        analysis, lyric_text=lyric_text, bpm=bpm, snap=snap,
        subdiv=_al.SNAP_SUBDIV, total_dur=total_dur,
        source_note_dur=src_note_dur)
    plan_notes = list(plan_obj.get("notes") or [])
    for w in (plan_obj.get("warnings") or []):
        log("    ⚠ " + str(w))
        errors.append("plan: " + str(w))
    plan_path = _pl.save(plan_obj, os.path.join(out_dir, PLAN_NAME))
    log("    乐谱 %d 个音符（歌词 %d 字）→ %s"
        % (len(plan_notes), len(lyric_text or ""), os.path.basename(plan_path)))

    # ---------- ⑤ 逐音符合成 ----------
    log("③ 逐音符合成（%d 个音符，音色锚定同一参考）…" % len(plan_notes))
    cb = _prog("singer")
    clips, sdiags = singer.sing_plan(
        plan_obj, progress=cb or _progress_printer("singer"))
    n_degraded = int(sum(1 for d in sdiags if d.get("degraded")))
    if n_degraded:
        log("    ⚠ %d 个音符降级为静音（详见 diag）" % n_degraded)
        errors.append("%d 个音符合成失败已静音占位" % n_degraded)
    if not clips:
        raise ClearVocalError("合成没有产出任何片段。")

    if keep_clips:
        try:
            singer.dump_clips(clips, clips_dir)
            log("    片段已落盘：%s" % clips_dir)
        except Exception as e:
            errors.append("dump_clips 失败：%s" % e)

    # ---------- ⑥ 确定每个片段的"自身音高"（旋律搬移的基准） ----------
    #
    # 为什么不能逐个片段去测（实跑得出的结论）
    # ----------------------------------------
    # 直觉做法是"对每个合成片段跑 pyin，测它实际唱在哪个音高"。实测**不可行**：
    # 8 个真实片段得到的中位数是 ``[40.00, 40.00, 40.15, 47.95, 无, 84.50, 49.60, 85.40]``
    # —— 多个读数**钉死在 pyin 搜索边界**（40 = fmin，84~86 = fmax），
    # 还有 1 个完全没有 voiced 帧。原因：TTS 输出的是**短促（~0.6s）非稳态语音**，
    # pyin 在这种素材上不返回"低置信度"，而是直接饱和 —— 看起来像正常数值。
    #
    # 正确做法：**锚定参考音色自身的音域**
    # ------------------------------------
    # 同一参考 wav 合成的所有片段，都唱在该音色的固有音域附近（TTS 不按文本变调）。
    # 所以只在**参考音频**（长、稳态、pyin 可靠）上测一次基准音高，
    # 所有音符共用这个基准 —— 比逐片段测更稳，也更符合音乐语义：
    # 整首歌相对"这个嗓音的自然音高"整体移调，而不是每个音各自乱搬。
    log("④ 确定搬移基准：实测参考音色自身音域（每片段测 pyin 不可靠）…")
    anchor = None
    if reference_wav:
        try:
            ry, rsr = _sg._read_wav(reference_wav)
            anchor = singer.median_midi_of(ry, rsr, default=None)
        except Exception as e:
            errors.append("参考音色测音高失败：%s" % e)
    if anchor is None:
        # 测不出参考音色 → 整首不搬移。绝不猜：搬错八度比不搬更糟。
        anchor = None
        log("    ⚠ 无法确定基准音高 → 本曲不做音高搬移（保持原样，最安全）")
    else:
        log("    参考音色基准音高 ≈ %.2f MIDI" % anchor)

    # ---------- ⑥b 整体八度折叠（把乐谱对齐到这条嗓子的音域）----------
    #
    # 为什么必须折叠，而不是简单调高限幅
    # --------------------------------
    # 锚点一确定就能看出：若乐谱落在 C4-G4 而参考音色唱 D3，逐个音符的搬移量会是
    # +9 ~ +16 半音。**即便限幅器不拦、音高精确到位，听感也是错的** —— 那是把低沉的
    # 胸声硬拽进高音区，必然发抖、发紧。所以问题不在限幅器，而在"乐谱没为这条嗓子
    # 考虑音域"。
    #
    # 解法：把整个乐谱**平移整数个八度**，使平均搬移量最小。
    # 关键性质：整数八度平移**不改变任何音程** —— 旋律（音与音之间的距离）逐音符
    # 完全保真，只改"整体定调"。这是唯一能同时满足"旋律一致"与"自然度"的操作。
    #
    # 实测（8 音符 C4-G4 + 锚点 50.70）：原始 mean|shift|=11.67、3 个音被限幅；
    # 折叠 -1 八度后 mean|shift|=2.30、0 个被限幅，且音程数组逐项相等。
    octave_fold = 0
    if anchor is not None and plan_notes:
        base = np.asarray([float(n.get("midi", 60.0)) for n in plan_notes],
                          dtype=np.float64)
        best = None
        for octv in range(-4, 5):
            d = (base + 12.0 * octv) - float(anchor)
            score = float(np.abs(d).mean())
            if best is None or score < best[0]:
                best = (score, octv)
        if best is not None:
            octave_fold = int(best[1])
        if octave_fold:
            log("    整体移调 %+d 个八度（音程不变，仅对齐音域）" % octave_fold)
        else:
            log("    无需整体移调（乐谱已在音域内）")
        for n in plan_notes:
            n["midi_octave_fold"] = octave_fold
            n["midi_planned"] = float(n.get("midi", 60.0))
            n["midi"] = float(n.get("midi", 60.0)) + 12.0 * octave_fold

    n_measured = 0
    n_clamped = 0
    for i, n in enumerate(plan_notes):
        if anchor is None:
            n["source_midi"] = float(n.get("midi", 60.0))   # source == target → 不搬移
            continue
        n["source_midi"] = float(anchor)
        n["shift_basis"] = "reference_anchor"
        # 折叠后仍越界 → 素材本身病态，如实记账（由 aligner 钳制并计入诊断）
        if abs(float(n["midi"]) - float(anchor)) >= abs(float(max_semitones)):
            n_clamped += 1
        n_measured += 1
    n_rejected = len(plan_notes) - n_measured
    if n_clamped:
        errors.append(
            "%d 个音符在八度折叠后仍需超过 %.0f 半音的搬移，已钳制（旋律将不准确）"
            % (n_clamped, abs(float(max_semitones))))

    # 诊断里也带上"实测 vs 目标"，方便人工核对
    plan_with_src = dict(plan_obj)
    plan_with_src["notes"] = plan_notes
    plan_with_src["measured_source_midi"] = True
    _dumps(plan_with_src, plan_path)

    # ---------- ⑦ 对齐 + 拼接 ----------
    log("⑤ 精确对齐并拼接（时长误差目标 0 采样）…")
    audio, diags = _al.align_all(
        plan_notes, clips, sr=sr,
        bpm=float(plan_obj.get("bpm") or 0.0),
        origin=float(plan_obj.get("origin") or 0.0),
        snap=snap, total_dur=total_dur,
        max_semitones=max_semitones,
        progress=_prog("align"))

    out_wav = _write_wav(os.path.join(out_dir, OUTPUT_NAME), audio, sr)
    diag = {"per_note": diags,
            "summary": _al.summary(diags, max_semitones=max_semitones)}
    _dumps(diag, os.path.join(out_dir, DIAG_NAME))

    sm = diag["summary"]
    log("    对齐完成：拉伸中位 %.3f（越界 %d 个），音高搬移最大 %.2f 半音"
        % (sm.get("stretch_mean", 0.0), sm.get("stretch_outliers", 0),
           sm.get("semitone_max_abs", 0.0)))

    result = {
        "ok": True,
        "out_wav": out_wav,
        "out_dir": out_dir,
        "analysis_path": os.path.join(out_dir, ANALYSIS_NAME),
        "plan_path": plan_path,
        "diag_path": os.path.join(out_dir, DIAG_NAME),
        "clips_dir": clips_dir if keep_clips else None,
        "n_notes": len(plan_notes),
        "n_measured": n_measured,
        "n_measure_rejected": n_rejected,
        "n_degraded": n_degraded,
        "octave_fold": int(octave_fold),
        "n_pitch_clamped": int(n_clamped),
        "lyric_source": (analysis.get("auto_lyric") and "auto") or ("text" if (lyric_text or "").strip() else "hum"),
        "duration": round(float(len(audio) / float(sr)), 4),
        "summary": sm,
        "errors": errors,
        "elapsed": round(time.time() - t0, 2),
    }
    _dumps({k: v for k, v in result.items() if k != "summary"},
           os.path.join(out_dir, SUMMARY_NAME))

    _STATE["last_run"] = result
    _STATE["n_runs"] += 1
    _STATE["tmp_dir"] = out_dir
    log("完成，用时 %.1fs → %s" % (result["elapsed"], out_wav))
    return result


def _resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """线性重采样（仅用于**分析**路径：节拍/音高对轻微插值误差不敏感）。

    刻意不引入 soxr/scipy：分析用 22.05k，线性插值对节拍与 F0 的影响远小于
    pyin 自身的量化步长；而合成路径**绝不重采样**（那会伤音色）。
    """
    if sr_in == sr_out or y.size == 0:
        return np.asarray(y, dtype=np.float32)
    n_out = int(round(y.size * sr_out / float(sr_in)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, y.size, endpoint=False)
    x = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x, xp, np.asarray(y, dtype=np.float64)).astype(np.float32)


# ============================================================ 插件入口
def setup(ctx) -> None:
    """插件启动。只读设置 + 建数据目录，**不做任何合成**。"""
    s = ctx.settings or {}
    _STATE["ready"] = True
    _STATE["n_runs"] = 0
    _STATE["n_failed"] = 0
    _STATE["data_dir"] = ctx.data_dir
    ctx.log("clear_vocal 就绪（数据目录 %s）" % ctx.data_dir)


def teardown(ctx) -> None:
    """插件停止：无需释放长驻资源，仅复位状态并汇报用量。"""
    _STATE["ready"] = False
    ctx.log("clear_vocal 已停止（本次进程累计 %d 次运行）" % _STATE["n_runs"])


def on_report_enrich(payload):
    """把清唱插件的状态挂到合成报告里（**只读**，不触发任何生成）。

    合并策略是"只增不改"：若同名键已存在，plugins.emit 会拒绝并记日志，
    所以这里用带前缀的键名，避免与核心字段撞名。
    """
    last = _STATE.get("last_run") or {}
    return {
        "clear_vocal": {
            "ready": bool(_STATE.get("ready")),
            "n_runs": int(_STATE.get("n_runs") or 0),
            "last_ok": bool(last.get("ok")) if last else None,
            "last_out_wav": last.get("out_wav"),
            "last_n_notes": last.get("n_notes"),
        }
    }


def on_api_routes(payload):
    """注册清唱插件的 HTTP 路由：查询状态与产物路径，供 UI 展示/下载。

    ⚠️ 这里**刻意只提供读取**。真正跑流水线需要一个"提交任务 → 后台线程 →
    轮询进度"的完整任务系统（一首歌几百次合成，必然超过任何 HTTP 超时），
    且必须保证合成发生在 ``_infer_lock`` 之外 —— 那是下一步的工作。
    现在暴露一个半成品接口（同步阻塞跑合成）会把"服务挂死"当成功能交出去，
    所以宁可不提供。
    """
    app = payload.get("app") if isinstance(payload, dict) else None
    if app is None:
        return None
    try:
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/api/plugins/clear_vocal/status")
        def _status():
            last = _STATE.get("last_run") or {}
            return {
                "ready": bool(_STATE.get("ready")),
                "n_runs": int(_STATE.get("n_runs") or 0),
                "n_failed": int(_STATE.get("n_failed") or 0),
                "last_run": last or None,
            }

        app.include_router(router)
    except Exception as e:  # 插件坏掉绝不能影响服务启动
        print("[clear_vocal] 注册路由失败（已忽略）：%s: %s"
              % (type(e).__name__, e), flush=True)
    return None
