"""清唱生成插件 · 唱歌合成器
==============================
按 ``note_plan`` 逐音符生成短句音频。**这是整个插件里唯一接触模型的模块。**

    输入: note_plan['notes'] + 参考音频
    输出: [每音符的短句音频（未对齐，时长≈ source_note_dur）]

⚠️⚠️ 最重要的约束：**绝不能在插件钩子内调用本模块**
--------------------------------------------------------
``server.py:214`` 的 ``_infer_lock = threading.Lock()`` **非可重入**。
插件钩子是在持有该锁的调用路径中被触发的（``server.py:3591``
``with _infer_lock:`` → ``synthesize_stable`` → 内部 ``emit(...)``）。
若某个钩子再回调本模块去合成 → **同一线程二次获取同一把锁 → 永久死锁**，
无异常、无日志，服务直接挂死。

因此本模块的设计是：

* **模型与锁都从外部注入**（``Singer(model, lock=...)``），自己不 import ``server``；
* 编排层（``plugin.py``）负责**在锁外**调用；
* ``singer`` 内部加一道自检：若检测到自己是在钩子上下文里被调用，**立刻抛错**
  而不是安静地死锁（见 ``_assert_not_in_hook``）。

为什么要把"短句时长"当成一等公民（``source_note_dur``）
----------------------------------------------------------
VoxCPM2 是自回归 TTS，生成时长**不可直接指定**。我们能控制的只有文本长度。
一个音符可能只有 0.15 秒（十六分音符），但合成"啊"这个字最少也要 ~0.3 秒。
所以策略必须是：

    让每个音符合成**尽量短**的文本 → 得到接近 ``source_note_dur`` 的片段
    → 再由 ``aligner`` 拉伸/压缩到音符的精确时长

``source_note_dur`` 就是"一次短句合成的典型时长"这个基准。它有两个用途：
  1. ``planner`` 用它算 ``target_ratio``（诊断用，让用户看到拉伸得多狠）；
  2. 本模块用它决定**要不要拆字**（一个字合成出来太长时，减少字数）。

实测参考：单字（如"啊"）在默认参数下约 0.3~0.5 秒。
故默认值取 0.35，并允许按实测自动校准（``calibrate_source_dur``）。
"""
from __future__ import annotations

import os
import tempfile
import wave
from typing import Any, Callable

import numpy as np

#: 单次合成的文本上限。⚠️ 必须很小！
#: server.py 里 ``max_chars=60`` 是给"朗读长文"用的；清唱里若把 60 个字
#: 合成进一个 0.4 秒的音符，aligner 需要压缩 ~30 倍 → 完全崩坏。
#: 这里限到 6，保证每次合成的是"一个短句"。
MAX_CHARS_PER_NOTE = 6

#: 合成的默认最小字符数：太短（1 字）容易得到过短的音频，加一个弱起音。
MIN_TEXT_CHARS = 1

#: 哼鸣兜底字符（无歌词时用）
HUM_CHAR = "啊"


class HookContextError(RuntimeError):
    """在插件钩子上下文里尝试合成 → 会死锁，故显式拒绝。"""


def _assert_not_in_hook() -> None:
    """若当前处于插件钩子内，抛错（避免二次获取非可重入锁导致死锁）。

    检测方式：读 ``voice_clone.plugin_core`` 的"当前钩子"状态。
    该模块提供 ``current_hook()``（见 plugins.py），钩子执行期间返回钩子名。
    """
    try:
        from voice_clone import plugin_core as _plugins
    except Exception:
        return
    hook = None
    try:
        getter = getattr(_plugins, "current_hook", None)
        if callable(getter):
            hook = getter()
    except Exception:
        hook = None
    if hook:
        raise HookContextError(
            "清唱合成不能在插件钩子 '%s' 内调用：server._infer_lock 非可重入，"
            "会永久死锁且不报错。请在钩子之外调用 singer.Singer。" % hook)


def _write_wav(path: str, y: np.ndarray, sr: int) -> str:
    """写 16-bit PCM wav（VoxCPM2 的参考音频入口吃文件路径）。"""
    y = np.asarray(y, dtype=np.float32)
    pk = float(np.abs(y).max()) if y.size else 0.0
    if pk > 1.0:
        y = y / pk
    pcm = np.clip(y * 32767.0, -32768, 32767).astype("<i2")
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return path


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(a, dtype=np.float32), int(sr)


class Singer:
    """逐音符合成器。

    参数
    ----
    model        : VoxCPM2 模型（由调用方在锁外持有）
    sr           : 参考音频采样率（TTS 的 ``sr_tts``）
    lock         : 可选，``server._infer_lock``。传了就**只在推理瞬间**加锁；
                   不传则由调用方自己保证串行。
    reference_wav: 参考音色音频路径（**音色锁定的关键**：全程复用同一个，
                   保证每个音符的音色一致 —— 这是用户验收项之一）
    emotion      : 情绪标签（可选，走既有情绪链）
    cfg_value / inference_timesteps : 透传给模型
    tmp_dir      : 中间 wav 落盘目录（便于调试；None 则用系统临时目录）
    """

    #: pyin 的搜索音域（与 ``analyzer.extract_f0`` 的默认值保持一致）。
    #: 实测片段音高时用它做"边界钉死"判据：中位数贴到边界 = pyin 失效。
    F0_NOTE_RANGE = (40, 88)
    #: 距搜索边界小于该半音数即视为"钉死"（pyin 失效的可靠信号）。
    BOUNDARY_GUARD = 1.0
    #: voiced 帧占比低于此值即判为不可信（短促非稳态语音上 pyin 常低于 0.3）。
    MIN_VOICED_RATIO = 0.25

    def __init__(self, model, sr: int, lock=None,
                 reference_wav: str | None = None,
                 prompt_wav: str | None = None,
                 prompt_text: str | None = None,
                 emotion: str = "",
                 cfg_value: float = 2.0,
                 inference_timesteps: int = 10,
                 max_chars: int = MAX_CHARS_PER_NOTE,
                 tmp_dir: str | None = None,
                 keep_tmp: bool = False):
        self.model = model
        self.sr = int(sr)
        self.lock = lock
        self.reference_wav = reference_wav
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.emotion = emotion or ""
        self.cfg_value = float(cfg_value)
        self.inference_timesteps = int(inference_timesteps)
        self.max_chars = int(max_chars)
        self.tmp_dir = tmp_dir or tempfile.gettempdir()
        self.keep_tmp = bool(keep_tmp)
        self.errors: list[dict] = []

    # ------------------------------------------------------------ 单次合成
    def _synth_text(self, text: str) -> tuple[np.ndarray, int]:
        """合成一段文本，返回 (音频, 采样率)。"""
        _assert_not_in_hook()

        from voice_clone import synthesis_stab as _stab
        # 极短文本 + 无停顿，尽量得到"一个干净短句"
        kwargs: dict[str, Any] = {}
        if self.lock is not None:
            with self.lock:
                wav, _rep = _stab.synthesize_stable(
                    self.model, text,
                    reference_wav_path=self.reference_wav,
                    sr_tts=self.sr,
                    prompt_wav_path=self.prompt_wav,
                    prompt_text=self.prompt_text,
                    max_chars=self.max_chars,
                    pause=0.0,             # ⚠️ 清唱不需要句间停顿
                    breath=0.0,            # ⚠️ 不要呼吸声（会污染音符边界）
                    emotion=self.emotion,
                    cfg_value=self.cfg_value,
                    inference_timesteps=self.inference_timesteps,
                    **kwargs)
        else:
            wav, _rep = _stab.synthesize_stable(
                self.model, text,
                reference_wav_path=self.reference_wav,
                sr_tts=self.sr,
                prompt_wav_path=self.prompt_wav,
                prompt_text=self.prompt_text,
                max_chars=self.max_chars,
                pause=0.0, breath=0.0,
                emotion=self.emotion,
                cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
                **kwargs)
        return np.asarray(wav, dtype=np.float32), int(self.sr)

    # ------------------------------------------------------------ 批量
    def sing_plan(self, plan_obj: dict,
                  progress: Callable[[int, int, dict], None] | None = None,
                  on_error: str = "skip") -> tuple[list[np.ndarray], list[dict]]:
        """按 note_plan 逐音符合成。

        返回 ``(clips, diags)``：
          * ``clips[i]`` —— 第 i 个音符的**未对齐**短句音频
          * ``diags[i]`` —— 该音符的合成诊断（文本、时长、是否降级）

        ``on_error``：
          * ``"skip"``（默认）—— 某音符合成失败则用**静音**占位，
            记录到 ``self.errors``，不中断整首（一首歌几百个音，
            不能因为一个音失败就全军覆没）；
          * ``"raise"`` —— 立刻抛出（调试用）。

        音色一致性怎么保证
        ------------------
        每个音符都用**同一个** ``reference_wav`` 作为参考（而不是用上一段的输出
        继续链式生成）。链式生成会让音色逐段漂移，这是本项目历史的已知问题；
        锚定同一参考是唯一可靠解法，代价是首音略"生"（可接受）。
        """
        notes = list(plan_obj.get("notes") or [])
        clips: list[np.ndarray] = []
        diags: list[dict] = []
        total = len(notes)

        for i, n in enumerate(notes):
            raw = (n.get("lyric") or "").strip()
            text = raw if raw else HUM_CHAR
            item: dict[str, Any] = {
                "index": i, "text": text, "lyric": raw,
                "midi": n.get("midi"), "dur": n.get("dur"),
                "degraded": False, "error": None, "src_dur": 0.0,
            }
            try:
                y, sr = self._synth_text(text)
                item["src_dur"] = round(len(y) / float(sr), 4)
                item["sr"] = sr
                if y.size == 0:
                    raise RuntimeError("合成为空音频")
                clips.append(y)
            except HookContextError:
                raise
            except Exception as e:
                item["degraded"] = True
                item["error"] = "%s: %s" % (type(e).__name__, e)
                self.errors.append(dict(item))
                if on_error == "raise":
                    raise
                clips.append(np.zeros(int(0.05 * self.sr), dtype=np.float32))
            diags.append(item)
            if progress is not None:
                try:
                    progress(i + 1, total, item)
                except Exception:
                    pass
        return clips, diags

    # ------------------------------------------------------------ 实测片段音高
    def median_midi_of(self, clip: np.ndarray, sr: int | None = None,
                       default: float | None = None,
                       target_midi: float | None = None,
                       tol_semitones: float = 7.0) -> float | None:
        """实测一个合成片段的**中位音高**（MIDI），供 ``aligner`` 当 ``source_midi``。

        为什么不直接猜 TTS 的"基准音高"
        -------------------------------
        ``aligner.align_note()`` 的搬移量 = ``target_midi - source_midi``。
        若 ``source_midi`` 错一个八度（12 半音），整首歌会被**整体平移一个八度**，
        而单音符 ±12 半音的钳制恰好不会拦下 12 这个值 —— 错误会静默通过。

        ⚠️ 为什么不能"无条件相信 pyin"（实跑踩到的真 bug）
        --------------------------------------------------
        实测：在真实合成的 8 个片段上直接跑 pyin，得到
        ``[54.8, 无, 40.00, 86.00, 无, 40.50, 65.9, 46.4]`` —— 其中
        **40.00 恰好钉在 fmin 边界、86.00 钉在 fmax 边界**，另有 2 个完全没有
        voiced 帧。这是 pyin 在**短促（~0.6s）非稳态语音**上的典型失效模式：
        它不返回"低置信度"，而是**饱和到搜索边界**，看起来像个正常数值。
        若照单全收，4 个音符会被贴着 12 半音上限搬运，听感完全错。

        故本函数做三重**合理性闸门**，任一不过就返回 ``default``（→ 不搬移）：
          1. **边界钉死**：中位数距 fmin/fmax 边界 < 1 半音 → 判为失效；
          2. **可信度**：voiced 帧占比 < ``min_voiced_ratio`` → 判为失效；
          3. **与目标音高的一致性**：给了 ``target_midi`` 时，实测值若偏离超过
             ``tol_semitones`` 半音，说明测得不可信（调式内音程极少超过 7 半音），
             此时**信目标**而不是信测量。

        宁可返回 None（→ 不搬移，保持自然）也不返回一个"看起来合理"的错值 ——
        不搬移只是"没做到"，搬错八度是"做坏了"。
        """
        from . import analyzer as _an
        y = np.asarray(clip, dtype=np.float32).reshape(-1)
        if y.size == 0:
            return default
        use_sr = int(sr or self.sr)
        try:
            f0 = _an.extract_f0(y, use_sr)
        except Exception as e:
            self.errors.append({"stage": "median_midi_of",
                                "error": "%s: %s" % (type(e).__name__, e)})
            return default
        midi = np.asarray(f0.get("midi", []), dtype=np.float64)
        voiced = np.asarray(f0.get("voiced", []), dtype=bool)
        if midi.size == 0 or not bool(voiced.any()):
            return default

        # 闸门 2：可信度
        ratio = float(np.mean(voiced))
        if ratio < self.MIN_VOICED_RATIO:
            return default

        vals = midi[voiced]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return default
        med = float(np.median(vals))

        # 闸门 1：边界钉死（pyin 失效的可靠信号）
        lo = float(self.F0_NOTE_RANGE[0])
        hi = float(self.F0_NOTE_RANGE[1])
        if med <= lo + self.BOUNDARY_GUARD or med >= hi - self.BOUNDARY_GUARD:
            return default

        # 闸门 3：与目标音高的一致性（调式内音程极少超过 7 半音）
        if target_midi is not None and abs(med - float(target_midi)) > tol_semitones:
            return default

        return med

    def measure_clips(self, clips: list[np.ndarray], sr: int | None = None,
                      default: float | None = None,
                      target_midis: list[float] | None = None
                      ) -> list[float | None]:
        """批量实测片段音高（测不出/不可信时为 ``default``）。

        ``target_midis`` 给出每个片段"本应唱到"的音高，用于启用一致性闸门。
        """
        out: list[float | None] = []
        for i, c in enumerate(clips):
            tgt = None
            if target_midis is not None and i < len(target_midis):
                tgt = target_midis[i]
            out.append(self.median_midi_of(c, sr=sr, default=default,
                                           target_midi=tgt))
        return out

    # ------------------------------------------------------------ 校准
    def calibrate_source_dur(self, sample_texts: tuple[str, ...] = ("啊", "啦", "米")) -> float:
        """实测"单次短句合成"的典型时长，用于校准 ``planner`` 的 ``source_note_dur``。

        为什么要实测而不是写死：TTS 输出时长取决于参考音频的音色与语速，
        不同音色包可能差 2 倍。写死会导致 ``target_ratio`` 诊断失真。
        取多个样本的**中位数**（抗单次异常）。
        """
        durs: list[float] = []
        for t in sample_texts:
            try:
                y, sr = self._synth_text(t)
                if y.size:
                    durs.append(len(y) / float(sr))
            except HookContextError:
                raise
            except Exception as e:
                self.errors.append({"index": -1, "text": t, "degraded": True,
                                    "error": "校准失败: %s" % e})
        if not durs:
            return 0.35
        return float(np.median(durs))

    # ------------------------------------------------------------ 落盘（调试）
    def dump_clips(self, clips: list[np.ndarray], out_dir: str) -> list[str]:
        """把每个音符的片段落盘（便于人工试听定位哪个音不对）。"""
        os.makedirs(out_dir, exist_ok=True)
        paths: list[str] = []
        for i, c in enumerate(clips):
            p = os.path.join(out_dir, "note_%04d.wav" % i)
            _write_wav(p, c, self.sr)
            paths.append(p)
        return paths
