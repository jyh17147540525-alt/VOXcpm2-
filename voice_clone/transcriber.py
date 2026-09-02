"""
长音频自动转写 (Transcriber)
============================
用 faster-whisper（CTranslate2，CPU int8）把用户上传的长语音自动切句 + 逐句转写，
产出「候选训练样本」供前端审阅（可改文本、勾选），确认后按时间戳切片导入
training_store。

背景：训练样本是「文本↔音频」监督对，需要短句（官方 packer 按整段音频编码，
过长会被 max_len=4096 静默截断）。直接放开上传到 10 分钟是错的 —— 正确姿势是
把长音频切成 1~30s 的句子并逐句转写，再入库。

模块职责：
  init(base_dir)                     —— 指定根目录，初始化 job 目录
  is_busy()                          —— 是否有转写任务在跑
  start_transcribe(src_path, label)  —— 后台线程启动一次转写，返回 job_id
  get_job(job_id)                    —— 查询 job 状态 / segments
  import_segments(job_id, items)     —— 按 [{idx,text}] 切片导入为训练样本

状态约定（job）：
  {job_id, status: pending|processing|done|error, progress: 0-100,
   message, error, segments: [{idx,text,start,end,duration}], lang}
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import soundfile as sf

# ------------------------------------------------------------------ 配置
MODEL_SIZE = "small"          # faster-whisper 模型规格: tiny/base/small/medium
MODEL_DIR_NAME = "faster-whisper-models"
MAX_JOB_SECONDS = 600.0       # 单次转写音频上限（与推理参考音频一致：10 分钟）
MIN_SEGMENT_SECONDS = 1.0     # 切片入库时长下限（对齐 training_store）
MAX_SEGMENT_SECONDS = 30.0    # 切片入库时长上限（对齐 training_store）
JOB_TTL_HOURS = 24            # 残留 job 目录自动清理阈值

# ------------------------------------------------------------------ 全局
JOBS_ROOT: Path | None = None
MODEL_ROOT: Path | None = None
_cleanup_done = False
_jobs: dict[str, dict] = {}          # job_id -> 共享状态（跨线程读）
_jobs_lock = threading.Lock()
_busy = threading.Lock()             # 同一时间只跑一个转写任务
_model = None
_model_lock = threading.Lock()
_model_failed = ""


# ------------------------------------------------------------------ 初始化
def init(base_dir):
    global JOBS_ROOT, MODEL_ROOT, _cleanup_done
    base = Path(base_dir)
    JOBS_ROOT = base / "training_data" / "transcribe_jobs"
    MODEL_ROOT = base / "models"
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    if not _cleanup_done:
        _cleanup_stale_jobs()
        _cleanup_done = True


def _cleanup_stale_jobs():
    """清理超过 TTL 的 job 目录（用户放弃导入时避免残留堆积）。"""
    if JOBS_ROOT is None:
        return
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    try:
        for p in JOBS_ROOT.iterdir():
            if p.is_dir():
                try:
                    if p.stat().st_mtime < cutoff:
                        shutil.rmtree(str(p), ignore_errors=True)
                except OSError:
                    pass
    except OSError:
        pass


# ------------------------------------------------------------------ 工具
def _find_ffmpeg():
    """定位 ffmpeg（PATH → imageio_ffmpeg → F 盘常见安装路径）。"""
    import glob
    import os
    import shutil as _sh
    cand = os.environ.get("FFMPEG") or ""
    if cand:
        p = _sh.which(cand) or (cand if Path(cand).exists() else None)
        if p:
            return p
    p = _sh.which("ffmpeg")
    if p:
        return p
    for pat in ("F:/ffmpeg*/bin/ffmpeg.exe", "F:/VoxCPM2/ffmpeg*/bin/ffmpeg.exe",
                "C:/ffmpeg*/bin/ffmpeg.exe"):
        ms = sorted(glob.glob(pat))
        if ms:
            return ms[0]
    try:
        import imageio_ffmpeg as _iff
        return _iff.get_ffmpeg_exe()
    except Exception:
        return None


def to_16k_mono_wav(src: Path, dst: Path) -> Path:
    """用 ffmpeg 把任意音频转成 16k 单声道 wav（whisper 输入标准）。

    返回 dst 路径；失败抛 RuntimeError（含中文原因）。
    """
    ff = _find_ffmpeg()
    if ff is None:
        raise RuntimeError("需要 ffmpeg 才能转写（用于音频解码），请先安装 ffmpeg 并加入 PATH")
    r = subprocess.run(
        [ff, "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(dst)],
        capture_output=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("音频解码失败（请确认是有效的音频文件）: "
                           + r.stderr.decode("utf-8", "ignore")[-200:])
    return dst


def _probe_duration(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        # sf 读不了（如 m4a）就交给 ffmpeg，转完再量
        return -1.0


# ------------------------------------------------------------------ 模型
def _model_dir():
    """本地模型目录：F:\\VoxCPM2\\models\\faster-whisper-<size>\\。

    手动下载/缓存优先放这里（四个文件：config.json/model.bin/tokenizer.json/vocabulary.txt）。
    """
    if MODEL_ROOT is None:
        return None
    d = MODEL_ROOT / f"faster-whisper-{MODEL_SIZE}"
    if d.exists() and (d / "model.bin").exists() and (d / "config.json").exists():
        return str(d)
    return None


def _ensure_model():
    """懒加载 faster-whisper 模型（CPU int8）。首次需要模型权重（本地或自动下载）。

    返回模型；失败抛 RuntimeError（含下载指引）。
    """
    global _model, _model_failed
    with _model_lock:
        if _model is not None:
            return _model
        if _model_failed:
            raise RuntimeError(_model_failed)
    try:
        from faster_whisper import WhisperModel  # noqa
    except Exception as e:
        _model_failed = f"faster-whisper 未安装或导入失败: {e}"
        raise RuntimeError(_model_failed)

    local = _model_dir()
    dl_root = str(MODEL_ROOT / "faster-whisper-hub") if MODEL_ROOT else None
    try:
        env_offline = __import__("os").environ.pop("HF_HUB_OFFLINE", None)
        env_xet = __import__("os").environ.pop("HF_HUB_DISABLE_XET", None)
        try:
            if local:
                m = WhisperModel(local, device="cpu", compute_type="int8")
            else:
                # 自动下载到 hub 缓存目录（下载脚本产出本地目录后走上一分支）
                m = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8",
                                 download_root=dl_root)
        finally:
            if env_offline is not None:
                __import__("os").environ["HF_HUB_OFFLINE"] = env_offline
            if env_xet is not None:
                __import__("os").environ["HF_HUB_DISABLE_XET"] = env_xet
        with _model_lock:
            _model = m
        return m
    except Exception as e:
        hint = (
            f"whisper 模型（{MODEL_SIZE}）加载失败: {type(e).__name__}: {e}\n"
            "如果是因为模型权重缺失或网络无法访问 HuggingFace，请手动下载一次：\n"
            "  1) 下载这 4 个文件到 F:\\VoxCPM2\\models\\faster-whisper-small\\ 目录：\n"
            "     https://hf-mirror.com/Systran/faster-whisper-small/resolve/main/{config.json,model.bin,tokenizer.json,vocabulary.txt}\n"
            "  2) 完成后回到页面重新转写。"
        )
        _model_failed = hint
        raise RuntimeError(hint)


# ------------------------------------------------------------------ 转写任务
def is_busy() -> bool:
    with _jobs_lock:
        return any(j.get("status") in ("pending", "processing") for j in _jobs.values())


def _new_job() -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "排队中…",
        "error": None,
        "lang": None,
        "segments": [],
        "created_at": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def drop_job(job_id: str):
    with _jobs_lock:
        _jobs.pop(job_id, None)
    if JOBS_ROOT is not None:
        try:
            shutil.rmtree(str(JOBS_ROOT / job_id), ignore_errors=True)
        except OSError:
            pass


def start_transcribe(src_path, label: str = "", transcript: str = "") -> dict:
    """后台线程启动一次转写。src_path 会被复制进 job 目录（16k mono wav）。

    传 transcript（权威台词全文）时：转写完成后自动按各分段时长把台词
    匹配进 text 字段（whisper 只提供时间边界，识别文本不参与匹配）。

    返回 job 字典；失败（忙/参数错）抛 RuntimeError。
    """
    global JOBS_ROOT
    if JOBS_ROOT is None:
        raise RuntimeError("transcriber 未初始化")
    if not _busy.acquire(blocking=False):
        raise RuntimeError("已有转写任务在运行，请稍候")
    src = Path(src_path)
    if not src.exists():
        _busy.release()
        raise RuntimeError("音频文件不存在")
    dur = _probe_duration(src)
    if 0.0 < dur < 1.0:
        _busy.release()
        raise RuntimeError(f"音频太短（{dur:.1f}s），没有可转写的内容")
    if dur > MAX_JOB_SECONDS:
        _busy.release()
        raise RuntimeError(
            f"音频太长（{dur:.0f}s），转写上限为 {MAX_JOB_SECONDS / 60:g} 分钟。"
            "如需更长请自行裁剪分段后分别转写。")

    job = _new_job()
    job_dir = JOBS_ROOT / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    wav16 = job_dir / "audio_16k.wav"
    try:
        to_16k_mono_wav(src, wav16)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["message"] = "转码失败"
        drop_job(job["job_id"])
        _busy.release()
        raise RuntimeError(str(e))

    def _worker(job_id: str, wav: Path, lb: str, tr: str):
        job = None
        try:
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                return
            job["status"] = "processing"
            job["message"] = "加载 whisper 模型…"
            job["progress"] = 5
            model = _ensure_model()

            job["message"] = "正在转写（长音频需要一点时间）…"
            job["progress"] = 20
            segments, info = model.transcribe(
                str(wav),
                language=None,            # 自动检测（中文/粤语/英语等）
                beam_size=5,
                vad_filter=True,          # 跳过静音段
                vad_parameters={"min_silence_duration_ms": 300},
                initial_prompt="。，！？",
                condition_on_previous_text=True,
                word_timestamps=True,     # 词级时间戳（台词对齐的强锚点）
            )
            lang = getattr(info, "language", None)
            job["lang"] = lang
            keep_overlong = bool(tr)      # 台词对齐模式：保留 >30s 段，保证时间轴连续
            segs = []
            timeline = []                 # 词级锚点 [{u,s,e}]：全音频展平、按时间有序
            raw_tail = []                 # 全部“像语音”的 whisper 原始段（含 <1s），台词覆盖不到时兜底
            total = 0.0
            for seg in segments:
                text = (seg.text or "").strip()
                st = float(getattr(seg, "start", 0))
                en = float(getattr(seg, "end", 0))
                d = en - st
                total = max(total, en)
                if not _looks_like_speech(text):
                    continue
                raw_tail.append({
                    "text": text, "start": round(st, 3),
                    "end": round(en, 3), "duration": round(d, 2),
                })
                for w in (getattr(seg, "words", None) or []):
                    _add_word_anchors(timeline, w)
                if d < MIN_SEGMENT_SECONDS:
                    continue
                if not keep_overlong and d > MAX_SEGMENT_SECONDS:
                    continue
                segs.append({
                    "idx": len(segs),
                    "text": text,
                    "start": round(st, 3),
                    "end": round(en, 3),
                    "duration": round(d, 2),
                })
                if len(segs) % 20 == 0:
                    job["progress"] = min(85, 30 + len(segs))
            job["words_timeline"] = timeline     # 供 align 与 align_job(改台词后重对齐) 复用
            job["raw_tail"] = raw_tail
            job["audio_total"] = round(total, 3)
            note = ""
            if tr and (segs or timeline):
                segs, note = align_text_to_segments(
                    segs, tr,
                    timeline=timeline, raw_tail=raw_tail, audio_total=total)
            job["segments"] = segs
            job["progress"] = 100
            job["status"] = "done"
            if tr:
                job["aligned"] = bool(segs)
                job["align_note"] = note or ("未识别到语音分段，无法匹配台词" if not segs else "")
            job["message"] = (f"转写完成：{len(segs)} 段"
                              + (f"（{lang}）" if lang else ""))
            if tr:
                job["message"] += "；" + (note or "未识别到语音分段")
        except Exception as e:
            if job is not None:
                job["status"] = "error"
                job["error"] = str(e)
                job["message"] = f"转写失败: {e}"
        finally:
            _busy.release()

    threading.Thread(target=_worker,
                     args=(job["job_id"], wav16, label, transcript or ""),
                     daemon=True).start()
    return dict(job)


def _looks_like_speech(text: str) -> bool:
    """过滤空段/纯标点/纯符号段。要求至少含一个 CJK 或字母数字字符。"""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            return True
    return False


# ------------------------------------------------------------------ 台词对齐
_SENT_PUNCT = set("。！？…!?；;")
_CLAUSE_PUNCT = set("，,、：:")
_TRAIL_PUNCT = set("”’」』】》)\"'】]…~—–·—-—")


def _tokenize_transcript(text: str) -> list:
    """把完整台词拆成"语音单元"（token = 单个 CJK 字 / 连续 ASCII 字母数字串），
    返回 list[dict]，每项含：
      w     —— 语音权重（CJK 字=1；英文按 0.5 字/字母折算，至少 1）
      cut   —— 该 token 在原文的切片终点（含其后粘连的标点/空白，字符下标）
      kind  —— 尾随断句强度：2=句末(。！？…换行) 1=逗号类 0=句中无停顿
    token 之间的空白/标点会"吸"到前一个 token 的 cut 上，保证切出来的
    每个分段都不以句号/逗号开头。
    """
    def _cjk(ch):
        return "\u3400" <= ch <= "\u4dbf" or "\u4e00" <= ch <= "\u9fff"

    tokens = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch.isspace() or ch in _SENT_PUNCT or ch in _CLAUSE_PUNCT or ch in "（）()【】[]<>《》““”‘’「」『』·—":
            i += 1
            continue
        if _cjk(ch):
            i += 1
            w = 1
        elif ch.isascii() and ch.isalnum():
            s0 = i
            while i < n and text[i].isascii() and text[i].isalnum():
                i += 1
            w = max(1, int(round((i - s0) * 0.5)))
        else:  # 假名/谚文等非常见字：按 1 字算
            i += 1
            w = 1
        # 尾随标点/空白归前，并记录断句强度
        kind = 0
        j = i
        while j < n:
            c2 = text[j]
            if c2.isspace():
                if c2 in "\n\r":
                    kind = 2
                j += 1
                continue
            if c2 in _SENT_PUNCT:
                kind = 2
            elif c2 in _CLAUSE_PUNCT or c2 in _TRAIL_PUNCT:
                kind = max(kind, 1)
            else:
                break
            j += 1
        tokens.append({"w": w, "cut": j, "kind": kind})
        i = j
    return tokens


# ================================================================== 台词对齐 v2
#
# 旧版（_align_proportional，仅作无时间戳时的兜底）假设"语速均匀"，按
# 总字数/总时长把台词摊到各段 —— 语速变化、音乐间奏、口误重复都会整体错位。
#
# v2 采用"外部位对齐"思路（同 WhisperX / Gentle / MFA 一类）：
#   1. 转写时 faster-whisper 已开 word_timestamps=True，每个 word 被展开成
#      "字符级/词级时间锚点"（见 _add_word_anchors）—— 锚点文本 ↔ 真实时刻，
#      这是比"字数/时长比例"可信得多的直接证据。
#   2. 把台词切成对齐原子（CJK 逐字一原子、拉丁字母/数字按"词"一原子），
#      与锚点文本做一次全局编辑对齐（Needleman–Wunsch：等值 0 / 替换 1 / 插删 1）。
#   3. 只有完全等值的对齐对才是"硬锚点"（台词原子直接获得锚点真实时刻）；
#      其余原子在最近两个硬锚点之间按字符序号线性插值，保证"文本序 → 时刻"单调。
#   4. 语音分段两两之间取"间隙中点"作文本切界，二分时刻表即可切出每段真正
#      对应的台词；台词覆盖不到的段保留 whisper 原文（aligned=False）供勾除，
#      台词末尾超出音频总长的部分不再硬塞进末段。
import bisect as _bisect


def _add_word_anchors(timeline: list, w) -> None:
    """把一个 whisper word 展开为时间锚点追加进 timeline。

    w —— faster-whisper 的 word 对象（word/start/end）
    锚点规则：CJK 等非 ASCII 字符逐字一个锚点（词区间按字数等分）；
    连续 ASCII 字母/数字整段一个锚点（词对齐天然落在词边界上）。
    锚点 = {u: 文本, s: 起, e: 止, t: 中心时刻}。标点/空白不产出锚点。
    """
    ws = float(getattr(w, "start", 0) or 0)
    we = float(getattr(w, "end", 0) or 0)
    txt = (getattr(w, "word", "") or "").strip()
    if we <= ws or not txt:
        return
    n = len(txt)
    i = 0
    while i < n:
        ch = txt[i]
        if not ch.isalnum():
            i += 1
            continue
        if ch.isascii():
            j = i + 1
            while j < n and txt[j].isascii() and txt[j].isalnum():
                j += 1
            a, b = ws, we
            u = txt[i:j]
            timeline.append({"u": u, "s": round(a, 4), "e": round(b, 4),
                             "t": round((a + b) / 2, 4)})
            i = j
        else:
            a = ws + (we - ws) * (i) / n
            b = ws + (we - ws) * (i + 1) / n
            timeline.append({"u": ch, "s": round(a, 4), "e": round(b, 4),
                             "t": round((a + b) / 2, 4)})
            i += 1


def _fold(ch: str) -> str:
    """单个字符归一化：全角→半角、大写→小写（用于识别文本与台词比对）。"""
    o = ord(ch)
    if 0xFF01 <= o <= 0xFF5E:
        ch = chr(o - 0xFEE0)
    return ch.lower() if ch.isalpha() else ch


def _text_atoms(text: str) -> list:
    """台词 → 对齐原子列表，每项 {ch: 文本, o: 在原文的下标}。

    CJK 等非 ASCII 可发音字符逐字一个原子；连续 ASCII 字母/数字按"词"
    一个原子（与 _add_word_anchors 的锚点粒度一致）。标点/空白不产出
    原子，只作为切分文本时的"胶水"。
    """
    atoms = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if not ch.isalnum():
            i += 1
            continue
        if ch.isascii():
            j = i + 1
            while j < n and text[j].isascii() and text[j].isalnum():
                j += 1
            atoms.append({"ch": text[i:j], "o": i})
            i = j
        else:
            atoms.append({"ch": ch, "o": i})
            i += 1
    return atoms


def _match_atom_times(atoms: list, timeline: list, n_max_cells: int = 15_000_000,
                      tail_floor: float = 0.0):
    """全局字符级编辑对齐 → 每个台词原子的估计时刻（秒）。

    Needleman–Wunsch 在"台词原子"与"锚点文本"之间求最小代价路径
    （等值 0 / 替换 1 / 插删 1），完全等值的原子↔锚点对构成硬锚点，
    其余原子在相邻硬锚点间按原子序号线性插值（首部未匹配用首个锚点时刻
    平推，结果单调不减）。

    超出最后一个硬锚点的尾部原子不再外推猜时刻，而是统一排在
    max(音频总长, 末锚点时刻) 之后 —— 它们被认为是"台词超出已发声部分"，
    由调用方统计 dropped 并提示，而不是污染最后一个分段的文本。

    返回 list[float]（与 atoms 等长）；timeline 为空、无等值锚点或规模
    超限时返回 None —— 调用方据此退回比例法。
    """
    m = len(timeline)
    n = len(atoms)
    if not n or not m:
        return None
    ref = ["".join(_fold(c) for c in a["ch"]) for a in atoms]
    rec = []
    anc_idx = []            # rec[k] 对应 timeline[anc_idx[k]]
    for j, an in enumerate(timeline):
        u = an.get("u")
        if not u:
            continue
        rec.append("".join(_fold(c) for c in u))
        anc_idx.append(j)
    m2 = len(rec)
    if not m2:
        return None
    if n * m2 > n_max_cells:
        return None

    # DP：行=台词原子，列=锚点文本。prev/cur 只留一行代价，dirs 记方向供回溯。
    prev = list(range(m2 + 1))                      # dp[0][*] = j（删识别文本）
    dirs = [bytearray([2]) * (m2 + 1)]              # 首行方向全是"左"
    dirs[0][0] = 0
    for i in range(1, n + 1):
        ri = ref[i - 1]
        cur = [i] + [0] * m2                        # dp[i][0] = i（删台词原子）
        drow = bytearray(m2 + 1)
        drow[0] = 1
        p, cj = prev, cur
        for j in range(1, m2 + 1):
            diag = p[j - 1] + (0 if ri == rec[j - 1] else 1)
            up = p[j] + 1
            left = cj[j - 1] + 1
            if diag <= up and diag <= left:
                drow[j] = 0
                cj[j] = diag
            elif up <= left:
                drow[j] = 1
                cj[j] = up
            else:
                drow[j] = 2
                cj[j] = left
        prev = cur
        dirs.append(drow)

    # 回溯收集硬锚点对 (台词原子下标, 锚点下标)
    pair = []
    i, j = n, m2
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            dr = dirs[i][j]
            if dr == 0:
                if ref[i - 1] == rec[j - 1]:
                    pair.append((i - 1, anc_idx[j - 1]))
                i -= 1
                j -= 1
            elif dr == 1:
                i -= 1
            else:
                j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
    pair.reverse()
    if not pair:
        return None

    mpos = [p[0] for p in pair]
    mt = [timeline[p[1]]["t"] for p in pair]
    est = [None] * n
    for ri, aj in pair:
        est[ri] = timeline[aj]["t"]

    times = [0.0] * n
    ptr = 0
    npair = len(pair)
    last = None            # (台词原子下标, 时刻) 最近一个硬锚点
    for i in range(n):
        if est[i] is not None:
            times[i] = est[i]
            last = (i, est[i])
            continue
        while ptr < npair and mpos[ptr] <= i:
            ptr += 1
        if ptr < npair:
            nri, nt = mpos[ptr], mt[ptr]
            if last is None:
                times[i] = max(0.0, nt)          # 首部未匹配：平推首个锚点
            else:
                li, lt = last
                times[i] = lt + (nt - lt) * (i - li) / (nri - li)
        elif last is not None:
            li, lt = last
            # 尾部原子：整体排到 max(音频总长, 末锚点) 之后 → 不被任何分段承接
            base = max(tail_floor, lt)
            times[i] = base + 20.0 + 0.1 * (i - li)
        else:
            times[i] = 0.0
    for i in range(1, n):
        if times[i] < times[i - 1]:
            times[i] = times[i - 1]
    return times


def _merge_short_speech(speech: list, min_dur: float = MIN_SEGMENT_SECONDS) -> list:
    """把过短的语音段并入相邻段（优先并入间隙更小的一侧）。

    返回新列表（不改入参）：每段 {start,end,text,...}，按 start 排序、
    相互不重叠；目标是把所有 < min_dur 的碎片并进邻居，避免产出
    无法入库的极短切片，同时保证时间轴连续。
    """
    w = [dict(s) for s in speech]
    w.sort(key=lambda x: x["start"])
    guard = 0
    while guard < 500:
        guard += 1
        merged = False
        for k in range(len(w)):
            if (w[k]["end"] - w[k]["start"]) >= min_dur - 1e-9:
                continue
            prev = w[k - 1] if k > 0 else None
            nxt = w[k + 1] if k + 1 < len(w) else None
            if prev is None and nxt is None:
                break
            gap_p = (w[k]["start"] - prev["end"]) if prev is not None else 1e18
            gap_n = (nxt["start"] - w[k]["end"]) if nxt is not None else 1e18
            if nxt is not None and (prev is None or gap_n <= gap_p):
                nxt["start"] = w[k]["start"]
                if w[k].get("text"):
                    nxt["text"] = ((w[k]["text"] or "") + " " + (nxt.get("text") or "")).strip()
            else:
                prev["end"] = w[k]["end"]
                if w[k].get("text"):
                    prev["text"] = ((prev.get("text") or "") + " " + (w[k]["text"] or "")).strip()
            w.pop(k)
            merged = True
            break
        if not merged:
            break
    return w


def _cut_segments_by_times(speech: list, atoms: list, times: list,
                           transcript: str) -> tuple:
    """用时刻表把各语音分段切出真正对应的台词文本。

    相邻语音段之间取"间隙中点"作为文本切界（台词原子归属最近的发声段），
    末段用其 end 收口 —— 台词末尾超出音频总长的原子不再硬塞进末段。

    返回 (out, dropped)：out 为最终输出分段 [{idx,text,start,end,duration,
    orig_text,aligned}]；dropped 为超出音频未被任何段承接的台词原子数。
    """
    n_atoms = len(atoms)
    text = transcript
    speech = list(speech)
    ns = len(speech)
    # 逐段文本切界（原子下标），两两间隙中点
    bounds = []
    lo = 0
    for k in range(ns):
        if k == ns - 1:
            bt = speech[k]["end"] + 0.06
        else:
            bt = (speech[k]["end"] + speech[k + 1]["start"]) / 2.0
        hi = _bisect.bisect_right(times, bt, lo)
        bounds.append(hi)
        lo = hi
    max_hi = bounds[-1] if bounds else 0
    dropped = n_atoms - max_hi
    out = []
    prev = 0
    for k, s in enumerate(speech):
        hi = bounds[k]
        orig = (s.get("text") or "").strip()
        piece = ""
        if hi > prev and prev < n_atoms:
            a0 = atoms[prev]["o"]
            a1 = atoms[hi - 1]["o"] + 1
            while a1 < len(text) and not text[a1].isalnum():
                a1 += 1
            piece = text[a0:a1].strip()
        prev = hi
        out.append({
            "idx": len(out),
            "text": piece if piece else orig,
            "start": s["start"],
            "end": s["end"],
            "duration": round(s["end"] - s["start"], 2),
            "orig_text": orig,
            "aligned": bool(piece),
        })
    return out, dropped


def _align_proportional(speech: list, text: str) -> tuple:
    """兜底：无词级时间戳时按"语速均匀"把台词摊到各段（旧 v1 逻辑）。"""
    total_dur = sum((s["end"] - s["start"]) for s in speech)
    if total_dur <= 0:
        return [], ""
    tokens = _tokenize_transcript(text)
    if not tokens:
        return [], "台词里没有可朗读的文字（请直接粘贴纯文本）"
    n = len(tokens)
    cum = [0.0] * n
    acc = 0.0
    for i, tk in enumerate(tokens):
        acc += tk["w"]
        cum[i] = acc
    rate = acc / total_dur
    c = 0
    covered = 0
    for s in speech:
        if c >= n:
            break
        base = cum[c - 1] if c else 0.0
        target = base + (s["end"] - s["start"]) * rate
        t = c
        while t < n and cum[t] < target:
            t += 1
        cands = []
        if t < n:
            cands.append(t + 1)
        if t - 1 >= c:
            cands.append(t)
        if not cands and c < n:
            cands.append(n)
        if not cands:
            break

        def _err(cc):
            return abs((cum[cc - 1] if cc else 0.0) - target)

        best = min(cands, key=lambda cc: (_err(cc), tokens[cc - 1]["kind"]))
        lo = tokens[c - 1]["cut"] if c else 0
        hi = tokens[best - 1]["cut"] if best else lo
        piece = text[lo:hi].strip()
        if not piece and best > c:
            piece = text[tokens[c]["cut"] if c < n else lo:tokens[best - 1]["cut"]].strip()
        s["orig_text"] = s.get("text", "")
        s["text"] = piece
        s["aligned"] = True
        covered += 1
        c = best

    parts = [f"已按台词匹配 {covered}/{len(speech)} 段"]
    if c < n and covered == len(speech):
        lo = tokens[c - 1]["cut"] if c else 0
        hi = tokens[n - 1]["cut"]
        tail = text[lo:hi].strip()
        if tail:
            speech[-1]["text"] = (speech[-1]["text"] + tail).strip()
            parts.append("台词比语音长，末尾已并入最后一段，请试听核对")
    if covered < len(speech):
        parts.append(f"台词不足，后 {len(speech) - covered} 段保留原识别文本")
    out = []
    for s in speech:
        out.append({
            "idx": len(out),
            "text": s.get("text") or "",
            "start": s["start"],
            "end": s["end"],
            "duration": round(s["end"] - s["start"], 2),
            "orig_text": s.get("orig_text") or "",
            "aligned": bool(s.get("aligned")),
        })
    return out, "；".join(parts)


def align_text_to_segments(segments, transcript: str, timeline=None,
                           raw_tail=None, audio_total: float = 0.0):
    """把用户提供的完整台词精确匹配到各语音分段的 text 字段（v2）。

    segments   —— whisper 分段（>=1s，含 start/end/text/duration），
                 仅当 raw_tail 缺失时作为语音边界来源
    transcript —— 用户权威台词全文（可含换行/标点）
    timeline   —— 词级/字符级时间锚点 [{u,s,e,t}]（由 _add_word_anchors 产出）
    raw_tail   —— 全部"像语音"的 whisper 原始分段（含 <1s，用于合并边界）
    audio_total—— 音频总时长（秒），用于判断台词是否超出音频

    有 timeline 时走"全局编辑对齐 + 锚点时刻映射 + 间隙中点切分"；
    无 timeline 或对齐不收敛时退回 v1 比例法。

    返回 (segments, note)。台词命中的段 text 为台词、aligned=True；
    未命中的段保留原识别文本、aligned=False。
    """
    text = (transcript or "").strip()
    if not text:
        return list(segments), "台词为空"
    if len(text) > 200000:
        return list(segments), "台词过长（超过 20 万字），请分段处理"
    atoms = _text_atoms(text)
    if not atoms:
        return list(segments), "台词里没有可朗读的文字（请直接粘贴纯文本）"

    src = raw_tail if raw_tail else segments
    speech = []
    for s in src:
        st = float(s.get("start") or 0)
        en = float(s.get("end") or 0)
        if en > st:
            speech.append({"start": st, "end": en,
                           "text": (s.get("text") or "").strip()})
    speech = _merge_short_speech(speech)
    speech = [s for s in speech
              if (s["end"] - s["start"]) >= MIN_SEGMENT_SECONDS - 1e-9]
    if not speech:
        return list(segments), "未识别到有效语音分段（时长 ≥1s），无法匹配台词"

    times = None
    fallback_reason = None
    if timeline:
        times = _match_atom_times(atoms, timeline, tail_floor=audio_total)
        if times is None:
            fallback_reason = "词级对齐未收敛（文本与识别结果差异过大或文本过长）"
    if times is None:
        segs, note = _align_proportional(speech, text)
        pre = "（未获得词级时间戳，退回按时长比例分配）" if fallback_reason is None \
            else f"（{fallback_reason}，退回按时长比例分配）"
        note = pre if not note else pre + "；" + note
        return segs, note

    segs, dropped = _cut_segments_by_times(speech, atoms, times, text)
    matched = sum(1 for s in segs if s.get("aligned"))
    parts = [f"已按词级锚点精确匹配 {matched}/{len(segs)} 段"]
    fb = len(segs) - matched
    if fb:
        parts.append(f"{fb} 段疑似误识别/无声，保留原识别文本，建议勾除")
    if dropped > 3:
        parts.append(f"台词末尾约 {dropped} 字超出音频时长，未匹配（请核对台词与音频是否对应）")
    return segs, "；".join(parts)


def align_job(job_id: str, transcript: str) -> dict:
    """对已完成转写的 job 重新做一次台词对齐（不改时间戳/音频，只替换文本）。
    供前端在转写完成后粘贴/修改台词时调用。返回更新后的 job 副本。
    """
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            raise RuntimeError("转写任务不存在或已过期，请重新转写")
        if j.get("status") != "done":
            raise RuntimeError("转写尚未完成，请稍候")
        segs0 = list(j.get("raw_tail") or j.get("segments") or [])
        timeline = j.get("words_timeline") or None
        raw_tail = j.get("raw_tail") or None
        total = float(j.get("audio_total") or 0.0)
    segs, note = align_text_to_segments(
        segs0, transcript, timeline=timeline,
        raw_tail=raw_tail, audio_total=total)
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["segments"] = segs
            _jobs[job_id]["aligned"] = True
            _jobs[job_id]["align_note"] = note
            j = _jobs[job_id]
    return dict(j)


def load_segment_audio(job_id: str, seg: dict):
    """按 segment 时间戳从 job 的 16k 音频切片，返回 (numpy 数组, 采样率)。

    供前端逐段试听与导入切片共用。
    """
    if JOBS_ROOT is None:
        raise RuntimeError("transcriber 未初始化")
    wav16 = JOBS_ROOT / job_id / "audio_16k.wav"
    if not wav16.exists():
        raise RuntimeError("转写音频已丢失，请重新转写")
    data, sr = sf.read(str(wav16), dtype="float32")
    total = len(data) / sr
    start = min(max(0.0, float(seg["start"])), total)
    end = min(max(start + 0.2, float(seg["end"])), total)
    i0, i1 = int(start * sr), int(end * sr)
    if i1 - i0 < int(MIN_SEGMENT_SECONDS * sr) or i0 >= len(data):
        raise RuntimeError("片段时长过短，无法切片")
    return data[i0:i1], sr


def import_segments(job_id: str, items) -> dict:
    """把 job 里勾选的 segment（可带用户修改后的文本）切片导入 training_store。

    items: [{idx:int, text:str}, ...]（text 为空/过短会被跳过）
    返回 {"imported": n, "skipped": m, "error": str|None}。
    """
    import voice_clone.training_store as tstore
    job = get_job(job_id)
    if job is None:
        raise RuntimeError("转写任务不存在或已过期，请重新转写")
    if job["status"] != "done":
        raise RuntimeError("转写尚未完成")
    seg_map = {s["idx"]: s for s in job["segments"]}
    if not tstore.SAMPLES_DIR:
        raise RuntimeError("training_store 未初始化")

    imported, skipped = 0, 0
    errors = []
    for it in items or []:
        try:
            idx = int(it.get("idx"))
            text = (it.get("text") or "").strip()
        except (TypeError, ValueError):
            skipped += 1
            continue
        seg = seg_map.get(idx)
        if seg is None or not text:
            skipped += 1
            continue
        try:
            chunk, sr = load_segment_audio(job_id, seg)
        except RuntimeError as e:
            skipped += 1
            errors.append(f"段 {idx + 1}: {e}")
            continue
        tmp = (JOBS_ROOT / job_id / f"chunk_{idx}.wav") if JOBS_ROOT else None
        if tmp is None:
            skipped += 1
            continue
        try:
            sf.write(str(tmp), chunk, sr)
            tstore.add_sample(
                str(tmp),
                text,
                name=f"转写{idx + 1:02d}",
            )
            imported += 1
        except ValueError as e:
            skipped += 1
            errors.append(f"段 {idx + 1}: {e}")
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # 导入完成即清理 job（残留由 TTL 兜底）
    drop_job(job_id)
    return {"imported": imported, "skipped": skipped,
            "error": ("; ".join(errors[:5]) if errors else None)}
