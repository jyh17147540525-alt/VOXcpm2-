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


def start_transcribe(src_path, label: str = "") -> dict:
    """后台线程启动一次转写。src_path 会被复制进 job 目录（16k mono wav）。

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

    def _worker(job_id: str, wav: Path, lb: str):
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
            )
            lang = getattr(info, "language", None)
            job["lang"] = lang
            segs = []
            for seg in segments:
                text = (seg.text or "").strip()
                d = float(getattr(seg, "end", 0) - getattr(seg, "start", 0))
                if d < MIN_SEGMENT_SECONDS or d > MAX_SEGMENT_SECONDS:
                    continue
                if not _looks_like_speech(text):
                    continue
                segs.append({
                    "idx": len(segs),
                    "text": text,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "duration": round(d, 2),
                })
                if len(segs) % 20 == 0:
                    job["progress"] = min(85, 30 + len(segs))
            job["segments"] = segs
            job["progress"] = 100
            job["status"] = "done"
            job["message"] = (f"转写完成：{len(segs)} 段"
                              + (f"（{lang}）" if lang else ""))
        except Exception as e:
            if job is not None:
                job["status"] = "error"
                job["error"] = str(e)
                job["message"] = f"转写失败: {e}"
        finally:
            _busy.release()

    threading.Thread(target=_worker, args=(job["job_id"], wav16, label),
                     daemon=True).start()
    return dict(job)


def _looks_like_speech(text: str) -> bool:
    """过滤空段/纯标点/纯符号段。要求至少含一个 CJK 或字母数字字符。"""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            return True
    return False


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
