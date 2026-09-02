"""
训练数据 (Training Data) 本地持久化管理
=====================================
让用户「加语音、填台词，让模型持续学习」：每条样本 = 一段语音 + 对应的
逐字文本，保存在本地。训练时导出为 VoxCPM training 需要的 JSONL manifest
（{"text": ..., "audio": ...} 格式），直接喂给官方 dataset/packer。

存储结构：  F:\\VoxCPM2\\training_data\\
              manifest.json              —— 所有样本的元数据列表
              samples/<id>.wav           —— 训练语音（保持原采样率，训练时重采样）
              loras/<name>/              —— LoRA 权重（由 trainer 管理）

对外接口：
  init(base_dir)              —— 指定根目录并创建目录结构
  list_samples()              —— 返回样本元数据列表（供前端展示）
  add_sample(src_path, text, name) —— 保存一条样本，返回元数据
  delete_sample(sample_id)    —— 删除样本及其音频文件
  export_manifest(out_path)   —— 导出训练用 JSONL manifest，返回路径
  get_stats()                 —— 统计（样本数 / 总时长 / 总字数）
  loras_dir(), list_loras(), delete_lora() —— LoRA 权重目录管理
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import soundfile as sf

TRAIN_DIR: Path | None = None
SAMPLES_DIR: Path | None = None
LORA_DIR: Path | None = None
MANIFEST_PATH: Path | None = None
_lock = threading.Lock()

# 单条样本的时长约束（秒）：太短学不到东西，太长显存吃不消且会被 max_len 截断
MIN_SAMPLE_SECONDS = 1.0
MAX_SAMPLE_SECONDS = 30.0
# 台词长度约束（字）
MIN_TEXT_CHARS = 1
MAX_TEXT_CHARS = 400


def init(base_dir):
    """指定根目录并创建 training_data 目录结构。"""
    global TRAIN_DIR, SAMPLES_DIR, LORA_DIR, MANIFEST_PATH
    base = Path(base_dir)
    TRAIN_DIR = base / "training_data"
    SAMPLES_DIR = TRAIN_DIR / "samples"
    LORA_DIR = TRAIN_DIR / "loras"
    MANIFEST_PATH = TRAIN_DIR / "manifest.json"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text("[]", encoding="utf-8")


# ------------------------------------------------------------------ 内部工具
def _load() -> list:
    if MANIFEST_PATH is None or not MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(items: list):
    if MANIFEST_PATH is None:
        return
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def _probe_audio(path: Path) -> tuple[float, int]:
    """返回 (时长秒, 采样率)。无法解码时抛 ValueError。"""
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate)
    except Exception as e:
        raise ValueError(f"音频无法解码（请用 wav/mp3/flac 等常见格式）: {e}")


# ------------------------------------------------------------------ 对外接口
def list_samples() -> list:
    """返回样本元数据列表（按创建时间倒序）。"""
    with _lock:
        items = _load()
    return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)


def add_sample(src_path, text: str, name: str = "") -> dict:
    """保存一条训练样本（语音 + 逐字文本）。

    src_path: 临时音频文件路径（会复制进 samples 目录）
    text:     该段语音对应的逐字台词（必须与音频内容一致）
    name:     可选备注名，便于识别
    """
    if SAMPLES_DIR is None:
        raise RuntimeError("training_store 未初始化，请先调用 init()")
    src = Path(src_path)
    if not src.exists():
        raise ValueError("音频文件不存在")

    text = (text or "").strip()
    if not (MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS):
        raise ValueError(f"台词长度需在 {MIN_TEXT_CHARS}~{MAX_TEXT_CHARS} 字之间")

    duration, sr = _probe_audio(src)
    if duration < MIN_SAMPLE_SECONDS:
        raise ValueError(f"音频太短（{duration:.1f}s），至少需要 {MIN_SAMPLE_SECONDS:g} 秒")
    if duration > MAX_SAMPLE_SECONDS:
        raise ValueError(f"音频太长（{duration:.1f}s），请裁剪到 {MAX_SAMPLE_SECONDS:g} 秒以内")

    sid = uuid.uuid4().hex[:12]
    dst = SAMPLES_DIR / f"{sid}.wav"
    # 统一存为 wav，避免训练时依赖各种解码器
    if src.suffix.lower() == ".wav":
        shutil.copy2(str(src), str(dst))
    else:
        try:
            data, orig_sr = sf.read(str(src))
            sf.write(str(dst), data, orig_sr)
        except Exception as e:
            raise ValueError(f"音频转码失败: {e}")

    meta = {
        "id": sid,
        "name": (name or "").strip() or f"样本{time.strftime('%m%d_%H%M%S')}",
        "text": text,
        "audio": str(dst),
        "duration": round(duration, 2),
        "sample_rate": sr,
        "chars": len(text),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with _lock:
        items = _load()
        items.append(meta)
        _save(items)
    return meta


def delete_sample(sample_id: str) -> bool:
    """删除样本及其音频文件，返回是否成功。"""
    with _lock:
        items = _load()
        target = next((x for x in items if x.get("id") == sample_id), None)
        if target is None:
            return False
        items = [x for x in items if x.get("id") != sample_id]
        _save(items)
    try:
        p = Path(target["audio"])
        if p.exists():
            p.unlink()
    except Exception:
        pass
    return True


def export_manifest(out_path=None) -> Path:
    """导出 VoxCPM training 需要的 JSONL manifest（每行 {"text","audio"}）。"""
    if TRAIN_DIR is None:
        raise RuntimeError("training_store 未初始化，请先调用 init()")
    out = Path(out_path) if out_path else (TRAIN_DIR / "train_manifest.jsonl")
    items = list_samples()
    lines = []
    for it in items:
        audio = Path(it["audio"])
        if not audio.exists():
            continue
        lines.append(json.dumps({"text": it["text"], "audio": str(audio)}, ensure_ascii=False))
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def get_stats() -> dict:
    """统计信息：样本数 / 总时长 / 总字数。"""
    items = list_samples()
    return {
        "count": len(items),
        "total_duration": round(sum(float(x.get("duration", 0)) for x in items), 2),
        "total_chars": sum(int(x.get("chars", 0)) for x in items),
    }


# ------------------------------------------------------------------ LoRA 管理
def loras_dir() -> Path:
    """LoRA 权重存放目录（不存在则创建）。"""
    if LORA_DIR is None:
        raise RuntimeError("training_store 未初始化，请先调用 init()")
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    return LORA_DIR


def list_loras() -> list:
    """列出已训练的 LoRA 权重。"""
    d = loras_dir()
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        ckpt = p / "lora_weights.ckpt"
        meta_f = p / "meta.json"
        meta = {}
        if meta_f.exists():
            try:
                meta = json.loads(meta_f.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        out.append({
            "name": p.name,
            "path": str(p),
            "has_weights": ckpt.exists(),
            "size_mb": round(ckpt.stat().st_size / 1e6, 2) if ckpt.exists() else 0,
            "created_at": meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S",
                                                             time.localtime(p.stat().st_mtime))),
            "base_samples": meta.get("base_samples", 0),
            "steps": meta.get("steps", 0),
            "final_loss": meta.get("final_loss"),
            "config": meta.get("config", {}),
        })
    return out


def delete_lora(name: str) -> bool:
    """删除一个 LoRA 权重目录。"""
    d = loras_dir() / name
    if not d.exists() or not d.is_dir():
        return False
    shutil.rmtree(str(d), ignore_errors=True)
    return not d.exists()
