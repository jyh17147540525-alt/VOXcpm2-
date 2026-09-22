"""
音色包 (Voice Pack) 本地持久化管理
=================================
让用户“一次提取、长期复用”：上传参考音频并抽取/清洗音色后，把这条声线
以音色包形式保存在本地（manifest.json + wav 文件），之后每次克隆只需从
已保存列表中选择，无需再次上传一长段音频。

存储结构：  <项目根目录>\voice_packs\
              manifest.json              —— 所有音色包的元数据列表
              <id>.wav                  —— 抽取/清洗后的代表参考音频（送入模型用）
              <id>_preview.wav          —— 前若干秒试听片段

对外接口：
  init(base_dir)             —— 指定根目录并创建 voice_packs 目录
  list_packs()               —— 返回元数据列表（不含文件路径，供前端展示）
  create_pack(...)           —— 抽取音色，保存为音色包，返回元数据
  get_pack_paths(pack_id)    —— 返回 (processed_wav, preview_wav) Path，缺失返回 (None,None)
  delete_pack(pack_id)       —— 删除音色包及其文件，返回是否成功
"""
from __future__ import annotations

import os
import json
import time
import uuid
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

VOICE_PACK_DIR: Path | None = None
MANIFEST_PATH: Path | None = None
_lock = threading.Lock()

# 试听片段时长（秒）
PREVIEW_SECONDS = 8.0
# 抽取目标时长（秒）：长音频会自动融合为约该长度的代表参考
TARGET_DUR = 25.0


def init(base_dir) -> None:
    global VOICE_PACK_DIR, MANIFEST_PATH
    VOICE_PACK_DIR = Path(base_dir) / "voice_packs"
    VOICE_PACK_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH = VOICE_PACK_DIR / "manifest.json"


def _load() -> list:
    if MANIFEST_PATH is None or not MANIFEST_PATH.exists():
        return []
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(records: list) -> None:
    assert MANIFEST_PATH is not None
    MANIFEST_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_packs() -> list:
    """返回给前端的列表（剔除内部文件路径字段）。"""
    recs = _load()
    out = []
    for r in recs:
        d = dict(r)
        d.pop("file", None)
        d.pop("preview_file", None)
        out.append(d)
    return out


def get_pack_paths(pack_id: str):
    """返回 (processed_wav_path, preview_wav_path)；找不到返回 (None, None)。"""
    recs = _load()
    for r in recs:
        if r.get("id") == pack_id:
            wav = VOICE_PACK_DIR / r["file"] if r.get("file") else None
            prev = (
                VOICE_PACK_DIR / r["preview_file"]
                if r.get("preview_file")
                else None
            )
            if wav is not None and wav.exists():
                return wav, prev
            return None, None
    return None, None


def get_pack_meta(pack_id: str) -> dict | None:
    """返回某音色包的完整元数据（含 accelerated 等），找不到返回 None。"""
    for r in _load():
        if r.get("id") == pack_id:
            return r
    return None


def create_pack(
    name: str,
    ref_path: str,
    denoise: bool = True,
    remove_bg: bool = False,
    target_dur: float = TARGET_DUR,
    source_name: str = "",
    accelerated: bool = False,
) -> dict:
    """
    抽取音色包：
      1) 复用 voice_clone.prepare_reference 做降噪/去背景/长音频分段融合，
         得到“干净有界”的代表参考音频；
      2) 落盘为 <id>.wav，并截取前 PREVIEW_SECONDS 秒作为试听片段；
      3) 写入 manifest.json。
    返回元数据 dict。
    """
    from voice_clone import prepare_reference

    proc_path, rep = prepare_reference(
        ref_path,
        denoise=denoise,
        remove_bg=remove_bg,
        target_dur=target_dur,
        use_cache=False,
    )
    y, sr = sf.read(proc_path)
    if y.ndim > 1:
        y = y[:, 0]

    pid = uuid.uuid4().hex[:10]
    file_name = f"{pid}.wav"
    sf.write(str(VOICE_PACK_DIR / file_name), y, sr)

    # 试听片段：取前 PREVIEW_SECONDS 秒（不足则取全部）
    prev_len = int(min(PREVIEW_SECONDS, len(y) / sr) * sr)
    prev = y[:prev_len]
    prev_name = f"{pid}_preview.wav"
    sf.write(str(VOICE_PACK_DIR / prev_name), prev, sr)

    rec = {
        "id": pid,
        "name": (name or f"音色包_{time.strftime('%m%d_%H%M')}").strip(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name,
        "source_duration": rep.get("input_duration"),
        "processed_duration": round(len(y) / sr, 2),
        "sample_rate": int(sr),
        "params": {"denoise": denoise, "remove_bg": remove_bg, "target_dur": target_dur},
        "adaptation": rep.get("adaptation", {}),
        "accelerated": bool(accelerated),
        "file": file_name,
        "preview_file": prev_name,
    }
    with _lock:
        recs = _load()
        recs.append(rec)
        _save(recs)
    return rec


def delete_pack(pack_id: str) -> bool:
    """删除音色包及其音频文件。返回是否删除成功。"""
    with _lock:
        recs = _load()
        target = next((r for r in recs if r.get("id") == pack_id), None)
        if target is None:
            return False
        for f in (target.get("file"), target.get("preview_file")):
            if f:
                p = VOICE_PACK_DIR / f
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        new = [r for r in recs if r.get("id") != pack_id]
        _save(new)
        return True
