"""
声音资产 (Voice Asset) 本地持久化管理
=====================================
与「音色包」的分工，别混淆（这是本模块存在的理由）：

  · 音色包（voice_packs）—— **已提纯的代表参考**。
    创建时就做降噪 / 长音频分段融合，产物是约 25 秒的干净参考，定位是
    「一条可直接喂模型的声线」。它一旦生成，音色就冻结了。

  · 声音资产（voice_assets）—— **未加工的原始素材**。
    导入的音频原样保存（采样率、内容都不动；只有视频会先提取音轨），
    定位是「可反复复用的音频素材」：既能当清唱的原曲，也能当克隆的参考
    （此时按该页的参数现场做预处理）。

为什么不合成一个库：两者的**处理契约相反**。音色包若再喂一次预处理会二次
降噪、音色漂移；资产若被提前提纯就失去了当原曲的能力。混存会让这两套语义
无声地互相污染 —— 所以宁可两个目录、两个 manifest。

存储结构：  F:\VoxCPM2\voice_assets\
              manifest.json        —— 所有资产的元数据列表
              <id>.wav             —— 单声道 wav（保留源采样率）
              <id>_preview.wav     —— 前若干秒试听片段

对外接口：
  init(base_dir)             —— 指定根目录并创建 voice_assets 目录
  list_assets()              —— 元数据列表（剔除内部文件路径字段）
  import_asset(...)          —— 落库：读源 → 归一化为单声道 wav + 试听片段
  get_asset_paths(asset_id)  —— 返回 (wav, preview) Path，缺失返回 (None, None)
  get_asset_meta(asset_id)   —— 完整元数据（含 file 字段）
  rename_asset(asset_id, nm) —— 重命名，返回更新后的元数据或 None
  delete_asset(asset_id)     —— 删除资产及其文件，返回是否成功
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

VOICE_ASSET_DIR: Path | None = None
MANIFEST_PATH: Path | None = None
_lock = threading.Lock()

# 试听片段时长（秒）
PREVIEW_SECONDS = 8.0
# 单个资产允许的最大时长（秒）。
# 放宽到 20 分钟：清唱的原曲往往是完整歌曲，若套用「参考音频 10 分钟」那条
# 上限，正常素材会被挡在门外。资产的时长由使用者各自的流程去把关。
MAX_ASSET_SECONDS = 1200.0
# 分块读写的块大小（帧）：保证长音频导入时内存占用恒定，不随文件变大
_BLOCK = 1 << 16


def init(base_dir) -> None:
    global VOICE_ASSET_DIR, MANIFEST_PATH
    VOICE_ASSET_DIR = Path(base_dir) / "voice_assets"
    VOICE_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH = VOICE_ASSET_DIR / "manifest.json"


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


def list_assets() -> list:
    """返回给前端的列表（剔除内部文件路径字段）。"""
    out = []
    for r in _load():
        d = dict(r)
        d.pop("file", None)
        d.pop("preview_file", None)
        out.append(d)
    return out


def get_asset_meta(asset_id: str) -> dict | None:
    for r in _load():
        if r.get("id") == asset_id:
            return r
    return None


def get_asset_paths(asset_id: str):
    """返回 (wav_path, preview_path)；找不到返回 (None, None)。"""
    for r in _load():
        if r.get("id") == asset_id:
            wav = VOICE_ASSET_DIR / r["file"] if r.get("file") else None
            prev = VOICE_ASSET_DIR / r["preview_file"] if r.get("preview_file") else None
            if wav is not None and wav.exists():
                return wav, prev
            return None, None
    return None, None


def _stream_to_wav(src: str, dst: Path, prev: Path) -> tuple[int, float]:
    """分块把 src 转成单声道 wav 写到 dst，同时把开头 PREVIEW_SECONDS 秒写进 prev。

    返回 (采样率, 时长秒)。任何解码/写入失败都抛 ValueError（由调用方兜底转译）。
    """
    import soundfile as sf

    with sf.SoundFile(src) as fin:
        sr = int(fin.samplerate)
        if sr <= 0:
            raise ValueError("源音频采样率无效")
        prev_left = int(PREVIEW_SECONDS * sr)
        total = 0
        with sf.SoundFile(str(dst), "w", samplerate=sr, channels=1,
                          subtype="PCM_16") as fo, sf.SoundFile(
                str(prev), "w", samplerate=sr, channels=1,
                subtype="PCM_16") as fp:
            while True:
                blk = fin.read(_BLOCK, dtype="float32", always_2d=True)
                if blk.shape[0] == 0:
                    break
                mono = blk.mean(axis=1) if blk.shape[1] > 1 else blk[:, 0]
                total += int(mono.shape[0])
                if total / sr > MAX_ASSET_SECONDS:
                    raise ValueError(
                        "音频过长（超过 %d 分钟），请先裁剪后再导入"
                        % (MAX_ASSET_SECONDS // 60)
                    )
                fo.write(mono)
                if prev_left > 0:
                    n = min(int(mono.shape[0]), prev_left)
                    fp.write(mono[:n])
                    prev_left -= n
    if total == 0:
        raise ValueError("音频内容为空（没有可用的采样点）")
    return sr, total / sr


def import_asset(source_path: str, name: str = "", source_name: str = "",
                 kind: str = "audio", ffmpeg_path: str | None = None,
                 note: str = "") -> dict:
    """把一个源文件（音频或视频）导入为声音资产。

    流程：优先直接解码（wav/flac/ogg/mp3 视 libsndfile 版本而定）；解不开且
    提供了 ffmpeg_path 时，先转成单声道 wav 再入库（视频走的就是这条）。
    原文件不会被修改；产物是资产目录下的单声道 wav（保留源采样率）+ 试听片段。
    """
    if VOICE_ASSET_DIR is None:
        raise RuntimeError("voice_assets 未初始化（缺少 init 调用）")

    aid = uuid.uuid4().hex[:10]
    dst = VOICE_ASSET_DIR / f"{aid}.wav"
    prev = VOICE_ASSET_DIR / f"{aid}_preview.wav"

    sr = 0
    dur = 0.0
    first_err: Exception | None = None
    try:
        sr, dur = _stream_to_wav(str(source_path), dst, prev)
    except Exception as e:                                  # noqa: BLE001
        first_err = e

    if sr == 0 and ffmpeg_path:
        # 解不开就用 ffmpeg 转一道（视频、以及 libsndfile 不认的编码走这里）。
        # 不指定 -ar：保留源采样率，避免无谓的重采样损失。
        import subprocess
        import tempfile
        td = Path(tempfile.mkdtemp())
        tmp_wav = td / "src.wav"
        try:
            r = subprocess.run(
                [ffmpeg_path, "-y", "-i", str(source_path), "-vn", "-ac", "1",
                 str(tmp_wav)],
                capture_output=True,
            )
            if r.returncode != 0 or not tmp_wav.exists():
                raise ValueError(
                    "音轨提取/转码失败：" + r.stderr.decode("utf-8", "ignore")[:200]
                )
            sr, dur = _stream_to_wav(str(tmp_wav), dst, prev)
        except Exception as e:                              # noqa: BLE001
            first_err = e
        finally:
            try:
                tmp_wav.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                td.rmdir()
            except OSError:
                pass

    if sr == 0:
        # 清理失败尝试留下的半成品，避免占着目录却不在 manifest 里
        for p in (dst, prev):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        detail = str(first_err) if first_err else "未知错误"
        if not ffmpeg_path:
            detail += "（当前未检测到 ffmpeg，视频与部分压缩格式需要它）"
        raise ValueError("无法导入该素材：" + detail)

    rec = {
        "id": aid,
        "name": (name or "").strip() or f"声音资产_{time.strftime('%m%d_%H%M')}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": source_name or Path(str(source_path)).name,
        "source_kind": "video" if str(kind).lower() == "video" else "audio",
        "duration": round(float(dur), 2),
        "sample_rate": int(sr),
        "note": (note or "").strip(),
        "file": dst.name,
        "preview_file": prev.name,
    }
    with _lock:
        recs = _load()
        recs.append(rec)
        _save(recs)
    return rec


def rename_asset(asset_id: str, name: str) -> dict | None:
    """重命名资产（不改动音频文件）。找不到返回 None。"""
    new = (name or "").strip()
    if not new:
        return None
    with _lock:
        recs = _load()
        for r in recs:
            if r.get("id") == asset_id:
                r["name"] = new
                _save(recs)
                return r
    return None


def delete_asset(asset_id: str) -> bool:
    """删除资产及其音频文件。返回是否删除成功。"""
    with _lock:
        recs = _load()
        target = next((r for r in recs if r.get("id") == asset_id), None)
        if target is None:
            return False
        for f in (target.get("file"), target.get("preview_file")):
            if f:
                p = VOICE_ASSET_DIR / f
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        _save([r for r in recs if r.get("id") != asset_id])
        return True
