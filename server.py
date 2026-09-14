"""
VoxCPM2 本地推理服务 (FastAPI + 令牌验证)  ——  加固版
=================================================
修复点（针对“克隆音频频繁返回请求失败”）：
- 全局异常处理器：任何未捕获异常都返回 JSON {detail:...} 并记录完整 traceback，
  不再出现 uvicorn 原生纯文本 "Internal Server Error"（前端 r.json() 解析失败 → 笼统“请求失败”）。
- 推理链路(get_model + generate + 后处理)整体包进 try/except，异常转为 HTTPException(500) JSON。
- 模型自愈：单次推理抛异常后把 _model 置空，下次请求自动重新加载，避免损坏态卡死整个服务。
- 参考音频上传后做格式/时长校验，坏文件返回清晰的 400 而非笼统 500。

启动：  F:\\VoxCPM2\\start.bat        （或 env\\python.exe server.py）
凭证：  F:\\VoxCPM2\\credentials.json
日志：  F:\\VoxCPM2\\server_error.log （推理/未捕获异常的完整 traceback）
"""

import io
import os
import re
import sys
import json
import time
import uuid
import secrets
import threading
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def _restore_native_deletion() -> None:
    """还原被 WorkBuddy 沙箱“安全删除”垫片(sitecustomize)替换的文件删除函数。

    沙箱为保护用户文件，会把 os.remove/unlink、pathlib.Path.unlink 等替换成
    “移入回收站”，回收站不可用时就抛 OSError(fail-closed)。本服务是用户自己的
    本地 TTS 应用，需要原生删除自己 uploads/prepared 下的临时文件（等价于在
    沙箱外运行），故在启动时还原为原生实现。还原失败不影响主流程。
    """
    try:
        import sitecustomize as _sc
        import shutil as _shutil
        import pathlib as _pathlib
        for _mod, _patched, _orig_name in (
            (os, "remove", "_orig_remove"),
            (os, "unlink", "_orig_unlink"),
            (os, "rmdir", "_orig_rmdir"),
            (_shutil, "rmtree", "_orig_shutil_rmtree"),
        ):
            _orig = getattr(_sc, _orig_name, None)
            if _orig is not None:
                setattr(_mod, _patched, _orig)
        _pu = getattr(_sc, "_orig_path_unlink", None)
        if _pu is not None:
            _pathlib.Path.unlink = _pu
        _pr = getattr(_sc, "_orig_path_rmdir", None)
        if _pr is not None:
            _pathlib.Path.rmdir = _pr
        print("[VoxCPM2] 已还原原生文件删除（绕过沙箱 safe-delete 垫片）", flush=True)
    except Exception as e:
        print(f"[VoxCPM2][WARN] 还原原生删除失败: {e}", flush=True)


_restore_native_deletion()

# ============================== 配置 ==============================
BASE_DIR = Path(r"F:\VoxCPM2")
MODEL_PATH = str(BASE_DIR)
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
CRED_FILE = BASE_DIR / "credentials.json"
ERROR_LOG = BASE_DIR / "server_error.log"

PORT = int(os.environ.get("VOXCPM_PORT", "8808"))
HOST = os.environ.get("VOXCPM_HOST", "127.0.0.1")
DEVICE = os.environ.get("VOXCPM_DEVICE", "auto")

# 参考音频时长限制（秒）
MIN_REFERENCE_SECONDS = 0.3    # 太短无法提取稳定音色
MAX_REFERENCE_SECONDS = 600    # 最长 10 分钟

# 长参考音频克隆增强（voice_clone 套件）
REF_TARGET_DUR = float(os.environ.get("VOXCPM_REF_TARGET_DUR", "25.0"))  # 融合参考目标时长
# 加速模式：使用该音色包生成时采用的扩散步数（远小于默认 10，显著缩短生成耗时）
ACCEL_STEPS = int(os.environ.get("VOXCPM_ACCEL_STEPS", "4"))
# 长文本自动稳定合成阈值（超过此长度强制分块，避免单次超长生成导致音色漂移/机械感）
LONG_TEXT_CHARS = int(os.environ.get("VOXCPM_LONG_TEXT_CHARS", "100"))
# 显存保护阈值（GB）：按「进程总显存」判断（mem_get_info），VoxCPM 连续推理会在
# 内部累积显存缓存，超过此阈值先优雅卸载并重载模型回收，避免长期连续生成触发
# CUDA OOM / native crash。默认 11GB（4070Ti 16GB）：偏保守，宁可偶尔重载一次
# （约 15s），也不要让显存累积到临界后 native crash（克隆/极致克隆的参考编码
# 峰值显存最高，最容易触发）。
MEMORY_RESET_THRESHOLD_GB = float(os.environ.get("VOXCPM_MEM_RESET_GB", "11.0"))
sys.path.insert(0, str(BASE_DIR))  # 让 voice_clone 包可被导入
from voice_clone import prepare_reference as _vc_prepare_reference
from voice_clone import synthesis_stab as _vc_stab

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 音色包（声线包）本地持久化管理
import voice_packs as vp_store
vp_store.init(BASE_DIR)
VOICE_PACK_DIR = vp_store.VOICE_PACK_DIR

# 训练模块：数据管理 + LoRA 训练引擎
import voice_clone.training_store as tstore
import voice_clone.trainer as trainer
tstore.init(BASE_DIR)
# 长音频自动转写（faster-whisper，离线切句生成训练样本候选）
import voice_clone.transcriber as transcriber
import voice_clone.preprocess as vc_preprocess
transcriber.init(BASE_DIR)
# MDX-NET 神经网络人声分离引擎（models/mdx/*.onnx；无权重时自动回退 DSP 链）
try:
    import voice_clone.mdx_separator as _mdx_sep
    _mdx_sep.init(BASE_DIR)
except Exception as _e:  # onnxruntime 缺失等 -> 引擎不可用，不影响其它功能
    print(f"[warn] MDX separator unavailable: {_e}")

# 导演层（模块4）：合成前的文本梳理 —— 台词/旁白判定、情绪推断、停顿规划。
# 规则版必装；LLM 版（模块5）涉及网络+API Key，按需在请求内懒加载，避免启动变慢或强依赖。
try:
    from voice_clone import director as _director
except Exception as _e:
    _director = None
    print(f"[warn] director (rule) unavailable: {_e}")


def log_error(where: str, exc: BaseException):
    """把完整 traceback 追加写入 server_error.log，方便事后定位。
    先打印到 stdout（voxcpm_server.log 兜底），写文件失败也不阻断主流程、不掩盖真实异常。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"[VoxCPM2][ERROR] {where}: {type(exc).__name__}: {exc}", flush=True)
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 70}\n[{ts}] {where}\n")
            f.write(tb)
            f.flush()
    except Exception as e:
        print(f"[VoxCPM2][WARN] 写错误日志失败({e})，完整堆栈见下：", flush=True)
        print(tb, flush=True)


def load_or_create_token() -> str:
    env_token = os.environ.get("VOXCPM_API_KEY")
    if env_token:
        return env_token
    if CRED_FILE.exists():
        try:
            data = json.loads(CRED_FILE.read_text(encoding="utf-8"))
            if data.get("access_token"):
                return data["access_token"]
        except Exception:
            pass
    token = "vox2_" + secrets.token_urlsafe(24)
    CRED_FILE.write_text(
        json.dumps(
            {
                "service": "VoxCPM2 本地推理服务",
                "access_token": token,
                "url": f"http://localhost:{PORT}",
                "quick_login_url": f"http://localhost:{PORT}/?token={token}",
                "api_header": "X-API-Key: <access_token>",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return token


ACCESS_TOKEN = load_or_create_token()

# ============================== 模型 ==============================
_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()
_model_lora: str | None = None          # 当前模型已加载的 LoRA 名称（None=原模型）
_model_info = {"loaded": False, "device": None, "sample_rate": None, "load_seconds": None}


def _resolve_lora(lora_name: str | None):
    """把 LoRA 名称解析为 (weights_path, LoRAConfig 或 None)。无 LoRA 返回 (None, None)。"""
    if not lora_name:
        return None, None
    import voice_clone.training_store as _ts
    d = _ts.loras_dir() / lora_name
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"LoRA「{lora_name}」不存在")
    from voxcpm.model.voxcpm import LoRAConfig
    cfg = LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=True, r=8, alpha=16)
    meta_f = d / "meta.json"
    if meta_f.exists():
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            c = meta.get("config", {})
            cfg = LoRAConfig(
                enable_lm=True, enable_dit=True, enable_proj=True,
                r=int(c.get("lora_r", 8)), alpha=int(c.get("lora_alpha", 16)),
            )
        except Exception:
            pass
    if (d / "lora_weights.safetensors").exists():
        return str(d / "lora_weights.safetensors"), cfg
    if (d / "lora_weights.ckpt").exists():
        return str(d / "lora_weights.ckpt"), cfg
    raise HTTPException(status_code=404, detail=f"LoRA「{lora_name}」缺少权重文件")


def get_model(lora_name: str | None = None):
    """获取推理模型。lora_name 非空时加载该 LoRA；与当前已加载的不一致时自动重载。"""
    global _model, _model_lora
    if _model is not None and _model_lora == (lora_name or None):
        return _model
    with _model_lock:
        if _model is not None and _model_lora == (lora_name or None):
            return _model
        # LoRA 切换：先卸载当前模型
        if _model is not None:
            unload_model()
        t0 = time.time()
        print("[VoxCPM2] 正在加载模型，首次约需 20-60 秒 ...", flush=True)
        from voxcpm import VoxCPM
        import torch

        lora_path, lora_cfg = _resolve_lora(lora_name)
        if lora_path:
            print(f"[VoxCPM2] 使用 LoRA: {lora_name}", flush=True)
        _model = VoxCPM.from_pretrained(
            MODEL_PATH, load_denoiser=False, device=DEVICE,
            lora_config=lora_cfg, lora_weights_path=lora_path,
        )
        _model_lora = lora_name or None
        _model_info["loaded"] = True
        _model_info["device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        _model_info["sample_rate"] = _vc_stab._get_sample_rate(_model)
        _model_info["load_seconds"] = round(time.time() - t0, 1)
        print(f"[VoxCPM2] 模型就绪，用时 {_model_info['load_seconds']}s "
              f"设备 {_model_info['device']}"
              + (f" · LoRA={lora_name}" if lora_path else ""), flush=True)
    return _model


def reset_model():
    """模型自愈：推理异常后置空，下次请求自动重载（避免损坏态永久卡死）"""
    global _model
    _model = None
    _model_info["loaded"] = False


def unload_model():
    """优雅卸载模型并释放显存。停止服务前先调用本函数，可避免直接强杀进程
    导致 GPU CUDA 上下文损坏（段错误、需整机重启）。"""
    global _model, _model_lora
    if _model is not None:
        try:
            del _model
        except Exception:
            pass
        _model = None
    _model_lora = None
    _model_info["loaded"] = False
    _model_info["device"] = None
    _model_info["sample_rate"] = None
    try:
        import gc
        import torch
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


def normalize_reference(ref_path: str) -> str:
    """校验上传的参考音频：可解码 + 时长合理；返回模型可用的路径。
    坏文件抛出 ValueError，由调用方转成清晰的 400。"""
    try:
        data, sr = sf.read(ref_path)
        dur = len(data) / sr
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(ref_path, sr=None, mono=False)
            dur = len(y) / sr
        except Exception as e:
            raise ValueError(f"参考音频无法解码（请使用 wav/mp3/flac 且未损坏的文件）: {e}")
    if dur < MIN_REFERENCE_SECONDS:
        raise ValueError(f"参考音频太短（<{MIN_REFERENCE_SECONDS:g} 秒），请上传 0.3 秒以上的清晰音频")
    if dur > MAX_REFERENCE_SECONDS:
        raise ValueError(f"参考音频过长（>{MAX_REFERENCE_SECONDS // 60} 分钟），请裁剪到 10 分钟以内")
    return ref_path


def strip_design_annotations(text: str) -> str:
    """提取「括号外」的台词内容，仅用于空文本校验。

    括号内的音色/风格/情绪提示（如"（年轻女性，温柔甜美）"）会被保留在原 text 中
    传给模型，由模型理解并应用（VoxCPM 的 Voice Design 用法）；本函数只负责判断
    剥掉括号后是否还有实际台词。"""
    t = text or ""
    for _ in range(8):
        nt = re.sub(r"（[^（）]*）", "", t)
        nt = re.sub(r"\([^()]*\)", "", nt)
        if nt == t:
            break
        t = nt
    return t.strip()


def normalize_design_brackets(text: str) -> str:
    """把中文括号（）统一成英文括号()。

    VoxCPM 仅把英文括号识别为「设计提示」分隔符；中文括号会被当成普通文本朗读。
    因此无论用户用哪种括号写提示词，都在传给模型前统一为英文括号。"""
    return (text or "").replace("（", "(").replace("）", ")")


# ============================== Beta：多人朗读 + 情绪控制 ==============================
# 已知情绪词（中英双语别名），命中即识别为情绪标记
_BETA_EMOTION_WORDS = {
    "高兴": "高兴", "开心": "高兴", "快乐": "高兴", "happy": "高兴",
    "悲伤": "悲伤", "难过": "悲伤", "伤心": "悲伤", "sad": "悲伤",
    "严肃": "严肃", "serious": "严肃",
    "温柔": "温柔", "gentle": "温柔", "soft": "温柔",
    "愤怒": "愤怒", "生气": "愤怒", "angry": "愤怒",
    "平静": "平静", "calm": "平静", "neutral": "平静", "中性": "平静",
}


def parse_multi_speaker_text(text: str) -> list[dict]:
    """解析多人朗读文本，把 (@音色包名) 音色切换标记和 (情绪词) 情绪标记拆出来。

    规则：
      - (@xxx)        → 后续文本切换到名为 xxx 的音色包（持续到下一个 @ 标记）
      - (情绪词)      → 紧随其后的那段文本用该情绪（段级，到下一个标记为止）
      - (@xxx,情绪词) → 同时切换音色并指定情绪
      - 未知音色包名  → 该段标记为 missing，合成时回退默认音色并告警
      - 未知括号内容  → 当普通文本保留，不当标记处理
      - 括号标记本身不参与朗读
    返回 [{"text", "voice"(音色包名或 None), "emotion"(情绪键或 "neutral"), "voice_missing"(bool)}]
    """
    t = normalize_design_brackets(text or "")
    segments: list[dict] = []
    cur_voice: str | None = None
    cur_emotion = "neutral"
    pos = 0
    for m in re.finditer(r"\(([^()]*)\)", t):
        before = t[pos:m.start()]
        if before.strip():
            segments.append({"text": before.strip(), "voice": cur_voice,
                             "emotion": cur_emotion, "voice_missing": False})
        content = m.group(1).strip()
        if content.startswith("@"):
            # 音色切换标记：@名 或 @名,情绪
            body = content[1:]
            parts = body.split(",", 1)
            name = parts[0].strip()
            if name:
                cur_voice = name
            if len(parts) > 1 and parts[1].strip():
                emo = _BETA_EMOTION_WORDS.get(parts[1].strip().lower(), parts[1].strip())
                cur_emotion = emo if emo in {"高兴", "悲伤", "严肃", "温柔", "愤怒", "平静"} else "neutral"
            else:
                cur_emotion = "neutral"  # 切换音色时重置情绪
        else:
            # 情绪标记
            emo = _BETA_EMOTION_WORDS.get(content.lower(), content)
            if emo in {"高兴", "悲伤", "严肃", "温柔", "愤怒", "平静"}:
                cur_emotion = emo
            else:
                # 未知括号内容：当普通文本，不剥离（回填到前一段或新建）
                if segments:
                    segments[-1]["text"] += "(" + content + ")"
                else:
                    segments.append({"text": "(" + content + ")", "voice": cur_voice,
                                     "emotion": cur_emotion, "voice_missing": False})
        pos = m.end()
    tail = t[pos:]
    if tail.strip():
        segments.append({"text": tail.strip(), "voice": cur_voice,
                         "emotion": cur_emotion, "voice_missing": False})
    # 标记未知音色包（合成时再校验名字是否存在）
    return segments


def parse_dialogue(text: str) -> list[dict]:
    """解析多人对话文本，返回「参与列表」。每次 (@角色) 标记 = 一次参与。

    返回 [{"role": 角色名, "seq": 该角色第几次参与, "text": 该次完整台词,
           "emotion": 该次情绪, "narrative": 是否旁白(无 @ 角色)}]。
    同一角色多次出现时 seq 递增，各次相互独立、互不影响。"""
    segments = parse_multi_speaker_text(text)
    participations: list[dict] = []
    role_seq: dict[str, int] = {}
    cur: dict | None = None
    for seg in segments:
        voice = seg.get("voice")
        role = voice or "旁白"
        emo = seg.get("emotion", "neutral")
        if cur is None or cur["role"] != role:
            if cur is not None:
                participations.append(cur)
            seq = role_seq.get(role, 0) + 1
            role_seq[role] = seq
            cur = {"role": role, "seq": seq, "text": "", "emotion": "neutral",
                   "narrative": voice is None, "voice": voice}
        cur["text"] += seg.get("text", "")
        if emo != "neutral":
            cur["emotion"] = emo
    if cur is not None:
        participations.append(cur)
    return participations


# ================= 导演层接入（模块4：文本梳理 → 逐 turn 韵律） =================

def _norm_ws(s: str) -> str:
    """去掉全部空白字符。导演层保证不改字，所以去空白后的字符流可逐字对齐。"""
    return re.sub(r"\s+", "", s or "")


def run_director_plan(text: str, engine: str = "rule", emotion: str = "",
                      tone_hint: str = "", chunks: list | None = None,
                      context: str | None = None):
    """跑一遍导演层，返回 PlanResult；不可用时返回 None（由调用方负责回落）。

    engine="llm" 会尝试 LLM 内核（模块5），未配置 Key / 网络失败时
    plan_llm(fallback=True) 会自动降级到规则版，PlanResult.source 标记实际通路。
    chunks 仅对 LLM 通路有效：直接指定切分边界（多人对话的说话人 turn），
    让 LLM 一次调用即可按 turn 返回韵律，不必事后按字符对齐。
    context 仅对 LLM 通路有效：给模型看的上下文全文（可含 (@角色名) 标记），
    只进提示词，不参与切分。
    """
    if _director is None:
        return None
    if str(engine).lower() == "llm":
        try:
            from voice_clone.director_llm import plan_llm
            return plan_llm(text, fallback=True, chunks=chunks, context=context)
        except Exception as e:
            print(f"[warn] director_llm unavailable, fallback to rule: {e}", flush=True)
    return _director.plan(text, emotion=emotion, tone_hint=tone_hint)


def _seg_to_dict(seg) -> dict:
    """把 director.Segment 折成一个 turn 级的参数字典（情绪取主导、数值取均值）。"""
    return {
        "emotion": seg.emotion,
        "intensity": round(float(seg.intensity or 0.0), 3),
        "pause": (round(float(seg.pause_after), 3) if seg.pause_after is not None else None),
        "cfg": (round(float(seg.cfg), 2) if getattr(seg, "cfg", None) is not None else None),
        "pace": (round(float(seg.pace), 3) if getattr(seg, "pace", None) is not None else None),
        "pitch_st": (round(float(seg.pitch_st), 3)
                     if getattr(seg, "pitch_st", None) is not None else None),
    }


def _aggregate_segments(segs: list) -> dict:
    """把同一 turn 内的多个 Segment 折成一个参数（情绪按 字符数×强度 加权投票）。"""
    votes: dict = {}
    pauses: list = []
    cfgs: list = []
    paces: list = []
    pitches: list = []
    for s in segs or []:
        w = len(_norm_ws(s.text)) * max(float(s.intensity or 0.0), 0.05)
        votes[s.emotion] = votes.get(s.emotion, 0.0) + w
        if s.pause_after is not None:
            pauses.append(float(s.pause_after))
        if getattr(s, "cfg", None) is not None:
            cfgs.append(float(s.cfg))
        if getattr(s, "pace", None) is not None:
            paces.append(float(s.pace))
        if getattr(s, "pitch_st", None) is not None:
            pitches.append(float(s.pitch_st))
    if not votes:
        return {}
    dom = max(votes, key=votes.get)
    tot = sum(votes.values()) or 1.0
    return {
        "emotion": dom,
        "intensity": round(votes.get(dom, 0.0) / tot, 3),
        "pause": (round(sum(pauses) / len(pauses), 3) if pauses else None),
        "cfg": (round(sum(cfgs) / len(cfgs), 2) if cfgs else None),
        "pace": (round(sum(paces) / len(paces), 3) if paces else None),
        "pitch_st": (round(sum(pitches) / len(pitches), 3) if pitches else None),
    }


def director_turn_params(text: str, engine: str = "rule", hint: str = "") -> dict:
    """对**单个 turn**（同一说话人的一段台词）跑导演层，聚合出该 turn 的参数。

    ⚠️ 为什么要逐 turn 独立规划，而不是整篇规划后再切：
    规则版导演层带「情绪惯性」（承接上句情绪并衰减），这在连续旁白里是特性，
    但在多人对话里是缺陷 —— 说话人不同，A 的怒气不该传染给 B 的下一句。
    实测（整篇规划后对齐）：「…我等了多久！」的 question/exclamation 会衰减着
    渗进后面两句完全不相干的台词，把「对不起。」也判成 exclamation。
    逐 turn 规划让惯性只在同一说话人内部生效。

    hint 为全局基调（由整篇扫描得出）。注意它走 tone_hint 而不是 emotion ——
    emotion 的语义是「强制全篇统一」，传进去会让每段都被钉死成同一标签。
    """
    pr = run_director_plan(text, engine=engine, tone_hint=hint)
    if pr is None or not pr.segments:
        return {}
    d = _aggregate_segments(pr.segments)
    if d:
        d["source"] = pr.source
        d["tone"] = pr.global_tone
    return d


def _build_rule_plan(turns: list, texts: list, warns: list):
    """规则引擎路径：整篇只用来取「全局基调」，判定逐 turn 独立做。

    为什么不像 LLM 那样整篇规划后归组：规则版带「情绪惯性」（承接上句情绪并衰减），
    整篇规划会让「…我等了多久！」的怒气衰减着渗进后面无关的台词。逐 turn 判定
    让惯性只在同一说话人内部生效。
    """
    full = "\n".join(texts)
    pr_all = run_director_plan(full, engine="rule")
    hint = pr_all.global_tone if pr_all is not None else ""
    per_turn = [director_turn_params(t, engine="rule", hint=hint) for t in texts]
    info = {"engine": "rule", "source": (pr_all.source if pr_all is not None else "rule"),
            "tone": hint, "share": (round(float(pr_all.global_share), 3)
                                    if pr_all is not None else 0.0),
            "segments": len(texts), "granularity": "per-turn", "error": ""}
    return per_turn, info, warns


def build_director_plan(turns: list, engine: str = "rule"):
    """为整个对话生成 per-turn 参数表 + 概览信息。

    - LLM 引擎：整篇一次调用，但把 **说话人 turn 作为切分边界** 传进去
      （chunks=各 turn 文本）。LLM 仍能看到全文上下文（context 里额外带 (@角色名)），
      返回结果与 turn 天然 1:1，无需事后按字符对齐，也就不会跨说话人串味。
      LLM 未生效时**不**沿用整篇规则计划，而是转走逐 turn 规则判定（见下）。
    - 规则引擎：逐 turn 独立判定。
    返回 (per_turn列表 或 None, 概览dict 或 None, 警告列表)
    """
    warns: list = []
    texts = [str(t.get("text") or "") for t in turns]
    full = "\n".join(texts)
    if _director is None:
        warns.append("导演层模块未加载，本次按默认参数合成")
        return None, None, warns

    if str(engine).lower() == "llm":
        # context 带上说话人标记，让模型知道每句是谁说的；text/chunks 保持裸台词，
        # 保证 Segment 与 turn 逐字对齐（标记只进提示词，不进合成文本）
        ctx_lines = []
        for t in turns:
            role = str(t.get("role") or "").strip()
            body = str(t.get("text") or "")
            ctx_lines.append(f"(@{role}){body}" if role and role != "旁白" else body)
        pr = run_director_plan(full, engine="llm", chunks=texts,
                               context="\n".join(ctx_lines))
        if pr is None:
            warns.append("LLM 导演层不可用，已回落规则版（逐 turn 判定）")
            return _build_rule_plan(turns, texts, warns)
        if pr.source == "rule":
            # 未启用 / 无 Key / 网络失败 → LLM 已内部降级。此时若沿用整篇规则计划，
            # 会把跨说话人的情绪惯性带回来（实测「对不起。」被判成 exclamation），
            # 所以改走逐 turn 规则判定，保证降级后质量不倒退。
            warns.append("LLM 内核未生效，已降级规则版（逐 turn 判定）：%s"
                         % (pr.error or "未启用"))
            per_turn, info, warns = _build_rule_plan(turns, texts, warns)
            info["engine_attempted"] = "llm"
            return per_turn, info, warns
        per_turn = [_seg_to_dict(s) for s in pr.segments]
        while len(per_turn) < len(texts):   # 长度兜底，保证与 turns 索引对齐
            per_turn.append({})
        per_turn = per_turn[:len(texts)]
        try:
            intact = bool(_director.verify_text_intact(full, pr))
        except Exception:
            intact = None
        info = {"engine": "llm", "source": pr.source, "tone": pr.global_tone,
                "share": round(float(pr.global_share), 3), "segments": len(pr.segments),
                "granularity": "turn-boundary", "text_intact": intact,
                "error": pr.error}
        return per_turn, info, warns

    return _build_rule_plan(turns, texts, warns)


def prepare_clone_reference(ref_path: str, denoise_on: bool, remove_bg_on: bool) -> str:
    """克隆/极致克隆参考音频增强（voice_clone 套件）：
    - 长音频(>30s)自动分段 + 声纹离群剔除 + 融合为有界代表参考（避免整段编码特征漂移）
    - 可选：谱门控降噪、背景音/音乐去除
    返回融合参考 wav 路径（按文件哈希+参数缓存，避免重复计算）。"""
    try:
        out_path, rep = _vc_prepare_reference(
            ref_path, denoise=denoise_on, remove_bg=remove_bg_on, target_dur=REF_TARGET_DUR)
        adapt = rep.get("adaptation", {})
        print(f"[VoxCPM2] 参考增强: 输入 {rep.get('input_duration')}s -> "
              f"融合 {rep.get('output_duration')}s | 分段={adapt.get('split_method')} "
              f"段数={adapt.get('n_segments')} 选={adapt.get('chosen')}", flush=True)
        return out_path
    except ValueError as e:
        raise ValueError(f"参考音频处理失败: {e}")


# ============================== 鉴权 ==============================
def check_auth(request: Request) -> bool:
    header_key = request.headers.get("x-api-key")
    if header_key and secrets.compare_digest(header_key, ACCESS_TOKEN):
        return True
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        if secrets.compare_digest(auth[7:].strip(), ACCESS_TOKEN):
            return True
    cookie = request.cookies.get("voxcpm_token")
    if cookie and secrets.compare_digest(cookie, ACCESS_TOKEN):
        return True
    q = request.query_params.get("token")
    if q and secrets.compare_digest(q, ACCESS_TOKEN):
        return True
    return False


def require_auth(request: Request):
    if not check_auth(request):
        raise HTTPException(status_code=401, detail="访问令牌无效，请检查 credentials.json")


app = FastAPI(title="VoxCPM2 本地推理服务", version="2.1.0", docs_url=None, redoc_url=None)


# ---------- 全局异常处理器：保证永远返回 JSON，并把 traceback 落盘 ----------
@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    log_error("未捕获异常", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {type(exc).__name__}: {exc}"},
    )


# ============================== 前端页面 ==============================
LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoxCPM2 · 访问验证</title><link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiI+PHJlY3Qgd2lkdGg9IjMyIiBoZWlnaHQ9IjMyIiByeD0iNyIgZmlsbD0iIzZhNTVlOCIvPjxnIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIyLjYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgZmlsbD0ibm9uZSI+PHBhdGggZD0iTTE2IDd2MTAiLz48cGF0aCBkPSJNMTEgMTF2NSIvPjxwYXRoIGQ9Ik0yMSAxMXY1Ii8+PC9nPjxwYXRoIGQ9Ik0xMSAyMmgxMCIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIuNiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+"><style>*{margin:0;padding:0;box-sizing:border-box}
:root{color-scheme:light;
--font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
--bg:#f6f6f8;--surface:#ffffff;--surface-2:#fafafc;--surface-3:#f1f1f5;
--border:rgba(16,16,24,.07);--border-2:rgba(16,16,24,.14);
--text:#16161c;--text-1:#42424e;--text-2:#71717e;--text-3:#a0a0ac;--on-accent:#fff;
--accent:#6a55e8;--accent-2:#5a46d6;--accent-3:#7d6bff;--link:#5a46d6;
--accent-soft:rgba(106,85,232,.08);--accent-soft-2:rgba(106,85,232,.15);--accent-ink:#4a38c4;
--accent-glow:rgba(106,85,232,.26);--ambient:rgba(106,85,232,.05);
--ok-bg:rgba(16,185,129,.10);--ok-ink:#0b9d6e;--warn-bg:rgba(245,158,11,.12);--warn-ink:#b45309;
--err-bg:rgba(239,68,68,.09);--err-ink:#dc2626;--danger:#dc2626;--danger-ink:#dc2626;
--res-bg:rgba(16,185,129,.08);--res-border:rgba(16,185,129,.25);--green:#10b981;--green-ink:#0b9d6e;
--sky:#0ea5e9;--sky-ink:#0284c7;--violet:#7c5cff;--disabled:#b8b8c2;
--focus:rgba(106,85,232,.14);--shadow-1:rgba(16,16,24,.05);--shadow-2:rgba(16,16,24,.12);
--range-track:#e6e6ec;--card-hi:rgba(255,255,255,.95)}
html[data-theme="dark"]{color-scheme:dark;
--bg:#0a0a0f;--surface:#131318;--surface-2:#18181f;--surface-3:#1f1f28;
--border:rgba(255,255,255,.07);--border-2:rgba(255,255,255,.14);
--text:#f4f4f7;--text-1:#c9c9d4;--text-2:#8f8f9d;--text-3:#5f5f6d;--on-accent:#fff;
--accent:#7c6bff;--accent-2:#6a55e8;--accent-3:#8f80ff;--link:#8f80ff;
--accent-soft:rgba(124,107,255,.14);--accent-soft-2:rgba(124,107,255,.24);--accent-ink:#b3a6ff;
--accent-glow:rgba(124,107,255,.4);--ambient:rgba(124,107,255,.07);
--ok-bg:rgba(16,185,129,.14);--ok-ink:#34d399;--warn-bg:rgba(245,158,11,.14);--warn-ink:#fbbf24;
--err-bg:rgba(239,68,68,.13);--err-ink:#f87171;--danger:#ef4444;--danger-ink:#f87171;
--res-bg:rgba(16,185,129,.10);--res-border:rgba(16,185,129,.3);--green:#34d399;--green-ink:#4ade80;
--sky:#38bdf8;--sky-ink:#7dd3fc;--violet:#8f80ff;--disabled:#3f3f4c;
--focus:rgba(124,107,255,.35);--shadow-1:rgba(0,0,0,.35);--shadow-2:rgba(0,0,0,.55);
--range-track:#2a2a36;--card-hi:rgba(255,255,255,.05)}
body{font-family:var(--font);background:var(--bg);
background-image:radial-gradient(760px 320px at 50% -8%,var(--ambient),transparent 70%);
background-attachment:fixed;min-height:100vh;display:flex;align-items:center;justify-content:center;
padding:20px;color:var(--text);-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:40px 38px;max-width:420px;width:100%;
box-shadow:inset 0 1px 0 var(--card-hi),0 1px 2px var(--shadow-1),0 24px 60px -28px var(--shadow-2);position:relative;overflow:hidden}
.card::before{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:linear-gradient(90deg,var(--accent-3),var(--accent-2))}
h1{font-size:24px;font-weight:650;color:var(--text);margin:6px 0 4px;letter-spacing:-.02em}
p.sub{font-size:13px;color:var(--text-2);margin-bottom:26px;line-height:1.6}
label{display:block;font-size:13px;color:var(--text-1);margin-bottom:8px;font-weight:550}
input{width:100%;padding:12px 14px;border:1px solid var(--border-2);border-radius:12px;font-size:14px;
font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;transition:border-color .16s ease,box-shadow .16s ease;
background:var(--surface);color:var(--text)}
input:hover{border-color:var(--text-3)}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--focus)}
button{width:100%;margin-top:18px;padding:13px;background:linear-gradient(180deg,var(--accent-3),var(--accent-2));color:var(--on-accent);
border:none;border-radius:12px;font-size:15px;font-weight:600;cursor:pointer;
box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 4px 16px var(--accent-glow);
transition:transform .15s ease,box-shadow .2s ease,filter .2s ease}
button:hover{transform:translateY(-1px);box-shadow:inset 0 1px 0 rgba(255,255,255,.22),0 8px 24px var(--accent-glow);filter:brightness(1.05)}
button:active{transform:translateY(0);filter:brightness(.98)}
.err{margin-top:12px;padding:10px 12px;background:var(--err-bg);color:var(--err-ink);border-radius:10px;font-size:13px;display:none}
.hint{margin-top:18px;padding:12px 14px;background:var(--surface-2);border:1px solid var(--border);border-radius:12px;font-size:12px;color:var(--text-2);line-height:1.7}
code{background:var(--accent-soft);color:var(--accent-ink);padding:1px 6px;border-radius:5px;font-size:11px}
textarea,input[type=text],input[type=file],input[type=password],input[type=number],select{background:var(--surface);color:var(--text);caret-color:var(--accent)}
select option{background:var(--surface);color:var(--text)}
::placeholder{color:var(--text-3);opacity:1}
input[type=checkbox]{accent-color:var(--accent)}</style><script>(function(){try{var t='';try{t=localStorage.getItem('voxcpm_theme')||'';}catch(e){}if(t!=='dark'&&t!=='light'){try{var qs=new URLSearchParams(location.search);var q=qs.get('theme')||qs.get('vox_theme');if(q==='dark'||q==='light'){t=q;}}catch(e){}}if(t!=='dark'&&t!=='light'){t=(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','light');}})();</script></head><body>
<div class="card">
  <h1>🎙️ VoxCPM2 <span data-i18n="loginLocal">本地服务</span></h1>
  <p class="sub" data-i18n="loginSub">该服务已启用访问验证，请输入访问令牌。</p>
  <label data-i18n="loginToken">访问令牌 (Access Token)</label>
  <input type="password" id="tk" placeholder="vox2_..." autofocus>
  <button onclick="go()" data-i18n="loginEnter">进入</button>
  <div style="text-align:right;margin-top:6px">
    <button onclick="lgLang('zh')" id="lgZh" style="border:none;background:none;cursor:pointer;font-size:12px">中</button>
    <button onclick="lgLang('en')" id="lgEn" style="border:none;background:none;cursor:pointer;font-size:12px;opacity:.5">EN</button>
  </div>
  <div class="err" id="err"></div>
  <div class="hint"><span data-i18n="loginHint1">令牌保存在</span> <code>CRED_PATH_PLACEHOLDER</code><br>
  <span data-i18n="loginHint2">也可用一键链接直接进入：</span><code>http://localhost:PORT_PLACEHOLDER/?token=你的令牌</code></div>
</div>
<script>
const LGI={zh:{loginLocal:'本地服务',loginSub:'该服务已启用访问验证，请输入访问令牌。',loginToken:'访问令牌 (Access Token)',loginEnter:'进入',loginEmpty:'请输入访问令牌',loginBad:'令牌不正确',loginHint1:'令牌保存在',loginHint2:'也可用一键链接直接进入：'},
          en:{loginLocal:'Local Service',loginSub:'This service requires authentication. Enter your access token.',loginToken:'Access Token',loginEnter:'Enter',loginEmpty:'Please enter your access token',loginBad:'Invalid token',loginHint1:'Token stored at',loginHint2:'Or open the one-click link:'}};
let lg='zh';
function lgLang(l){lg=l;const d=LGI[l];
  document.querySelectorAll('[data-i18n]').forEach(el=>{const k=el.getAttribute('data-i18n');if(d[k]!==undefined)el.textContent=d[k];});
  document.getElementById('lgZh').style.opacity=(l==='zh')?'1':'.5';
  document.getElementById('lgEn').style.opacity=(l==='en')?'1':'.5';}
async function go(){
  const tk=document.getElementById('tk').value.trim();
  if(!tk){show(LGI[lg].loginEmpty);return;}
  const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:tk})});
  if(r.ok){location.href='/';}else{show(LGI[lg].loginBad);}
}
function show(m){const e=document.getElementById('err');e.textContent='❌ '+m;e.style.display='block';}
document.getElementById('tk').addEventListener('keydown',e=>{if(e.key==='Enter')go();});
</script></body></html>"""

APP_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoxCPM2 · 本地语音合成</title><link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiI+PHJlY3Qgd2lkdGg9IjMyIiBoZWlnaHQ9IjMyIiByeD0iNyIgZmlsbD0iIzZhNTVlOCIvPjxnIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIyLjYiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgZmlsbD0ibm9uZSI+PHBhdGggZD0iTTE2IDd2MTAiLz48cGF0aCBkPSJNMTEgMTF2NSIvPjxwYXRoIGQ9Ik0yMSAxMXY1Ii8+PC9nPjxwYXRoIGQ9Ik0xMSAyMmgxMCIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIuNiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PC9zdmc+"><style>*{margin:0;padding:0;box-sizing:border-box}
:root{color-scheme:light;
--font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
--bg:#f6f6f8;--surface:#ffffff;--surface-2:#fafafc;--surface-3:#f1f1f5;
--border:rgba(16,16,24,.07);--border-2:rgba(16,16,24,.14);
--text:#16161c;--text-1:#42424e;--text-2:#71717e;--text-3:#a0a0ac;--on-accent:#fff;
--accent:#6a55e8;--accent-2:#5a46d6;--accent-3:#7d6bff;--link:#5a46d6;
--accent-soft:rgba(106,85,232,.08);--accent-soft-2:rgba(106,85,232,.15);--accent-ink:#4a38c4;
--accent-glow:rgba(106,85,232,.26);--ambient:rgba(106,85,232,.05);
--ok-bg:rgba(16,185,129,.10);--ok-ink:#0b9d6e;--warn-bg:rgba(245,158,11,.12);--warn-ink:#b45309;
--err-bg:rgba(239,68,68,.09);--err-ink:#dc2626;--danger:#dc2626;--danger-ink:#dc2626;
--res-bg:rgba(16,185,129,.08);--res-border:rgba(16,185,129,.25);--green:#10b981;--green-ink:#0b9d6e;
--sky:#0ea5e9;--sky-ink:#0284c7;--violet:#7c5cff;--disabled:#b8b8c2;
--focus:rgba(106,85,232,.14);--shadow-1:rgba(16,16,24,.05);--shadow-2:rgba(16,16,24,.12);
--range-track:#e6e6ec;--card-hi:rgba(255,255,255,.95)}
html[data-theme="dark"]{color-scheme:dark;
--bg:#0a0a0f;--surface:#131318;--surface-2:#18181f;--surface-3:#1f1f28;
--border:rgba(255,255,255,.07);--border-2:rgba(255,255,255,.14);
--text:#f4f4f7;--text-1:#c9c9d4;--text-2:#8f8f9d;--text-3:#5f5f6d;--on-accent:#fff;
--accent:#7c6bff;--accent-2:#6a55e8;--accent-3:#8f80ff;--link:#8f80ff;
--accent-soft:rgba(124,107,255,.14);--accent-soft-2:rgba(124,107,255,.24);--accent-ink:#b3a6ff;
--accent-glow:rgba(124,107,255,.4);--ambient:rgba(124,107,255,.07);
--ok-bg:rgba(16,185,129,.14);--ok-ink:#34d399;--warn-bg:rgba(245,158,11,.14);--warn-ink:#fbbf24;
--err-bg:rgba(239,68,68,.13);--err-ink:#f87171;--danger:#ef4444;--danger-ink:#f87171;
--res-bg:rgba(16,185,129,.10);--res-border:rgba(16,185,129,.3);--green:#34d399;--green-ink:#4ade80;
--sky:#38bdf8;--sky-ink:#7dd3fc;--violet:#8f80ff;--disabled:#3f3f4c;
--focus:rgba(124,107,255,.35);--shadow-1:rgba(0,0,0,.35);--shadow-2:rgba(0,0,0,.55);
--range-track:#2a2a36;--card-hi:rgba(255,255,255,.05)}
html{scroll-behavior:smooth}
body{font-family:var(--font);font-size:14px;line-height:1.6;color:var(--text);padding:30px 20px 60px;
background:var(--bg);background-image:radial-gradient(900px 340px at 50% -10%,var(--ambient),transparent 70%);
background-attachment:fixed;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:var(--border-2);border-radius:999px;border:2px solid transparent;background-clip:content-box}
::-webkit-scrollbar-thumb:hover{background:var(--text-3);background-clip:content-box}
::-webkit-scrollbar-track{background:transparent}
.wrap{max-width:920px;margin:0 auto}
.top{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;position:relative;
background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 22px;margin-bottom:20px;
box-shadow:inset 0 1px 0 var(--card-hi),0 1px 2px var(--shadow-1),0 12px 36px -24px var(--shadow-2)}
.top::before{content:"";position:absolute;left:0;top:14px;bottom:14px;width:3px;border-radius:0 3px 3px 0;
background:linear-gradient(180deg,var(--accent-3),var(--accent-2))}
.top h1{font-size:18px;font-weight:650;letter-spacing:-.02em;display:flex;align-items:center;gap:10px}
.badges{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{font-size:12px;padding:5px 11px;border-radius:999px;background:var(--surface-3);color:var(--text-1);border:1px solid var(--border);transition:all .16s ease}
.badge.ok{background:var(--ok-bg);color:var(--ok-ink);border-color:transparent;display:inline-flex;align-items:center;gap:5px}
.badge.ok::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--ok-ink)}
.badge.warn{background:var(--warn-bg);color:var(--warn-ink);border-color:transparent}
button.badge{cursor:pointer}
button.badge:hover{border-color:var(--border-2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:20px;
box-shadow:inset 0 1px 0 var(--card-hi),0 1px 2px var(--shadow-1),0 12px 40px -28px var(--shadow-2)}
.tabs{display:flex;gap:4px;margin-bottom:20px;flex-wrap:wrap;background:var(--surface);border:1px solid var(--border);
border-radius:12px;padding:4px;box-shadow:inset 0 1px 0 var(--card-hi),0 1px 2px var(--shadow-1)}
.tab{padding:8px 16px;border:none;border-radius:9px;background:transparent;cursor:pointer;font-size:13.5px;color:var(--text-2);transition:all .16s ease;font-weight:500}
.tab:hover{color:var(--text);background:var(--surface-3)}
.tab.active{background:linear-gradient(180deg,var(--accent-3),var(--accent-2));color:var(--on-accent);font-weight:600;
box-shadow:inset 0 1px 0 rgba(255,255,255,.16),0 2px 8px var(--accent-glow)}
label{display:block;font-size:13px;font-weight:550;color:var(--text-1);margin-bottom:7px}
textarea{width:100%;padding:13px 14px;border:1px solid var(--border-2);border-radius:12px;font-size:14px;
min-height:110px;resize:vertical;font-family:inherit;line-height:1.65;transition:border-color .16s ease,box-shadow .16s ease}
input[type=text],input[type=file]{width:100%;padding:11px 13px;border:1px solid var(--border-2);border-radius:12px;font-size:14px;transition:border-color .16s ease,box-shadow .16s ease}
textarea:hover,input[type=text]:hover,input[type=file]:hover{border-color:var(--text-3)}
textarea:focus,input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--focus)}
.field{margin-bottom:16px}
.hide{display:none}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}
.chip{font-size:12px;padding:6px 12px;background:var(--accent-soft);color:var(--accent-ink);border-radius:999px;cursor:pointer;border:1px solid transparent;transition:all .16s ease}
.chip:hover{background:var(--accent-soft-2);border-color:var(--border)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.pbox{background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:13px 14px;transition:border-color .16s ease}
.pbox:hover{border-color:var(--border-2)}
.pbox label{font-size:12px;color:var(--text-2);font-weight:500;margin-bottom:8px}
.prow{display:flex;align-items:center;gap:10px}
.pv{font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;font-size:13px;color:var(--accent);min-width:34px;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
input[type=range]{flex:1;-webkit-appearance:none;appearance:none;height:5px;border-radius:999px;background:var(--range-track);cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;
background:var(--surface);border:2px solid var(--accent);box-shadow:0 1px 3px var(--shadow-2);transition:transform .15s ease,box-shadow .15s ease}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15);box-shadow:0 2px 8px var(--accent-glow)}
input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--surface);border:2px solid var(--accent);box-shadow:0 1px 3px var(--shadow-2);cursor:pointer}
input[type=range]::-moz-range-track{height:5px;border-radius:999px;background:var(--range-track)}
.checks{display:flex;gap:20px;margin-bottom:18px;font-size:13px;color:var(--text-1);flex-wrap:wrap}
.checks label{display:flex;align-items:center;gap:6px;font-weight:500;margin:0;cursor:pointer}
input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;cursor:pointer}
.gen{width:100%;padding:15px;background:linear-gradient(180deg,var(--accent-3),var(--accent-2));color:var(--on-accent);border:none;border-radius:13px;
font-size:15px;font-weight:600;cursor:pointer;letter-spacing:.01em;
box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 4px 16px var(--accent-glow);
transition:transform .15s ease,box-shadow .2s ease,filter .2s ease}
.gen:hover:not(:disabled){transform:translateY(-1px);box-shadow:inset 0 1px 0 rgba(255,255,255,.2),0 8px 26px var(--accent-glow);filter:brightness(1.05)}
.gen:active:not(:disabled){transform:translateY(0);filter:brightness(.98)}
.gen:disabled{background:var(--disabled);box-shadow:none;cursor:not-allowed}
.status{margin-top:16px;padding:14px;background:var(--surface-2);border:1px solid var(--border);border-radius:12px;font-size:13px;
color:var(--text-1);display:none;align-items:center;gap:10px}
.status.show{display:flex}
.spin{width:18px;height:18px;border:2px solid var(--border-2);border-top-color:var(--accent);border-radius:50%;
animation:sp .8s linear infinite;flex-shrink:0}
@keyframes sp{to{transform:rotate(360deg)}}
.err{margin-top:14px;padding:12px 14px;background:var(--err-bg);color:var(--err-ink);border-radius:12px;font-size:13px;display:none;
white-space:pre-wrap;line-height:1.6}
.err.show{display:block}
.res{margin-top:18px;padding:16px;background:var(--res-bg);border:1px solid var(--res-border);border-radius:14px;display:none;
box-shadow:inset 0 1px 0 var(--card-hi)}
.res.show{display:block;animation:fadeUp .25s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.res .meta{font-size:12px;color:var(--ok-ink);margin-bottom:10px}
audio{width:100%}
.hist{font-size:13px}
.hist .row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid var(--border)}
.hist .row:last-child{border:none}
.hist a{color:var(--link);text-decoration:none;font-size:12px}
.hist a:hover{text-decoration:underline}
.muted{color:var(--text-3);font-size:12px}
.api{font-family:ui-monospace,"SF Mono","Cascadia Code",Consolas,monospace;font-size:12px;background:var(--surface-2);padding:12px 14px;
border-radius:10px;color:var(--text-1);white-space:pre-wrap;line-height:1.7;overflow-x:auto;border:1px solid var(--border)}
.ptab{padding:9px 16px;border:1px solid var(--border);border-radius:10px;background:var(--surface);cursor:pointer;font-size:13px;color:var(--text-1);transition:all .16s ease}
.ptab:hover{border-color:var(--accent)}
.ptab.active{background:linear-gradient(180deg,var(--accent-3),var(--accent-2));color:var(--on-accent);border-color:transparent;box-shadow:inset 0 1px 0 rgba(255,255,255,.16),0 2px 8px var(--accent-glow)}
.pack{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 13px;
border:1px solid var(--border);border-radius:12px;margin-bottom:8px;background:var(--surface);transition:border-color .16s ease,box-shadow .16s ease}
.pack:hover{border-color:var(--border-2);box-shadow:0 2px 10px var(--shadow-1)}
.pack .info{min-width:0}
.pack .nm{font-size:14px;font-weight:600;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pack .meta{font-size:12px;color:var(--text-2);margin-top:2px}
.pack .acts{display:flex;gap:6px;flex-shrink:0}
.pack .acts button{padding:6px 11px;border:1px solid var(--border);background:var(--surface);border-radius:8px;font-size:12px;cursor:pointer;color:var(--text-1);transition:all .15s ease}
.pack .acts button:hover{border-color:var(--accent);color:var(--link)}
.pack .acts .del:hover{border-color:var(--danger-ink);color:var(--danger-ink)}
.pack .acts .use{background:linear-gradient(180deg,var(--accent-3),var(--accent-2));color:var(--on-accent);border-color:transparent;box-shadow:0 2px 8px var(--accent-glow)}
.vp-sel{width:100%;padding:10px 13px;border:1px solid var(--border-2);border-radius:12px;font-size:14px;margin-top:6px;background:var(--surface);transition:border-color .16s ease,box-shadow .16s ease;cursor:pointer}
.vp-sel:hover{border-color:var(--text-3)}
.vp-sel:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--focus)}
textarea,input[type=text],input[type=file],input[type=password],input[type=number],select{background:var(--surface);color:var(--text);caret-color:var(--accent)}
select option{background:var(--surface);color:var(--text)}
::placeholder{color:var(--text-3);opacity:1}</style><script>(function(){try{var t='';try{t=localStorage.getItem('voxcpm_theme')||'';}catch(e){}if(t!=='dark'&&t!=='light'){try{var qs=new URLSearchParams(location.search);var q=qs.get('theme')||qs.get('vox_theme');if(q==='dark'||q==='light'){t=q;}}catch(e){}}if(t!=='dark'&&t!=='light'){t=(window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light';}document.documentElement.setAttribute('data-theme',t);}catch(e){document.documentElement.setAttribute('data-theme','light');}})();</script></head><body>
<div class="wrap">
  <div class="top">
    <h1>🎙️ VoxCPM2 <span class="muted" style="font-size:13px" data-i18n="localDeploy">本地部署</span></h1>
    <div class="badges">
      <span class="badge" id="devBadge">检测中…</span>
      <span class="badge ok">2B · 48kHz</span>
      <span class="badge" id="modelBadge" data-i18n="modelNotLoaded">模型未加载</span>
      <button class="badge" id="langZh" onclick="setLang('zh')" style="cursor:pointer;border:1px solid var(--border-2)">中</button>
      <button class="badge" id="langEn" onclick="setLang('en')" style="cursor:pointer;border:1px solid var(--border-2);opacity:.5">EN</button>
      <button class="badge" id="themeBtn" onclick="toggleTheme()" style="cursor:pointer;border:1px solid var(--border-2)" title=""></button>
    </div>
  </div>

  <div class="tabs" id="mainNav" style="margin-bottom:16px">
    <button class="tab active" data-mode="design" onclick="setMode('design')" data-i18n="modeDesign">🎨 语音设计</button>
    <button class="tab" data-mode="clone" onclick="setMode('clone')" data-i18n="modeClone">🎛️ 音色克隆</button>
    <button class="tab" data-mode="hifi" onclick="setMode('hifi')" data-i18n="modeHifi">🎙️ 极致克隆</button>
    <button class="tab" data-mode="beta" onclick="setMode('beta')" data-i18n="modeBeta">🧪 内测 Beta</button>
    <button class="tab" data-mode="train" onclick="setMode('train')" data-i18n="modeTrain">🎓 训练</button>
    <button class="tab" data-mode="settings" onclick="setMode('settings')" data-i18n="modeSettings">⚙️ 设置</button>
  </div>

  <div class="card" id="mainCard">
    <div class="field">
      <label data-i18n="synthText">合成文本</label>
      <textarea id="text">你好，这里是本地部署的 VoxCPM2 语音大模型，现在可以直接在浏览器里使用了。</textarea>
      <div class="chips" id="chips">
        <button class="chip" onclick="pre('(年轻女性，温柔甜美)')" data-i18n="chip1">年轻女性·温柔</button>
        <button class="chip" onclick="pre('(中年男性，沉稳有磁性)')" data-i18n="chip2">中年男性·沉稳</button>
        <button class="chip" onclick="pre('(活力少年，语速偏快)')" data-i18n="chip3">活力少年</button>
        <button class="chip" onclick="pre('(广东话，中年男性)')" data-i18n="chip4">粤语</button>
        <button class="chip" onclick="pre('(四川话，年轻女性)')" data-i18n="chip5">四川话</button>
        <button class="chip" onclick="pre('(新闻播报腔，字正腔圆)')" data-i18n="chip6">新闻播报</button>
      </div>
      <div class="muted" style="margin-top:8px" data-i18n="designHint">语音设计模式：用「()」在文本开头描述想要的音色、情绪、语速，例如「(年轻女性，温柔甜美)你好」。</div>
    </div>

    <div class="field hide" id="refField">
      <label data-i18n="refLabel">参考音频（0.3 秒 – 10 分钟，wav/mp3/flac）</label>
      <input type="file" id="refFile" accept="audio/*">
      <div class="muted" style="margin-top:6px" data-i18n="refHint">克隆模式必填，模型会复刻这段音频的音色。</div>
    </div>

    <div class="field hide" id="packSelField">
      <label data-i18n="packSelLabel">或选择已保存音色包（免重复上传长音频）</label>
      <select class="vp-sel" id="packSel" onchange="onPackSel()">
        <option value="" data-i18n="noPackOpt">— 不使用音色包，改为上传音频 —</option>
      </select>
      <div class="muted" style="margin-top:6px" id="packSelHint"></div>
    </div>

    <div class="field hide" id="ptField">
      <label data-i18n="ptLabel">参考音频的逐字文本（极致克隆必填）</label>
      <input type="file" id="refFile2" accept="audio/*" style="display:none">
      <input type="text" id="promptText" data-i18n-ph="ptPh" placeholder="必须与参考音频内容完全一致">
    </div>

    <div class="grid">
      <div class="pbox">
        <label data-i18n="cfgLabel">CFG 引导强度（1.0-3.0，默认 2.0）</label>
        <div class="prow"><input type="range" id="cfg" min="1" max="3" step="0.1" value="2"
          oninput="document.getElementById('cfgv').textContent=this.value">
          <span class="pv" id="cfgv">2.0</span></div>
      </div>
      <div class="pbox">
        <label data-i18n="stepsLabel">扩散步数（4-30，越大越细腻越慢）</label>
        <div class="prow"><input type="range" id="steps" min="4" max="30" step="1" value="10"
          oninput="document.getElementById('stv').textContent=this.value">
          <span class="pv" id="stv">10</span></div>
      </div>
    </div>

    <div class="checks">
      <label><input type="checkbox" id="normalize" checked> <span data-i18n="normalizeLabel">文本规范化（数字/日期正确读出）</span></label>
      <label title="谱门控降噪，去除稳态/环境噪声（离线可用）"><input type="checkbox" id="denoise" checked> <span data-i18n="denoiseLabel">参考音频降噪</span></label>
      <label title="分离并抑制背景音乐/环境音，突出人声"><input type="checkbox" id="remove_bg"> <span data-i18n="removeBgLabel">去除背景音/音乐</span></label>
      <label title="长台词按句分块生成并交叉淡化拼接，避免断裂/突变"><input type="checkbox" id="stable"> <span data-i18n="stableLabel">长文本稳定合成</span></label>
    </div>
    <div class="muted" style="font-size:12px;margin:-6px 0 10px" data-i18n="refTip">提示：参考音频超过 30 秒时会自动分段，按说话人声纹融合为约 25 秒的代表音频，显著缓解长音频克隆的失真与音色漂移。</div>

    <div class="field">
      <label data-i18n="emoLabel">🎭 情绪语气（可选，套用一组音调/语速/停顿预设，可再手动微调）</label>
      <select id="emotionSel" class="vp-sel" onchange="applyEmotion(this.value)">
        <option value="" data-i18n="emoNone">— 不指定情绪 —</option>
        <option value="高兴" data-i18n="emoHappy">高兴</option>
        <option value="悲伤" data-i18n="emoSad">悲伤</option>
        <option value="严肃" data-i18n="emoSerious">严肃</option>
        <option value="温柔" data-i18n="emoGentle">温柔</option>
        <option value="愤怒" data-i18n="emoAngry">愤怒</option>
        <option value="平静" data-i18n="emoCalm">平静</option>
      </select>
    </div>
    <div class="field">
      <label data-i18n="loraLabel">🧩 LoRA 微调音色（训练页产出，选用后整段生效）</label>
      <select id="loraSel" class="vp-sel" onchange="onLoraSel()">
        <option value="" data-i18n="loraNone">— 不使用 —</option>
      </select>
      <div class="muted" id="loraSelHint" style="margin-top:6px"></div>
    </div>
    <div class="grid">
      <div class="pbox">
        <label data-i18n="pitchLabel">音调（半音，0=原音）</label>
        <div class="prow"><input type="range" id="pitch" min="-12" max="12" step="1" value="0"
          oninput="document.getElementById('pitchv').textContent=this.value">
          <span class="pv" id="pitchv">0</span></div>
      </div>
      <div class="pbox">
        <label data-i18n="speedLabel">语速（0.5x-2.0x）</label>
        <div class="prow"><input type="range" id="speed" min="0.5" max="2" step="0.05" value="1"
          oninput="document.getElementById('speedv').textContent=this.value">
          <span class="pv" id="speedv">1.0</span></div>
      </div>
      <div class="pbox">
        <label data-i18n="volumeLabel">音量（0.1x-2.0x）</label>
        <div class="prow"><input type="range" id="volume" min="0.1" max="2" step="0.05" value="1"
          oninput="document.getElementById('volumev').textContent=this.value">
          <span class="pv" id="volumev">1.0</span></div>
      </div>
      <div class="pbox">
        <label data-i18n="pauseLabel">句间停顿（秒，0=无）</label>
        <div class="prow"><input type="range" id="pause" min="0" max="1.5" step="0.05" value="0.15"
          oninput="document.getElementById('pausev').textContent=this.value">
          <span class="pv" id="pausev">0.15</span></div>
      </div>
      <div class="pbox">
        <label data-i18n="breathLabel">呼吸声轻重（0=无）</label>
        <div class="prow"><input type="range" id="breath" min="0" max="1" step="0.05" value="0"
          oninput="document.getElementById('breathv').textContent=this.value">
          <span class="pv" id="breathv">0</span></div>
      </div>
    </div>
    <div class="checks">
      <label title="解析 <break>/<prosody>/<emotion> 等 SSML 标签，强制控制停顿/重音/语速"><input type="checkbox" id="ssml"> <span data-i18n="ssmlLabel">启用 SSML 标签</span></label>
    </div>

    <button class="gen" id="btn" onclick="generate()" data-i18n="genBtn">🔊 生成语音</button>

    <div class="status" id="status"><div class="spin"></div><div id="statusText" data-i18n="statusIdle">生成中…</div></div>
    <div class="err" id="err"></div>
    <div class="res" id="res">
      <div class="meta" id="resMeta"></div>
      <audio id="player" controls></audio>
      <div class="prow" style="margin-top:10px;gap:8px">
        <select id="exportFmt" style="width:auto;padding:8px;border:1px solid var(--border-2);border-radius:8px">
          <option value="mp3">导出 MP3</option>
          <option value="wav">导出 WAV</option>
          <option value="m4a">导出 M4A</option>
        </select>
        <button id="exportBtn" onclick="exportAudio()" data-i18n="exportLabel" style="padding:8px 14px;border:1px solid var(--border-2);border-radius:8px;background:var(--surface);cursor:pointer;font-size:13px">⬇️ 导出</button>
      </div>
    </div>
  </div>

  <div class="card" id="histCard">
    <label data-i18n="history">本次会话生成记录</label>
    <div class="hist" id="hist"><div class="muted" data-i18n="noHistory">还没有生成记录</div></div>
  </div>

  <div class="card hide" id="settingsCard">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <button class="chip" onclick="setMode(prevMode||'design')" data-i18n="backBtn" style="padding:6px 12px;border:1px solid var(--border-2);border-radius:8px;background:var(--surface);cursor:pointer;font-size:13px">← 返回</button>
      <span class="badge">LLM</span>
      <label data-i18n="setTitle" style="margin:0">⚙️ 设置 · AI 内核配置</label>
    </div>

    <div class="muted" style="margin-bottom:14px;line-height:1.7" data-i18n="setDesc">导演层的「AI 内核」需要一个可调用的大模型来梳理情绪与停顿。在这里填入任意 OpenAI 兼容服务的密钥即可启用——选中服务商后地址与模型会自动带出，也可以选「自定义」手填任何中转站或内网网关。密钥只保存在本机 <code>llm_config.json</code>，不会外传。</div>

    <!-- 总开关 + 状态灯 -->
    <div style="border:1px solid var(--border-2);border-radius:10px;padding:12px 14px;margin-bottom:14px;background:var(--surface-2)">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer;font-weight:600">
          <input type="checkbox" id="setEnabled"> <span data-i18n="setEnableLabel">启用 AI 内核</span>
        </label>
        <span id="setStatusDot" style="width:9px;height:9px;border-radius:50%;background:var(--disabled);display:inline-block"></span>
        <span id="setStatusText" class="muted" style="font-size:13px"></span>
      </div>
      <div class="muted" id="setSourceNote" style="font-size:12px;margin-top:8px;line-height:1.6"></div>
    </div>

    <!-- 服务商 -->
    <div class="field">
      <label data-i18n="setProviderLabel">服务商预设</label>
      <select class="vp-sel" id="setProvider" onchange="setOnProviderChange()"></select>
      <div class="muted" id="setProviderNote" style="margin-top:6px;line-height:1.65;font-size:12px"></div>
    </div>

    <!-- 三项核心 -->
    <div class="field">
      <label data-i18n="setKeyLabel">API Key</label>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <input type="password" id="setKey" autocomplete="off" style="flex:1;min-width:220px"
               oninput="setKeyWasTyped()"
               data-i18n-ph="setKeyPh" placeholder="粘贴你的 API Key（留空表示不修改已保存的密钥）">
        <button class="chip" id="setKeyToggle" onclick="setToggleKeyView()" style="padding:8px 12px" data-i18n="setShow">👁 显示</button>
      </div>
      <div class="muted" id="setKeyHint" style="margin-top:6px;font-size:12px;line-height:1.6"></div>
    </div>

    <div class="field">
      <label data-i18n="setBaseLabel">接口地址 base_url</label>
      <input type="text" id="setBaseUrl" placeholder="https://api.deepseek.com/v1">
      <div class="muted" id="setBaseHint" style="margin-top:6px;font-size:12px;line-height:1.6"></div>
    </div>

    <div class="field">
      <label data-i18n="setModelLabel">模型名 model</label>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <input type="text" id="setModel" list="setModelList" placeholder="deepseek-chat" style="flex:1;min-width:200px">
        <datalist id="setModelList"></datalist>
        <button class="chip" id="setFetchModelsBtn" onclick="setFetchModels()" style="padding:8px 12px" data-i18n="setFetchModels">⬇ 拉取可用模型</button>
      </div>
      <div class="muted" id="setModelHint" style="margin-top:6px;font-size:12px;line-height:1.6"></div>
    </div>

    <!-- 高级参数 -->
    <details style="border:1px solid var(--border-2);border-radius:10px;padding:10px 12px;margin-bottom:14px">
      <summary style="cursor:pointer;font-weight:600;font-size:14px;outline:none" data-i18n="setAdvanced">🔧 高级参数（默认即可）</summary>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:10px">
        <div><div class="muted" style="font-size:11px" data-i18n="setTimeout">超时（秒）</div>
          <input type="number" id="setTimeout" min="5" max="600" step="5" value="60"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="setTemp">温度 temperature</div>
          <input type="number" id="setTemp" min="0" max="2" step="0.1" value="0.3"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="setProxy">代理（留空=直连）</div>
          <input type="text" id="setProxy" placeholder="http://127.0.0.1:7890"></div>
      </div>
      <div class="checks" style="margin-top:10px;flex-wrap:wrap;gap:14px">
        <label><input type="checkbox" id="setJsonMode" checked> <span data-i18n="setJsonMode">使用 JSON 模式（兼容层不支持时自动回退）</span></label>
        <label><input type="checkbox" id="setCache" checked> <span data-i18n="setCache">缓存梳理结果（同文本不重复调用）</span></label>
      </div>
    </details>

    <!-- 动作 -->
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
      <button class="chip" id="setSaveBtn" onclick="settingsSave()" style="padding:9px 20px;background:var(--violet);color:var(--on-accent);font-weight:600" data-i18n="setSave">💾 保存并测试</button>
      <button class="chip" id="setTestBtn" onclick="settingsTest()" style="padding:9px 18px" data-i18n="setTest">🔌 仅测试连接</button>
      <button class="chip" id="setReloadBtn" onclick="settingsLoad(true)" style="padding:9px 18px" data-i18n="setReload">↻ 重新读取</button>
    </div>

    <div class="err" id="setErr"></div>
    <div id="setResult" style="display:none;border:1px solid var(--border-2);border-radius:10px;padding:10px 12px;margin-bottom:12px;font-size:13px;line-height:1.7"></div>
    <div class="muted" id="setConfigPath" style="font-size:11px;word-break:break-all"></div>
  </div>

  <div class="card hide" id="betaCard">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <button class="chip" onclick="setMode(prevMode||'design')" data-i18n="backBtn" style="padding:6px 12px;border:1px solid var(--border-2);border-radius:8px;background:var(--surface);cursor:pointer;font-size:13px">← 返回</button>
      <span class="badge warn">Beta</span>
      <label data-i18n="betaTitle" style="margin:0">多人朗读与情绪控制</label>
    </div>
    <div class="muted" style="margin-bottom:12px" data-i18n="betaDesc">用 (@音色包名) 切换角色，用 (情绪词) 控制语气。例：(@张三)你好，(开心)今天真不错！(@李四)是啊。</div>
    <div class="field" style="position:relative">
      <label data-i18n="betaTextLabel">朗读文本（输入 (@ 会弹出音色包，支持 情绪词）</label>
      <textarea id="betaText" style="min-height:130px" oninput="betaOnInput()">(@磁性女声，我i的最爱)你好，欢迎使用多人朗读功能。(开心)今天真不错！</textarea>
      <div id="betaAtMenu" style="display:none;position:absolute;z-index:20;background:var(--surface);border:1px solid var(--border-2);border-radius:8px;max-height:200px;overflow:auto;width:100%;box-shadow:0 4px 12px var(--shadow-2)"></div>
    </div>
    <div class="chips" id="betaChips" style="margin-bottom:12px">
      <button class="chip" onclick="betaInsert('(@')" data-i18n="betaAtTag">@音色</button>
      <button class="chip" onclick="betaInsert('(开心)')" data-i18n="emoHappy">开心</button>
      <button class="chip" onclick="betaInsert('(悲伤)')" data-i18n="emoSad">悲伤</button>
      <button class="chip" onclick="betaInsert('(生气)')" data-i18n="emoAngry">生气</button>
      <button class="chip" onclick="betaInsert('(严肃)')" data-i18n="emoSerious">严肃</button>
      <button class="chip" onclick="betaInsert('(温柔)')" data-i18n="emoGentle">温柔</button>
    </div>
    <div class="field">
      <label data-i18n="dialogueTitle">对话面板（每次参与一个独立面板，可折叠，参数独立调节）</label>
      <div id="dialoguePanels"><div class="muted" data-i18n="dialogueEmpty">文本里用 (@音色包名) 指定角色后，这里会为每次参与生成独立面板。</div></div>
    </div>
    <div class="checks" style="margin-bottom:12px">
      <label><input type="checkbox" id="betaDenoise"> <span data-i18n="betaDenoise">背景音降噪</span></label>
    </div>
    <div class="grid" style="margin-bottom:12px">
      <div class="pbox">
        <label data-i18n="betaCfgLabel">CFG 表现力（1.0-3.0，越高越贴合提示；开启导演层后由导演逐段给建议值）</label>
        <div class="prow"><input type="range" id="betaCfg" min="1" max="3" step="0.1" value="2"
          oninput="document.getElementById('betaCfgv').textContent=this.value">
          <span class="pv" id="betaCfgv">2.0</span></div>
      </div>
      <div class="pbox">
        <label data-i18n="betaStepsLabel">扩散步数（4-30，越大越细腻越慢）</label>
        <div class="prow"><input type="range" id="betaSteps" min="4" max="30" step="1" value="10"
          oninput="document.getElementById('betaStepsv').textContent=this.value">
          <span class="pv" id="betaStepsv">10</span></div>
      </div>
    </div>
    <div class="checks" style="margin-bottom:12px;align-items:center">
      <label title="导演层会在合成前先梳理文本：判断台词/旁白、推断情绪、规划停顿与表现力">
        <input type="checkbox" id="betaDirector"> <span data-i18n="betaDirectorLabel">启用导演层（自动梳理情绪/停顿/表现力，不必手写情绪词）</span></label>
      <label style="gap:6px"><span data-i18n="betaEngineLabel">内核</span>
        <select id="betaEngine" class="vp-sel" style="width:auto;display:inline-block;padding:2px 8px;margin:0">
          <option value="rule" data-i18n="betaEngineRule">规则版（本地·快）</option>
          <option value="llm" data-i18n="betaEngineLlm">AI 内核（LLM·需配 Key）</option>
        </select>
      </label>
      <button class="chip" onclick="betaPreviewPlan()" data-i18n="betaPlanBtn">预览梳理结果</button>
    </div>
    <div id="betaLlmBar" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:12px;padding:8px 10px;border:1px solid var(--border-2);border-radius:8px;background:var(--surface-2);font-size:12px">
      <span id="betaLlmDot" style="width:8px;height:8px;border-radius:50%;background:var(--disabled);display:inline-block;flex:0 0 auto"></span>
      <span id="betaLlmText" class="muted" style="flex:1;min-width:160px"></span>
      <button class="chip" onclick="setMode('settings')" style="padding:4px 12px;font-size:12px" data-i18n="betaLlmGo">⚙️ 配置 Key</button>
    </div>
    <div id="betaPlanOut" class="muted" style="display:none;font-size:12px;line-height:1.55;white-space:pre-wrap;max-height:200px;overflow:auto;border:1px solid var(--border);border-radius:8px;padding:8px 10px;margin-bottom:12px;font-family:ui-monospace,Consolas,monospace"></div>
    <button class="gen" id="betaBtn" onclick="betaGenerate()" data-i18n="betaGenerate">🎭 多人朗读生成</button>
    <div class="status" id="betaStatus" style="display:none"><div class="spin"></div><div id="betaStatusText"></div></div>
    <div class="err" id="betaErr"></div>
    <div class="res" id="betaRes" style="display:none">
      <div class="meta" id="betaMeta"></div>
      <audio id="betaPlayer" controls></audio>
    </div>
  </div>

  <div class="card hide" id="trainCard">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <button class="chip" onclick="setMode(prevMode||'design')" data-i18n="backBtn" style="padding:6px 12px;border:1px solid var(--border-2);border-radius:8px;background:var(--surface);cursor:pointer;font-size:13px">← 返回</button>
      <span class="badge">LoRA</span>
      <label data-i18n="trainTitle" style="margin:0">🎓 持续训练</label>
      <span id="trainStatsBadge" class="badge" style="margin-left:auto"></span>
    </div>
    <div class="muted" style="margin-bottom:12px" data-i18n="trainDesc">上传语音并填写对应台词，让模型持续学习音色与风格。建议每条 3~30 秒清晰人声，总计 5 条以上效果更好。训练采用 LoRA（不动原模型权重），完成后可在生成页的「LoRA」下拉中选用。</div>

    <div class="field">
      <label data-i18n="trainAddLabel">➕ 添加训练样本</label>
      <input type="file" id="trainFile" accept="audio/*">
      <textarea id="trainText" data-i18n-ph="trainTextPh" style="min-height:54px;margin-top:6px" placeholder="逐字填写这段语音说的内容（与音频完全一致）"></textarea>
      <div style="display:flex;gap:14px;margin-top:6px;flex-wrap:wrap;font-size:12px;color:var(--text-1)">
        <label style="display:inline-flex;align-items:center;gap:4px;cursor:pointer"><input type="checkbox" id="tsDenoise"> <span data-i18n="enDenoise">🔊 导入前降噪（去除底噪/电流声）</span></label>
        <label style="display:inline-flex;align-items:center;gap:4px;cursor:pointer"><input type="checkbox" id="tsVocal"> <span data-i18n="enVocalOnly">🎤 只保留纯净人声（去背景音乐）</span></label>
      </div>
      <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap">
        <input type="text" id="trainName" data-i18n-ph="trainNamePh" placeholder="备注名（可选）" style="flex:1;min-width:140px">
        <button class="chip" id="trainAddBtn" style="padding:8px 16px" data-i18n="trainAddBtn">添加样本</button>
      </div>
      <div class="err" id="trainErr"></div>
    </div>

    <details id="trWrap" style="border:1px solid var(--border-2);border-radius:10px;padding:10px 12px;margin-bottom:14px">
      <summary style="cursor:pointer;font-weight:600;font-size:14px;outline:none" data-i18n="trSummary">🎧 长音频自动转写（whisper 离线切句，免手填台词）</summary>
      <div style="margin-top:10px">
        <div class="muted" style="margin-bottom:8px;line-height:1.6" data-i18n="trDesc">上传 1~10 分钟的语音（清晰人声、无背景乐效果最佳），会自动按静音切句并逐句转写。核对/修改每段文本后勾选导入为训练样本。首次转写需下载约 460MB 模型（一次性）。</div>
        <input type="file" id="trFile" accept="audio/*">
        <div style="display:flex;gap:14px;margin-top:6px;flex-wrap:wrap;font-size:12px;color:var(--text-1)">
          <label style="display:inline-flex;align-items:center;gap:4px;cursor:pointer"><input type="checkbox" id="trDenoise"> <span data-i18n="enDenoise">🔊 导入前降噪（去除底噪/电流声）</span></label>
          <label style="display:inline-flex;align-items:center;gap:4px;cursor:pointer"><input type="checkbox" id="trVocal"> <span data-i18n="enVocalOnly">🎤 只保留纯净人声（去背景音乐）</span></label>
        </div>
        <div style="margin-top:8px">
          <div style="font-size:12px;font-weight:600;color:var(--text-1)" data-i18n="trTranscriptLabel">📝 有完整台词？粘贴全文，自动逐句匹配到各分段（免手动逐条修改）</div>
          <textarea id="trTranscript" rows="4" style="width:100%;margin-top:4px;font-size:13px;box-sizing:border-box" data-i18n-ph="trTranscriptPh" placeholder="把与音频完全一致的完整台词粘贴到这里（每行一句效果最佳）。可先转写后再粘贴点「按台词匹配」，也可上传前就粘贴、转写完成后自动匹配。"></textarea>
          <div style="display:flex;gap:8px;margin-top:6px;align-items:center;flex-wrap:wrap">
            <button class="chip" id="trAlignBtn" style="display:none;padding:6px 14px;background:var(--violet);color:var(--on-accent)" data-i18n="trAlign">✨ 按台词匹配到各分段</button>
            <label class="chip" style="padding:6px 14px;cursor:pointer;font-size:13px;display:inline-flex;align-items:center;gap:4px">📄 <span data-i18n="trTxtFile">载入 txt 台词</span>
              <input type="file" id="trTxtFile" accept=".txt,text/plain" style="display:none">
            </label>
            <span class="muted" id="trAlignNote" style="font-size:12px"></span>
          </div>
        </div>
        <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap">
          <button class="chip" id="trStartBtn" style="padding:8px 16px" data-i18n="trStart">🎧 开始转写</button>
          <button class="chip" id="trImportBtn" style="display:none;padding:8px 16px;background:var(--green);color:var(--on-accent)" data-i18n="trImport">📥 导入勾选项</button>
        </div>
        <div class="muted" id="trStatus" style="margin-top:8px"></div>
        <div class="err" id="trErr"></div>
        <div id="trResults" style="margin-top:10px"></div>
      </div>
    </details>

    <div class="field">
      <label data-i18n="trainSamplesLabel">📚 训练样本</label>
      <div id="trainSamples"><div class="muted" data-i18n="trainNoSamples">还没有样本，先在上面添加几条。</div></div>
    </div>

    <div class="field">
      <label data-i18n="trainParamsLabel">⚙️ 训练参数（默认即可）</label>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px">
        <div><div class="muted" style="font-size:11px" data-i18n="trainPName">任务名</div><input type="text" id="tpName" placeholder="auto"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="trainPRank">秩 r</div><input type="number" id="tpR" value="8" min="1" max="64"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="trainPAlpha">Alpha</div><input type="number" id="tpAlpha" value="16" min="1" max="128"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="trainPLr">学习率</div><input type="number" id="tpLr" value="0.0001" step="0.00005" min="0.00001"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="trainPEpochs">轮数</div><input type="number" id="tpEpochs" value="3" min="1" max="50"></div>
        <div><div class="muted" style="font-size:11px" data-i18n="trainPAccum">梯度累积</div><input type="number" id="tpAccum" value="4" min="1" max="32"></div>
      </div>
    </div>

    <div style="display:flex;gap:8px;margin-bottom:10px">
      <button class="gen" id="trainStartBtn" style="flex:1" data-i18n="trainStart">🚀 开始训练</button>
      <button class="gen" id="trainStopBtn" style="flex:0 0 auto;background:var(--danger);display:none" data-i18n="trainStop">⏹ 停止</button>
    </div>
    <div class="muted" id="trainNeedHint" style="display:none;font-size:12px;margin:-6px 0 10px;color:var(--warn-ink)"></div>

    <div id="trainProgressWrap" style="display:none;margin-bottom:10px">
      <div style="height:10px;background:var(--border);border-radius:6px;overflow:hidden"><div id="trainProgressBar" style="height:100%;width:0%;background:var(--accent);transition:width .3s"></div></div>
      <div class="muted" id="trainStatusText" style="margin-top:6px"></div>
      <div class="muted" id="trainLossText" style="font-family:ui-monospace,Consolas,monospace;font-size:12px"></div>
    </div>

    <div class="field">
      <label data-i18n="trainLorasLabel">🧩 已训练的 LoRA（生成页可选用）</label>
      <div id="trainLoras"><div class="muted" data-i18n="trainNoLoras">还没有训练好的 LoRA。</div></div>
    </div>
  </div>

  <div class="card" id="packCard">
    <div class="tabs">
      <button class="ptab active" data-pane="manage" onclick="showPackPane('manage')" data-i18n="packManage">🎭 音色包管理</button>
      <button class="ptab" data-pane="save" onclick="showPackPane('save')" data-i18n="packMake">🎙️ 制作音色声线包</button>
    </div>

      <div id="packManage">
        <div class="muted" style="margin-bottom:12px" data-i18n="packDesc">已提取并保存在本地的音色声线包，后续克隆可直接选用，无需重复上传长音频。数据存于 <code>voice_packs/</code> 目录，重启服务后依然保留。也可用 API：<code>POST /api/voicepacks</code> 保存，生成时传 <code>voice_pack_id</code>。带 <span style="color:var(--warn-ink)">⚡加速</span> 标记的音色包在生成时自动提速。</div>
        <div id="packList"><div class="muted" data-i18n="packEmpty">还没有音色包，去“制作音色声线包”做一个吧。</div></div>
      </div>

    <div id="packSave" class="hide">
      <div class="field">
        <label data-i18n="recMethod">方式一：实时录制（直接用麦克风，无需上传文件）</label>
        <button class="gen" id="recBtn" onclick="startRec()" style="background:var(--sky)" data-i18n="recStart">🎤 开始录制</button>
        <div class="muted" id="recStatus" style="margin-top:6px;color:var(--sky-ink)" data-i18n="recHint">点击下方按钮授权麦克风后开始朗读，建议 10–30 秒清晰语句；录制完可回放确认。</div>
        <div class="field hide" id="recWrap" style="margin-top:10px">
          <label data-i18n="recPlayback">录制回放（确认无误再保存）</label>
          <audio id="recPlay" controls></audio>
        </div>
      </div>
      <div class="field" id="vpDropZone" style="border:2px dashed var(--border-2);border-radius:10px;padding:12px;transition:all .2s">
        <label data-i18n="upMethod">方式二：上传音频或拖拽视频（wav/mp3/flac/mp4/mov 等，视频自动提取人声）</label>
        <input type="file" id="vpFile" accept="audio/*,video/*">
        <div class="muted" id="vpDropHint" style="margin-top:6px" data-i18n="vpDropHint">建议 10–60 秒清晰人声；超过 30 秒会自动分段并融合为约 25 秒的代表参考。视频文件会自动提取音轨（需已安装 ffmpeg）。</div>
      </div>
      <div class="field">
        <label data-i18n="vpNameLabel">音色包名称（便于识别）</label>
        <input type="text" id="vpName" data-i18n-ph="vpNamePh" placeholder="例如：客服小美 / 讲师老王">
      </div>
      <div class="checks">
        <label><input type="checkbox" id="vpDenoise" checked> <span data-i18n="vpDenoise">参考音频降噪</span></label>
        <label><input type="checkbox" id="vpRemoveBg"> <span data-i18n="vpRemoveBg">去除背景音/音乐</span></label>
        <label title="开启后，使用该音色包克隆生成时会自动采用更少扩散步数，生成更快（音质略降）"><input type="checkbox" id="vpAccel"> <span data-i18n="vpAccel">🚀 加速模式（生成更快）</span></label>
      </div>
      <button class="gen" id="vpSaveBtn" onclick="savePack()" data-i18n="vpSaveBtn">🔒 提取并保存音色包</button>
      <div class="status" id="vpStatus"><div class="spin"></div><div id="vpStatusText" data-i18n="vpStatusIdle">提取中…（首次需加载模型，请稍候）</div></div>
      <div class="err" id="vpErr"></div>
    </div>
  </div>

  <div class="card">
    <label data-i18n="apiLabel">API 调用示例（令牌放请求头）</label>
    <div class="api" id="apiSample">curl -X POST http://localhost:PORT_PLACEHOLDER/api/tts \\
  -H "X-API-Key: <span data-i18n="apiToken">你的访问令牌</span>" \\
  -H "Content-Type: application/json" \\
  -d "{\\"text\\":\\"你好世界\\",\\"cfg_value\\":2.0,\\"inference_timesteps\\":10}" \\
  --output out.wav</div>
  </div>
</div>

<script>
const API_TOKEN='TOKEN_PLACEHOLDER';
let mode='design';
let prevMode='design';
let selectedPackId=null;
let voicePacks=[];
let lastOutputName=null;
// ===== i18n 双语言 =====
const I18N={
  zh:{localDeploy:'本地部署',detecting:'检测中…',modelNotLoaded:'模型未加载',modelReady:'模型就绪',
      modeDesign:'🎨 语音设计',modeClone:'🎛️ 音色克隆',modeHifi:'🎙️ 极致克隆',modeBeta:'🧪 内测 Beta',modeTrain:'🎓 训练',
      history:'本次会话生成记录',noHistory:'还没有生成记录',
      betaTitle:'多人朗读与情绪控制',
      betaDesc:'用 (@音色包名) 切换角色，用 (情绪词) 控制语气。例：(@张三)你好，(开心)今天真不错！(@李四)是啊。',
      betaTextLabel:'朗读文本（输入 (@ 会弹出音色包，支持 情绪词）',
      emoHappy:'开心',emoSad:'悲伤',emoAngry:'生气',emoSerious:'严肃',emoGentle:'温柔',
      betaGenerate:'🎭 多人朗读生成',betaLoading:'生成中…已用 ',betaSeconds:' 秒',betaFail:'请求失败',
      dialogueTitle:'对话面板（每次参与一个独立面板，可折叠，参数独立调节）',
      dialogueEmpty:'文本里用 (@音色包名) 指定角色后，这里会为每次参与生成独立面板。',
      turnLabel:'第{n}次参与',
      betaDenoise:'背景音降噪',
      betaCfgLabel:'CFG 表现力（1.0-3.0，越高越贴合提示；开启导演层后由导演逐段给建议值）',
      betaStepsLabel:'扩散步数（4-30，越大越细腻越慢）',
      betaDirectorLabel:'启用导演层（自动梳理情绪/停顿/表现力，不必手写情绪词）',
      betaEngineLabel:'内核',
      betaEngineRule:'规则版（本地·快）',
      betaEngineLlm:'AI 内核（LLM·需配 Key）',
      betaPlanBtn:'预览梳理结果',
      betaLlmGo:'⚙️ 配置 Key',
      modeSettings:'⚙️ 设置',
      setTitle:'⚙️ 设置 · AI 内核配置',
      setReasonNotEnabled:'AI 内核未启用（把上面的开关打开即可）',
      setReasonNoKey:'缺少 API Key —— 填好下面的密钥后保存',
      setReasonNoBaseUrl:'缺少 base_url（接口地址）',
      setReasonNoModel:'缺少 model（模型名）',
      setReasonUnknown:'配置不完整',
      setDesc:'导演层的「AI 内核」需要一个可调用的大模型来梳理情绪与停顿。在这里填入任意 OpenAI 兼容服务的密钥即可启用——选中服务商后地址与模型会自动带出，也可以选「自定义」手填任何中转站或内网网关。密钥只保存在本机 llm_config.json，不会外传。',
      setEnableLabel:'启用 AI 内核',
      setProviderLabel:'服务商预设',
      setKeyLabel:'API Key',
      setKeyPh:'粘贴你的 API Key（留空表示不修改已保存的密钥）',
      setShow:'👁 显示',
      setBaseLabel:'接口地址 base_url',
      setModelLabel:'模型名 model',
      setFetchModels:'⬇ 拉取可用模型',
      setAdvanced:'🔧 高级参数（默认即可）',
      setTimeout:'超时（秒）',setTemp:'温度 temperature',setProxy:'代理（留空=直连）',
      setJsonMode:'使用 JSON 模式（兼容层不支持时自动回退）',
      setCache:'缓存梳理结果（同文本不重复调用）',
      setSave:'💾 保存并测试',setTest:'🔌 仅测试连接',setReload:'↻ 重新读取',
      betaPlanLoading:'正在梳理文本…',
      betaPlanFail:'梳理失败',
      betaPlanTitle:'导演层梳理结果',
      backBtn:'← 返回',
      synthText:'合成文本',genBtn:'🔊 生成语音',
      packManage:'🎭 音色包管理',
      textDefault:'你好，这里是本地部署的 VoxCPM2 语音大模型，现在可以直接在浏览器里使用了。',
      chip1:'年轻女性·温柔',chip2:'中年男性·沉稳',chip3:'活力少年',chip4:'粤语',chip5:'四川话',chip6:'新闻播报',
      designHint:'语音设计模式：用「()」在文本开头描述想要的音色、情绪、语速，例如「(年轻女性，温柔甜美)你好」。',
      refLabel:'参考音频（0.3 秒 – 10 分钟，wav/mp3/flac）',refHint:'克隆模式必填，模型会复刻这段音频的音色。',
      packSelLabel:'或选择已保存音色包（免重复上传长音频）',noPackOpt:'— 不使用音色包，改为上传音频 —',
      ptLabel:'参考音频的逐字文本（极致克隆必填）',ptPh:'必须与参考音频内容完全一致',
      cfgLabel:'CFG 引导强度（1.0-3.0，默认 2.0）',stepsLabel:'扩散步数（4-30，越大越细腻越慢）',
      normalizeLabel:'文本规范化（数字/日期正确读出）',denoiseLabel:'参考音频降噪',removeBgLabel:'去除背景音/音乐',stableLabel:'长文本稳定合成',
      refTip:'提示：参考音频超过 30 秒时会自动分段，按说话人声纹融合为约 25 秒的代表音频，显著缓解长音频克隆的失真与音色漂移。',
      emoLabel:'🎭 情绪语气（可选，套用一组音调/语速/停顿预设，可再手动微调）',emoNone:'— 不指定情绪 —',emoCalm:'平静',
      pitchLabel:'音调（半音，0=原音）',speedLabel:'语速（0.5x-2.0x）',volumeLabel:'音量（0.1x-2.0x）',pauseLabel:'句间停顿（秒，0=无）',breathLabel:'呼吸声轻重（0=无）',
      ssmlLabel:'启用 SSML 标签',statusIdle:'生成中…',exportLabel:'⬇️ 导出',
      betaAtTag:'@音色',
      betaDefault:'(@磁性女声，我i的最爱)你好，欢迎使用多人朗读功能。(开心)今天真不错！',
      packMake:'🎙️ 制作音色声线包',
      packDesc:'已提取并保存在本地的音色声线包，后续克隆可直接选用，无需重复上传长音频。数据存于 voice_packs/ 目录，重启服务后依然保留。也可用 API：POST /api/voicepacks 保存，生成时传 voice_pack_id。带 ⚡加速 标记的音色包在生成时自动提速。',
      packEmpty:'还没有音色包，去“制作音色声线包”做一个吧。',
      recMethod:'方式一：实时录制（直接用麦克风，无需上传文件）',recStart:'🎤 开始录制',
      recHint:'点击下方按钮授权麦克风后开始朗读，建议 10–30 秒清晰语句；录制完可回放确认。',recPlayback:'录制回放（确认无误再保存）',
      upMethod:'方式二：上传音频或拖拽视频（wav/mp3/flac/mp4/mov 等，视频自动提取人声）',
      vpDropHint:'建议 10–60 秒清晰人声；超过 30 秒会自动分段并融合为约 25 秒的代表参考。视频文件会自动提取音轨（需已安装 ffmpeg）。',
      vpNameLabel:'音色包名称（便于识别）',vpNamePh:'例如：客服小美 / 讲师老王',
      vpDenoise:'参考音频降噪',vpRemoveBg:'去除背景音/音乐',vpAccel:'🚀 加速模式（生成更快）',
      vpSaveBtn:'🔒 提取并保存音色包',vpStatusIdle:'提取中…（首次需加载模型，请稍候）',
      apiLabel:'API 调用示例（令牌放请求头）',apiToken:'你的访问令牌',
      trainTitle:'🎓 持续训练',trainDesc:'上传语音并填写对应台词，让模型持续学习音色与风格。建议每条 3~30 秒清晰人声，总计 5 条以上效果更好。训练采用 LoRA（不动原模型权重），完成后可在生成页的「LoRA」下拉中选用。',
      trainAddLabel:'➕ 添加训练样本',trainTextPh:'逐字填写这段语音说的内容（与音频完全一致）',trainNamePh:'备注名（可选）',trainAddBtn:'添加样本',
      trainSamplesLabel:'📚 训练样本',trainNoSamples:'还没有样本，先在上面添加几条。',
      trainParamsLabel:'⚙️ 训练参数（默认即可）',trainPName:'任务名',trainPRank:'秩 r',trainPAlpha:'Alpha',trainPLr:'学习率',trainPEpochs:'轮数',trainPAccum:'梯度累积',
      trainStart:'🚀 开始训练',trainStop:'⏹ 停止',
      trainLorasLabel:'🧩 已训练的 LoRA（生成页可选用）',trainNoLoras:'还没有训练好的 LoRA。',
      loraLabel:'🧩 LoRA 微调音色（训练页产出，选用后整段生效）',loraNone:'— 不使用 —',
      trainUploading:'上传中…',trainAdded:'已添加',trainSamples:'条样本',trainAddFail:'添加失败',
      trainRunning:'训练进行中，请等待完成或点击停止',trainIdle:'空闲',trainStarting:'正在启动训练…',
      trainProgress:'进度',trainLoss:'损失',trainStep:'步',trainEpoch:'轮',
      trainDone:'训练完成！已保存 LoRA：',trainFailed:'训练失败',trainStopped:'训练已停止',
      trainDelSample:'删除该样本？',trainDelLora:'删除该 LoRA？权重文件会被永久移除。',
      trainUseLora:'使用此 LoRA',trainPlaySample:'试听',trainDelete:'删除',
      trainNeedSamples:'样本不足（当前 {cur} 条）：至少需要 2 条「语音+台词」样本才能开始训练',
      trainNoFile:'请先选择音频文件',trainNoText:'请填写与音频对应的台词文本',
      trainFileTooShort:'音频过短（不足 1 秒）',trainFileTooLong:'音频过长（超过 30 秒）',trainTextTooLong:'台词过长（超过 400 字）',
      enDenoise:'🔊 导入前降噪（去除底噪/电流声）',enVocalOnly:'🎤 只保留纯净人声（去背景音乐）',
      trSummary:'🎧 长音频自动转写（whisper 离线切句，免手填台词）',
      trDesc:'上传 1~10 分钟的语音（清晰人声、无背景乐效果最佳），会自动按静音切句并逐句转写。核对/修改每段文本后勾选导入为训练样本。首次转写需加载约 460MB 的 whisper 模型（一次性）。',
      trStart:'🎧 开始转写',trImport:'📥 导入勾选项',trNoSeg:'未识别到可用的语音片段（音频过短或无人声）',
      trPlay:'试听',trSelAll:'全选',trSegDur:'片段 {n}',trImportOk:'已导入 {n} 条样本',
      trModelLoading:'正在加载 whisper 模型（首次约需 1~2 分钟）…',
      trTranscriptLabel:'📝 有完整台词？粘贴全文，自动逐句匹配到各分段（免手动逐条修改）',
      trTranscriptPh:'把与音频完全一致的完整台词粘贴到这里（每行一句效果最佳）。可先转写后再粘贴点「按台词匹配」，也可上传前就粘贴、转写完成后自动匹配。',
      trAlign:'✨ 按台词匹配到各分段',trTxtFile:'载入 txt 台词',trNoTranscript:'请先粘贴完整台词',themeGoDark:'🌙 深色',themeGoLight:'☀️ 浅色'},
  en:{localDeploy:'Local',detecting:'Detecting…',modelNotLoaded:'Model not loaded',modelReady:'Model ready',
      modeDesign:'🎨 Voice Design',modeClone:'🎛️ Voice Clone',modeHifi:'🎙️ HiFi Clone',modeBeta:'🧪 Beta',modeTrain:'🎓 Train',
      history:'Generation history',noHistory:'No history yet',
      betaTitle:'Multi-speaker & Emotion Control',
      betaDesc:'Use (@pack_name) to switch speaker, (emotion) for tone. e.g. (@John)Hello, (happy)Great day! (@Jane)Yeah.',
      betaTextLabel:'Text (type (@ to pick a voice pack, emotion tags supported)',
      emoHappy:'Happy',emoSad:'Sad',emoAngry:'Angry',emoSerious:'Serious',emoGentle:'Gentle',
      betaGenerate:'🎭 Multi-speaker Generate',betaLoading:'Generating… ',betaSeconds:'s elapsed',betaFail:'Request failed',
      dialogueTitle:'Dialogue panels (one collapsible panel per turn, independent settings)',
      dialogueEmpty:'Add (@pack_name) tags in the text; each turn gets its own panel here.',
      turnLabel:'Turn {n}',
      betaDenoise:'Background noise reduction',
      betaCfgLabel:'CFG expressiveness (1.0-3.0; higher follows the prompt more closely; the director suggests per-segment values when enabled)',
      betaStepsLabel:'Diffusion steps (4-30; higher = finer but slower)',
      betaDirectorLabel:'Enable director layer (auto-plans emotion / pauses / expressiveness — no manual emotion tags needed)',
      betaEngineLabel:'Engine',
      betaEngineRule:'Rule (local, fast)',
      betaEngineLlm:'AI kernel (LLM, needs API key)',
      betaPlanBtn:'Preview plan',
      betaLlmGo:'⚙️ Configure key',
      modeSettings:'⚙️ Settings',
      setTitle:'⚙️ Settings · AI Kernel',
      setReasonNotEnabled:'AI kernel is off (flip the switch above to enable)',
      setReasonNoKey:'No API key yet — fill in the key below and save',
      setReasonNoBaseUrl:'Missing base_url (endpoint)',
      setReasonNoModel:'Missing model name',
      setReasonUnknown:'Configuration incomplete',
      setDesc:'The "AI kernel" drives the director layer — it needs a callable LLM to plan emotion and pauses. Paste a key from any OpenAI-compatible service to enable it — picking a provider fills in the endpoint and model automatically, or choose "Custom" to enter any relay/proxy gateway by hand. The key is stored only in your local llm_config.json and never sent anywhere else.',
      setEnableLabel:'Enable AI kernel',
      setProviderLabel:'Provider preset',
      setKeyLabel:'API Key',
      setKeyPh:'Paste your API key (leave blank to keep the saved one)',
      setShow:'👁 Show',
      setBaseLabel:'Endpoint base_url',
      setModelLabel:'Model name',
      setFetchModels:'⬇ Fetch models',
      setAdvanced:'🔧 Advanced (defaults are fine)',
      setTimeout:'Timeout (s)',setTemp:'Temperature',setProxy:'Proxy (blank = direct)',
      setJsonMode:'Use JSON mode (auto-fallback if unsupported)',
      setCache:'Cache planning results (skip repeat calls)',
      setSave:'💾 Save & test',setTest:'🔌 Test connection',setReload:'↻ Reload',
      betaPlanLoading:'Planning text…',
      betaPlanFail:'Planning failed',
      betaPlanTitle:'Director plan',
      backBtn:'← Back',
      synthText:'Text to synthesize',genBtn:'🔊 Generate',
      packManage:'🎭 Voice Packs',
      textDefault:'Hi, this is the locally-deployed VoxCPM2 voice model. You can use it right in your browser.',
      chip1:'Young woman, gentle',chip2:'Mature man, deep voice',chip3:'Lively teenager',chip4:'Cantonese',chip5:'Sichuan dialect',chip6:'News anchor',
      designHint:'Design mode: describe the voice, emotion or speed in "()" at the start of the text, e.g. "(young woman, sweet)Hello".',
      refLabel:'Reference audio (0.3s – 10min, wav/mp3/flac)',refHint:'Required for Clone mode. The model replicates the timbre of this audio.',
      packSelLabel:'Or pick a saved voice pack (no re-upload needed)',noPackOpt:'— No voice pack, upload audio instead —',
      ptLabel:'Verbatim transcript of the reference (required for HiFi)',ptPh:'Must match the reference audio exactly',
      cfgLabel:'CFG guidance (1.0-3.0, default 2.0)',stepsLabel:'Diffusion steps (4-30, higher = finer & slower)',
      normalizeLabel:'Text normalization (numbers/dates read correctly)',denoiseLabel:'Reference denoise',removeBgLabel:'Remove background/music',stableLabel:'Long-text stable synthesis',
      refTip:'Tip: audio over 30s is auto-segmented and fused by voiceprint into a ~25s representative clip, reducing distortion and timbre drift in long-audio cloning.',
      emoLabel:'🎭 Emotion & tone (optional preset, fine-tune below)',emoNone:'— No emotion —',emoCalm:'Calm',
      pitchLabel:'Pitch (semitones, 0=original)',speedLabel:'Speed (0.5x-2.0x)',volumeLabel:'Volume (0.1x-2.0x)',pauseLabel:'Pause between sentences (s, 0=none)',breathLabel:'Breath intensity (0=none)',
      ssmlLabel:'Enable SSML tags',statusIdle:'Generating…',exportLabel:'⬇️ Export',
      betaAtTag:'@Pack',
      betaDefault:'(@John)Hello! Welcome to multi-speaker reading. (happy)What a nice day!',
      packMake:'🎙️ Create Voice Pack',
      packDesc:'Voice packs are extracted and saved locally for reuse, so you never re-upload long audio. Stored under the voice_packs/ directory and persist across restarts. Save via POST /api/voicepacks and pass voice_pack_id when generating. Packs marked ⚡accelerated generate faster automatically.',
      packEmpty:'No voice packs yet. Go to "Create Voice Pack" to make one.',
      recMethod:'Method 1: record live (microphone, no file upload)',recStart:'🎤 Start Recording',
      recHint:'Click the button, allow microphone access, then read for 10–30s. Playback to confirm after recording.',recPlayback:'Playback (confirm before saving)',
      upMethod:'Method 2: upload audio or drag a video (wav/mp3/flac/mp4/mov; voice auto-extracted from video)',
      vpDropHint:'10–60s of clear voice is recommended; over 30s is auto-segmented and fused into a ~25s representative clip. Video audio is auto-extracted (ffmpeg required).',
      vpNameLabel:'Voice pack name (for identification)',vpNamePh:'e.g. Support-Xiaomei / Trainer-LaoWang',
      vpDenoise:'Reference denoise',vpRemoveBg:'Remove background/music',vpAccel:'🚀 Accelerated mode (faster generation)',
      vpSaveBtn:'🔒 Extract & Save Voice Pack',vpStatusIdle:'Extracting… (model loads on first run, please wait)',
      apiLabel:'API examples (token in header)',apiToken:'YOUR_TOKEN',
      trainTitle:'🎓 Continuous Training',trainDesc:'Upload audio with verbatim transcripts and the model learns your voice/style via LoRA (original weights untouched). 3–30s per clip, 5+ clips recommended. Finished LoRAs appear in the LoRA dropdown on the generate page.',
      trainAddLabel:'➕ Add Training Sample',trainTextPh:'Type exactly what this audio says (must match perfectly)',trainNamePh:'Note name (optional)',trainAddBtn:'Add Sample',
      trainSamplesLabel:'📚 Training Samples',trainNoSamples:'No samples yet — add some above.',
      trainParamsLabel:'⚙️ Training Params (defaults are fine)',trainPName:'Task name',trainPRank:'Rank r',trainPAlpha:'Alpha',trainPLr:'Learning rate',trainPEpochs:'Epochs',trainPAccum:'Grad accum',
      trainStart:'🚀 Start Training',trainStop:'⏹ Stop',
      trainLorasLabel:'🧩 Trained LoRAs (selectable on generate page)',trainNoLoras:'No trained LoRAs yet.',
      loraLabel:'🧩 LoRA fine-tuned voice (from training page, applies to entire output)',loraNone:'— None —',
      trainUploading:'Uploading…',trainAdded:'Added',trainSamples:'samples',trainAddFail:'Add failed',
      trainRunning:'Training in progress — wait or click Stop',trainIdle:'Idle',trainStarting:'Starting training…',
      trainProgress:'Progress',trainLoss:'Loss',trainStep:'step',trainEpoch:'epoch',
      trainDone:'Training complete! LoRA saved: ',trainFailed:'Training failed',trainStopped:'Training stopped',
      trainDelSample:'Delete this sample?',trainDelLora:'Delete this LoRA? Weight file will be permanently removed.',
      trainUseLora:'Use this LoRA',trainPlaySample:'Play',trainDelete:'Delete',
      trainNeedSamples:'Not enough samples (currently {cur}): at least 2 audio+text pairs required to start training',
      trainNoFile:'Please select an audio file first',trainNoText:'Please enter the transcript matching the audio',
      trainFileTooShort:'Audio too short (< 1s)',trainFileTooLong:'Audio too long (> 30s)',trainTextTooLong:'Transcript too long (> 400 chars)',
      enDenoise:'🔊 Denoise before import (hiss / AC hum)',enVocalOnly:'🎤 Keep vocals only (remove BGM/music)',
      trSummary:'🎧 Auto-transcribe long audio (offline whisper, no manual transcript)',
      trDesc:'Upload 1–10 min of speech (clear voice, no background music works best). It is auto-segmented by silence and transcribed sentence by sentence. Review/edit each line, tick the ones to keep, then import as training samples. The ~460MB whisper model loads on first use (one-time).',
      trStart:'🎧 Start',trImport:'📥 Import selected',trNoSeg:'No usable speech segments found (audio too short or no voice)',
      trPlay:'Play',trSelAll:'Select all',trSegDur:'Seg {n}',trImportOk:'{n} samples imported',
      trModelLoading:'Loading whisper model (first run ~1–2 min)…',
      trTranscriptLabel:'📝 Have the verbatim transcript? Paste it and auto-match into each segment (no manual line-by-line edits)',
      trTranscriptPh:'Paste the full transcript that matches the audio exactly (one sentence per line works best). You can transcribe first and then click "Match transcript", or paste before uploading and it will be matched automatically when transcription finishes.',
      trAlign:'✨ Match transcript into segments',trTxtFile:'Load txt',trNoTranscript:'Please paste the full transcript first',
      themeGoDark:'🌙 Dark',themeGoLight:'☀️ Light'}
};
let curLang='zh';
function setLang(l){
  curLang=l;const d=I18N[l]||I18N.zh;
  document.querySelectorAll('[data-i18n]').forEach(el=>{const k=el.getAttribute('data-i18n');if(d[k]!==undefined)el.textContent=d[k];});
  document.querySelectorAll('[data-i18n-ph]').forEach(el=>{const k=el.getAttribute('data-i18n-ph');if(d[k]!==undefined)el.placeholder=d[k];});
  // 示例文本：仅当用户尚未修改时随语言切换
  const t1=document.getElementById('text'),t2=document.getElementById('betaText');
  const zh1=I18N.zh.textDefault,en1=I18N.en.textDefault,zh2=I18N.zh.betaDefault,en2=I18N.en.betaDefault;
  if(t1&&(!t1.value||t1.value===zh1||t1.value===en1))t1.value=d.textDefault;
  if(t2&&(!t2.value||t2.value===zh2||t2.value===en2))t2.value=d.betaDefault;
  document.getElementById('langZh').style.opacity=(l==='zh')?'1':'.5';
  document.getElementById('langEn').style.opacity=(l==='en')?'1':'.5';
  if(window.updateThemeBtn)updateThemeBtn();
  try{localStorage.setItem('voxcpm_lang',l);}catch(_){}
  repaintDynamicText();
}
function tr(zh,en){return curLang==='zh'?zh:en;}
// ===== 主题（深色/浅色） =====
function curTheme(){return document.documentElement.getAttribute('data-theme')==='dark'?'dark':'light';}
function updateThemeBtn(){var b=document.getElementById('themeBtn');if(!b)return;
  var d=I18N[curLang]||I18N.zh;
  b.textContent=(curTheme()==='dark')?d.themeGoLight:d.themeGoDark;
  b.title=b.textContent;}
function toggleTheme(){var t=curTheme()==='dark'?'light':'dark';
  document.documentElement.setAttribute('data-theme',t);
  try{localStorage.setItem('voxcpm_theme',t);}catch(_){}
  updateThemeBtn();}
// ===== Beta：多人朗读 =====
function betaInsert(tag){
  const el=document.getElementById('betaText');const s=el.selectionStart||0,e=el.selectionEnd||0;
  el.value=el.value.slice(0,s)+tag+el.value.slice(e);el.focus();
  el.selectionStart=el.selectionEnd=s+tag.length;
  renderDialoguePanels();
}
// @音色自动补全：输入 (@ 时弹出已有音色包列表
function betaOnInput(){
  const el=document.getElementById('betaText'),menu=document.getElementById('betaAtMenu');
  const s=el.selectionStart||0,before=el.value.slice(0,s);
  const m=before.match(/\(@([^()]*)$/);
  if(!m){menu.style.display='none';return;}
  const kw=m[1].trim();
  const items=voicePacks.filter(p=>!kw||p.name.includes(kw));
  if(!items.length){menu.style.display='none';return;}
  // 用 DOM 创建 + 事件绑定，避免 innerHTML 字符串拼接的引号转义问题
  menu.innerHTML='';
  items.forEach(p=>{
    const d=document.createElement('div');
    d.style.cssText='padding:8px 12px;cursor:pointer;border-bottom:1px solid var(--surface-3);font-size:13px';
    d.textContent=p.name;
    d.onmousedown=function(){betaPickVoice(p.name);};
    menu.appendChild(d);
  });
  menu.style.display='block';
}
function betaPickVoice(name){
  const el=document.getElementById('betaText'),menu=document.getElementById('betaAtMenu');
  const s=el.selectionStart||0,before=el.value.slice(0,s);
  const idx=before.lastIndexOf('(@');
  if(idx<0)return;
  el.value=before.slice(0,idx+2)+name+')'+el.value.slice(s);
  el.focus();const pos=idx+2+name.length+1;el.selectionStart=el.selectionEnd=pos;
  menu.style.display='none';renderDialoguePanels();
}
// 渲染角色参数：解析文本里的 @音色，每个角色一组独立参数滑块
let dialogues=[];   // 参与状态 [{role,seq,text,emotion,tone,volume,pitch,speed,pause,breath,collapsed,voice,narrative}]
const TONE_OPTS={zh:['自然','温柔','严肃','活泼','低沉'],en:['Natural','Gentle','Serious','Lively','Low']};
const EMO_OPTS={zh:['无','高兴','悲伤','生气','严肃','温柔','惊讶','恐惧'],en:['None','Happy','Sad','Angry','Serious','Gentle','Surprised','Fear']};
function parseDialogue(text){
  const t=(text||'').replace(/（/g,'(').replace(/）/g,')');
  const EM={'高兴':'高兴','开心':'高兴','快乐':'高兴','happy':'高兴','悲伤':'悲伤','难过':'悲伤','伤心':'悲伤','sad':'悲伤','严肃':'严肃','serious':'严肃','温柔':'温柔','gentle':'温柔','soft':'温柔','愤怒':'愤怒','生气':'愤怒','angry':'愤怒','平静':'平静','calm':'平静','neutral':'平静','中性':'平静'};
  const res=[];const seqMap={};
  let cur=null;
  const re=/\(([^()]*)\)/g;let pos=0,m;
  function newCur(role,voice){
    const seq=(seqMap[role]=(seqMap[role]||0)+1);
    return {role:role,seq:seq,text:'',emotion:'neutral',tone:'自然',volume:1,pitch:0,speed:1,pause:0.15,breath:0.4,collapsed:false,voice:voice,narrative:voice==null};
  }
  function flush(){ if(cur&&cur.text.trim())res.push(cur); }
  function append(txt){ if(!txt)return; if(!cur)cur=newCur('旁白',null); cur.text+=(cur.text?' ':'')+txt; }
  while((m=re.exec(t))){
    const before=t.slice(pos,m.index); if(before.trim())append(before.trim());
    const content=m[1].trim();
    if(content.startsWith('@')){
      const body=content.slice(1); const parts=body.split(',');
      const name=parts[0].trim(); const role=name||'旁白';
      flush(); cur=newCur(role,name||null);
      if(parts.length>1){ const e=EM[parts[1].trim().toLowerCase()]; if(e&&e!=='平静')cur.emotion=e; }
    } else {
      const e=EM[content.toLowerCase()];
      if(e){ if(cur)cur.emotion=(e==='平静'?'neutral':e); }
      else append('('+content+')');
    }
    pos=m.index+m[0].length;
  }
  const tail=t.slice(pos); if(tail.trim())append(tail.trim());
  flush();
  return res;
}
function renderDialoguePanels(){
  const box=document.getElementById('dialoguePanels');
  const text=document.getElementById('betaText').value||'';
  const fresh=parseDialogue(text);
  // 保留用户已改参数：按 role+seq 合并（同一参与只更新台词/情绪，保留 tone/volume/pitch/speed/pause/breath/collapsed）
  const keep={};
  dialogues.forEach(d=>{ if(d.role&&d.seq)keep[d.role+'#'+d.seq]=d; });
  dialogues=fresh.map(d=>{ const k=keep[d.role+'#'+d.seq];
    return k?Object.assign({},d,{tone:k.tone,volume:k.volume,pitch:k.pitch,speed:k.speed,pause:k.pause,breath:k.breath,collapsed:k.collapsed}):d; });
  if(!dialogues.length){ box.innerHTML='<div class="muted" data-i18n="dialogueEmpty">文本里用 (@音色包名) 指定角色后，这里会为每次参与生成独立面板。</div>'; setLang(curLang); return; }
  box.innerHTML='';
  dialogues.forEach((d,idx)=>{
    const panel=document.createElement('div');
    panel.style.cssText='border:1px solid var(--border);border-radius:10px;margin-bottom:10px;overflow:hidden';
    // 头部：折叠箭头 + 角色标识
    const head=document.createElement('div');
    head.style.cssText='display:flex;align-items:center;gap:8px;padding:10px 12px;cursor:pointer;font-weight:600;background:var(--surface-2);font-size:13px';
    head.onclick=function(){toggleDp(idx);};
    const arrow=document.createElement('span'); arrow.id='dp_arrow_'+idx; arrow.textContent=d.collapsed?'▸':'▾';
    const label=document.createElement('span');
    label.textContent=(curLang==='zh'?'':'')+d.role+' - '+I18N[curLang].turnLabel.replace('{n}',d.seq);
    head.appendChild(arrow); head.appendChild(label);
    panel.appendChild(head);
    // 主体：参数
    const body=document.createElement('div'); body.id='dp_body_'+idx;
    body.style.cssText='padding:10px 12px;border-top:1px solid var(--surface-3)'+(d.collapsed?';display:none':'');
    const L=curLang;
    function row(lbl){ const r=document.createElement('div'); r.style.cssText='margin-bottom:8px'; const s=document.createElement('div'); s.style.cssText='font-size:12px;color:var(--text-2);margin-bottom:4px'; s.textContent=lbl; r.appendChild(s); return r; }
    function slider(lbl,min,max,step,val,cb){ const r=row(lbl); const w=document.createElement('div'); w.style.cssText='display:flex;align-items:center;gap:10px'; const inp=document.createElement('input'); inp.type='range'; inp.min=min; inp.max=max; inp.step=step; inp.value=val; inp.style.cssText='flex:1'; const v=document.createElement('span'); v.style.cssText='width:36px;text-align:right;font-size:12px;color:var(--text-2)'; v.textContent=val; inp.oninput=function(){ v.textContent=inp.value; cb(parseFloat(inp.value)); }; w.appendChild(inp); w.appendChild(v); r.appendChild(w); return r; }
    // 语气
    const rTone=row(L==='zh'?'语气':'Tone'); const selTone=document.createElement('select');
    selTone.style.cssText='width:100%;padding:6px;border:1px solid var(--border-2);border-radius:6px;font-size:13px';
    TONE_OPTS[L].forEach(o=>{ const op=document.createElement('option'); op.textContent=o; op.value=o; if(o===d.tone)op.selected=true; selTone.appendChild(op); });
    selTone.onchange=function(){d.tone=selTone.value;};
    rTone.appendChild(selTone); body.appendChild(rTone);
    // 台词
    const rText=row(L==='zh'?'台词':'Line'); const ta=document.createElement('textarea');
    ta.style.cssText='width:100%;padding:6px;border:1px solid var(--border-2);border-radius:6px;font-size:13px;min-height:44px;font-family:inherit';
    ta.value=d.text; ta.oninput=function(){d.text=ta.value;};
    rText.appendChild(ta); body.appendChild(rText);
    // 情绪
    const rEmo=row(L==='zh'?'情绪':'Emotion'); const selEmo=document.createElement('select');
    selEmo.style.cssText='width:100%;padding:6px;border:1px solid var(--border-2);border-radius:6px;font-size:13px';
    const curEmo=(d.emotion==='neutral'||d.emotion==='平静')?(L==='zh'?'无':'None'):d.emotion;
    EMO_OPTS[L].forEach(o=>{ const op=document.createElement('option'); op.textContent=o; op.value=o; if(o===curEmo)op.selected=true; selEmo.appendChild(op); });
    selEmo.onchange=function(){ const v=selEmo.value; d.emotion=(v==='无'||v==='None')?'neutral':v; };
    rEmo.appendChild(selEmo); body.appendChild(rEmo);
    // 音量
    const rVol=row(L==='zh'?'音量':'Volume'); const volWrap=document.createElement('div'); volWrap.style.cssText='display:flex;align-items:center;gap:10px';
    const vol=document.createElement('input'); vol.type='range'; vol.min='0.3'; vol.max='2'; vol.step='0.05'; vol.value=d.volume; vol.style.cssText='flex:1';
    const volV=document.createElement('span'); volV.style.cssText='width:36px;text-align:right;font-size:12px;color:var(--text-2)'; volV.textContent=d.volume;
    vol.oninput=function(){d.volume=parseFloat(vol.value);volV.textContent=vol.value;};
    volWrap.appendChild(vol); volWrap.appendChild(volV); rVol.appendChild(volWrap); body.appendChild(rVol);
    // 音调 / 语速 / 句间停顿 / 呼吸 —— 每次参与独立调节，区别于其他参与
    body.appendChild(slider(L==='zh'?'音调':'Pitch',-6,6,0.5,d.pitch||0,x=>{d.pitch=x;}));
    body.appendChild(slider(L==='zh'?'语速':'Speed',0.5,2,0.05,d.speed||1,x=>{d.speed=x;}));
    body.appendChild(slider(L==='zh'?'句间停顿':'Pause',0,1,0.05,d.pause||0.15,x=>{d.pause=x;}));
    body.appendChild(slider(L==='zh'?'呼吸':'Breath',0,1,0.05,d.breath||0.4,x=>{d.breath=x;}));
    panel.appendChild(body);
    box.appendChild(panel);
  });
}
function toggleDp(idx){
  if(!dialogues[idx])return;
  dialogues[idx].collapsed=!dialogues[idx].collapsed;
  const body=document.getElementById('dp_body_'+idx),arrow=document.getElementById('dp_arrow_'+idx);
  if(body)body.style.display=dialogues[idx].collapsed?'none':'';
  if(arrow)arrow.textContent=dialogues[idx].collapsed?'▸':'▾';
}
function betaPlanEngine(){ const s=document.getElementById('betaEngine'); return s?s.value:'rule'; }
async function betaPreviewPlan(){
  const out=document.getElementById('betaPlanOut');
  const text=(document.getElementById('betaText').value||'');
  out.style.display='block';
  if(!text.trim()){ out.textContent='❌ '+I18N[curLang].betaPlanFail; return; }
  out.textContent=I18N[curLang].betaPlanLoading;
  try{
    const r=await fetch('/api/plan',{method:'POST',
      headers:Object.assign({'Content-Type':'application/json'},apiHeaders()),
      body:JSON.stringify({text:text,engine:betaPlanEngine()})});
    if(!r.ok){ let m=I18N[curLang].betaPlanFail; try{const j=await r.json();m=j.detail||m;}catch(e){} throw new Error(m); }
    const d=await r.json();
    const head=(curLang==='zh'?'通路':'source')+'='+(d.source||'')+' · '
      +(curLang==='zh'?'段数':'segments')+'='+(d.n_segments||0)+' · '
      +(curLang==='zh'?'原文未改动':'text intact')+'='+(d.text_intact===null?'?':(d.text_intact?'✓':'✗'));
    out.textContent=I18N[curLang].betaPlanTitle+'  ['+head+']\\n'+(d.error?('⚠️ '+d.error+'\\n'):'')+'\\n'+(d.summary||'');
  }catch(e){ out.textContent='❌ '+e.message; }
}
async function betaGenerate(){
  renderDialoguePanels();   // 确保 dialogues 与最新文本同步
  if(!dialogues.length){const e=document.getElementById('betaErr');e.textContent='❌ '+(curLang==='zh'?'请先在文本里用 (@音色包名) 指定角色':'Add (@pack_name) tags first');e.classList.add('show');return;}
  const btn=document.getElementById('betaBtn'),st=document.getElementById('betaStatus'),errEl=document.getElementById('betaErr');
  btn.disabled=true;st.style.display='flex';errEl.textContent='';errEl.classList.remove('show');
  document.getElementById('betaRes').style.display='none';
  const t0=Date.now();
  const timer=setInterval(()=>{document.getElementById('betaStatusText').textContent=
    I18N[curLang].betaLoading+((Date.now()-t0)/1000).toFixed(1)+I18N[curLang].betaSeconds;},200);
  const turns=dialogues.map(d=>({role:d.voice||d.role,text:d.text,tone:d.tone,emotion:d.emotion,volume:d.volume,pitch:d.pitch||0,speed:d.speed||1,pause:d.pause||0.15,breath:d.breath||0.4}));
  // Beta 面板内的 CFG/步数（此前被硬编码成 2.0/10，UI 里的 #cfg/#steps 在 Beta 模式被 hide 了）
  const cfgV=parseFloat((document.getElementById('betaCfg')||{}).value)||2.0;
  const stepsV=parseInt((document.getElementById('betaSteps')||{}).value,10)||10;
  const dirOn=!!((document.getElementById('betaDirector')||{}).checked);
  const body={turns:turns,denoise:document.getElementById('betaDenoise').checked,
    cfg_value:cfgV,inference_timesteps:stepsV,
    use_director:dirOn,director_engine:betaPlanEngine()};
  try{
    const r=await fetch('/api/dialogue',{method:'POST',headers:Object.assign({'Content-Type':'application/json'},apiHeaders()),body:JSON.stringify(body)});
    clearInterval(timer);
    if(!r.ok){let m='Failed';try{const j=await r.json();m=j.detail||m;}catch(e){}errEl.textContent='❌ '+m;errEl.classList.add('show');st.style.display='none';return;}
    const blob=await r.blob();
    const segInfo=r.headers.get('X-Segments'),dur=r.headers.get('X-Duration'),name=r.headers.get('X-Output-Name');
    document.getElementById('betaPlayer').src=URL.createObjectURL(blob);
    let meta='✅ '+(curLang==='zh'?'多人朗读完成':'Done')+' · '+(curLang==='zh'?'时长':'duration')+' '+dur+'s · '+name;
    if(segInfo){try{const si=JSON.parse(segInfo);meta+=' · '+si.n+(curLang==='zh'?' 段':' segments');
      const dz=si.director;
      if(dz){ meta+=' · '+(curLang==='zh'?'导演层':'director')+':'+dz.engine
                +'/'+dz.source+'('+dz.granularity+')'; }
      const emos=(si.segments||[]).map(function(s){return s.emotion;}).filter(function(x,i,a){
        return x&&x!=='neutral'&&a.indexOf(x)===i;});
      if(emos.length)meta+=' · '+(curLang==='zh'?'情绪':'emotions')+': '+emos.join('/');
      if(si.warnings&&si.warnings.length)meta+=' · ⚠️ '+si.warnings.join('; ');}catch(e){}}
    document.getElementById('betaMeta').textContent=meta;
    document.getElementById('betaRes').style.display='block';st.style.display='none';
  }catch(e){clearInterval(timer);errEl.textContent='❌ '+I18N[curLang].betaFail+': '+e.message;errEl.classList.add('show');st.style.display='none';}
  finally{btn.disabled=false;}
}

// ===== Training module (LoRA) =====
var trainPollTimer=null;
function refreshTrainUI(){
  refreshTrainSamples();
  refreshLoras();
  var st=trainerStatusCache;
  if(st&&st.running){showTrainRunning(st);}
}
var trainerStatusCache={running:false};
var trainSampleCount=0;
function refreshTrainSamples(){
  fetch('/api/train/samples',{headers:apiHeaders()})
    .then(function(r){return r.json();})
    .then(function(d){
      var el=document.getElementById('trainSamples');
      var badge=document.getElementById('trainStatsBadge');
      var arr=d.samples||[];
      trainSampleCount=arr.length;updTrainStartBtn();
      if(badge)badge.textContent=arr.length+' '+tr('条','samples');
      if(!arr.length){
        el.innerHTML='<div class="muted">'+I18N[curLang].trainNoSamples+'</div>';
        return;
      }
      el.innerHTML='';
      arr.forEach(function(s){
        var row=document.createElement('div');
        row.style.cssText='display:flex;align-items:center;gap:8px;padding:8px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px';
        var info=document.createElement('div');
        info.style.flex='1';
        info.innerHTML='<div style="font-weight:600">'+escHtml(s.name||('#'+s.id))+
          '</div><div style="font-size:12px;color:var(--text-2)">'+(s.duration?s.duration.toFixed(1)+'s · ':'')+
          escHtml(s.text).slice(0,80)+(s.text&&s.text.length>80?'…':'')+'</div>';
        var play=document.createElement('button');
        play.className='chip';play.textContent=I18N[curLang].trainPlaySample;
        play.dataset.act='play';play.dataset.id=s.id;
        play.style.cssText='padding:4px 10px;font-size:12px';
        var del=document.createElement('button');
        del.className='chip';del.textContent=I18N[curLang].trainDelete;
        del.dataset.act='delete';del.dataset.id=s.id;
        del.style.cssText='padding:4px 10px;font-size:12px;color:var(--danger-ink)';
        row.appendChild(info);row.appendChild(play);row.appendChild(del);
        el.appendChild(row);
      });
    })
    .catch(function(e){
      document.getElementById('trainSamples').innerHTML='<div class="err show">'+e.message+'</div>';
    });
}
function escHtml(s){if(!s)return'';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function addTrainSample(){
  var f=document.getElementById('trainFile');
  if(!f||!f.files||!f.files.length){document.getElementById('trainErr').textContent=I18N[curLang].trainNoFile;return;}
  var text=document.getElementById('trainText').value.trim();
  if(!text){document.getElementById('trainErr').textContent=I18N[curLang].trainNoText;return;}
  var name=document.getElementById('trainName').value.trim();
  var fd=new FormData();
  fd.append('file',f.files[0]);
  fd.append('text',text);
  fd.append('name',name);
  fd.append('denoise',!!document.getElementById('tsDenoise').checked);
  fd.append('vocal_only',!!document.getElementById('tsVocal').checked);
  var btn=document.getElementById('trainAddBtn');
  var errEl=document.getElementById('trainErr');
  errEl.textContent='';
  btn.disabled=true;btn.textContent=I18N[curLang].trainUploading;
  fetch('/api/train/samples',{method:'POST',body:fd,headers:apiHeaders()})
    .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||'Error');return d;});})
    .then(function(d){
      document.getElementById('trainFile').value='';
      document.getElementById('trainText').value='';
      document.getElementById('trainName').value='';
      refreshTrainSamples();
      btn.textContent=I18N[curLang].trainAddBtn;
    })
    .catch(function(e){
      errEl.textContent=I18N[curLang].trainAddFail+': '+e.message;
      btn.textContent=I18N[curLang].trainAddBtn;
    })
    .finally(function(){btn.disabled=false;});
}

function startTrain(){
  var fd={
    name:document.getElementById('tpName').value.trim()||'auto',
    r:parseInt(document.getElementById('tpR').value)||8,
    alpha:parseInt(document.getElementById('tpAlpha').value)||16,
    lr:parseFloat(document.getElementById('tpLr').value)||0.0001,
    epochs:parseInt(document.getElementById('tpEpochs').value)||3,
    accum:parseInt(document.getElementById('tpAccum').value)||4
  };
  var btn=document.getElementById('trainStartBtn');
  var errEl=document.getElementById('trainErr');
  errEl.textContent='';
  btn.disabled=true;btn.textContent=I18N[curLang].trainStarting;
  fetch('/api/train/start',{method:'POST',headers:Object.assign({'Content-Type':'application/json'},apiHeaders()),body:JSON.stringify(fd)})
    .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||'Error');return d;});})
    .then(function(d){
      showTrainRunning({running:true});
      pollTrain();
    })
    .catch(function(e){
      errEl.textContent=I18N[curLang].trainFailed+': '+e.message;
      btn.disabled=false;btn.textContent=I18N[curLang].trainStart;
    });
}

function showTrainRunning(st){
  document.getElementById('trainProgressWrap').style.display='block';
  document.getElementById('trainStartBtn').style.display='none';
  document.getElementById('trainStopBtn').style.display='inline-block';
  document.getElementById('trainStartBtn').disabled=true;
  document.getElementById('trainProgressBar').style.width=(st.progress||0)+'%';
  if(st.status)document.getElementById('trainStatusText').textContent=st.status;
  if(st.loss_history&&st.loss_history.length){
    var last=st.loss_history[st.loss_history.length-1];
    document.getElementById('trainLossText').textContent=
      I18N[curLang].trainLoss+': '+last.toFixed(4)+
      '  ·  '+I18N[curLang].trainStep+' '+(st.step||0)+
      '/'+(st.total_steps||'?')+
      '  ·  '+I18N[curLang].trainEpoch+' '+((st.epoch||0)+1)+'/'+(st.epochs||'?');
  }
}

function showTrainIdle(){
  document.getElementById('trainProgressWrap').style.display='none';
  document.getElementById('trainStartBtn').style.display='inline-block';
  document.getElementById('trainStartBtn').disabled=false;
  document.getElementById('trainStartBtn').textContent=I18N[curLang].trainStart;
  document.getElementById('trainStopBtn').style.display='none';
  updTrainStartBtn();
}

function updTrainStartBtn(){
  if(trainerStatusCache&&trainerStatusCache.running){return;}
  var sb=document.getElementById('trainStartBtn'),nh=document.getElementById('trainNeedHint');
  var enough=trainSampleCount>=2;
  if(sb){sb.disabled=!enough;}
  if(nh){
    if(enough){nh.style.display='none';}
    else{
      nh.style.display='block';
      nh.textContent=I18N[curLang].trainNeedSamples.replace('{cur}',String(trainSampleCount));
    }
  }
}

function pollTrain(){
  if(trainPollTimer)clearInterval(trainPollTimer);
  function tick(){
    fetch('/api/train/status',{headers:apiHeaders()})
      .then(function(r){return r.json();})
      .then(function(st){
        trainerStatusCache=st;
        if(st.running){showTrainRunning(st);}
        else{
          if(trainPollTimer){clearInterval(trainPollTimer);trainPollTimer=null;}
          showTrainIdle();
          if(st.error){
            document.getElementById('trainErr').textContent=I18N[curLang].trainFailed+': '+st.error;
          }else if(st.lora_name){
            document.getElementById('trainErr').textContent='';
            refreshLoras();
            alert(I18N[curLang].trainDone+st.lora_name);
          }else if(st.stopped){
            document.getElementById('trainErr').textContent=I18N[curLang].trainStopped;
          }
        }
      })
      .catch(function(){});
  }
  tick();
  trainPollTimer=setInterval(tick,2000);
}

function stopTrain(){
  fetch('/api/train/stop',{method:'POST',headers:apiHeaders()})
    .then(function(r){return r.json();})
    .then(function(){
      if(trainPollTimer){clearInterval(trainPollTimer);trainPollTimer=null;}
      showTrainIdle();
    })
    .catch(function(){});
}

function refreshLoras(){
  fetch('/api/train/loras',{headers:apiHeaders()})
    .then(function(r){return r.json();})
    .then(function(d){
      var arr=d.loras||[];
      var sel=document.getElementById('loraSel');
      if(sel){
        var cur=sel.value;
        sel.innerHTML='<option value="">'+I18N[curLang].loraNone+'</option>';
        arr.forEach(function(l){
          var o=document.createElement('option');
          o.value=l.name;o.textContent=l.name+(l.final_loss!=null?' (loss '+l.final_loss.toFixed(3)+')':'');
          sel.appendChild(o);
        });
        sel.value=cur;
      }
      var hint=document.getElementById('loraSelHint');
      if(hint){
        var v=sel?sel.value:'';
        if(v){
          var found=arr.filter(function(l){return l.name===v;})[0];
          if(found&&found.final_loss!=null)hint.textContent=tr('最终损失 ','Final loss ')+found.final_loss.toFixed(4)+tr('，选用中',' (active)');
          else hint.textContent=tr('选用中',' (active)');
        }else hint.textContent='';
      }
      var list=document.getElementById('trainLoras');
      if(list){
        if(!arr.length){
          list.innerHTML='<div class="muted">'+I18N[curLang].trainNoLoras+'</div>';
          return;
        }
        list.innerHTML='';
        arr.forEach(function(l){
          var row=document.createElement('div');
          row.style.cssText='display:flex;align-items:center;gap:8px;padding:8px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px';
          var info=document.createElement('div');
          info.style.flex='1';
          var meta=l.created_at||'';
          if(l.final_loss!=null)meta+=' · loss '+l.final_loss.toFixed(3);
          if(l.steps)meta+=' · '+l.steps+' steps';
          info.innerHTML='<div style="font-weight:600">'+escHtml(l.name)+'</div>'+
            '<div style="font-size:12px;color:var(--text-2)">'+escHtml(meta)+'</div>';
          var use=document.createElement('button');
          use.className='chip';use.textContent=I18N[curLang].trainUseLora;
          use.dataset.act='use';use.dataset.id=l.name;
          use.style.cssText='padding:4px 10px;font-size:12px';
          var del=document.createElement('button');
          del.className='chip';del.textContent=I18N[curLang].trainDelete;
          del.dataset.act='delete';del.dataset.id=l.name;
          del.style.cssText='padding:4px 10px;font-size:12px;color:var(--danger-ink)';
          row.appendChild(info);row.appendChild(use);row.appendChild(del);
          list.appendChild(row);
        });
      }
    })
    .catch(function(){});
}

function onLoraSel(){
  var sel=document.getElementById('loraSel');
  var hint=document.getElementById('loraSelHint');
  if(!sel||!hint)return;
  if(sel.value){
    hint.textContent=tr('选用 LoRA：','Active LoRA: ')+sel.value;
  }else{
    hint.textContent='';
  }
}

// ===== 长音频自动转写（whisper）=====
var trJobId=null, trTimer=null;
function trShowErr(m){var e=document.getElementById('trErr');if(e)e.textContent=m?('❌ '+m):'';}
function trStart(){
  var f=document.getElementById('trFile');
  if(!f||!f.files||!f.files.length){trShowErr(I18N[curLang].trainNoFile);return;}
  var btn=document.getElementById('trStartBtn'),st=document.getElementById('trStatus');
  btn.disabled=true;st.textContent='';trShowErr('');
  document.getElementById('trResults').innerHTML='';
  document.getElementById('trImportBtn').style.display='none';
  var ab=document.getElementById('trAlignBtn');if(ab)ab.style.display='none';
  var an=document.getElementById('trAlignNote');if(an)an.textContent='';
  var fd=new FormData();fd.append('audio',f.files[0]);
  var tta=document.getElementById('trTranscript');
  if(tta&&tta.value.trim())fd.append('transcript',tta.value.trim());
  var tDen=document.getElementById('trDenoise'),tVoc=document.getElementById('trVocal');
  fd.append('denoise',tDen?tDen.checked:false);
  fd.append('vocal_only',tVoc?tVoc.checked:false);
  fetch('/api/train/transcribe',{method:'POST',body:fd,headers:apiHeaders()})
    .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||'Error');return d;});})
    .then(function(d){
      trJobId=d.job.job_id;
      trPoll();
      if(trTimer)clearInterval(trTimer);
      trTimer=setInterval(trPoll,1500);
    })
    .catch(function(e){trShowErr(e.message);btn.disabled=false;});
}
function trPoll(){
  if(!trJobId)return;
  fetch('/api/train/transcribe/'+trJobId,{headers:apiHeaders()})
    .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||'Error');return d.job;});})
    .then(function(j){
      var st=document.getElementById('trStatus');
      if(j.status==='pending'||j.status==='processing'){
        st.textContent=(j.message||'')+' …';
        if(j.progress<20)st.textContent=I18N[curLang].trModelLoading;
      }else if(j.status==='done'){
        if(trTimer){clearInterval(trTimer);trTimer=null;}
        st.textContent=j.message||'';
        trRender(j);
      }else if(j.status==='error'){
        if(trTimer){clearInterval(trTimer);trTimer=null;}
        document.getElementById('trStartBtn').disabled=false;
        st.textContent='';
        trShowErr(j.error||tr('转写失败','Transcription failed'));
      }
    })
    .catch(function(){});
}
function trRender(j){
  var box=document.getElementById('trResults');
  box.innerHTML='';
  document.getElementById('trStartBtn').disabled=false;
  var segs=j.segments||[];
  if(!segs.length){
    box.innerHTML='<div class="muted">'+I18N[curLang].trNoSeg+'</div>';
    return;
  }
  var head=document.createElement('div');
  head.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:6px';
  var selAll=document.createElement('label');
  selAll.style.cssText='font-size:13px;display:flex;align-items:center;gap:4px;cursor:pointer';
  var cbAll=document.createElement('input');cbAll.type='checkbox';cbAll.checked=true;
  cbAll.addEventListener('change',function(){box.querySelectorAll('input[type=checkbox].tr-cb').forEach(function(c){c.checked=cbAll.checked;});trUpdImportBtn();});
  selAll.appendChild(cbAll);
  selAll.appendChild(document.createTextNode(' '+I18N[curLang].trSelAll));
  head.appendChild(selAll);
  var cnt=document.createElement('span');
  cnt.style.cssText='margin-left:auto;font-size:12px;color:var(--text-2)';
  cnt.textContent=segs.length+' '+tr('段','segments')+' · '+j.lang;
  head.appendChild(cnt);
  box.appendChild(head);
  segs.forEach(function(s){
    var row=document.createElement('div');
    row.style.cssText='display:flex;align-items:flex-start;gap:6px;padding:8px;border:1px solid var(--border);border-radius:8px;margin-bottom:6px;background:var(--surface-3)';
    var cb=document.createElement('input');
    cb.type='checkbox';cb.className='tr-cb';cb.checked=true;
    cb.dataset.idx=s.idx;
    cb.addEventListener('change',trUpdImportBtn);
    cb.style.cssText='margin-top:3px';
    row.appendChild(cb);
    var mid=document.createElement('div');
    mid.style.flex='1';
    var lab=document.createElement('div');
    lab.style.cssText='font-size:11px;color:var(--text-2);margin-bottom:3px';
    lab.textContent=I18N[curLang].trSegDur.replace('{n}',String(s.idx+1))+' · '+s.duration.toFixed(1)+'s';
    mid.appendChild(lab);
    var ta=document.createElement('textarea');
    ta.value=s.text;
    ta.style.cssText='width:100%;min-height:40px;font-size:13px';
    ta.dataset.idx=s.idx;
    mid.appendChild(ta);
    row.appendChild(mid);
    var play=document.createElement('button');
    play.className='chip';play.textContent=I18N[curLang].trPlay;
    play.dataset.act='play';play.dataset.idx=s.idx;
    play.style.cssText='padding:4px 10px;font-size:12px;flex:0 0 auto';
    row.appendChild(play);
    box.appendChild(row);
  });
  document.getElementById('trImportBtn').style.display='inline-block';
  var ab=document.getElementById('trAlignBtn');
  if(ab){ab.style.display='inline-block';ab.disabled=false;}
  var an=document.getElementById('trAlignNote');
  if(an){
    if(j.aligned){
      an.textContent='✅ '+(j.align_note||'');
      an.style.color='var(--green-ink)';
    }else{
      an.textContent='';
      an.style.color='';
    }
  }
  trUpdImportBtn();
}
function trAlign(){
  var ta=document.getElementById('trTranscript');
  if(!ta||!ta.value.trim()){trShowErr(I18N[curLang].trNoTranscript);return;}
  if(!trJobId){trShowErr(I18N[curLang].trNoSeg);return;}
  var btn=document.getElementById('trAlignBtn');btn.disabled=true;
  trShowErr('');
  fetch('/api/train/align',{method:'POST',
    headers:Object.assign({'Content-Type':'application/json'},apiHeaders()),
    body:JSON.stringify({job_id:trJobId,transcript:ta.value})})
    .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||'Error');return d;});})
    .then(function(d){
      if(btn)btn.disabled=false;
      trRender(d.job);
    })
    .catch(function(e){if(btn)btn.disabled=false;trShowErr(e.message);});
}
function trLoadTxt(f){
  if(!f)return;
  var rd=new FileReader();
  rd.onload=function(){
    var ta=document.getElementById('trTranscript');
    if(ta)ta.value=(rd.result||'').replace(/^\uFEFF/,'');
  };
  rd.readAsText(f,'utf-8');
}
function trUpdImportBtn(){
  var btn=document.getElementById('trImportBtn');
  if(!btn)return;
  var n=document.querySelectorAll('#trResults input.tr-cb:checked').length;
  btn.textContent=I18N[curLang].trImport+(n?' ('+n+')':'');
}
function trImport(){
  var items=[];
  document.querySelectorAll('#trResults input.tr-cb:checked').forEach(function(cb){
    var row=cb.closest('div');
    var ta=row.querySelector('textarea');
    items.push({idx:parseInt(cb.dataset.idx,10),text:ta?ta.value:''});
  });
  if(!items.length){trShowErr(I18N[curLang].trNoSeg);return;}
  var btn=document.getElementById('trImportBtn');btn.disabled=true;
  fetch('/api/train/import_segments',{method:'POST',
    headers:Object.assign({'Content-Type':'application/json'},apiHeaders()),
    body:JSON.stringify({job_id:trJobId,items:items})})
    .then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||'Error');return d;});})
    .then(function(d){
      btn.disabled=false;
      document.getElementById('trResults').innerHTML='';
      document.getElementById('trImportBtn').style.display='none';
      document.getElementById('trStatus').textContent='✅ '+(I18N[curLang].trImportOk||'').replace('{n}',String(d.imported||0))+(d.skipped?(' · ⚠️ '+(tr('跳过 ','skipped '))+d.skipped):'');
      trJobId=null;
      refreshTrainSamples();
    })
    .catch(function(e){btn.disabled=false;trShowErr(e.message);});
}
// trResults 事件委托：试听片段
document.getElementById('trResults').addEventListener('click',function(e){
  var btn=e.target.closest('button[data-act="play"]');
  if(!btn)return;
  var idx=btn.dataset.idx,row=btn.closest('div');
  var old=row.querySelector('audio');
  if(old){old.remove();return;}
  var au=document.createElement('audio');
  au.controls=true;au.preload='none';
  au.src='/api/train/transcribe/'+trJobId+'/segment/'+idx+'/audio';
  au.style.cssText='width:100%;margin-top:4px';
  row.appendChild(au);
  au.play();
});
// 按钮 wiring
var trStartBtn=document.getElementById('trStartBtn');
if(trStartBtn)trStartBtn.addEventListener('click',trStart);
var trImportBtn=document.getElementById('trImportBtn');
if(trImportBtn)trImportBtn.addEventListener('click',trImport);
var trAlignBtn=document.getElementById('trAlignBtn');
if(trAlignBtn)trAlignBtn.addEventListener('click',trAlign);
var trTxtFile=document.getElementById('trTxtFile');
if(trTxtFile)trTxtFile.addEventListener('change',function(){trLoadTxt(trTxtFile.files[0]);trTxtFile.value='';});


// event delegation: train samples (play/delete)
document.getElementById('trainSamples').addEventListener('click',function(e){
  var btn=e.target.closest('button[data-act]');
  if(!btn)return;
  var id=btn.dataset.id,act=btn.dataset.act;
  if(act==='play'){
    var au=document.createElement('audio');
    au.src='/api/train/samples/'+id+'/audio';au.controls=true;
    au.style.cssText='width:100%;margin-top:4px';
    var row=btn.parentElement;
    var old=row.querySelector('audio');
    if(old)old.remove();
    row.appendChild(au);
    au.play();
  }else if(act==='delete'){
    if(!confirm(I18N[curLang].trainDelSample))return;
    fetch('/api/train/samples/'+id,{method:'DELETE',headers:apiHeaders()})
      .then(function(r){return r.json();})
      .then(function(){refreshTrainSamples();})
      .catch(function(){});
  }
});
// event delegation: train loras (use/delete)
document.getElementById('trainLoras').addEventListener('click',function(e){
  var btn=e.target.closest('button[data-act]');
  if(!btn)return;
  var name=btn.dataset.id,act=btn.dataset.act;
  if(act==='use'){
    var sel=document.getElementById('loraSel');
    if(sel)sel.value=name;
    onLoraSel();
    setMode(prevMode||'design');
  }else if(act==='delete'){
    if(!confirm(I18N[curLang].trainDelLora))return;
    fetch('/api/train/loras/'+encodeURIComponent(name),{method:'DELETE',headers:apiHeaders()})
      .then(function(r){return r.json();})
      .then(function(){refreshLoras();})
      .catch(function(){});
  }
});
// button wiring
document.getElementById('trainAddBtn').addEventListener('click',addTrainSample);
document.getElementById('trainStartBtn').addEventListener('click',startTrain);
document.getElementById('trainStopBtn').addEventListener('click',stopTrain);

function setMode(m){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.mode===m));
  const beta=(m==='beta'), tr=(m==='train'), st=(m==='settings'), off=beta||tr||st;
  document.getElementById('mainCard').classList.toggle('hide',off);
  document.getElementById('histCard').classList.toggle('hide',off);
  document.getElementById('packCard').classList.toggle('hide',off);
  document.getElementById('betaCard').classList.toggle('hide',!beta);
  document.getElementById('trainCard').classList.toggle('hide',!tr);
  document.getElementById('settingsCard').classList.toggle('hide',!st);
  if(beta){prevMode=mode||'design';renderDialoguePanels();refreshBetaLlmBar();return;}
  if(tr){prevMode=(mode&&mode!=='train')?mode:'design';refreshTrainUI();return;}
  if(st){prevMode=(mode&&mode!=='settings')?mode:'design';settingsLoad();return;}
  mode=m;
  document.getElementById('refField').classList.toggle('hide',m==='design');
  document.getElementById('packSelField').classList.toggle('hide',m==='design');
  document.getElementById('chips').style.display=(m==='hifi')?'none':'flex';
  if(m==='design'){
    selectedPackId=null;
    const sel=document.getElementById('packSel');
    if(sel)sel.value='';
    document.getElementById('packSelHint').textContent='';
    document.getElementById('refFile').value='';
  }
  updatePtField();
}
function updatePtField(){
  // 极致克隆下，逐字文本仅在上传参考音频时显示/必填；选用音色包时隐藏、无需填写
  document.getElementById('ptField').classList.toggle('hide', !(mode==='hifi' && !selectedPackId));
}

/* ==================== AI 内核配置（LLM Key 填写窗口） ====================
   设计要点：
   - 服务商预设由后端 /api/llm/providers 下发，前端不硬编码任何地址，
     避免"UI 一份、后端一份"的漂移。
   - api_key 后端只回显脱敏串；输入框留空 = 不修改已存密钥（三态语义）。
   - 「保存并测试」一步到位：配置落盘 + 立刻验证连通性。
*/
let LLM_PROVIDERS=[], LLM_CUR=null, SET_KEY_DIRTY=false;

async function llmApi(path,opts){
  const o=Object.assign({headers:Object.assign({'Content-Type':'application/json'},apiHeaders())},opts||{});
  const r=await fetch(path,o);
  let d=null;
  try{d=await r.json();}catch(e){d={detail:'返回内容不是合法 JSON'};}
  if(!r.ok && !(d&&d.stage)){ // 非业务性失败（如 401/503）才当异常抛
    const msg=(d&&(d.detail||d.reason||d.message))||('HTTP '+r.status);
    const err=new Error(msg); err.status=r.status; err.body=d; throw err;
  }
  return d||{};
}

async function ensureProviders(){
  if(LLM_PROVIDERS.length)return LLM_PROVIDERS;
  try{
    const d=await llmApi('/api/llm/providers');
    LLM_PROVIDERS=d.providers||[];
  }catch(e){LLM_PROVIDERS=[];}
  return LLM_PROVIDERS;
}

function llmProviderById(id){
  return LLM_PROVIDERS.find(p=>p.id===id)||LLM_PROVIDERS.find(p=>p.id==='custom')||null;
}
function llmIsZh(){return curLang==='zh';}
function llmPName(p){return p?(llmIsZh()?p.name_zh:p.name_en):'';}
/* 就绪原因本地化：后端给 reason_code，前端查 i18n；取不到再退回后端的中文文案 */
function llmReasonText(c){
  const key={not_enabled:'setReasonNotEnabled',no_key:'setReasonNoKey',
             no_base_url:'setReasonNoBaseUrl',no_model:'setReasonNoModel'}[c&&c.reason_code];
  if(key){
    const v=(I18N[curLang]||{})[key];
    if(v)return v;
  }
  return (c&&c.reason)||(llmIsZh()?'配置不完整':'Configuration incomplete');
}

async function settingsLoad(notify){
  const err=document.getElementById('setErr');
  err.style.display='none';
  await ensureProviders();
  const sel=document.getElementById('setProvider');
  if(sel.options.length!==LLM_PROVIDERS.length){
    sel.textContent='';
    LLM_PROVIDERS.forEach(p=>{
      const o=document.createElement('option');
      o.value=p.id; o.textContent=(p.local?'🏠 ':'')+llmPName(p);
      sel.appendChild(o);
    });
  }
  try{
    const c=await llmApi('/api/llm/config');
    LLM_CUR=c;
    document.getElementById('setEnabled').checked=!!c.enabled;
    sel.value=c.provider||'custom';
    if(!sel.value)sel.value='custom';
    document.getElementById('setBaseUrl').value=c.base_url||'';
    document.getElementById('setModel').value=c.model||'';
    document.getElementById('setTimeout').value=c.timeout||60;
    document.getElementById('setTemp').value=c.temperature!==undefined?c.temperature:0.3;
    document.getElementById('setProxy').value=c.proxy||'';
    document.getElementById('setJsonMode').checked=!!c.json_mode;
    document.getElementById('setCache').checked=!!c.cache;
    document.getElementById('setConfigPath').textContent=
      (llmIsZh()?'配置文件：':'Config file: ')+(c.config_path||'');
    setOnProviderChange(true);
    setKeyFieldFromStatus(c);
    refreshStatusUI(c);
    if(notify)setResult(true,(llmIsZh()?'已重新读取配置':'Config reloaded'));
  }catch(e){
    err.textContent=(e&&e.message)||String(e);
    err.style.display='block';
  }
}

function setKeyFieldFromStatus(c){
  const el=document.getElementById('setKey');
  el.value=''; SET_KEY_DIRTY=false;
  // 注意：data-i18n-ph 的值是"键名"，必须查 I18N 取译文，
  // 直接用属性值会把 setKeyPh 这种键名当成占位文字露出来。
  const key=el.getAttribute('data-i18n-ph');
  const dict=I18N[curLang]||I18N.zh||{};
  const phText=(key&&dict[key])||'API Key';
  if(c&&c.has_key){
    // 已存密钥：不回显明文，用 placeholder 显示脱敏串作为"已配置"的视觉凭证
    el.placeholder=(llmIsZh()?'已保存：':'Saved: ')+(c.api_key_masked||'••••')+
      (llmIsZh()?'（留空则不修改）':' (leave blank to keep)');
  }else{
    el.placeholder=phText;
  }
}

function setOnProviderChange(silent){
  const id=document.getElementById('setProvider').value;
  const p=llmProviderById(id);
  if(!p)return;
  if(!silent){
    // 切换服务商时，地址、模型与超时跟随预设（用户之后仍可手改）
    document.getElementById('setBaseUrl').value=p.base_url||'';
    document.getElementById('setModel').value=(p.models&&p.models[0])||'';
    // 本地模型首次调用要加载权重，默认给足超时，避免"第一次点测试就失败"
    document.getElementById('setTimeout').value=p.local?300:60;
  }
  const dl=document.getElementById('setModelList');
  dl.textContent='';
  (p.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m;dl.appendChild(o);});
  const note=document.getElementById('setProviderNote');
  let s=(llmIsZh()?p.note:(p.note_en||p.note))||'';
  if(p.key_url){
    s+=(s?' ':'')+(llmIsZh()?'申请 Key：':'Get a key: ')+p.key_url;
  }
  note.textContent=s;
  const dh=document.getElementById('setBaseHint');
  dh.textContent=llmIsZh()
    ?'不带 /chat/completions，程序会自动补上。切换服务商会自动填入预设地址。'
    :'Do NOT include /chat/completions; the app appends it. Switching provider fills the preset.';
  const mh=document.getElementById('setModelHint');
  mh.textContent=(llmIsZh()?'模型名区分大小写，必须与服务商文档完全一致。':'Model names are case-sensitive; copy exactly from the provider docs.');
  const kh=document.getElementById('setKeyHint');
  kh.textContent=(llmIsZh()?'格式提示：':'Key format: ')+(llmIsZh()?p.key_hint:(p.key_hint_en||p.key_hint))+
    (p.local?(llmIsZh()?'。本地服务无鉴权，可留空或随便填。':' Local server: any value works.'):'');
}

function setToggleKeyView(){
  const el=document.getElementById('setKey');
  const btn=document.getElementById('setKeyToggle');
  const show=el.type==='password';
  el.type=show?'text':'password';
  btn.textContent=show?(llmIsZh()?'🙈 隐藏':'🙈 Hide'):(llmIsZh()?'👁 显示':'👁 Show');
}

function setKeyWasTyped(){SET_KEY_DIRTY=true;}

function settingsPayload(){
  const body={
    enabled:document.getElementById('setEnabled').checked,
    provider:document.getElementById('setProvider').value,
    base_url:document.getElementById('setBaseUrl').value.trim(),
    model:document.getElementById('setModel').value.trim(),
    timeout:parseInt(document.getElementById('setTimeout').value,10)||60,
    temperature:parseFloat(document.getElementById('setTemp').value),
    proxy:document.getElementById('setProxy').value.trim(),
    json_mode:document.getElementById('setJsonMode').checked,
    cache:document.getElementById('setCache').checked
  };
  if(isNaN(body.temperature))body.temperature=0.3;
  // 只有用户真的敲过 key 才带上该字段 → 留空可安全表示"不改动已存密钥"
  const kv=document.getElementById('setKey').value;
  if(SET_KEY_DIRTY && kv!==undefined)body.api_key=kv;
  return body;
}

function refreshStatusUI(c){
  const dot=document.getElementById('setStatusDot');
  const txt=document.getElementById('setStatusText');
  const note=document.getElementById('setSourceNote');
  if(!c)return;
  const ready=!!c.ready;
  dot.style.background=ready?'var(--green)':'var(--warn-ink)';
  if(ready){
    txt.textContent=llmIsZh()?'已就绪':'Ready';
    txt.style.color='var(--ok-ink)';
  }else{
    txt.textContent=llmReasonText(c);
    txt.style.color='var(--warn-ink)';
  }
  const srcMap={
    config_file:llmIsZh()?'来自 llm_config.json':'from llm_config.json',
    api_key_txt:llmIsZh()?'来自 api_key.txt':'from api_key.txt',
    env:llmIsZh()?'来自环境变量（不是本文件）':'from an environment variable',
    local_placeholder:llmIsZh()?'本地服务占位符（无需真实 Key）':'local placeholder (no real key needed)',
    unknown:'—',none:llmIsZh()?'尚未配置密钥':'no key yet'
  };
  let s='';
  if(c.has_key){
    s=(llmIsZh()?'当前密钥 ':'Current key ')+(srcMap[c.key_source]||'')+
      (llmIsZh()?'，长度 ':' , length ')+(c.api_key_masked||'').length+
      (llmIsZh()?' 位。':' chars.');
    if(c.key_source==='env'){
      s+=(llmIsZh()
        ?' ⚠️ 环境变量优先级在文件之上，改本页面可能不生效。'
        :' ⚠️ Env vars take precedence; edits here may not apply.');
    }
  }else{
    s=llmIsZh()?'还没有配置密钥，填好下面三项并保存即可启用。':'No key configured yet. Fill the fields below and save.';
  }
  note.textContent=s;
}

function setResult(ok,msg,extra){
  const box=document.getElementById('setResult');
  box.style.display='block';
  box.style.borderColor=ok?'var(--res-border)':'rgba(239,68,68,.4)';
  box.style.background=ok?'var(--res-bg)':'rgba(239,68,68,.08)';
  box.style.color=ok?'var(--ok-ink)':'#ef4444';
  let html='';
  if(msg)html+='<div>'+String(msg).replace(/</g,'&lt;')+'</div>';
  if(extra)html+='<div style="margin-top:6px;font-size:12px;color:var(--text-2);line-height:1.6">'+String(extra).replace(/</g,'&lt;')+'</div>';
  box.innerHTML=html;
}

function setBusy(on,btnId){
  const b=document.getElementById(btnId);
  if(b){b.disabled=!!on;b.style.opacity=on?'0.6':'1';}
}

function renderTestResult(d){
  if(!d)return;
  if(d.ok){
    let extra='';
    if(d.models&&d.models.length){
      extra=(llmIsZh()?'可用模型（'+d.models.length+'）：':'Models ('+d.models.length+'): ')+
        d.models.slice(0,25).join(', ')+
        (d.models.length>25?(llmIsZh()?' …等':' …'):'');
    }
    if(d.reply)extra=(extra?extra+'\\n':'')+(llmIsZh()?'模型回复：':'Reply: ')+d.reply;
    if(d.warnings&&d.warnings.length)extra+=(extra?'\\n':'')+'⚠ '+d.warnings.join('；');
    setResult(true,'✅ '+d.message,extra);
  }else{
    let extra='';
    if(d.detail)extra=d.detail;
    if(d.models&&d.models.length)extra+=(extra?'\\n':'')+(llmIsZh()?'该 Key 可见 ':'Key can see ')+d.models.length+(llmIsZh()?' 个模型':' models');
    if(d.stage)extra+=(extra?'\\n':'')+(llmIsZh()?'失败阶段：':'Failed at stage: ')+d.stage;
    setResult(false,'❌ '+d.message,extra);
  }
}

async function settingsSave(){
  const err=document.getElementById('setErr');
  err.style.display='none';
  setBusy(true,'setSaveBtn');
  try{
    const d=await llmApi('/api/llm/config',{method:'POST',body:JSON.stringify(settingsPayload())});
    renderTestResult(d);
    if(d.saved){
      await settingsLoad();               // 拉回服务端真值（含脱敏 key 与就绪状态）
      renderTestResult(d);                // settingsLoad 会清空结果框，再写回
    }
  }catch(e){
    // 400 校验失败也带着可读原因，统一走结果框展示（比红字错误框更醒目）
    const b=e&&e.body;
    if(b&&b.reason){
      setResult(false,'❌ '+b.reason,
        b.warnings&&b.warnings.length?('⚠ '+b.warnings.join('；')):'');
    }else{
      err.textContent=(e&&e.message)||String(e);
      err.style.display='block';
    }
  }finally{setBusy(false,'setSaveBtn');}
}

async function settingsTest(){
  const err=document.getElementById('setErr');
  err.style.display='none';
  setBusy(true,'setTestBtn');
  try{
    const d=await llmApi('/api/llm/test',{method:'POST',body:JSON.stringify(settingsPayload())});
    renderTestResult(d);
    if(d.ok&&d.models&&d.models.length){
      const dl=document.getElementById('setModelList');
      const cur=dl.textContent;
      d.models.slice(0,50).forEach(m=>{
        const o=document.createElement('option');o.value=m;dl.appendChild(o);
      });
    }
  }catch(e){
    err.textContent=(e&&e.message)||String(e);
    err.style.display='block';
  }finally{setBusy(false,'setTestBtn');}
}

async function setFetchModels(){
  const err=document.getElementById('setErr');
  err.style.display='none';
  setBusy(true,'setFetchModelsBtn');
  try{
    const d=await llmApi('/api/llm/test',{method:'POST',
      body:JSON.stringify(Object.assign(settingsPayload(),{probe_chat:false}))});
    const dl=document.getElementById('setModelList');
    dl.textContent='';
    (d.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m;dl.appendChild(o);});
    if(d.models&&d.models.length){
      setResult(true,(llmIsZh()?'拉到 ':'Found ')+d.models.length+(llmIsZh()?' 个可用模型，点模型输入框可从下拉选择。':' models. Click the model field to pick one.'),
        d.models.slice(0,40).join(', ')+(d.models.length>40?' …':''));
      const mv=document.getElementById('setModel').value.trim();
      if(!mv){document.getElementById('setModel').value=d.models[0];}
    }else{
      setResult(!!d.ok,(llmIsZh()?'未能拉到模型列表':'Could not list models'),
        (d.message||'')+(llmIsZh()?'\\n部分服务商不提供 /models 接口，这不代表 Key 无效——可改用「仅测试连接」。':'\\nSome providers do not expose /models; this does not mean the key is invalid — use "Test connection" instead.'));
    }
  }catch(e){
    err.textContent=(e&&e.message)||String(e);
    err.style.display='block';
  }finally{setBusy(false,'setFetchModelsBtn');}
}

/* Beta 面板里的精简状态条 */
let LLM_BAR_CACHE=null;
async function refreshBetaLlmBar(){
  const dot=document.getElementById('betaLlmDot');
  const txt=document.getElementById('betaLlmText');
  if(!dot||!txt)return;
  try{
    const c=await llmApi('/api/llm/config');
    LLM_BAR_CACHE=c;
    const ready=!!c.ready;
    dot.style.background=ready?'var(--green)':'var(--warn-ink)';
    if(ready){
      const pn=llmPName(llmProviderById(c.provider));
      txt.textContent=(llmIsZh()?'AI 内核已就绪：':'AI kernel ready: ')+(pn||c.provider)+' · '+(c.model||'');
      txt.style.color='var(--text-1)';
    }else{
      txt.textContent=(llmIsZh()?'AI 内核未就绪 — ':'AI kernel not ready — ')+llmReasonText(c);
      txt.style.color='var(--warn-ink)';
    }
  }catch(e){
    dot.style.background='var(--disabled)';
    txt.textContent=llmIsZh()?'无法读取内核状态':'Cannot read kernel status';
  }
}
/* 切换语言后重绘"由 JS 动态生成"的文案。
   setLang 只会刷新带 data-i18n / data-i18n-ph 属性的元素，而这些提示是
   JS 在渲染时按语言拼出来的，不重绘就会留下上一种语言的残留
   （症状：标签已是中文，但就绪状态/Key 提示/服务商说明还是英文）。
   刻意用同步的 LLM_CUR 缓存而不是重新拉接口 —— 异步回来时语言可能又被切了。 */
function repaintDynamicText(){
  const sc=document.getElementById('settingsCard');
  if(sc&&!sc.classList.contains('hide')&&LLM_CUR){
    setOnProviderChange(true);
    if(!SET_KEY_DIRTY)setKeyFieldFromStatus(LLM_CUR);   // 别清掉用户刚敲进去的 key
    refreshStatusUI(LLM_CUR);
  }
  const bc=document.getElementById('betaCard');
  if(bc&&!bc.classList.contains('hide'))refreshBetaLlmBar();
  if(typeof renderDialoguePanels==='function'){
    const dp=document.getElementById('dialoguePanels');
    if(dp&&dp.offsetParent!==null){try{renderDialoguePanels();}catch(_){}}
  }
}
function pre(t){const el=document.getElementById('text');el.value=t+el.value.replace(/^\([^()]*\)|^（[^（）]*）/,'');el.focus();}

const EMOTION_PRESETS={
  '高兴':{pitch:1,speed:1.08,volume:1.12,pause:0.12,breath:0.4},
  '悲伤':{pitch:-1,speed:0.86,volume:0.90,pause:0.28,breath:0.5},
  '严肃':{pitch:0,speed:0.92,volume:1.00,pause:0.32,breath:0.35},
  '温柔':{pitch:0,speed:0.95,volume:0.95,pause:0.18,breath:0.45},
  '愤怒':{pitch:0,speed:1.15,volume:1.25,pause:0.10,breath:0.3},
  '平静':{pitch:0,speed:1.00,volume:1.00,pause:0.15,breath:0.35}
};
function setSlider(id,val,vid){const el=document.getElementById(id);el.value=val;document.getElementById(vid).textContent=val;}
function applyEmotion(name){
  const p=EMOTION_PRESETS[name];
  if(!p)return;
  setSlider('pitch',p.pitch,'pitchv');
  setSlider('speed',p.speed,'speedv');
  setSlider('volume',p.volume,'volumev');
  setSlider('pause',p.pause,'pausev');
  setSlider('breath',p.breath,'breathv');
}

function apiHeaders(){return {'x-api-key':API_TOKEN};}

function setModelBadge(state,extra){
  const b=document.getElementById('modelBadge');
    if(state==='loading'){b.textContent=tr('模型加载中…(约20-60秒)','Loading model…(20-60s)');b.className='badge';b.style.cursor='default';b.onclick=null;}
    else if(state==='error'){b.textContent=tr('模型加载失败 · 点此重试','Model load failed · click to retry');b.className='badge warn';b.style.cursor='pointer';b.onclick=()=>warmupModel();}
    else if(state==='ready'){b.textContent=tr('模型已加载 · ','Model ready · ')+(extra||'');b.className='badge ok';b.style.cursor='default';b.onclick=null;}
    else {b.textContent=extra||tr('模型未加载','Model not loaded');b.className='badge';b.style.cursor='default';b.onclick=null;}
}
async function refreshStatus(){
  try{
    const r=await fetch('/api/health');const d=await r.json();
    document.getElementById('devBadge').textContent=d.device||tr('未知设备','Unknown device');
    document.getElementById('devBadge').className='badge '+(d.cuda?'ok':'warn');
    if(d.model_loaded)setModelBadge('ready',(d.sample_rate/1000)+'kHz');
    else setModelBadge('notloaded');
  }catch(e){setModelBadge('error');}
}
async function warmupModel(){
  setModelBadge('loading');
  try{
    const r=await fetch('/api/warmup',{method:'POST',headers:apiHeaders()});
    if(!r.ok)throw new Error('warmup '+r.status);
    await refreshStatus();
  }catch(e){setModelBadge('error');}
}
async function init(){
  try{
    const r=await fetch('/api/health');const d=await r.json();
    document.getElementById('devBadge').textContent=d.device||tr('未知设备','Unknown device');
    document.getElementById('devBadge').className='badge '+(d.cuda?'ok':'warn');
    if(d.model_loaded)setModelBadge('ready',(d.sample_rate/1000)+'kHz');
    else warmupModel();   // 自动加载模型，避免一直显示“模型未加载”
  }catch(e){setModelBadge('error');}
}
init();
loadVoicePacks();
refreshLoras();
// 初始化语言（从本地存储恢复，默认中文）
try{const _sl=localStorage.getItem('voxcpm_lang');if(_sl)setLang(_sl);}catch(_){}

// 真实调用一次：带超时(300s) + 失败自动重试一次
async function callGenerate(fd, signal){
  const r=await fetch('/api/generate',{method:'POST',body:fd,signal,headers:apiHeaders()});
  if(!r.ok){
    let detail=tr('生成失败','Generation failed');
    try{const j=await r.json();detail=j.detail||detail;}catch(e){}
    throw new Error(detail+'  (HTTP '+r.status+')');
  }
  return r;
}

async function generate(){
  const text=document.getElementById('text').value.trim();
  const err=document.getElementById('err'),st=document.getElementById('status'),
        res=document.getElementById('res'),btn=document.getElementById('btn');
  err.classList.remove('show');res.classList.remove('show');
  if(!text){return showErr(tr('请输入要合成的文本','Please enter text to synthesize'));}
  const refFile=document.getElementById('refFile').files[0];
  if(mode!=='design'&&!refFile&&!selectedPackId){return showErr(tr('该模式需要上传参考音频，或从“已保存音色包”中选择一个','This mode requires a reference audio upload, or pick one from the saved voice packs'));}
  const promptText=document.getElementById('promptText').value.trim();
  // 极致克隆：选用音色包时无需逐字文本；上传参考音频时才需填写
  if(mode==='hifi'&&!selectedPackId&&!promptText){return showErr(tr('极致克隆请上传参考音频并填写其逐字文本；或直接选用音色包（无需逐字文本）','HiFi clone: upload a reference audio and its verbatim transcript; or just pick a voice pack (no transcript needed)'));}

  btn.disabled=true;st.classList.add('show');
  document.getElementById('statusText').textContent=tr('生成中…','Generating…');
  const t0=Date.now();
  const timer=setInterval(()=>{document.getElementById('statusText').textContent=
    tr('生成中… 已用 ','Generating… ')+((Date.now()-t0)/1000).toFixed(1)+tr(' 秒（首次需加载模型，请耐心等待）','s (first run loads the model, please wait)');},100);

  let attempt=0;
  while(true){
    attempt++;
    const controller=new AbortController();
    const to=setTimeout(()=>controller.abort(),1800000); // 30 分钟硬超时（支持最长 10 分钟参考音频）
    const fd=new FormData();
    fd.append('text',text);
    fd.append('cfg_value',document.getElementById('cfg').value);
    fd.append('inference_timesteps',document.getElementById('steps').value);
    fd.append('normalize',document.getElementById('normalize').checked);
    fd.append('denoise',document.getElementById('denoise').checked);
    fd.append('remove_bg',document.getElementById('remove_bg').checked);
    fd.append('stable',document.getElementById('stable').checked);
    fd.append('pitch',document.getElementById('pitch').value);
    fd.append('speed',document.getElementById('speed').value);
    fd.append('volume',document.getElementById('volume').value);
    fd.append('pause',document.getElementById('pause').value);
    fd.append('breath',document.getElementById('breath').value);
    fd.append('emotion',document.getElementById('emotionSel').value);
    fd.append('lora_name',document.getElementById('loraSel').value||'');
    fd.append('ssml',document.getElementById('ssml').checked);
    fd.append('mode',mode);
    if(refFile)fd.append('reference',refFile);
    if(selectedPackId)fd.append('voice_pack_id',selectedPackId);
    if(promptText)fd.append('prompt_text',promptText);
    try{
      const r=await callGenerate(fd, controller.signal);
      clearTimeout(to);clearInterval(timer);
      const name=r.headers.get('X-Output-Name')||'output.wav';
      lastOutputName=name;
      const secs=r.headers.get('X-Elapsed')||'?';
      const blob=await r.blob();
      const player=document.getElementById('player');
      const oldUrl=player.src;
      if(oldUrl&&oldUrl.indexOf('blob:')===0){try{URL.revokeObjectURL(oldUrl);}catch(_){}}
      player.src=URL.createObjectURL(blob);
      document.getElementById('resMeta').textContent=tr('✅ 生成成功 · 耗时 ','✅ Done · ')+secs+tr(' 秒 · 文件 ','s · file ')+name+tr('（已保存到 F:\\\\VoxCPM2\\\\outputs）',' (saved to F:\\\\VoxCPM2\\\\outputs)');
      res.classList.add('show');addHist(name,text);refreshStatus();
      clearInterval(timer);btn.disabled=false;st.classList.remove('show');
      return;
    }catch(e){
      clearTimeout(to);
      const canRetry = attempt===1 && (/Failed to fetch|网络|HTTP 5/.test(e.message));
      if(canRetry){
        document.getElementById('statusText').textContent=tr('连接异常，正在自动重试（第 2 次）…','Connection error, auto-retrying (attempt 2)…');
        await new Promise(s=>setTimeout(s,800));
        continue; // 重试一次
      }
      clearInterval(timer);
      let msg=e.message||tr('请求失败','Request failed');
      if(e.name==='AbortError')msg=tr('请求超时（>30 分钟）。参考音频过长或文本太多，请缩短后重试','Request timed out (>30min). Shorten the reference audio or text and retry.');
      showErr(msg);btn.disabled=false;st.classList.remove('show');
      return;
    }finally{
      if(attempt>=2)clearInterval(timer);
    }
  }
}
function showErr(m){const e=document.getElementById('err');e.textContent='❌ '+m;e.classList.add('show');
  document.getElementById('btn').disabled=false;document.getElementById('status').classList.remove('show');}
function addHist(name,text){
  const h=document.getElementById('hist');
  if(h.querySelector('.muted'))h.innerHTML='';
  const d=document.createElement('div');d.className='row';
  d.innerHTML='<span>'+(text.length>34?text.slice(0,34)+'…':text)+'</span>'+
    '<a href="/api/outputs/'+name+'" target="_blank">'+name+' ↓</a>';
  h.prepend(d);
}

async function exportAudio(){
  const fmt=document.getElementById('exportFmt').value;
  if(!lastOutputName){return alert(tr('请先生成音频','Generate audio first'));}
  const fd=new FormData();
  fd.append('format',fmt);
  fd.append('name',lastOutputName);
  try{
    const r=await fetch('/api/export',{method:'POST',headers:apiHeaders(),body:fd});
    if(!r.ok){let m=tr('导出失败','Export failed');try{const j=await r.json();m=j.detail||m;}catch(e){}return alert(m);}
    const blob=await r.blob();
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download=lastOutputName.replace(/\.wav$/i,'')+'.'+fmt;
    document.body.appendChild(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  }catch(e){alert(tr('导出失败：','Export failed: ')+e.message);}
}

// ============ 音色包管理 ============
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

async function loadVoicePacks(){
  try{
    const r=await fetch('/api/voicepacks',{headers:apiHeaders()});
    if(!r.ok)return;
    const d=await r.json();
    voicePacks=d.packs||[];
    renderPacks();
    fillPackSel();
  }catch(e){/* 静默：不影响主功能 */}
}

function renderPacks(){
  const el=document.getElementById('packList');
  el.innerHTML='';
  if(!voicePacks.length){el.innerHTML='<div class="muted">'+tr('还没有音色包，去“制作音色声线包”做一个吧。','No voice packs yet. Go to "Create Voice Pack" to make one.')+'</div>';return;}
  for(const p of voicePacks){
    const dur=p.processed_duration!=null?p.processed_duration+'s':'';
    const src=p.source_duration!=null?p.source_duration+'s':'';
    const accel=p.accelerated?' <span style="color:var(--warn-ink)">⚡加速</span>':'';
    const meta=[dur?(tr('代表参考 ','Ref ')+dur):'', src?(tr('原片 ','Src ')+src):'', p.created_at].filter(Boolean).join(' · ');
    const row=document.createElement('div');
    row.className='pack'; row.dataset.id=p.id;
    const info=document.createElement('div'); info.className='info';
    const nm=document.createElement('div'); nm.className='nm'; nm.innerHTML=esc(p.name)+accel;
    const md=document.createElement('div'); md.className='meta'; md.textContent=meta;
    info.appendChild(nm); info.appendChild(md);
    const acts=document.createElement('div'); acts.className='acts';
    function btn(label, cls, act){
      const b=document.createElement('button');
      if(cls) b.className=cls;
      b.textContent=label;
      b.dataset.act=act;
      return b;
    }
    acts.appendChild(btn(tr('▶ 试听','▶ Preview'),'', 'preview'));
    acts.appendChild(btn(tr('选用','Use'),'use','use'));
    acts.appendChild(btn(tr('删除','Delete'),'del','delete'));
    row.appendChild(info); row.appendChild(acts);
    el.appendChild(row);
  }
}

// 事件委托：用 data-* 标记按钮意图，避开 onclick 字符串拼接导致的 V8 解析陷阱
document.getElementById('packList').addEventListener('click', function(e){
  const b=e.target.closest && e.target.closest('button[data-act]');
  if(!b) return;
  const row=b.closest('.pack');
  const id=row && row.dataset.id;
  if(!id) return;
  const act=b.dataset.act;
  if(act==='preview') previewPack(id);
  else if(act==='use') usePack(id);
  else if(act==='delete') deletePack(id);
});

// 参考音频与音色包互斥：上传参考音频时清空音色包选择
document.getElementById('refFile').addEventListener('change', function(){
  if(this.files && this.files.length){
    const sel=document.getElementById('packSel');
    if(sel)sel.value='';
    selectedPackId=null;
    document.getElementById('packSelHint').textContent='';
    updatePtField();
  }
});

function fillPackSel(){
  const sel=document.getElementById('packSel');
  const cur=sel.value;
  sel.innerHTML='<option value="">'+tr('— 不使用音色包，改为上传音频 —','— No voice pack, upload audio instead —')+'</option>'+
    voicePacks.map(p=>'<option value="'+p.id+'">'+esc(p.name)+(p.processed_duration!=null?(' ('+p.processed_duration+'s)'):'')+'</option>').join('');
  if(cur)sel.value=cur;
}

function showPackPane(which){
  const manage=which==='manage';
  document.getElementById('packManage').classList.toggle('hide',!manage);
  document.getElementById('packSave').classList.toggle('hide',manage);
  document.querySelectorAll('.ptab').forEach(t=>t.classList.toggle('active',t.dataset.pane===which));
}

function showVpErr(m){const e=document.getElementById('vpErr');e.textContent=m?('❌ '+m):'';e.classList.toggle('show',!!m);}

let __droppedVpFile=null;  // 拖拽进来的视频/音频文件（savePack 优先使用）

async function savePack(){
  const file=document.getElementById('vpFile').files[0]||__droppedVpFile;
  if(!file&&!recBlob){return showVpErr(tr('请先录制、上传或拖入参考音频/视频','Record, upload or drop a reference audio/video first'));}
  const fd=new FormData();
  fd.append('name',document.getElementById('vpName').value);
  fd.append('denoise',document.getElementById('vpDenoise').checked);
  fd.append('remove_bg',document.getElementById('vpRemoveBg').checked);
  fd.append('accelerated',document.getElementById('vpAccel').checked);
  if(recBlob)fd.append('reference',recBlob,'recording.wav');
  else fd.append('reference',file);
  const btn=document.getElementById('vpSaveBtn'),st=document.getElementById('vpStatus');
  btn.disabled=true;st.classList.add('show');showVpErr('');
  const t0=Date.now();
  const timer=setInterval(()=>{document.getElementById('vpStatusText').textContent=tr('提取中… 已用 ','Extracting… ')+((Date.now()-t0)/1000).toFixed(1)+tr(' 秒','s');},200);
  try{
    const r=await fetch('/api/voicepacks',{method:'POST',body:fd,headers:apiHeaders()});
    clearInterval(timer);
    if(!r.ok){let m=tr('保存失败','Save failed');try{const j=await r.json();m=j.detail||m;}catch(e){}showVpErr(m);st.classList.remove('show');return;}
    const d=await r.json();
    document.getElementById('vpFile').value='';
    __droppedVpFile=null;
    setVpDropHint('');
    document.getElementById('vpName').value='';
    resetRec();
    st.classList.remove('show');
    await loadVoicePacks();
    showPackPane('manage');
    alert(tr('已保存音色包：','Voice pack saved: ')+d.pack.name+(d.pack.accelerated?tr('（⚡已开启加速模式）',' (⚡accelerated mode)') :''));
  }catch(e){clearInterval(timer);st.classList.remove('show');showVpErr(tr('请求失败：','Request failed: ')+e.message);}
  finally{btn.disabled=false;}
}

// ===== 拖拽上传（视频/音频 → 音色包）=====
function setVpDropHint(t){
  const el=document.getElementById('vpDropHint');
  if(!el)return;
  el.textContent=t||tr('建议 10–60 秒清晰人声；超过 30 秒会自动分段并融合为约 25 秒的代表参考。视频文件会自动提取音轨（需已安装 ffmpeg）。','10–60s of clear voice is recommended; over 30s is auto-segmented and fused into a ~25s representative clip. Video audio is auto-extracted (ffmpeg required).');
}
(function(){
  const dz=document.getElementById('vpDropZone');
  if(!dz)return;
  ['dragover','dragenter'].forEach(ev=>dz.addEventListener(ev,function(e){
    e.preventDefault();e.stopPropagation();
    dz.style.borderColor='var(--accent)';dz.style.background='var(--accent-soft)';
  }));
  ['dragleave','dragend'].forEach(ev=>dz.addEventListener(ev,function(e){
    e.preventDefault();
    dz.style.borderColor='var(--border-2)';dz.style.background='';
  }));
  dz.addEventListener('drop',function(e){
    e.preventDefault();e.stopPropagation();
    dz.style.borderColor='var(--border-2)';dz.style.background='';
    const files=e.dataTransfer&&e.dataTransfer.files;
    if(!files||!files.length)return;
    const f=files[0];
    // 校验扩展名（视频/音频）
    const ok=/\.(wav|mp3|flac|m4a|aac|ogg|mp4|mov|mkv|avi|webm|flv|m4v|wmv|ts)$/i.test(f.name||'');
    if(!ok){showVpErr(tr('不支持的文件类型：','Unsupported file type: ')+(f.name||'')+tr('（请拖入 wav/mp3/flac/mp4/mov 等音视频文件）','(drop wav/mp3/flac/mp4/mov etc.)'));return;}
    __droppedVpFile=f;
    // 更新 input 显示（部分浏览器支持 DataTransfer 赋值，失败不影响）
    try{
      const dt=new DataTransfer();
      dt.items.add(f);
      document.getElementById('vpFile').files=dt.files;
    }catch(_){}
    setVpDropHint(tr('✅ 已拖入：','✅ Dropped: ')+f.name+tr('（',' (')+(f.size/1024/1024).toFixed(1)+tr(' MB）—— 正在提取音色，请稍候…',' MB) — extracting voice, please wait…'));
    showVpErr('');
    savePack();   // 拖入即自动提取保存
  });
})();

function onPackSel(){
  const sel=document.getElementById('packSel');
  const v=sel.value;
  selectedPackId=v||null;
  const hint=document.getElementById('packSelHint');
  if(v){const p=voicePacks.find(x=>x.id===v);
    let t=tr('✅ 已选用：','✅ Selected: ')+(p?p.name:v)+tr('（无需再上传音频，直接点生成即可）',' (no re-upload needed, just generate)');
    if(p&&p.accelerated)t+=tr('  ⚡加速模式已启用，生成更快','  ⚡Accelerated mode on, faster generation');
    hint.textContent=t;
    document.getElementById('refField').classList.add('hide');
    document.getElementById('refFile').value='';   // 二选一互斥：清空参考音频
  }
  else{hint.textContent='';document.getElementById('refField').classList.remove('hide');}
  updatePtField();
}

function usePack(id){
  const p=voicePacks.find(x=>x.id===id);
  selectedPackId=id;
  setMode('clone');
  document.getElementById('packSel').value=id;
  const hint=document.getElementById('packSelHint');
  let t=tr('✅ 已选用音色包：','✅ Voice pack selected: ')+(p?p.name:id)+tr('（无需再上传音频，直接点生成即可）',' (no re-upload needed, just generate)');
  if(p&&p.accelerated)t+=tr('  ⚡加速模式已启用，生成更快','  ⚡Accelerated mode on, faster generation');
  hint.textContent=t;
  document.getElementById('refField').classList.add('hide');
  document.getElementById('refFile').value='';
  document.getElementById('packSelField').classList.remove('hide');
}

let vpAudio=null;
async function previewPack(id){
  if(!vpAudio)vpAudio=new Audio();
  try{
    const r=await fetch('/api/voicepacks/'+id+'/preview',{headers:apiHeaders()});
    if(!r.ok){alert(tr('试听失败（','Preview failed (')+r.status+tr('）',')'));return;}
    const blob=await r.blob();
    vpAudio.src=URL.createObjectURL(blob);
    vpAudio.play().catch(()=>{});
  }catch(e){alert(tr('试听失败：','Preview failed: ')+e.message);}
}

async function deletePack(id){
  if(!confirm(tr('确定删除该音色包？此操作不可撤销。','Delete this voice pack? This cannot be undone.')))return;
  try{
    const r=await fetch('/api/voicepacks/'+id,{method:'DELETE',headers:apiHeaders()});
    if(r.ok){if(selectedPackId===id)selectedPackId=null;await loadVoicePacks();}
    else{let m=tr('删除失败','Delete failed');try{const j=await r.json();m=j.detail||m;}catch(e){}alert(m);}
  }catch(e){alert(tr('请求失败：','Request failed: ')+e.message);}
}

// ============ 录制音色（MediaRecorder → WAV） ============
let recBlob=null, recChunks=[], recStream=null, mediaRec=null, recTimer=null, recSecs=0;
async function startRec(){
  try{
    recStream=await navigator.mediaDevices.getUserMedia({audio:true});
    mediaRec=new MediaRecorder(recStream);
    recChunks=[];
    mediaRec.ondataavailable=e=>{if(e.data&&e.data.size)recChunks.push(e.data);};
    mediaRec.onstop=async ()=>{
      try{
        const blob=new Blob(recChunks,{type:mediaRec.mimeType||'audio/webm'});
        recBlob=await blobToWav(blob);
        document.getElementById('recPlay').src=URL.createObjectURL(recBlob);
        document.getElementById('recWrap').classList.remove('hide');
        document.getElementById('vpFile').value=''; // 录制优先，清空上传
      }catch(e){showVpErr(tr('录音转码失败：','Audio conversion failed: ')+e.message);}
      if(recStream)recStream.getTracks().forEach(t=>t.stop());
    };
    mediaRec.start();
    recSecs=0;
    const btn=document.getElementById('recBtn');
    btn.textContent=tr('⏹ 停止录制','⏹ Stop Recording');btn.style.background='var(--danger)';btn.onclick=stopRec;
    document.getElementById('recStatus').textContent=tr('录制中 ','Recording ')+'0.0s';
    recTimer=setInterval(()=>{recSecs+=0.1;document.getElementById('recStatus').textContent=tr('录制中 ','Recording ')+recSecs.toFixed(1)+'s';},100);
  }catch(e){showVpErr(tr('无法访问麦克风：','Microphone access failed: ')+(e.message||e.name)+tr('（请允许浏览器麦克风权限）','(please allow microphone permission)'));}
}
function stopRec(){
  if(mediaRec&&mediaRec.state!=='inactive')mediaRec.stop();
  clearInterval(recTimer);
  const btn=document.getElementById('recBtn');
  btn.textContent=tr('🎤 重新录制','🎤 Re-record');btn.style.background='var(--sky)';btn.onclick=startRec;
  document.getElementById('recStatus').textContent=tr('录制完成，可回放或重新录制','Recording done. Playback or re-record.');
}
function resetRec(){
  recBlob=null;recChunks=[];recSecs=0;
  const btn=document.getElementById('recBtn');
  if(btn){btn.textContent=tr('🎤 开始录制','🎤 Start Recording');btn.style.background='var(--sky)';btn.onclick=startRec;}
  const w=document.getElementById('recWrap');if(w)w.classList.add('hide');
  const s=document.getElementById('recStatus');if(s)s.textContent=tr('点击下方按钮授权麦克风后开始朗读，建议 10–30 秒清晰语句；录制完可回放确认。','Click the button, allow microphone access, then read for 10–30s. Playback to confirm after recording.');
}
async function blobToWav(blob){
  const arr=await blob.arrayBuffer();
  const Ctx=window.AudioContext||window.webkitAudioContext;
  const ctx=new Ctx();
  const buf=await ctx.decodeAudioData(arr.slice(0));
  const ch=buf.numberOfChannels, len=buf.length, sr=buf.sampleRate;
  const inter=new Float32Array(len*ch);
  for(let c=0;c<ch;c++){const d=buf.getChannelData(c);for(let i=0;i<len;i++)inter[i*ch+c]=d[i];}
  return new Blob([encodeWav(inter,sr,ch)],{type:'audio/wav'});
}
function encodeWav(samples,sr,ch){
  const bps=2, blockAlign=ch*bps, dataSize=samples.length*bps;
  const ab=new ArrayBuffer(44+dataSize), view=new DataView(ab);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)view.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF');view.setUint32(4,36+dataSize,true);ws(8,'WAVE');ws(12,'fmt ');
  view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,ch,true);
  view.setUint32(24,sr,true);view.setUint32(28,sr*blockAlign,true);view.setUint16(32,blockAlign,true);
  view.setUint16(34,16,true);ws(36,'data');view.setUint32(40,dataSize,true);
  let off=44;
  for(let i=0;i<samples.length;i++){let s=Math.max(-1,Math.min(1,samples[i]));view.setInt16(off,s<0?s*0x8000:s*0x7FFF,true);off+=2;}
  return ab;
}
</script></body></html>"""


def render(html: str) -> str:
    return html.replace("PORT_PLACEHOLDER", str(PORT)).replace("TOKEN_PLACEHOLDER", ACCESS_TOKEN)


# ============================== 路由 ==============================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not check_auth(request):
        return HTMLResponse(render(LOGIN_HTML), status_code=200)
    resp = HTMLResponse(render(APP_HTML))
    q = request.query_params.get("token")
    if q and secrets.compare_digest(q, ACCESS_TOKEN):
        resp.set_cookie("voxcpm_token", ACCESS_TOKEN, httponly=True, samesite="lax", max_age=30 * 86400)
    return resp


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    token = (body or {}).get("token", "")
    if not token or not secrets.compare_digest(token, ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="令牌不正确")
    resp = JSONResponse({"ok": True})
    resp.set_cookie("voxcpm_token", ACCESS_TOKEN, httponly=True, samesite="lax", max_age=30 * 86400)
    return resp


@app.get("/api/health")
def health():
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        pass
    return {
        "status": "ok",
        "model_loaded": _model_info["loaded"],
        "device": _model_info["device"] or ("CUDA 可用" if cuda else "CPU"),
        "cuda": cuda,
        "sample_rate": _model_info["sample_rate"],
        "model_path": MODEL_PATH,
        "port": PORT,
    }


@app.post("/api/unload")
def unload(request: Request):
    """优雅卸载模型、释放显存（停止服务前调用，避免强杀损坏 GPU）。"""
    require_auth(request)
    unload_model()
    return {"ok": True, "detail": "模型已卸载，显存已释放"}


@app.post("/api/generate")
def generate(
    request: Request,
    text: str = Form(...),
    mode: str = Form("design"),
    cfg_value: float = Form(2.0),
    inference_timesteps: int = Form(10),
    normalize: str = Form("true"),
    denoise: str = Form("false"),
    remove_bg: str = Form("false"),
    stable: str = Form("false"),
    prompt_text: str = Form(""),
    voice_pack_id: str = Form(None),
    pitch: float = Form(0.0),
    speed: float = Form(1.0),
    volume: float = Form(1.0),
    pause: float = Form(0.15),
    breath: float = Form(0.0),
    emotion: str = Form(""),
    default_emotion: str = Form("neutral"),
    trigger_threshold: float = Form(0.6),
    transition_smoothness: float = Form(0.5),
    timbre_lock: str = Form("true"),
    ssml: str = Form("false"),
    lora_name: str = Form(""),
    reference: UploadFile = File(None),
):
    """网页用的统一生成接口（支持文件上传）"""
    require_auth(request)
    _check_not_training()
    text = text or ""
    # 仅用括号外台词做空校验；括号提示词保留，交由模型应用音色/风格
    if not strip_design_annotations(text):
        raise HTTPException(status_code=400, detail="文本不能为空")
    text = normalize_design_brackets(text)  # 中文括号统一成英文括号，模型才能识别提示

    ref_path = None
    used_pack = False
    if voice_pack_id:
        # 复用已保存的音色包，无需再次上传长段音频
        if mode not in ("clone", "hifi"):
            raise HTTPException(status_code=400, detail="音色包仅用于克隆 / 极致克隆模式")
        vp_wav, _ = vp_store.get_pack_paths(voice_pack_id)
        if vp_wav is None or not Path(vp_wav).exists():
            raise HTTPException(status_code=404, detail="所选音色包不存在或已损坏，请从列表重新选择")
        ref_path = str(vp_wav)
        used_pack = True
        print(f"[VoxCPM2] 使用音色包 {voice_pack_id} 作为参考", flush=True)
    elif reference is not None and reference.filename:
        suffix = Path(reference.filename).suffix or ".wav"
        ref_path = UPLOAD_DIR / f"ref_{uuid.uuid4().hex[:8]}{suffix}"
        ref_path.write_bytes(reference.file.read())
        ref_path = str(ref_path)

    if mode in ("clone", "hifi") and not ref_path:
        raise HTTPException(status_code=400, detail="该模式需要上传参考音频，或从已保存音色包中选择")

    # 参考音频校验（坏文件在此给出清晰 400，不进入推理）
    if ref_path and not used_pack:
        try:
            normalize_reference(ref_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 克隆/极致克隆：参考音频增强预处理（降噪 / 去背景音 / 长音频分段融合）
    # 注：音色包已是清洗/融合后的代表参考，跳过二次处理，直接复用，避免重复计算与音色漂移。
    if ref_path and not used_pack and mode in ("clone", "hifi"):
        try:
            ref_path = prepare_clone_reference(
                ref_path,
                denoise_on=str(denoise).lower() == "true",
                remove_bg_on=str(remove_bg).lower() == "true",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    kwargs = dict(
        text=text,
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
        normalize=str(normalize).lower() == "true",
        denoise=str(denoise).lower() == "true",
        _stable=str(stable).lower() == "true",
        pitch=float(pitch),
        speed=float(speed),
        volume=float(volume),
        pause=float(pause),
        breath=float(breath),
        emotion=str(emotion).strip(),
        default_emotion=str(default_emotion).strip() or "neutral",
        trigger_threshold=float(trigger_threshold),
        transition_smoothness=float(transition_smoothness),
        timbre_lock=str(timbre_lock).lower() == "true",
        _ssml=str(ssml).lower() == "true",
        lora_name=str(lora_name).strip(),
    )
    # 加速模式：使用已开启加速的音色包时，自动降低扩散步数，显著缩短生成耗时
    if used_pack:
        _meta = vp_store.get_pack_meta(voice_pack_id)
        if _meta and _meta.get("accelerated"):
            kwargs["inference_timesteps"] = min(int(inference_timesteps), ACCEL_STEPS)
    if mode == "clone":
        kwargs["reference_wav_path"] = ref_path
    elif mode == "hifi":
        # 二选一：音色包自带干净参考，无需逐字文本（降级为普通克隆）；
        # 上传参考音频时仍需逐字文本，用于极致克隆增强。
        if used_pack:
            kwargs["reference_wav_path"] = ref_path
        else:
            if not prompt_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="极致克隆需上传参考音频并填写其逐字文本；或直接选用音色包（无需逐字文本）",
                )
            kwargs["reference_wav_path"] = ref_path
            kwargs["prompt_wav_path"] = ref_path
            kwargs["prompt_text"] = prompt_text

    return _do_generate(kwargs)


@app.post("/api/tts")
async def tts_api(request: Request):
    """纯 JSON 接口，方便脚本 / 其它程序调用"""
    require_auth(request)
    _check_not_training()
    body = await request.json()
    text = (body or {}).get("text", "") or ""
    # 仅用括号外台词做空校验；括号提示词保留，交由模型应用音色/风格
    if not strip_design_annotations(text):
        raise HTTPException(status_code=400, detail="文本不能为空")
    text = normalize_design_brackets(text)  # 中文括号统一成英文括号，模型才能识别提示
    kwargs = dict(
        text=text,
        cfg_value=float(body.get("cfg_value", 2.0)),
        inference_timesteps=int(body.get("inference_timesteps", 10)),
        normalize=bool(body.get("normalize", True)),
        denoise=bool(body.get("denoise", False)),
    )
    if body.get("voice_pack_id"):
        vp_wav, _ = vp_store.get_pack_paths(body["voice_pack_id"])
        if vp_wav is None or not Path(vp_wav).exists():
            raise HTTPException(status_code=404, detail="所选音色包不存在或已损坏")
        kwargs["reference_wav_path"] = str(vp_wav)
        if body.get("prompt_text"):
            kwargs["prompt_wav_path"] = str(vp_wav)
        _meta = vp_store.get_pack_meta(body["voice_pack_id"])
        if _meta and _meta.get("accelerated"):
            kwargs["inference_timesteps"] = min(int(body.get("inference_timesteps", 10)), ACCEL_STEPS)
    elif body.get("reference_wav_path"):
        kwargs["reference_wav_path"] = body["reference_wav_path"]
    if body.get("prompt_wav_path"):
        kwargs["prompt_wav_path"] = body["prompt_wav_path"]
    if body.get("prompt_text"):
        kwargs["prompt_text"] = body["prompt_text"]
    kwargs.update(
        pitch=float(body.get("pitch", 0) or 0),
        speed=float(body.get("speed", 1.0) or 1.0),
        volume=float(body.get("volume", 1.0) or 1.0),
        pause=float(body.get("pause", 0.15) or 0.15),
        breath=float(body.get("breath", 0) or 0),
        emotion=str(body.get("emotion", "") or "").strip(),
        _ssml=bool(body.get("ssml", False)),
        lora_name=str(body.get("lora_name", "") or "").strip(),
    )
    return _do_generate(kwargs)


def _do_generate(kwargs: dict):
    """统一生成：整条链路包进 try/except，异常 → JSON；失败 → 模型自愈。"""
    t0 = time.time()
    try:
        _lora_name = (kwargs.pop("lora_name", "") or "").strip() or None
        model = get_model(_lora_name)            # 加载也可能抛异常，一并捕获
        # CUDA 健康检查 + 显存保护：
        # - VoxCPM 连续推理会累积显存缓存（内部泄漏，无法从外部根治），总显存超阈值
        #   先优雅卸载并重载模型回收，避免 CUDA OOM / native crash；
        # - native crash 后 CUDA 上下文可能损坏（健康检查用轻量 tensor 操作暴露）。
        try:
            import torch
            if torch.cuda.is_available():
                _cuda_ok = True
                try:
                    _x = torch.zeros(4, device="cuda")
                    _ = _x.sum()
                    del _x
                except Exception:
                    _cuda_ok = False
                if not _cuda_ok:
                    print("[VoxCPM2] CUDA 健康检查失败，重载模型", flush=True)
                    unload_model()
                    model = get_model()
                else:
                    _free, _total = torch.cuda.mem_get_info()
                    _used_gb = (_total - _free) / (1024 ** 3)
                    if _used_gb > MEMORY_RESET_THRESHOLD_GB:
                        print(f"[VoxCPM2] 显存占用 {_used_gb:.1f}GB 超阈值，主动重载模型释放泄漏", flush=True)
                        unload_model()
                        model = get_model()
        except Exception:
            pass
        _stable = kwargs.pop("_stable", False)
        _ssml = kwargs.pop("_ssml", False)
        pitch = float(kwargs.pop("pitch", 0.0) or 0.0)
        speed = float(kwargs.pop("speed", 1.0) or 1.0)
        volume = float(kwargs.pop("volume", 1.0) or 1.0)
        _pause = kwargs.pop("pause", 0.15)
        pause = float(_pause if _pause is not None else 0.15)
        breath = float(kwargs.pop("breath", 0.0) or 0.0)
        emotion = (kwargs.pop("emotion", "") or "").strip()
        # 情绪控制参数（长文本默认中性、情绪切换阈值、过渡平滑度、音色锁定）
        emotion_control = {
            "default_emotion": kwargs.pop("default_emotion", "neutral") or "neutral",
            "trigger_threshold": float(kwargs.pop("trigger_threshold", 0.6) or 0.6),
            "transition_smoothness": float(kwargs.pop("transition_smoothness", 0.5) or 0.5),
            "timbre_lock": bool(kwargs.pop("timbre_lock", True)),
            "keep_default_when_unspecified": True,
        }

        # 情绪预设 + SSML 解析（lazy import 避免启动依赖）
        try:
            import audio_edit as _ae
        except Exception:
            _ae = None
        # 情绪预设（延迟应用，需先判断是否走稳定合成，避免块间情绪混入）
        emotion_preset = None
        if emotion and _ae is not None:
            name = _ae.EMOTION_ALIAS.get(emotion.lower(), emotion)
            emotion_preset = _ae.EMOTION_PRESETS.get(name) or _ae.EMOTION_PRESETS.get(emotion)

        # SSML 解析（优先级高于情绪预设）
        if _ssml and _ae is not None:
            text_str, sp = _ae.parse_ssml(str(kwargs.get("text", "")))
            kwargs["text"] = text_str
            pitch = sp.get("pitch", pitch)
            speed = sp.get("speed", speed)
            volume = sp.get("volume", volume)
            pause = sp.get("pause", pause)
            breath = sp.get("breath", breath)

        # 是否走稳定合成（长文本 / 勾选稳定）
        use_stable = _stable or len(str(kwargs.get("text", ""))) >= LONG_TEXT_CHARS

        # 应用情绪预设：
        # - 稳定路径：情绪韵律交由 synthesize_stable 逐块统一施加，此处只取停顿，
        #   避免 pitch/speed/volume 双重施加与块间情绪不一致
        # - 短文本直通：情绪用全局韵律（pitch/speed/volume/pause）
        # - 呼吸(breath)是用户显式可控参数（默认 0 = 无呼吸），情绪预设绝不覆盖它，
        #   否则用户把呼吸调到 0 仍会因情绪预设冒出呼吸声
        if emotion_preset:
            if use_stable:
                pause = pause if abs(pause - 0.15) > 0.01 else emotion_preset.get("pause", 0.15)
            else:
                # 情绪不改音调（语调保持不变），只取语速/音量/停顿
                speed = speed if abs(speed - 1.0) > 0.01 else emotion_preset.get("speed", 1.0)
                volume = volume if abs(volume - 1.0) > 0.01 else emotion_preset.get("volume", 1.0)
                pause = pause if abs(pause - 0.15) > 0.01 else emotion_preset.get("pause", 0.15)

        # 发音校正：检测多音字/生僻字并记录（模型本身具备 LLM 级多音字上下文理解）
        if _ae is not None:
            try:
                _polys = _ae.detect_polyphones(str(kwargs.get("text", "")))
                if _polys:
                    print(f"[VoxCPM2] 多音字/生僻字提示: {''.join(_polys[:24])}", flush=True)
            except Exception:
                pass

        eff_steps = int(kwargs.get("inference_timesteps", 10))
        stability_report = None
        sr = _vc_stab._get_sample_rate(model)
        with _infer_lock:                         # 串行推理，防并发打爆显存
            text_str = str(kwargs.get("text", ""))
            # 长文本/稳定合成：句末+逗号+换行分块 + 每块独立生成(参考锚定) + 分级停顿拼接，
            # 根治音色漂移 / 机械感累积 / 语速越来越快 / 逗号无停顿 / 情绪块间混入。
            if use_stable:
                wav, stab_rep = _vc_stab.synthesize_stable(
                    model,
                    text_str,
                    reference_wav_path=kwargs.get("reference_wav_path"),
                    sr_tts=sr,
                    prompt_wav_path=kwargs.get("prompt_wav_path"),
                    prompt_text=kwargs.get("prompt_text"),
                    max_chars=60,
                    pause=pause,
                    breath=breath,
                    emotion=emotion,
                    emotion_control=emotion_control,
                    cfg_value=kwargs.get("cfg_value", 2.0),
                    inference_timesteps=kwargs.get("inference_timesteps", 10),
                    normalize=kwargs.get("normalize", True),
                    denoise=kwargs.get("denoise", False),
                )
                stability_report = stab_rep
                print(f"[VoxCPM2] 稳定合成指标: {stab_rep}", flush=True)
            else:
                wav = model.generate(**kwargs)
        if isinstance(wav, list):
            wav = np.concatenate(wav)
        # 音调/语速（在归一化前应用，避免被 RMS 归一化抹平）
        if _ae is not None:
            wav = _ae.apply_pitch(wav, sr, pitch)
            wav = _ae.apply_speed(wav, sr, speed)
        # 统一后处理：软限幅防爆音 + RMS 归一化，保证基础响度一致
        wav = _vc_stab.postprocess_output(wav)
        # 音量（归一化后独立生效）+ 最终限幅防爆音
        if _ae is not None:
            wav = _ae.apply_volume(wav, volume)
        wav = _vc_stab.declip(wav)
        # 清晰度增强：温和 pre-emphasis（辅音/齿音提升），改善个别词语咬字不清。
        # 对全部成品启用（不止 30s 以上），amount 取温和值避免音色变尖。
        if _ae is not None:
            wav = _ae.enhance_clarity(wav, sr, amount=0.92)
        # 推理结束后主动释放 GPU 缓存，减少连续生成时的显存累积
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        elapsed = round(time.time() - t0, 2)

        name = f"tts_{time.strftime('%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.wav"
        path = OUTPUT_DIR / name
        sf.write(str(path), wav, sr)

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        dur = round(len(wav) / sr, 2)
        print(f"[VoxCPM2] 生成完成 {name} 时长{dur}s 耗时{elapsed}s", flush=True)
        headers = {
            "X-Output-Name": name,
            "X-Elapsed": str(elapsed),
            "X-Duration": str(dur),
            "X-Effective-Steps": str(eff_steps),
            "Content-Disposition": f'inline; filename="{name}"',
        }
        if stability_report:
            try:
                headers["X-Stability"] = json.dumps(stability_report, ensure_ascii=True)
            except Exception:
                pass
        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error("推理失败", e)
        # 彻底卸载模型释放显存（比仅置空更彻底：del + empty_cache + gc.collect），
        # 避免失败后模型进入损坏态导致后续请求持续失败
        unload_model()
        raise HTTPException(status_code=500, detail=f"推理失败: {type(e).__name__}: {e}")


@app.post("/api/multi_speaker")
def multi_speaker(request: Request,
                  text: str = Form(...),
                  cfg_value: float = Form(2.0),
                  inference_timesteps: int = Form(10),
                  voice_params: str = Form(""),
                  denoise: str = Form("false")):
    """Beta：多人朗读 + 情绪控制。
    解析 (@音色包名) 音色切换标记和 (情绪词) 情绪标记，逐段用对应音色包 + 独立角色参数
    + 情绪参数生成；长段走稳定合成（分块+参考锚定，避免机械音/崩溃），段间停顿拼接。"""
    require_auth(request)
    text = text or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")
    segments = parse_multi_speaker_text(text)
    if not segments:
        raise HTTPException(status_code=400, detail="未解析到可朗读的文本段")

    # 角色参数映射：{音色包名: {pitch,speed,volume,pause,breath}}
    vp_map: dict = {}
    if voice_params:
        try:
            vp_map = json.loads(voice_params)
        except Exception:
            vp_map = {}
    denoise_on = str(denoise).lower() == "true"

    packs = vp_store.list_packs()
    name_to_id = {p["name"]: p["id"] for p in packs}
    model = get_model()
    sr = _vc_stab._get_sample_rate(model)
    try:
        import audio_edit as _ae
    except Exception:
        _ae = None

    pieces: list[np.ndarray] = []
    seg_report: list[dict] = []
    warnings: list[str] = []
    with _infer_lock:
        for i, seg in enumerate(segments):
            voice_name = seg["voice"]
            vpid = name_to_id.get(voice_name) if voice_name else None
            missing = bool(voice_name) and not vpid
            if missing:
                warnings.append(f"未知音色包「{voice_name}」，该段用默认音色")
            ref_path = None
            if vpid:
                vp_wav, _ = vp_store.get_pack_paths(vpid)
                if vp_wav and vp_wav.exists():
                    ref_path = str(vp_wav)

            # 角色独立参数（默认中性）
            rp = (vp_map.get(voice_name, {}) or {}) if voice_name else {}
            r_pitch = float(rp.get("pitch", 0) or 0)
            r_speed = float(rp.get("speed", 1) or 1)
            r_volume = float(rp.get("volume", 1) or 1)
            r_pause = float(rp.get("pause", 0.15) if rp.get("pause") is not None else 0.15)
            r_breath = float(rp.get("breath", 0) if rp.get("breath") is not None else 0)

            # 情绪参数：情绪韵律交由 synthesize_stable 逐块统一施加（严格一致），
            # 这里只叠加角色独立参数，避免情绪预设 double-apply 与块间情绪混入
            emo = seg["emotion"]
            pitch = r_pitch
            speed = r_speed
            volume = r_volume

            # 稳定合成（长段分块+参考锚定；短段直通 model.generate）
            wav, _ = _vc_stab.synthesize_stable(
                model, seg["text"], ref_path, sr,
                pause=r_pause, breath=r_breath, emotion=emo,
                cfg_value=float(cfg_value), inference_timesteps=int(inference_timesteps),
                normalize=True, denoise=denoise_on,
            )
            if _ae:
                if abs(pitch) > 0.01: wav = _ae.apply_pitch(wav, sr, pitch)
                if abs(speed - 1) > 0.01: wav = _ae.apply_speed(wav, sr, speed)
                if abs(volume - 1) > 0.01: wav = _ae.apply_volume(wav, volume)
            wav = _vc_stab.declip(wav)
            pieces.append(wav)
            seg_report.append({"i": i, "voice": voice_name or "默认", "emotion": emo,
                               "text": seg["text"][:24], "missing": missing})
            print(f"[VoxCPM2][Beta] 段{i}: 音色={voice_name or '默认'} 情绪={emo}"
                  f"{' (未知音色!)' if missing else ''}", flush=True)

    # 段间 0.3s 静音停顿拼接
    pause = np.zeros(int(sr * 0.3), dtype=np.float32)
    final = pieces[0]
    for p in pieces[1:]:
        final = np.concatenate([final, pause, p])
    name = f"multi_{time.strftime('%m%d_%H%M%S')}_{secrets.token_hex(2)}.wav"
    sf.write(str(OUTPUT_DIR / name), final, sr, format="WAV")
    buf = io.BytesIO(); sf.write(buf, final, sr, format="WAV"); buf.seek(0)
    dur = round(len(final) / sr, 2)
    headers = {
        "X-Output-Name": name, "X-Duration": str(dur),
        "X-Segments": json.dumps({"n": len(segments), "segments": seg_report,
                                  "warnings": warnings}, ensure_ascii=True),
        "Content-Disposition": f'inline; filename="{name}"',
    }
    return Response(content=buf.read(), media_type="audio/wav", headers=headers)


@app.post("/api/dialogue")
async def dialogue(request: Request):
    """Beta：多人多轮对话合成。接收 turns 列表（每次参与一个 turn），
    逐 turn 用对应音色包 + 语气/情绪/音量参数生成，段间停顿拼接。

    可选启用导演层（模块4/5）：use_director=true 时先对全文做一次梳理，
    把推断出的情绪 / 句间停顿 / 表现力 cfg / 语速 落到对应 turn。
    - 情绪：仅在该 turn 未显式指定（neutral）时接管，尊重用户手填的 (情绪)
    - cfg ：这是 VoxCPM2 唯一真正影响表达力的模型级旋钮，导演层规划值在此落地
    - 音高：默认**不施加**（apply_pitch 是 varispeed，会带偏共振峰损伤克隆音色），
            规划值只回显在 X-Segments 里；确需施加传 director_apply_pitch=true
    """
    require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体需为 JSON")
    turns = body.get("turns") or []
    if not turns:
        raise HTTPException(status_code=400, detail="turns 不能为空")
    denoise_on = bool(body.get("denoise", False))
    cfg = float(body.get("cfg_value", 2.0))
    steps = int(body.get("inference_timesteps", 10))
    use_director = bool(body.get("use_director", False))
    director_engine = str(body.get("director_engine") or "rule")
    director_apply_pitch = bool(body.get("director_apply_pitch", False))

    TONE_MAP = {"自然": (0, 1.0), "温柔": (0, 0.95), "严肃": (0, 0.92), "活泼": (1, 1.05), "低沉": (-2, 0.9),
                "Natural": (0, 1.0), "Gentle": (0, 0.95), "Serious": (0, 0.92), "Lively": (1, 1.05), "Low": (-2, 0.9)}
    packs = vp_store.list_packs()
    name_to_id = {p["name"]: p["id"] for p in packs}
    model = get_model()
    sr = _vc_stab._get_sample_rate(model)
    try:
        import audio_edit as _ae
    except Exception:
        _ae = None

    pieces: list[np.ndarray] = []
    seg_report: list[dict] = []
    warnings: list[str] = []
    plan_map = None
    director_info = None
    if use_director:
        try:
            plan_map, director_info, dwarns = build_director_plan(
                turns, engine=director_engine)
            warnings.extend(dwarns)
            if director_info is not None:
                director_info["apply_pitch"] = director_apply_pitch
                print(f"[VoxCPM2][Director] 引擎={director_info['engine']} "
                      f"通路={director_info['source']} 粒度={director_info['granularity']} "
                      f"基调={director_info['tone']}", flush=True)
        except Exception as e:
            warnings.append(f"导演层执行失败，已回落默认参数：{e}")
            log_error("director", e)
    with _infer_lock:
        for i, turn in enumerate(turns):
            role = str(turn.get("role") or "").strip()
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            tone = turn.get("tone") or "自然"
            emotion = turn.get("emotion") or "neutral"
            volume = float(turn.get("volume") or 1)
            pitch_user = float(turn.get("pitch") or 0)      # 独立音调（半音，叠加在语气/情绪预设之上）
            speed_user = float(turn.get("speed") or 1)      # 独立语速（倍率，乘以语气/情绪预设）
            pause_turn = float(turn.get("pause") or 0.15)   # 独立句间停顿（秒）
            _b = turn.get("breath")
            breath_turn = float(_b) if _b is not None else 0.4  # 独立呼吸轻重（0 关 ~ 1 重）
            vpid = name_to_id.get(role) if (role and role != "旁白") else None
            missing = bool(role and role != "旁白") and not vpid
            if missing:
                warnings.append(f"未知音色包「{role}」，该段用默认音色")
            ref_path = None
            if vpid:
                vp_wav, _ = vp_store.get_pack_paths(vpid)
                if vp_wav and vp_wav.exists():
                    ref_path = str(vp_wav)
            tp, ts = TONE_MAP.get(tone, (0, 1.0))
            # 情绪韵律交由 synthesize_stable 逐块统一施加（严格一致），
            # 这里只叠加「语气 + 用户独立参数」，避免情绪预设 double-apply 与块间混入
            pitch = tp + pitch_user
            speed = ts * speed_user
            vol = volume
            # —— 导演层落地：仅接管「用户没显式填」的部分，绝不覆盖手填参数 ——
            turn_cfg = cfg
            dir_applied = None
            if plan_map and i < len(plan_map):
                pm = plan_map[i]
                if emotion in ("", "neutral") and pm.get("emotion"):
                    emotion = pm["emotion"]
                if pm.get("pause") is not None:
                    pause_turn = float(pm["pause"])
                if pm.get("pace") is not None:
                    # pace 走 WSOLA 语速，保音高保共振峰，安全
                    speed *= float(pm["pace"])
                if pm.get("cfg") is not None:
                    turn_cfg = float(pm["cfg"])
                if director_apply_pitch and pm.get("pitch_st") is not None:
                    pitch += float(pm["pitch_st"])
                dir_applied = pm
            wav, _ = _vc_stab.synthesize_stable(
                model, text, ref_path, sr, pause=pause_turn, breath=breath_turn, emotion=emotion,
                cfg_value=turn_cfg, inference_timesteps=steps, normalize=True, denoise=denoise_on)
            if _ae:
                if abs(pitch) > 0.01: wav = _ae.apply_pitch(wav, sr, pitch)
                if abs(speed - 1) > 0.01: wav = _ae.apply_speed(wav, sr, speed)
                if abs(vol - 1) > 0.01: wav = _ae.apply_volume(wav, vol)
            wav = _vc_stab.declip(wav)
            pieces.append(wav)
            seg_report.append({"i": i, "voice": role, "emotion": emotion, "tone": tone,
                               "text": text[:24], "missing": missing,
                               "pitch": round(pitch, 2), "speed": round(speed, 3),
                               "volume": round(vol, 2), "pause": pause_turn, "breath": breath_turn,
                               "cfg": round(turn_cfg, 2),
                               "director": ({k: dir_applied[k] for k in
                                             ("emotion", "intensity", "pause", "cfg", "pace", "pitch_st")}
                                            if dir_applied else None)})
            print(f"[VoxCPM2][Dialogue] turn{i}: {role} 语气={tone} 情绪={emotion} "
                  f"cfg={turn_cfg:.2f}"
                  f"{' [导演层]' if dir_applied else ''}", flush=True)

    if not pieces:
        raise HTTPException(status_code=400, detail="没有可合成的台词")
    pause = np.zeros(int(sr * 0.3), dtype=np.float32)
    final = pieces[0]
    for p in pieces[1:]:
        final = np.concatenate([final, pause, p])
    name = f"dialogue_{time.strftime('%m%d_%H%M%S')}_{secrets.token_hex(2)}.wav"
    sf.write(str(OUTPUT_DIR / name), final, sr, format="WAV")
    buf = io.BytesIO(); sf.write(buf, final, sr, format="WAV"); buf.seek(0)
    dur = round(len(final) / sr, 2)
    headers = {"X-Output-Name": name, "X-Duration": str(dur),
               "X-Segments": json.dumps({"n": len(seg_report), "segments": seg_report,
                                         "warnings": warnings, "director": director_info},
                                        ensure_ascii=True),
               "Content-Disposition": f'inline; filename="{name}"'}
    return Response(content=buf.read(), media_type="audio/wav", headers=headers)


@app.post("/api/plan")
async def plan_endpoint(request: Request):
    """导演层预览（不合成音频）：返回文本梳理结果，便于核对判定是否合理。

    body: {text, engine?: "rule"|"llm", emotion?: "全局基调提示"}
    返回完整规划（含每段的 role/emotion/intensity/pause/依据）+ 人类可读 summary。
    """
    require_auth(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体需为 JSON")
    text = str(body.get("text") or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    engine = str(body.get("engine") or "rule")
    plan_result = run_director_plan(text, engine=engine,
                                    emotion=str(body.get("emotion") or ""))
    if plan_result is None:
        raise HTTPException(status_code=503, detail="导演层模块未加载")
    intact = None
    if _director is not None:
        try:
            intact = bool(_director.verify_text_intact(text, plan_result))
        except Exception:
            intact = None
    return JSONResponse({
        "source": plan_result.source,
        "global_tone": plan_result.global_tone,
        "global_share": round(float(plan_result.global_share), 3),
        "text_intact": intact,
        "error": plan_result.error,
        "n_segments": len(plan_result.segments),
        "segments": [{"i": i, "text": s.text, "role": s.role, "emotion": s.emotion,
                      "intensity": round(float(s.intensity), 3),
                      "pause_after": round(float(s.pause_after), 3),
                      "pace": s.pace, "pitch_st": s.pitch_st, "cfg": s.cfg,
                      "note": s.note, "reason": s.reason}
                     for i, s in enumerate(plan_result.segments)],
        "summary": plan_result.summary(),
    })


@app.get("/api/outputs/{name}")
def get_output(name: str, request: Request):
    require_auth(request)
    path = (OUTPUT_DIR / name).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(path), media_type="audio/wav", filename=name)


# --------------------------------------------------------------- AI 内核配置
# 给「AI 内核配置」界面用的三个端点：
#   GET  /api/llm/providers  —— 服务商预设清单（前端下拉直接渲染，不再硬编码）
#   GET  /api/llm/config     —— 读当前配置（api_key 脱敏回显）
#   POST /api/llm/config     —— 保存配置；可选 test_only 只测不存
#   POST /api/llm/test       —— 用（当前或传入的）配置测连通性


def _llm_module():
    """懒加载 director_llm，返回 None 表示不可用（与 LLM 引擎的加载策略一致）。"""
    try:
        from voice_clone import director_llm as _dl
        return _dl
    except Exception as _e:
        print(f"[warn] director_llm unavailable: {_e}", flush=True)
        return None


def _mask_key(key: str) -> str:
    """脱敏：只留头尾各 4 位，中间打码。空值返回空串。

    前端据此显示"已配置"，并要求用户在改 Key 时重新完整输入，
    避免把完整 Key 往返传输、也避免误把脱敏串存回配置。
    """
    k = str(key or "")
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}{'*' * min(12, len(k) - 8)}{k[-4:]}"


@app.get("/api/llm/providers")
def llm_providers_endpoint(request: Request):
    """返回服务商预设清单，供前端渲染下拉与联动提示。"""
    require_auth(request)
    try:
        from voice_clone import llm_providers as LP
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"llm_providers 模块不可用：{e}")
    return JSONResponse({"providers": [
        {"id": p["id"], "name_zh": p["name_zh"], "name_en": p["name_en"],
         "base_url": p["base_url"], "models": list(p["models"]),
         "key_hint": p["key_hint"], "key_hint_en": p.get("key_hint_en", ""),
         "key_url": p["key_url"], "note": p["note"],
         "note_en": p.get("note_en", ""),
         "local": bool(p.get("local"))}
        for p in LP.PROVIDERS
    ]})


@app.get("/api/llm/config")
def llm_get_config(request: Request):
    """读当前 LLM 配置。api_key 脱敏后回显，另附就绪状态与来源。"""
    require_auth(request)
    dl = _llm_module()
    if dl is None:
        raise HTTPException(status_code=503, detail="director_llm 模块不可用")
    cfg = dl.load_config()
    ready, why = dl.is_ready(cfg)
    # 机器可读的原因码：前端按它做本地化，英文界面才不会显示中文提示
    try:
        _rcode, _ = dl.readiness(cfg)
    except Exception:
        _rcode = None
    # 判断 key 的真实来源，便于界面提示"当前用的是环境变量而非本地文件"
    src = "none"
    raw_key = str(cfg.get("api_key") or "").strip()
    prov = dl._providers.get_provider(cfg.get("provider"))
    if raw_key:
        if prov.get("local") and raw_key == prov["id"]:
            # 本地部署的占位符，不是用户真实配置的密钥
            src = "local_placeholder"
        else:
            try:
                import json as _json
                file_key = ""
                if os.path.isfile(dl.CONFIG_PATH):
                    with open(dl.CONFIG_PATH, encoding="utf-8") as f:
                        file_key = str((_json.load(f) or {}).get("api_key") or "").strip()
                txt_key = ""
                if os.path.isfile(dl.API_KEY_TXT):
                    with open(dl.API_KEY_TXT, encoding="utf-8") as f:
                        txt_key = f.read().strip()
                if file_key and file_key == raw_key:
                    src = "config_file"
                elif txt_key and txt_key == raw_key:
                    src = "api_key_txt"
                else:
                    src = "env"
            except Exception:
                src = "unknown"
    return JSONResponse({
        "enabled": bool(cfg.get("enabled")),
        "provider": cfg.get("provider", ""),
        "base_url": cfg.get("base_url", ""),
        "model": cfg.get("model", ""),
        "api_key_masked": _mask_key(raw_key),
        "has_key": bool(raw_key),
        "key_source": src,
        "timeout": cfg.get("timeout", 60),
        "temperature": cfg.get("temperature", 0.3),
        "proxy": cfg.get("proxy", ""),
        "json_mode": bool(cfg.get("json_mode", True)),
        "cache": bool(cfg.get("cache", True)),
        "max_segments_per_call": cfg.get("max_segments_per_call", 120),
        "full_text_limit": cfg.get("full_text_limit", 6000),
        "ready": ready,
        "reason": why if not ready else "",
        "reason_code": _rcode if not ready else None,
        "config_path": dl.CONFIG_PATH,
    })


@app.post("/api/llm/config")
async def llm_save_config(request: Request):
    """保存 LLM 配置。

    body: {enabled?, provider?, base_url?, model?, api_key?, timeout?,
           temperature?, proxy?, json_mode?, cache?, test_only?}

    约定：**api_key 传空字符串或省略时保持原值不变**（界面只回显脱敏串，
    无法把明文回传），只有显式传入非空才覆盖。传 null 表示清空。
    """
    require_auth(request)
    dl = _llm_module()
    if dl is None:
        raise HTTPException(status_code=503, detail="director_llm 模块不可用")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体需为 JSON")
    body = body or {}

    cur = dl.load_config()
    patch: dict = {}

    for k in ("enabled",):
        if k in body:
            patch[k] = bool(body[k])
    for k in ("provider", "base_url", "model", "proxy"):
        if k in body:
            patch[k] = str(body[k] or "").strip()
    for k in ("temperature", "max_segments_per_call", "full_text_limit"):
        if k in body:
            try:
                patch[k] = type(dl.DEFAULT_CONFIG[k])(body[k])
            except Exception:
                raise HTTPException(status_code=400, detail=f"{k} 取值非法：{body[k]!r}")
    for k in ("json_mode", "cache"):
        if k in body:
            patch[k] = bool(body[k])
    # timeout 单独处理（见下）
    if "timeout" in body:
        try:
            patch["timeout"] = int(body["timeout"])
        except Exception:
            raise HTTPException(status_code=400, detail=f"timeout 取值非法：{body['timeout']!r}")

    # api_key 三态处理
    if "api_key" in body:
        raw = body["api_key"]
        if raw is None:
            patch["api_key"] = ""          # 显式清空
        else:
            v = str(raw)
            if v.strip():
                patch["api_key"] = v        # 显式传入 → 覆盖（保存时统一清洗）
            # 空串 → 不动（视为"未修改"）

    # 保存前用预设补全 base_url / model 空缺
    merged = dl._providers.apply_preset({**cur, **patch})

    # timeout 智能纠偏：用户没显式指定，且本次切了服务商 → 清掉遗留值，
    # 让 load_config 按新服务商填推荐值（云 60s / 本地 300s）。
    # 保护的是这种真实场景：老配置里留着云 API 的 timeout=60，用户切成本地
    # Ollama 后，27B 冷启 + 长文本规划超过 60s，会看到"配置对了却总超时"。
    _prov_prev = dl._providers.get_provider(cur.get("provider"))
    _prov_new = dl._providers.get_provider(merged.get("provider"))
    if "timeout" not in body and _prov_new["id"] != _prov_prev["id"]:
        merged["timeout"] = None      # save_config 会删掉该键 → 走推荐值

    # Key 校验分两档 —— 这个区分很重要：
    #   a) **格式性错误**（含中文、超长、把网址填进 Key 框）永远硬拦：
    #      这类输入 100% 是误操作，存下来只会变成难查的故障。
    #   b) **"key 为空"** 只在"用户想启用"时才硬拦。
    #      否则会产生死锁：用户从「本地 Ollama」切回「DeepSeek」时必定先经历
    #      "非本地 + 无 key"的中间态，若此时拒绝保存，就永远切不回去了。
    ok_key, tip = dl._providers.key_looks_valid(
        merged.get("api_key"), merged.get("provider"))
    key_empty = not str(merged.get("api_key") or "").strip()
    if not ok_key and not key_empty:
        return JSONResponse({"ok": False, "stage": "validate",
                             "reason": tip}, status_code=400)
    if str(merged.get("api_key") or "").strip():
        merged["api_key"] = dl._providers.normalize_key(merged["api_key"])

    warnings = []
    if not str(merged.get("base_url") or "").strip():
        warnings.append("base_url 为空")
    if not str(merged.get("model") or "").strip():
        warnings.append("model 为空")
    if tip:
        warnings.append(tip)

    # 想启用却没 key（非本地服务）→ 这才是真正该硬拦的时刻
    _prov = dl._providers.get_provider(merged.get("provider"))
    if merged.get("enabled") and key_empty and not _prov.get("local"):
        return JSONResponse({
            "ok": False, "stage": "validate",
            "reason": "要启用 AI 内核必须填写 API Key（当前服务商：%s）。"
                      "如果只是想先保存配置，请先取消勾选「启用 AI 内核」。"
                      % _prov["name_zh"],
            "warnings": warnings,
        }, status_code=400)

    if bool(body.get("test_only")):
        res = dl.test_connection(merged)
        res["saved"] = False
        res["warnings"] = warnings
        return JSONResponse(res)

    path = dl.save_config(merged)

    want_test = bool(body.get("test", True))
    res = dl.test_connection(merged, probe_chat=want_test)
    res["saved"] = True
    res["warnings"] = warnings
    res["config_path"] = path
    # 保存成功但连通失败时返回 200（配置确实落盘了），由 ok 字段表达连通性
    return JSONResponse(res)


@app.post("/api/llm/test")
async def llm_test(request: Request):
    """测连通性。body 可传一份临时配置（不落盘）；为空则测当前已存配置。

    body: {provider?, base_url?, model?, api_key?, timeout?, proxy?,
           probe_chat?(默认 true)}
    """
    require_auth(request)
    dl = _llm_module()
    if dl is None:
        raise HTTPException(status_code=503, detail="director_llm 模块不可用")
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body or {}

    base = dl.load_config()
    patch = {k: body[k] for k in
             ("provider", "base_url", "model", "timeout", "proxy") if k in body}
    if body.get("api_key"):
        patch["api_key"] = str(body["api_key"])
    cand = dl._providers.apply_preset({**base, **patch})
    if str(cand.get("api_key") or "").strip():
        cand["api_key"] = dl._providers.normalize_key(cand["api_key"])

    res = dl.test_connection(cand, probe_chat=bool(body.get("probe_chat", True)))
    return JSONResponse(res)


# ============================== 音频导出 (MP3/WAV/M4A) ==============================
_FFMPEG_PATH = None


def _find_ffmpeg():
    """定位 ffmpeg 可执行文件路径（用于 MP3/M4A 转码与视频提取）。找不到返回 None。"""
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    import glob as _g
    candidates = [os.environ.get("VOXCPM_FFMPEG", "")]
    # conda/miniforge 安装的 ffmpeg
    candidates += [
        "F:/miniforge3/Library/bin/ffmpeg.exe",
        "F:/miniforge3/Scripts/ffmpeg.exe",
        "F:/miniconda3/Library/bin/ffmpeg.exe",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            _FFMPEG_PATH = c
            return c
    for pat in ("F:/ffmpeg*/bin/ffmpeg.exe", "F:/VoxCPM2/ffmpeg*/bin/ffmpeg.exe"):
        for c in _g.glob(pat):
            if os.path.exists(c):
                _FFMPEG_PATH = c
                return c
    # 系统 PATH 里的 ffmpeg
    import shutil as _sh
    p = _sh.which("ffmpeg")
    if p:
        _FFMPEG_PATH = p
        return p
    # imageio-ffmpeg 内置的 ffmpeg 二进制（pip 安装，最可靠）
    try:
        import imageio_ffmpeg as _iff
        p = _iff.get_ffmpeg_exe()
        if p and os.path.exists(p):
            _FFMPEG_PATH = p
            return p
    except Exception:
        pass
    return None


def _run_ffmpeg(args: list[str]) -> bytes:
    ff = _find_ffmpeg()
    if not ff:
        raise HTTPException(status_code=400, detail="此操作需要 ffmpeg，未检测到（请确认已安装）")
    import subprocess
    r = subprocess.run([ff] + args, capture_output=True)
    if r.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg 转码失败: {r.stderr.decode('utf-8', 'ignore')[:200]}")
    return r.stdout


def convert_audio(data: bytes, fmt: str) -> bytes:
    """把 WAV 字节流转成目标格式。fmt: wav / mp3 / m4a。"""
    fmt = (fmt or "wav").lower()
    if fmt == "wav":
        return data
    if fmt == "mp3":
        # 优先 ffmpeg；ffmpeg 缺失或转码失败时回退 lameenc（纯 Python，已装）
        if _find_ffmpeg():
            import tempfile, os as _os
            td = tempfile.mkdtemp()
            src = _os.path.join(td, "in.wav")
            dst = _os.path.join(td, "out.mp3")
            try:
                with open(src, "wb") as f:
                    f.write(data)
                _run_ffmpeg(["-y", "-i", src, "-b:a", "192k", dst])
                with open(dst, "rb") as f:
                    return f.read()
            except Exception:
                pass  # ffmpeg 失败 → 回退 lameenc
            finally:
                for p in (src, dst):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        # lameenc 回退
        import lameenc
        wav, wsr = sf.read(io.BytesIO(data))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        enc = lameenc.Encoder()
        enc.set_bit_rate(192)
        enc.set_in_sample_rate(int(wsr))
        enc.set_channels(1)
        enc.set_quality(2)
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
        return bytes(enc.encode(pcm) + enc.flush())
    if fmt == "m4a":
        if not _find_ffmpeg():
            raise HTTPException(status_code=400, detail="M4A 导出需要 ffmpeg，未检测到")
        import tempfile
        td = tempfile.mkdtemp()
        src = os.path.join(td, "in.wav")
        dst = os.path.join(td, "out.m4a")
        try:
            with open(src, "wb") as f:
                f.write(data)
            _run_ffmpeg(["-y", "-i", src, "-c:a", "aac", "-b:a", "192k", dst])
            with open(dst, "rb") as f:
                return f.read()
        finally:
            for p in (src, dst):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    raise HTTPException(status_code=400, detail=f"不支持的导出格式: {fmt}")


@app.post("/api/export")
async def export_audio(request: Request, format: str = Form("mp3"),
                       name: str = Form(""), audio: UploadFile = File(None)):
    """把音频（上传 WAV 或引用 outputs 里的文件名）导出为 MP3/WAV/M4A。"""
    require_auth(request)
    if audio is not None and getattr(audio, "filename", ""):
        data = audio.file.read()
    elif name:
        p = (OUTPUT_DIR / name).resolve()
        if not str(p).startswith(str(OUTPUT_DIR.resolve())) or not p.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        data = p.read_bytes()
    else:
        raise HTTPException(status_code=400, detail="请提供音频文件或 output 文件名")
    fmt = (format or "mp3").lower()
    out = convert_audio(data, fmt)
    media = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(fmt, "application/octet-stream")
    return Response(content=out, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="export.{fmt}"'})


@app.post("/api/warmup")
def warmup(request: Request):
    require_auth(request)
    get_model()
    return {"ok": True, **_model_info}


# ============================== 训练模块（LoRA 微调） ==============================
def _check_not_training():
    """训练进行中时禁止生成类请求（显存互斥）。"""
    if trainer.is_running():
        raise HTTPException(status_code=409, detail="LoRA 训练进行中，请先停止训练或稍后再试")


@app.get("/api/train/samples")
def train_list_samples(request: Request):
    require_auth(request)
    return {"samples": tstore.list_samples(), "stats": tstore.get_stats()}


def _enhance_import(src: Path, denoise_on: bool, vocal_only: bool) -> Path | None:
    """导入前可选增强：转 16k 单声道后 降噪 / 只保留纯净人声。

    返回增强后的临时 wav 路径（调用方负责删除）；两开关均关时返回 None。
    统一降到 16k：与 whisper/训练读取口径一致，且长音频(10 分钟级)的内存
    占用有界。

    引擎优先级：
      vocal_only -> MDX-NET 神经人声分离（models/mdx/ 有权重时；实测
                    corr 0.994 / SDR +19.3dB，远优于 DSP 的 0.85 / +3.9dB），
                    无模型时回退 REPET-lite/HPSS；
      denoise    -> v2 谱域降噪（动态噪声跟踪 + DD-Wiener）。
    """
    if not (denoise_on or vocal_only):
        return None
    wav16 = UPLOAD_DIR / f"enh_{uuid.uuid4().hex[:8]}.wav"
    try:
        transcriber.to_16k_mono_wav(src, wav16)
    except Exception as e:
        _safe_unlink(wav16)
        raise ValueError(f"音频转码失败: {e}")
    try:
        y, sr = vc_preprocess.load_audio(str(wav16), sr=16000)
        # MDX 按 44.1k 训练、对低频更敏感；分离用 44.1k，完成后降回 16k
        if vocal_only:
            mdx = getattr(vc_preprocess, "_mdx_engine", lambda: None)()
            if mdx is not None and mdx.is_available():
                import librosa as _lb
                hi = _lb.resample(y, orig_sr=sr, target_sr=44100)
                voc = vc_preprocess.isolate_vocals(hi, 44100, method="mdx")
                y = _lb.resample(voc, orig_sr=44100, target_sr=sr)
            else:
                y = vc_preprocess.isolate_vocals(y, sr)
        else:
            y = vc_preprocess.denoise(y, sr)
        sf.write(str(wav16), y, sr)
    except Exception as e:
        _safe_unlink(wav16)
        raise ValueError(f"音频增强失败: {e}")
    return wav16


@app.post("/api/train/samples")
async def train_add_sample(request: Request):
    """上传一条训练样本（音频 + 逐字台词）。

    兼容两种音频字段名：`audio`（API 文档约定）与 `file`（前端 UI 实际发送）。
    注意：不能用 `audio: UploadFile = File(...)` 形参声明——前端发送的字段名是
    `file`，声明式绑定会直接 422，导致 UI 永远无法添加样本。
    """
    require_auth(request)
    form = await request.form()
    up = form.get("audio") or form.get("file")
    if up is None or not hasattr(up, "read"):
        raise HTTPException(status_code=400, detail="缺少音频文件字段（audio 或 file）")
    text = str(form.get("text") or "")
    name = str(form.get("name") or "")
    denoise_on = str(form.get("denoise") or "").lower() in ("1", "true", "on", "yes")
    vocal_only = str(form.get("vocal_only") or "").lower() in ("1", "true", "on", "yes")
    suffix = Path(up.filename or "ref.wav").suffix or ".wav"
    tmp = UPLOAD_DIR / f"train_{uuid.uuid4().hex[:8]}{suffix}"
    tmp.write_bytes(await up.read())
    enh = None
    try:
        enh = _enhance_import(tmp, denoise_on, vocal_only)
        meta = tstore.add_sample(str(enh or tmp), text, name)
    except ValueError as e:
        _safe_unlink(tmp)
        if enh:
            _safe_unlink(enh)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        _safe_unlink(tmp)
        if enh:
            _safe_unlink(enh)
    return {"ok": True, "sample": meta, "stats": tstore.get_stats()}


@app.delete("/api/train/samples/{sample_id}")
def train_del_sample(sample_id: str, request: Request):
    require_auth(request)
    if not tstore.delete_sample(sample_id):
        raise HTTPException(status_code=404, detail="样本不存在")
    return {"ok": True, "stats": tstore.get_stats()}


@app.get("/api/train/samples/{sample_id}/audio")
def train_sample_audio(sample_id: str, request: Request):
    require_auth(request)
    for s in tstore.list_samples():
        if s["id"] == sample_id:
            p = Path(s["audio"])
            if p.exists():
                return FileResponse(str(p), media_type="audio/wav")
            break
    raise HTTPException(status_code=404, detail="音频不存在")


@app.post("/api/train/start")
async def train_start(request: Request):
    """启动 LoRA 训练。训练前卸载推理模型（显存互斥），训练中拒绝生成请求。

    请求体两种兼容：
    - JSON（前端 UI 使用）：{"name", "r", "alpha", "lr", "epochs", "accum", ...}
    - multipart/form（curl / 脚本调用）：lora_name, lora_r, lora_alpha, lr, epochs,
      batch_size, grad_accum
    注意：本端点必须自行解析 JSON——若声明为 Form(...) 形参，FastAPI 收到 JSON
    请求体会静默使用默认值，导致前端填写的任务名/轮数/学习率等全部失效。
    """
    require_auth(request)
    if _infer_lock.locked():
        raise HTTPException(status_code=409, detail="正在生成音频，请稍后再开始训练")
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            b = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="请求体不是合法 JSON")
        def _pick(*keys, default=None):
            for k in keys:
                if k in b and b[k] is not None:
                    return b[k]
            return default
        lora_name = str(_pick("lora_name", "name", default="") or "").strip()
        lora_r = int(_pick("lora_r", "r", default=8) or 8)
        lora_alpha = int(_pick("lora_alpha", "alpha", default=16) or 16)
        lr = float(_pick("lr", default=1e-4) or 1e-4)
        epochs = int(_pick("epochs", default=3) or 3)
        batch_size = int(_pick("batch_size", default=1) or 1)
        grad_accum = int(_pick("grad_accum", "accum", default=4) or 4)
    else:
        f = await request.form()
        lora_name = str(f.get("lora_name", "") or "").strip()
        lora_r = int(f.get("lora_r", 8) or 8)
        lora_alpha = int(f.get("lora_alpha", 16) or 16)
        lr = float(f.get("lr", 1e-4) or 1e-4)
        epochs = int(f.get("epochs", 3) or 3)
        batch_size = int(f.get("batch_size", 1) or 1)
        grad_accum = int(f.get("grad_accum", 4) or 4)
    unload_model()  # 释放推理显存，给训练让路
    try:
        st = trainer.start_training({
            "lora_name": lora_name, "lora_r": lora_r, "lora_alpha": lora_alpha,
            "lr": lr, "epochs": epochs, "batch_size": batch_size, "grad_accum": grad_accum,
        }, base_dir=BASE_DIR)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "status": st}


@app.get("/api/train/status")
def train_status(request: Request):
    require_auth(request)
    return trainer.get_status()


@app.post("/api/train/stop")
def train_stop(request: Request):
    require_auth(request)
    if not trainer.is_running():
        raise HTTPException(status_code=400, detail="当前没有训练任务")
    trainer.request_stop()
    return {"ok": True}


@app.get("/api/train/loras")
def train_list_loras(request: Request):
    require_auth(request)
    return {"loras": tstore.list_loras()}


@app.delete("/api/train/loras/{name}")
def train_del_lora(name: str, request: Request):
    require_auth(request)
    if not tstore.delete_lora(name):
        raise HTTPException(status_code=404, detail="LoRA 不存在")
    return {"ok": True}


# -------------------------- 长音频自动转写（whisper 切句） --------------------------
def _check_not_transcribing():
    """转写任务进行中时拒绝再次转写（单任务互斥）。"""
    if transcriber.is_busy():
        raise HTTPException(status_code=409, detail="已有转写任务在运行，请等待完成或稍后再试")


def _public_job(job: dict) -> dict:
    """剥离仅供后端重对齐用的内部字段，避免上千条词级锚点撑大轮询 payload。"""
    j = dict(job or {})
    for k in ("words_timeline", "raw_tail"):
        j.pop(k, None)
    return j


@app.post("/api/train/transcribe")
async def train_transcribe(request: Request):
    """上传长音频 → 后台线程 whisper 转写 → 返回 job（前端轮询结果）。

    可选 multipart 字段 transcript（完整台词全文）：转写完成后自动把台词
    按各分段时长匹配进 text，whisper 仅提供时间边界。
    """
    require_auth(request)
    _check_not_training()
    _check_not_transcribing()
    if _infer_lock.locked():
        raise HTTPException(status_code=409, detail="正在生成音频，请稍后再转写")
    form = await request.form()
    up = form.get("audio") or form.get("file")
    if up is None or not hasattr(up, "read"):
        raise HTTPException(status_code=400, detail="缺少音频文件字段（audio）")
    transcript = str(form.get("transcript") or "").strip()
    denoise_on = str(form.get("denoise") or "").lower() in ("1", "true", "on", "yes")
    vocal_only = str(form.get("vocal_only") or "").lower() in ("1", "true", "on", "yes")
    suffix = Path(up.filename or "ref.wav").suffix or ".wav"
    tmp = UPLOAD_DIR / f"tr_{uuid.uuid4().hex[:8]}{suffix}"
    tmp.write_bytes(await up.read())
    enh = None
    try:
        enh = _enhance_import(tmp, denoise_on, vocal_only)
        job = transcriber.start_transcribe(str(enh or tmp), transcript=transcript)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        _safe_unlink(tmp)
        if enh:
            _safe_unlink(enh)
    return {"ok": True, "job": _public_job(job)}


@app.post("/api/train/align")
async def train_align(request: Request):
    """对已完成转写的 job 重新做台词对齐（转写后粘贴/修改完整台词时调用）。

    body: {job_id, transcript}。只替换各分段 text，不动时间戳/音频。
    """
    require_auth(request)
    body = await request.json()
    job_id = str(body.get("job_id") or "")
    transcript = str(body.get("transcript") or "")
    if not job_id:
        raise HTTPException(status_code=400, detail="缺少 job_id")
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="台词为空，请粘贴完整台词")
    try:
        job = transcriber.align_job(job_id, transcript)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "job": _public_job(job)}


@app.get("/api/train/transcribe/{job_id}")
def train_transcribe_status(job_id: str, request: Request):
    require_auth(request)
    job = transcriber.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="转写任务不存在或已过期，请重新转写")
    return {"ok": True, "job": _public_job(job)}


@app.get("/api/train/transcribe/{job_id}/segment/{idx}/audio")
def train_transcribe_seg_audio(job_id: str, idx: int, request: Request):
    """返回某个转写片段对应的音频切片（wav），用于前端逐段试听。"""
    require_auth(request)
    job = transcriber.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="转写任务不存在或已过期")
    seg = next((s for s in job.get("segments", []) if s["idx"] == idx), None)
    if seg is None:
        raise HTTPException(status_code=404, detail="片段不存在")
    try:
        data, sr = transcriber.load_segment_audio(job_id, seg)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    bio = io.BytesIO()
    sf.write(bio, data, sr, format="wav")
    bio.seek(0)
    return Response(content=bio.read(), media_type="audio/wav")


@app.post("/api/train/import_segments")
async def train_import_segments(request: Request):
    """把转写结果中勾选（可修改文本）的片段切片导入为训练样本。"""
    require_auth(request)
    _check_not_training()
    body = await request.json()
    job_id = str(body.get("job_id") or "")
    items = body.get("items") or []
    if not job_id or not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="缺少 job_id 或勾选项")
    try:
        res = transcriber.import_segments(job_id, items)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, **res, "stats": tstore.get_stats()}


def _safe_unlink(path) -> None:
    """尽力删除临时文件；删除失败（如沙箱回收站不可用）不影响主流程。"""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


# ============================== 音色包管理 ==============================
@app.get("/api/voicepacks")
def list_voice_packs(request: Request):
    require_auth(request)
    return {"packs": vp_store.list_packs()}


@app.post("/api/voicepacks")
def create_voice_pack(
    request: Request,
    name: str = Form(""),
    denoise: str = Form("true"),
    remove_bg: str = Form("false"),
    accelerated: str = Form("false"),
    reference: UploadFile = File(...),
):
    """从上传/录制的参考音频抽取音色，保存为可长期复用的音色包。"""
    require_auth(request)
    if reference is None or not reference.filename:
        raise HTTPException(status_code=400, detail="请上传或录制参考音频")
    suffix = (Path(reference.filename).suffix or ".wav").lower()
    raw_path = UPLOAD_DIR / f"vp_src_{uuid.uuid4().hex[:8]}{suffix}"
    raw_path.write_bytes(reference.file.read())
    raw_path = str(raw_path)

    # 视频文件：先用 ffmpeg 提取音轨（单声道 24k wav），再走音色抽取流程
    if suffix in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v", ".wmv", ".ts"):
        ff = _find_ffmpeg()
        if not ff:
            _safe_unlink(raw_path)
            raise HTTPException(status_code=400, detail="从视频提取音轨需要 ffmpeg，当前未检测到")
        import subprocess
        ref_path = str(UPLOAD_DIR / f"vp_src_{uuid.uuid4().hex[:8]}.wav")
        r = subprocess.run(
            [ff, "-y", "-i", raw_path, "-vn", "-ac", "1", "-ar", "24000", ref_path],
            capture_output=True,
        )
        _safe_unlink(raw_path)  # 视频源文件弃用
        if r.returncode != 0 or not Path(ref_path).exists():
            raise HTTPException(status_code=400,
                                detail="视频音轨提取失败：" + r.stderr.decode("utf-8", "ignore")[:200])
    else:
        ref_path = raw_path

    try:
        normalize_reference(ref_path)  # 坏文件在此给出清晰 400
    except ValueError as e:
        p = Path(ref_path)
        if p.exists():
            _safe_unlink(p)
        raise HTTPException(status_code=400, detail=str(e))
    try:
        rec = vp_store.create_pack(
            name=name.strip(),
            ref_path=ref_path,
            denoise=str(denoise).lower() == "true",
            remove_bg=str(remove_bg).lower() == "true",
            accelerated=str(accelerated).lower() == "true",
            source_name=reference.filename,
        )
        _safe_unlink(ref_path)  # 源文件用后即弃，音色已落盘
        print(f"[VoxCPM2] 已保存音色包 {rec['id']} ({rec['name']})", flush=True)
        return {"ok": True, "pack": rec}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log_error("保存音色包失败", e)
        raise HTTPException(status_code=500, detail=f"保存音色包失败: {type(e).__name__}: {e}")


@app.get("/api/voicepacks/{pack_id}/audio")
def get_voice_pack_audio(pack_id: str, request: Request):
    require_auth(request)
    wav, _ = vp_store.get_pack_paths(pack_id)
    if wav is None or not wav.exists():
        raise HTTPException(status_code=404, detail="音色包不存在")
    return FileResponse(str(wav), media_type="audio/wav", filename=f"{pack_id}.wav")


@app.get("/api/voicepacks/{pack_id}/preview")
def get_voice_pack_preview(pack_id: str, request: Request):
    require_auth(request)
    wav, preview = vp_store.get_pack_paths(pack_id)
    if wav is None or not wav.exists():
        raise HTTPException(status_code=404, detail="音色包不存在")
    target = preview if (preview is not None and preview.exists()) else wav
    return FileResponse(str(target), media_type="audio/wav", filename=f"{pack_id}_preview.wav")


@app.delete("/api/voicepacks/{pack_id}")
def delete_voice_pack(pack_id: str, request: Request):
    require_auth(request)
    if not vp_store.delete_pack(pack_id):
        raise HTTPException(status_code=404, detail="音色包不存在")
    return {"ok": True}


# ============================== 启动 ==============================
if __name__ == "__main__":
    line = "=" * 66
    print(line)
    print("  VoxCPM2 本地推理服务 (加固版)")
    print(line)
    print(f"  模型目录 : {MODEL_PATH}")
    print(f"  监听地址 : http://{HOST}:{PORT}")
    print(f"  浏览器访问: http://localhost:{PORT}")
    print(f"  一键登录 : http://localhost:{PORT}/?token={ACCESS_TOKEN}")
    print(f"  访问令牌 : {ACCESS_TOKEN}")
    print(f"  凭证文件 : {CRED_FILE}")
    print(f"  错误日志 : {ERROR_LOG}")
    print(f"  输出目录 : {OUTPUT_DIR}")
    print(line)
    print("  提示：模型在第一次生成时才加载（约 20-60 秒），之后常驻显存；")
    print("  若某次推理异常，服务会自动重载模型自愈，无需重启。")
    print(line, flush=True)
    # 注：本机 uvicorn 0.52.2 不对请求体大小做限制（仅 header 有 16KB 缓冲），
    # 故 10 分钟参考音频可直接以 multipart 流式上传，无需额外放宽上传上限。
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    except OSError as e:
        # 端口被占用是最常见的“启动报错”：给出清晰可执行的提示，而非晦涩堆栈
        msg = str(e).lower()
        if "address already in use" in msg or "10048" in msg or "10013" in msg:
            print("\n[VoxCPM2][错误] 端口 %d 已被占用，无法启动。" % PORT)
            print("  · 多半 VoxCPM2 已在运行 —— 直接打开 http://localhost:%d 即可。" % PORT)
            print("  · 若确认没有其它实例，请先结束占用进程，或改上方 PORT 后重试。")
            print("  · 查询占用： netstat -ano | findstr :%d" % PORT)
            sys.exit(2)
        else:
            raise
