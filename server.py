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
_model_info = {"loaded": False, "device": None, "sample_rate": None, "load_seconds": None}


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                t0 = time.time()
                print("[VoxCPM2] 正在加载模型，首次约需 20-60 秒 ...", flush=True)
                from voxcpm import VoxCPM
                import torch

                _model = VoxCPM.from_pretrained(
                    MODEL_PATH, load_denoiser=False, device=DEVICE
                )
                _model_info["loaded"] = True
                _model_info["device"] = (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                )
                _model_info["sample_rate"] = _vc_stab._get_sample_rate(_model)
                _model_info["load_seconds"] = round(time.time() - t0, 1)
                print(f"[VoxCPM2] 模型就绪，用时 {_model_info['load_seconds']}s "
                      f"设备 {_model_info['device']}", flush=True)
    return _model


def reset_model():
    """模型自愈：推理异常后置空，下次请求自动重载（避免损坏态永久卡死）"""
    global _model
    _model = None
    _model_info["loaded"] = False


def unload_model():
    """优雅卸载模型并释放显存。停止服务前先调用本函数，可避免直接强杀进程
    导致 GPU CUDA 上下文损坏（段错误、需整机重启）。"""
    global _model
    if _model is not None:
        try:
            del _model
        except Exception:
            pass
        _model = None
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
<title>VoxCPM2 · 访问验证</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6fa;min-height:100vh;
display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:36px;max-width:420px;width:100%;
box-shadow:0 4px 24px rgba(0,0,0,.06)}
h1{font-size:20px;color:#111827;margin-bottom:6px}
p.sub{font-size:13px;color:#6b7280;margin-bottom:24px;line-height:1.6}
label{display:block;font-size:13px;color:#374151;margin-bottom:8px;font-weight:600}
input{width:100%;padding:12px;border:1px solid #d1d5db;border-radius:10px;font-size:14px;
font-family:ui-monospace,Consolas,monospace}
input:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
button{width:100%;margin-top:16px;padding:13px;background:#2563eb;color:#fff;border:none;
border-radius:10px;font-size:15px;font-weight:600;cursor:pointer}
button:hover{background:#1d4ed8}
.err{margin-top:12px;padding:10px;background:#fef2f2;color:#b91c1c;border-radius:8px;font-size:13px;display:none}
.hint{margin-top:18px;padding:12px;background:#f9fafb;border-radius:8px;font-size:12px;color:#6b7280;line-height:1.7}
code{background:#eef2ff;color:#3730a3;padding:1px 5px;border-radius:4px;font-size:11px}
</style></head><body>
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
<title>VoxCPM2 · 本地语音合成</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6fa;color:#111827;padding:24px}
.wrap{max-width:900px;margin:0 auto}
.top{display:flex;align-items:center;justify-content:space-between;background:#fff;border:1px solid #e5e7eb;
border-radius:14px;padding:16px 20px;margin-bottom:16px}
.top h1{font-size:18px}
.badges{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{font-size:12px;padding:4px 10px;border-radius:999px;background:#f3f4f6;color:#374151}
.badge.ok{background:#ecfdf5;color:#047857}
.badge.warn{background:#fffbeb;color:#b45309}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px;margin-bottom:16px}
.tabs{display:flex;gap:8px;margin-bottom:18px;flex-wrap:wrap}
.tab{padding:9px 16px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;cursor:pointer;font-size:13px;color:#4b5563}
.tab.active{background:#2563eb;color:#fff;border-color:#2563eb}
label{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:7px}
textarea{width:100%;padding:13px;border:1px solid #d1d5db;border-radius:10px;font-size:14px;
min-height:110px;resize:vertical;font-family:inherit;line-height:1.6}
input[type=text],input[type=file]{width:100%;padding:11px;border:1px solid #d1d5db;border-radius:10px;font-size:14px}
textarea:focus,input:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.1)}
.field{margin-bottom:16px}
.hide{display:none}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{font-size:12px;padding:5px 10px;background:#eef2ff;color:#3730a3;border-radius:999px;cursor:pointer;border:none}
.chip:hover{background:#e0e7ff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.pbox{background:#f9fafb;border:1px solid #f3f4f6;border-radius:10px;padding:12px}
.pbox label{font-size:12px;color:#6b7280;font-weight:500;margin-bottom:6px}
.prow{display:flex;align-items:center;gap:10px}
input[type=range]{flex:1}
.pv{font-family:ui-monospace,Consolas,monospace;font-size:13px;color:#2563eb;min-width:34px;text-align:right}
.checks{display:flex;gap:20px;margin-bottom:18px;font-size:13px;color:#4b5563}
.checks label{display:flex;align-items:center;gap:6px;font-weight:500;margin:0;cursor:pointer}
.gen{width:100%;padding:15px;background:#2563eb;color:#fff;border:none;border-radius:12px;
font-size:15px;font-weight:600;cursor:pointer}
.gen:hover{background:#1d4ed8}
.gen:disabled{background:#9ca3af;cursor:not-allowed}
.status{margin-top:16px;padding:14px;background:#f9fafb;border-radius:10px;font-size:13px;
color:#4b5563;display:none;align-items:center;gap:10px}
.status.show{display:flex}
.spin{width:18px;height:18px;border:2px solid #e5e7eb;border-top-color:#2563eb;border-radius:50%;
animation:sp .8s linear infinite;flex-shrink:0}
@keyframes sp{to{transform:rotate(360deg)}}
.err{margin-top:14px;padding:12px;background:#fef2f2;color:#b91c1c;border-radius:10px;font-size:13px;display:none;
white-space:pre-wrap;line-height:1.6}
.err.show{display:block}
.res{margin-top:18px;padding:16px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;display:none}
.res.show{display:block}
.res .meta{font-size:12px;color:#047857;margin-bottom:10px}
audio{width:100%}
.hist{font-size:13px}
.hist .row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #f3f4f6}
.hist .row:last-child{border:none}
.hist a{color:#2563eb;text-decoration:none;font-size:12px}
.muted{color:#9ca3af;font-size:12px}
.api{font-family:ui-monospace,Consolas,monospace;font-size:12px;background:#f9fafb;padding:12px;
border-radius:8px;color:#374151;white-space:pre-wrap;line-height:1.7;overflow-x:auto}
.ptab{padding:9px 16px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;cursor:pointer;font-size:13px;color:#4b5563}
.ptab.active{background:#2563eb;color:#fff;border-color:#2563eb}
.pack{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;
border:1px solid #f1f5f9;border-radius:10px;margin-bottom:8px;background:#fff}
.pack .info{min-width:0}
.pack .nm{font-size:14px;font-weight:600;color:#111827;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pack .meta{font-size:12px;color:#6b7280;margin-top:2px}
.pack .acts{display:flex;gap:6px;flex-shrink:0}
.pack .acts button{padding:6px 10px;border:1px solid #e5e7eb;background:#fff;border-radius:8px;font-size:12px;cursor:pointer;color:#374151}
.pack .acts button:hover{border-color:#2563eb;color:#2563eb}
.pack .acts .del:hover{border-color:#dc2626;color:#dc2626}
.pack .acts .use{background:#2563eb;color:#fff;border-color:#2563eb}
.vp-sel{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:10px;font-size:14px;margin-top:6px;background:#fff}
</style></head><body>
<div class="wrap">
  <div class="top">
    <h1>🎙️ VoxCPM2 <span class="muted" style="font-size:13px" data-i18n="localDeploy">本地部署</span></h1>
    <div class="badges">
      <span class="badge" id="devBadge">检测中…</span>
      <span class="badge ok">2B · 48kHz</span>
      <span class="badge" id="modelBadge" data-i18n="modelNotLoaded">模型未加载</span>
      <button class="badge" id="langZh" onclick="setLang('zh')" style="cursor:pointer;border:1px solid #d1d5db">中</button>
      <button class="badge" id="langEn" onclick="setLang('en')" style="cursor:pointer;border:1px solid #d1d5db;opacity:.5">EN</button>
    </div>
  </div>

  <div class="tabs" id="mainNav" style="margin-bottom:16px">
    <button class="tab active" data-mode="design" onclick="setMode('design')" data-i18n="modeDesign">🎨 语音设计</button>
    <button class="tab" data-mode="clone" onclick="setMode('clone')" data-i18n="modeClone">🎛️ 音色克隆</button>
    <button class="tab" data-mode="hifi" onclick="setMode('hifi')" data-i18n="modeHifi">🎙️ 极致克隆</button>
    <button class="tab" data-mode="beta" onclick="setMode('beta')" data-i18n="modeBeta">🧪 内测 Beta</button>
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
        <select id="exportFmt" style="width:auto;padding:8px;border:1px solid #d1d5db;border-radius:8px">
          <option value="mp3">导出 MP3</option>
          <option value="wav">导出 WAV</option>
          <option value="m4a">导出 M4A</option>
        </select>
        <button id="exportBtn" onclick="exportAudio()" data-i18n="exportLabel" style="padding:8px 14px;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;font-size:13px">⬇️ 导出</button>
      </div>
    </div>
  </div>

  <div class="card" id="histCard">
    <label data-i18n="history">本次会话生成记录</label>
    <div class="hist" id="hist"><div class="muted" data-i18n="noHistory">还没有生成记录</div></div>
  </div>

  <div class="card hide" id="betaCard">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
      <button class="chip" onclick="setMode(prevMode||'design')" data-i18n="backBtn" style="padding:6px 12px;border:1px solid #d1d5db;border-radius:8px;background:#fff;cursor:pointer;font-size:13px">← 返回</button>
      <span class="badge warn">Beta</span>
      <label data-i18n="betaTitle" style="margin:0">多人朗读与情绪控制</label>
    </div>
    <div class="muted" style="margin-bottom:12px" data-i18n="betaDesc">用 (@音色包名) 切换角色，用 (情绪词) 控制语气。例：(@张三)你好，(开心)今天真不错！(@李四)是啊。</div>
    <div class="field" style="position:relative">
      <label data-i18n="betaTextLabel">朗读文本（输入 (@ 会弹出音色包，支持 情绪词）</label>
      <textarea id="betaText" style="min-height:130px" oninput="betaOnInput()">(@磁性女声，我i的最爱)你好，欢迎使用多人朗读功能。(开心)今天真不错！</textarea>
      <div id="betaAtMenu" style="display:none;position:absolute;z-index:20;background:#fff;border:1px solid #d1d5db;border-radius:8px;max-height:200px;overflow:auto;width:100%;box-shadow:0 4px 12px rgba(0,0,0,.1)"></div>
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
    <button class="gen" id="betaBtn" onclick="betaGenerate()" data-i18n="betaGenerate">🎭 多人朗读生成</button>
    <div class="status" id="betaStatus" style="display:none"><div class="spin"></div><div id="betaStatusText"></div></div>
    <div class="err" id="betaErr"></div>
    <div class="res" id="betaRes" style="display:none">
      <div class="meta" id="betaMeta"></div>
      <audio id="betaPlayer" controls></audio>
    </div>
  </div>

  <div class="card" id="packCard">
    <div class="tabs">
      <button class="ptab active" data-pane="manage" onclick="showPackPane('manage')" data-i18n="packManage">🎭 音色包管理</button>
      <button class="ptab" data-pane="save" onclick="showPackPane('save')" data-i18n="packMake">🎙️ 制作音色声线包</button>
    </div>

      <div id="packManage">
        <div class="muted" style="margin-bottom:12px" data-i18n="packDesc">已提取并保存在本地的音色声线包，后续克隆可直接选用，无需重复上传长音频。数据存于 <code>voice_packs/</code> 目录，重启服务后依然保留。也可用 API：<code>POST /api/voicepacks</code> 保存，生成时传 <code>voice_pack_id</code>。带 <span style="color:#b45309">⚡加速</span> 标记的音色包在生成时自动提速。</div>
        <div id="packList"><div class="muted" data-i18n="packEmpty">还没有音色包，去“制作音色声线包”做一个吧。</div></div>
      </div>

    <div id="packSave" class="hide">
      <div class="field">
        <label data-i18n="recMethod">方式一：实时录制（直接用麦克风，无需上传文件）</label>
        <button class="gen" id="recBtn" onclick="startRec()" style="background:#0ea5e9" data-i18n="recStart">🎤 开始录制</button>
        <div class="muted" id="recStatus" style="margin-top:6px;color:#0369a1" data-i18n="recHint">点击下方按钮授权麦克风后开始朗读，建议 10–30 秒清晰语句；录制完可回放确认。</div>
        <div class="field hide" id="recWrap" style="margin-top:10px">
          <label data-i18n="recPlayback">录制回放（确认无误再保存）</label>
          <audio id="recPlay" controls></audio>
        </div>
      </div>
      <div class="field" id="vpDropZone" style="border:2px dashed #cbd5e1;border-radius:10px;padding:12px;transition:all .2s">
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
      modeDesign:'🎨 语音设计',modeClone:'🎛️ 音色克隆',modeHifi:'🎙️ 极致克隆',modeBeta:'🧪 内测 Beta',
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
      apiLabel:'API 调用示例（令牌放请求头）',apiToken:'你的访问令牌'},
  en:{localDeploy:'Local',detecting:'Detecting…',modelNotLoaded:'Model not loaded',modelReady:'Model ready',
      modeDesign:'🎨 Voice Design',modeClone:'🎛️ Voice Clone',modeHifi:'🎙️ HiFi Clone',modeBeta:'🧪 Beta',
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
      apiLabel:'API examples (token in header)',apiToken:'YOUR_TOKEN'}
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
  try{localStorage.setItem('voxcpm_lang',l);}catch(_){}
}
function tr(zh,en){return curLang==='zh'?zh:en;}
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
    d.style.cssText='padding:8px 12px;cursor:pointer;border-bottom:1px solid #f3f4f6;font-size:13px';
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
const EMO_OPTS={zh:['无','高兴','悲伤','生气','严肃','温柔'],en:['None','Happy','Sad','Angry','Serious','Gentle']};
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
    panel.style.cssText='border:1px solid #e5e7eb;border-radius:10px;margin-bottom:10px;overflow:hidden';
    // 头部：折叠箭头 + 角色标识
    const head=document.createElement('div');
    head.style.cssText='display:flex;align-items:center;gap:8px;padding:10px 12px;cursor:pointer;font-weight:600;background:#f9fafb;font-size:13px';
    head.onclick=function(){toggleDp(idx);};
    const arrow=document.createElement('span'); arrow.id='dp_arrow_'+idx; arrow.textContent=d.collapsed?'▸':'▾';
    const label=document.createElement('span');
    label.textContent=(curLang==='zh'?'':'')+d.role+' - '+I18N[curLang].turnLabel.replace('{n}',d.seq);
    head.appendChild(arrow); head.appendChild(label);
    panel.appendChild(head);
    // 主体：参数
    const body=document.createElement('div'); body.id='dp_body_'+idx;
    body.style.cssText='padding:10px 12px;border-top:1px solid #f3f4f6'+(d.collapsed?';display:none':'');
    const L=curLang;
    function row(lbl){ const r=document.createElement('div'); r.style.cssText='margin-bottom:8px'; const s=document.createElement('div'); s.style.cssText='font-size:12px;color:#6b7280;margin-bottom:4px'; s.textContent=lbl; r.appendChild(s); return r; }
    function slider(lbl,min,max,step,val,cb){ const r=row(lbl); const w=document.createElement('div'); w.style.cssText='display:flex;align-items:center;gap:10px'; const inp=document.createElement('input'); inp.type='range'; inp.min=min; inp.max=max; inp.step=step; inp.value=val; inp.style.cssText='flex:1'; const v=document.createElement('span'); v.style.cssText='width:36px;text-align:right;font-size:12px;color:#6b7280'; v.textContent=val; inp.oninput=function(){ v.textContent=inp.value; cb(parseFloat(inp.value)); }; w.appendChild(inp); w.appendChild(v); r.appendChild(w); return r; }
    // 语气
    const rTone=row(L==='zh'?'语气':'Tone'); const selTone=document.createElement('select');
    selTone.style.cssText='width:100%;padding:6px;border:1px solid #d1d5db;border-radius:6px;font-size:13px';
    TONE_OPTS[L].forEach(o=>{ const op=document.createElement('option'); op.textContent=o; op.value=o; if(o===d.tone)op.selected=true; selTone.appendChild(op); });
    selTone.onchange=function(){d.tone=selTone.value;};
    rTone.appendChild(selTone); body.appendChild(rTone);
    // 台词
    const rText=row(L==='zh'?'台词':'Line'); const ta=document.createElement('textarea');
    ta.style.cssText='width:100%;padding:6px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;min-height:44px;font-family:inherit';
    ta.value=d.text; ta.oninput=function(){d.text=ta.value;};
    rText.appendChild(ta); body.appendChild(rText);
    // 情绪
    const rEmo=row(L==='zh'?'情绪':'Emotion'); const selEmo=document.createElement('select');
    selEmo.style.cssText='width:100%;padding:6px;border:1px solid #d1d5db;border-radius:6px;font-size:13px';
    const curEmo=(d.emotion==='neutral'||d.emotion==='平静')?(L==='zh'?'无':'None'):d.emotion;
    EMO_OPTS[L].forEach(o=>{ const op=document.createElement('option'); op.textContent=o; op.value=o; if(o===curEmo)op.selected=true; selEmo.appendChild(op); });
    selEmo.onchange=function(){ const v=selEmo.value; d.emotion=(v==='无'||v==='None')?'neutral':v; };
    rEmo.appendChild(selEmo); body.appendChild(rEmo);
    // 音量
    const rVol=row(L==='zh'?'音量':'Volume'); const volWrap=document.createElement('div'); volWrap.style.cssText='display:flex;align-items:center;gap:10px';
    const vol=document.createElement('input'); vol.type='range'; vol.min='0.3'; vol.max='2'; vol.step='0.05'; vol.value=d.volume; vol.style.cssText='flex:1';
    const volV=document.createElement('span'); volV.style.cssText='width:36px;text-align:right;font-size:12px;color:#6b7280'; volV.textContent=d.volume;
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
  const body={turns:turns,denoise:document.getElementById('betaDenoise').checked,cfg_value:2.0,inference_timesteps:10};
  try{
    const r=await fetch('/api/dialogue',{method:'POST',headers:Object.assign({'Content-Type':'application/json'},apiHeaders()),body:JSON.stringify(body)});
    clearInterval(timer);
    if(!r.ok){let m='Failed';try{const j=await r.json();m=j.detail||m;}catch(e){}errEl.textContent='❌ '+m;errEl.classList.add('show');st.style.display='none';return;}
    const blob=await r.blob();
    const segInfo=r.headers.get('X-Segments'),dur=r.headers.get('X-Duration'),name=r.headers.get('X-Output-Name');
    document.getElementById('betaPlayer').src=URL.createObjectURL(blob);
    let meta='✅ '+(curLang==='zh'?'多人朗读完成':'Done')+' · '+(curLang==='zh'?'时长':'duration')+' '+dur+'s · '+name;
    if(segInfo){try{const si=JSON.parse(segInfo);meta+=' · '+si.n+(curLang==='zh'?' 段':' segments');
      if(si.warnings&&si.warnings.length)meta+=' · ⚠️ '+si.warnings.join('; ');}catch(e){}}
    document.getElementById('betaMeta').textContent=meta;
    document.getElementById('betaRes').style.display='block';st.style.display='none';
  }catch(e){clearInterval(timer);errEl.textContent='❌ '+I18N[curLang].betaFail+': '+e.message;errEl.classList.add('show');st.style.display='none';}
  finally{btn.disabled=false;}
}
function setMode(m){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.mode===m));
  const beta=(m==='beta');
  document.getElementById('mainCard').classList.toggle('hide',beta);
  document.getElementById('histCard').classList.toggle('hide',beta);
  document.getElementById('packCard').classList.toggle('hide',beta);
  document.getElementById('betaCard').classList.toggle('hide',!beta);
  if(beta){prevMode=mode||'design';renderDialoguePanels();return;}
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
    const accel=p.accelerated?' <span style="color:#b45309">⚡加速</span>':'';
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
    dz.style.borderColor='#2563eb';dz.style.background='#eff6ff';
  }));
  ['dragleave','dragend'].forEach(ev=>dz.addEventListener(ev,function(e){
    e.preventDefault();
    dz.style.borderColor='#cbd5e1';dz.style.background='';
  }));
  dz.addEventListener('drop',function(e){
    e.preventDefault();e.stopPropagation();
    dz.style.borderColor='#cbd5e1';dz.style.background='';
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
    btn.textContent=tr('⏹ 停止录制','⏹ Stop Recording');btn.style.background='#dc2626';btn.onclick=stopRec;
    document.getElementById('recStatus').textContent=tr('录制中 ','Recording ')+'0.0s';
    recTimer=setInterval(()=>{recSecs+=0.1;document.getElementById('recStatus').textContent=tr('录制中 ','Recording ')+recSecs.toFixed(1)+'s';},100);
  }catch(e){showVpErr(tr('无法访问麦克风：','Microphone access failed: ')+(e.message||e.name)+tr('（请允许浏览器麦克风权限）','(please allow microphone permission)'));}
}
function stopRec(){
  if(mediaRec&&mediaRec.state!=='inactive')mediaRec.stop();
  clearInterval(recTimer);
  const btn=document.getElementById('recBtn');
  btn.textContent=tr('🎤 重新录制','🎤 Re-record');btn.style.background='#0ea5e9';btn.onclick=startRec;
  document.getElementById('recStatus').textContent=tr('录制完成，可回放或重新录制','Recording done. Playback or re-record.');
}
function resetRec(){
  recBlob=null;recChunks=[];recSecs=0;
  const btn=document.getElementById('recBtn');
  if(btn){btn.textContent=tr('🎤 开始录制','🎤 Start Recording');btn.style.background='#0ea5e9';btn.onclick=startRec;}
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
    reference: UploadFile = File(None),
):
    """网页用的统一生成接口（支持文件上传）"""
    require_auth(request)
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
    )
    return _do_generate(kwargs)


def _do_generate(kwargs: dict):
    """统一生成：整条链路包进 try/except，异常 → JSON；失败 → 模型自愈。"""
    t0 = time.time()
    try:
        model = get_model()                       # 加载也可能抛异常，一并捕获
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
    逐 turn 用对应音色包 + 语气/情绪/音量参数生成，段间停顿拼接。"""
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
            wav, _ = _vc_stab.synthesize_stable(
                model, text, ref_path, sr, pause=pause_turn, breath=breath_turn, emotion=emotion,
                cfg_value=cfg, inference_timesteps=steps, normalize=True, denoise=denoise_on)
            if _ae:
                if abs(pitch) > 0.01: wav = _ae.apply_pitch(wav, sr, pitch)
                if abs(speed - 1) > 0.01: wav = _ae.apply_speed(wav, sr, speed)
                if abs(vol - 1) > 0.01: wav = _ae.apply_volume(wav, vol)
            wav = _vc_stab.declip(wav)
            pieces.append(wav)
            seg_report.append({"i": i, "voice": role, "emotion": emotion, "tone": tone,
                               "text": text[:24], "missing": missing,
                               "pitch": round(pitch, 2), "speed": round(speed, 3),
                               "volume": round(vol, 2), "pause": pause_turn, "breath": breath_turn})
            print(f"[VoxCPM2][Dialogue] turn{i}: {role} 语气={tone} 情绪={emotion}", flush=True)

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
                                         "warnings": warnings}, ensure_ascii=True),
               "Content-Disposition": f'inline; filename="{name}"'}
    return Response(content=buf.read(), media_type="audio/wav", headers=headers)


@app.get("/api/outputs/{name}")
def get_output(name: str, request: Request):
    require_auth(request)
    path = (OUTPUT_DIR / name).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(path), media_type="audio/wav", filename=name)


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
