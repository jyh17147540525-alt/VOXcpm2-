"""
管道编排与可调用接口 (Pipeline & Callable API)
=============================================
对外最高层 API：
  prepare_reference(input_path, denoise, remove_bg, target_dur=25)
      -> (fused_wav_path, report)
      预处理(降噪/去背景) -> 标准化 -> 长音频自动分段融合，产出“干净有界”的融合参考。
  synthesize_stable(model, text, reference_wav_path, sr_tts, ...)
      -> (audio_np, report)
      长台词分块 -> 同参考逐段生成 -> 后处理(限幅/归一化) -> 交叉淡化拼接。

模块边界清晰：
  预处理   : voice_clone.preprocess
  长度适配 : voice_clone.length_adapter
  合成稳定 : voice_clone.synthesis_stab
"""
from __future__ import annotations
import os
import hashlib
import time
import numpy as np
import soundfile as sf

from . import preprocess, length_adapter, synthesis_stab

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARED_DIR = os.path.join(BASE_DIR, "prepared")
os.makedirs(PREPARED_DIR, exist_ok=True)

# 长参考音频自动适配阈值（秒）：超过即分段融合
LONG_AUDIO_THRESHOLD = 30.0


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def prepare_reference(input_path: str, denoise: bool = True, remove_bg: bool = False,
                      target_dur: float = 25.0, max_ref_seconds: float = 600.0,
                      use_cache: bool = True, cache_key: str | None = None) -> tuple[str, dict]:
    """
    准备克隆用参考音频：
      1) 预处理：可选去背景音、降噪、标准化(采样率/静音/响度)
      2) 长度适配：若 >30s，按静音边界分段 + 声纹离群剔除 + 融合为 ~target_dur 的代表参考
    返回 (融合参考 wav 路径, 报告 dict)。支持按文件哈希+参数缓存，避免重复计算。
    """
    ck = cache_key or _file_hash(input_path)
    opt_tag = f"d{int(denoise)}b{int(remove_bg)}t{int(target_dur)}"
    cache_name = f"{ck}_{opt_tag}.wav"
    cache_path = os.path.join(PREPARED_DIR, cache_name)
    if use_cache and os.path.exists(cache_path):
        report = {"cached": True, "output_path": cache_path}
        y, sr = preprocess.load_audio(cache_path, sr=None)
        report["output_duration"] = round(len(y) / sr, 2)
        return cache_path, report

    t0 = time.time()
    y, sr, pre = preprocess.preprocess_file(
        input_path, denoise_on=denoise, remove_bg_on=remove_bg)
    pre_dur = len(y) / sr

    adaptation = {"status": "passthrough", "reason": "<=30s 无需分段"}
    if pre_dur > LONG_AUDIO_THRESHOLD:
        y, sr, adaptation = length_adapter.fuse_reference(
            y, sr, target_dur=target_dur)
        # 融合后再次标准化，保证送入模型的参考干净一致
        y, sr = preprocess.standardize(y, sr, target_sr=preprocess.TARGET_SR)

    sf.write(cache_path, y, sr)
    report = {
        "cached": False,
        "output_path": cache_path,
        "input_duration": pre.get("input_duration"),
        "output_duration": round(len(y) / sr, 2),
        "preprocess": pre,
        "adaptation": adaptation,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    return cache_path, report


# 直接复用 synthesis_stab 的编排
synthesize_stable = synthesis_stab.synthesize_stable


def run_demo(input_path: str, text: str, model=None, denoise: bool = True,
             remove_bg: bool = False, target_dur: float = 25.0) -> dict:
    """端到端示例：准备参考 + (若提供 model) 稳定合成。便于命令行/测试调用。"""
    ref_path, prep = prepare_reference(input_path, denoise=denoise,
                                       remove_bg=remove_bg, target_dur=target_dur)
    out = {"reference_path": ref_path, "prepare_report": prep}
    if model is not None:
        sr_tts = int(getattr(model.tts_model, "sample_rate", 24000))
        audio, rep = synthesize_stable(model, text, ref_path, sr_tts,
                                       cfg_value=2.0, inference_timesteps=10)
        out_path = os.path.join(PREPARED_DIR, "demo_clone.wav")
        sf.write(out_path, audio, sr_tts)
        out["audio_path"] = out_path
        out["synthesize_report"] = rep
    return out
