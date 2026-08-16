"""VoxCPM2 推理诊断脚本：逐项验证 design / clone / hifi 各模式。

用法（在项目根目录执行，需先下载模型权重）：
    python examples/diag.py [参考音频路径]

若不传参考音频，clone / hifi 测试会跳过。
"""
import os
import sys
import io
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_PATH = str(Path(__file__).resolve().parent.parent)

import numpy as np
import soundfile as sf


def run(label, **kw):
    print("\n" + "=" * 60)
    print(">>> " + label)
    print(">>> kwargs:", {k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
                         for k, v in kw.items()})
    print("=" * 60, flush=True)
    try:
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained(MODEL_PATH, load_denoiser=False, device="auto")
        wav = model.generate(**kw)
        if isinstance(wav, list):
            wav = np.concatenate(wav)
        sr = int(model.tts_model.sample_rate)
        out = "diag_out.wav"
        sf.write(out, wav, sr)
        print(f"<<< 成功! 时长={len(wav)/sr:.2f}s 采样率={sr} 已写 {out}", flush=True)
        return True
    except Exception as e:
        print("<<< 异常类型:", type(e).__name__)
        print("<<< 异常信息:", repr(str(e))[:800])
        traceback.print_exc()
        return False


# 1) design 模式（零样本 TTS）
run("DESIGN 模式", text="这是设计模式诊断。", cfg_value=2.0, inference_timesteps=10,
    normalize=True, denoise=False)

# 2) clone / hifi 模式（需提供参考音频）
ref = sys.argv[1] if len(sys.argv) > 1 else ""
if ref and Path(ref).exists():
    run("CLONE 模式", text="这是克隆波形诊断。", cfg_value=2.0, inference_timesteps=10,
        normalize=True, denoise=False, reference_wav_path=ref)
else:
    print("\n[跳过] 未提供参考音频，clone / hifi 测试跳过。"
          "用法：python examples/diag.py 你的参考音频.wav")

print("\n##### 诊断结束 #####", flush=True)
