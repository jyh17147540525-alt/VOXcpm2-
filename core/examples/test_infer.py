"""模型加载 + 推理自检脚本。

用法（在项目根目录执行，需先按 README 下载好模型权重）：
    python examples/test_infer.py
"""
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import torch
import soundfile as sf
from voxcpm import VoxCPM

# 模型权重目录：默认当前项目根目录（可用环境变量 VOXCPM_HOME 覆盖）
MODEL = Path(__file__).resolve().parent.parent

print("=" * 60)
print("显卡:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 CUDA")
print(f"加载模型（目录：{MODEL}）...")
t0 = time.time()
model = VoxCPM.from_pretrained(str(MODEL), load_denoiser=False, device="auto")
t_load = time.time() - t0
print(f"加载完成: {t_load:.1f}s")
print("采样率:", model.tts_model.sample_rate)
print(f"显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB "
      f"(峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB)")

text = "（年轻女性，温柔甜美）你好，这里是 VoxCPM2 语音合成自检，现在一切正常。"
print("\n生成测试:", text)
t1 = time.time()
wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10, normalize=True)
t_gen = time.time() - t1
sr = model.tts_model.sample_rate
dur = len(wav) / sr
out = MODEL / "outputs" / "selftest.wav"
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out), wav, sr)
print(f"生成完成: 耗时 {t_gen:.2f}s, 音频时长 {dur:.2f}s, RTF={t_gen/dur:.2f}")
print(f"显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print("输出文件:", out)
print("=" * 60)
print("SELFTEST_OK")
