"""
voice_clone —— 长参考音频语音克隆增强套件
========================================
模块划分：
  preprocess      : 模块1 音频预处理（降噪 / 背景音去除 / 标准化）
  length_adapter  : 模块2 参考音频长度适配（分段 / 声纹提取 / 融合）
  synthesis_stab  : 模块3 合成稳定性保障（限幅防爆音 / 长文分块 / 交叉淡化）
  director        : 模块4 文本梳理 / 导演层（只标注不改写：台词旁白·情绪·停顿）
  director_llm    : 模块5 AI 导演层（LLM 内核，可选，需 llm_config.json 配 Key）
  pipeline        : 管道编排与可调用接口

director_llm 不在下方自动导入 —— 它涉及网络调用与 API Key 读取，
按需显式导入即可：from voice_clone.director_llm import plan_llm

快速使用：
  from voice_clone import prepare_reference, synthesize_stable
  ref_path, report = prepare_reference("long_ref.wav", denoise=True, remove_bg=True)
  audio, rep = synthesize_stable(model, "很长的一段台词……", ref_path, sr_tts=24000)
"""
from . import preprocess, length_adapter, synthesis_stab, director
from .pipeline import (
    prepare_reference,
    synthesize_stable,
    run_demo,
    LONG_AUDIO_THRESHOLD,
    PREPARED_DIR,
)

__all__ = [
    "preprocess", "length_adapter", "synthesis_stab", "director",
    "prepare_reference", "synthesize_stable", "run_demo",
    "LONG_AUDIO_THRESHOLD", "PREPARED_DIR",
]
