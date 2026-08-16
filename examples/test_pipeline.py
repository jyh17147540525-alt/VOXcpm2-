"""语音克隆增强套件自测：合成 60s 含背景音乐+噪声的长参考音频，验证三模块。"""
import os, sys, time
import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from voice_clone import prepare_reference, synthesis_stab

SR = 44100

def gen_fixture(path, dur=60.0):
    rng = np.random.default_rng(0)
    t = np.linspace(0, dur, int(dur*SR), endpoint=False)
    sig = np.zeros_like(t)
    # 背景音乐：持续和弦 (220/277/330Hz) + 缓慢起伏
    chord = (np.sin(2*np.pi*220*t) + np.sin(2*np.pi*277*t)*0.8 + np.sin(2*np.pi*330*t)*0.6)
    chord *= (0.12 + 0.05*np.sin(2*np.pi*0.2*t))
    # 语音段：多段 formant 爆发，段间留静音(便于分段)
    formants = [500, 1500, 2500, 3300]
    seg_sec = 3.0
    gap = 1.0
    pos = 0.5
    while pos < dur - seg_sec:
        s0 = int(pos*SR); s1 = int((pos+seg_sec)*SR)
        tt = np.linspace(0, seg_sec, s1-s0, endpoint=False)
        env = np.sin(np.pi*tt/seg_sec)**1.5  # 起止渐弱
        vib = 1 + 0.02*np.sin(2*np.pi*5*tt)
        seg = np.zeros_like(tt)
        for f in formants:
            seg += np.sin(2*np.pi*f*vib*tt) / len(formants)
        seg *= env * 0.5
        sig[s0:s1] += seg
        pos += seg_sec + gap
    # 噪声
    noise = rng.standard_normal(t.shape) * 0.02
    mix = sig + chord + noise
    mix = mix / np.max(np.abs(mix)) * 0.9
    sf.write(path, mix.astype(np.float32), SR)
    return path

def main():
    fx = os.path.join(BASE, "fixture_long.wav")
    print(">>> 生成测试夹具(60s, 44100Hz, 含背景乐+噪声) ...")
    gen_fixture(fx, dur=60.0)
    print("    夹具:", fx, "存在:", os.path.exists(fx))

    print("\n>>> [A] 长参考 + 去背景 + 降噪 (target_dur=25) ...")
    t0 = time.time()
    p, rep = prepare_reference(fx, denoise=True, remove_bg=True, target_dur=25.0)
    print("    用时 %.2fs" % (time.time()-t0))
    print("    融合参考:", p)
    print("    输入时长: %s s  输出时长: %s s" % (rep.get('input_duration'), rep.get('output_duration')))
    print("    适配报告:", rep.get('adaptation'))
    y, sr = (lambda a: (a[0], a[1]))(sf.read(p))
    print("    输出验证: sr=%d 长度=%d 峰值=%.3f" % (sr, len(y), np.max(np.abs(y))))
    assert len(y) > 0 and np.isfinite(y).all(), "输出音频非法"

    print("\n>>> [B] 长参考 + 仅降噪(不去背景) ...")
    p2, rep2 = prepare_reference(fx, denoise=True, remove_bg=False, target_dur=25.0)
    print("    输出时长: %s s  适配: %s" % (rep2.get('output_duration'), rep2.get('adaptation', {}).get('status')))

    print("\n>>> [C] 合成稳定性组件单测 ...")
    chunks = synthesis_stab.split_long_text("这是第一句。这是第二句，稍微长一点用来验证分块。第三句结束。", max_chars=12)
    print("    长文分块:", chunks)
    # 交叉淡化：两段等长噪声
    a = np.sin(np.linspace(0, 20*np.pi, 24000))
    b = np.sin(np.linspace(0, 20*np.pi, 24000) + 1.0)
    c = synthesis_stab.crossfade_cat([a, b], 24000, fade=0.2)
    print("    交叉淡化拼接: %d -> %d 样本, 有限值=%s" % (len(a)+len(b), len(c), np.isfinite(c).all()))
    assert np.isfinite(c).all()

    print("\n>>> [D] 缓存命中校验 ...")
    p3, rep3 = prepare_reference(fx, denoise=True, remove_bg=True, target_dur=25.0)
    print("    cached =", rep3.get('cached'), " 路径一致 =", p3 == p)
    assert rep3.get('cached') is True

    print("\n✅ 全部自测通过")
    # 清理夹具(保留 fused 产物在 prepared/ 供后续用)
    os.remove(fx)
    print("已清理测试夹具。融合参考留在:", p)

if __name__ == "__main__":
    main()
