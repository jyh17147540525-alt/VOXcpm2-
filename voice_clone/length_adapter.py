"""
模块 2：参考音频长度适配 (Reference Length Adaptation)
====================================================
解决“长参考音频直接整段编码导致特征漂移/音色突变”的核心问题。

流程：
  1) 分段：按静音边界(librosa.effects.split) + 长段能量极小值再切，得到自然语音段；
  2) 声纹提取：每段取 MFCC 均值作为轻量说话人表征（无需外部说话人模型，离线可用）；
  3) 离群剔除 + 融合：以各段声纹到“说话人质心”的距离打分，剔除明显偏离者
     （音色突变/他人/残留音乐），按得分贪心拼出总体时长有界(~target_dur)的融合参考。
融合参考既保留说话人稳定性，又避免过长音频被模型一次性平均导致漂移。
"""
from __future__ import annotations
import numpy as np
import librosa


# ----------------------------------------------------------------------------- 分段
def _split_long_at_energy_min(y: np.ndarray, s: int, e: int, sr: int,
                              max_dur: float) -> list[tuple[int, int]]:
    """把超长段在能量最低点递归切成若干 <=max_dur 的子段（避免边界碎片化合并）。"""
    dur = (e - s) / sr
    if dur <= max_dur:
        return [(s, e)]
    sub = y[s:e]
    S = np.abs(librosa.stft(sub, n_fft=512, hop_length=128))
    energy = S.mean(axis=0)
    k = max(1, int(len(energy) * 0.1))
    local = energy[k:-k] if len(energy) > 2 * k else energy
    cut = int((int(np.argmin(local)) + k) * 128)
    mid = s + cut
    # 防止切点贴边导致一侧过短
    if (mid - s) / sr < 0.5 or (e - mid) / sr < 0.5:
        mid = s + (e - s) // 2
    return _split_long_at_energy_min(y, s, mid, sr, max_dur) + \
           _split_long_at_energy_min(y, mid, e, sr, max_dur)


def segment_by_silence(y: np.ndarray, sr: int, min_dur: float = 1.5,
                       max_dur: float = 30.0, top_db: float = 28.0):
    """按静音边界切分；随后合并被极短静音隔开的相邻段，再对超长段在能量最低点递归切分。
    返回自然语音段列表（样本起止）。"""
    intervals = librosa.effects.split(y, top_db=top_db)
    # 1) 合并被 <0.3s 静音隔开的相邻段（仅限 librosa 原始区间之间）
    merged = []
    for s, e in intervals:
        if merged and (s - merged[-1][1]) / sr < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # 2) 过滤过短 + 超长段递归切分
    segs = []
    for s, e in merged:
        dur = (e - s) / sr
        if dur < min_dur:
            continue
        if dur > max_dur:
            segs.extend(_split_long_at_energy_min(y, s, e, sr, max_dur))
        else:
            segs.append((s, e))
    return segs


def segment_fixed_window(y: np.ndarray, sr: int, win: float = 10.0,
                         hop: float = 5.0) -> list[tuple[int, int]]:
    """固定窗口切分（用于连续音频、无明显静音时的声纹投票候选）。"""
    n = len(y)
    step = int(hop * sr)
    w = int(win * sr)
    segs = []
    for s in range(0, n, step):
        e = min(s + w, n)
        if (e - s) / sr >= 1.0:
            segs.append((s, e))
        if e >= n:
            break
    return segs


# ----------------------------------------------------------------------------- 声纹
def extract_voiceprint(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """轻量说话人表征：voiced 帧的 MFCC 均值。离线、无外部模型。"""
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    energy = S.mean(axis=0)
    thr = energy.mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
    voiced = energy > thr
    if voiced.any():
        vec = mfcc[:, voiced].mean(axis=1)
    else:
        vec = mfcc.mean(axis=1)
    # 标准化，使不同段距离可比
    vec = (vec - vec.mean()) / (vec.std() + 1e-8)
    return vec.astype(np.float32)


# ----------------------------------------------------------------------------- 融合
def fuse_reference(y: np.ndarray, sr: int, target_dur: float = 25.0,
                   min_dur: float = 1.5, max_dur: float = 30.0,
                   top_db: float = 28.0) -> tuple[np.ndarray, int, dict]:
    """
    长参考音频适配：分段 -> 声纹 -> 离群剔除 -> 贪心融合为 ~target_dur 的干净代表参考。
    返回 (fused_y, sr, info)。
    """
    segs = segment_by_silence(y, sr, min_dur=min_dur, max_dur=max_dur, top_db=top_db)
    info = {"n_segments": len(segs), "split_method": "silence"}
    if len(segs) < 2:
        # 连续音频(无明显静音，如背景乐贯穿)时退化为固定窗口切分，以保证有候选可投票
        segs = segment_fixed_window(y, sr, win=10.0, hop=5.0)
        info["split_method"] = "fixed_window"
        info["n_segments"] = len(segs)
    if not segs:
        # 无有效语音段：退化返回整段标准化结果
        return y, sr, {**info, "status": "no_speech_segment", "chosen": 0,
                       "total_dur": round(len(y) / sr, 2)}
    vps = np.stack([extract_voiceprint(y[s:e], sr) for (s, e) in segs], axis=0)
    centroid = np.median(vps, axis=0)
    dist = np.linalg.norm(vps - centroid, axis=1)
    info["dist_to_centroid"] = [round(float(d), 3) for d in dist]
    # 得分：与质心越近越好；并轻度偏好更长片段(信息更充分)
    dur_arr = np.array([(e - s) / sr for (s, e) in segs], dtype=float)
    scores = 1.0 / (dist + 1e-6) * (1.0 + 0.2 * np.clip(dur_arr / target_dur, 0, 1))
    scores = scores / scores.sum()
    info["scores"] = [round(float(s), 3) for s in scores]
    order = np.argsort(-scores)
    chosen, total = [], 0.0
    for idx in order:
        s, e = segs[idx]
        d = (e - s) / sr
        # 以 target_dur 为软上限：已有足够且再加会明显超界则停止
        if chosen and total + d > target_dur * 1.25:
            break
        chosen.append((s, e))
        total += d
        if total >= target_dur:
            break
    chosen.sort()
    fused = np.concatenate([y[s:e] for (s, e) in chosen]) if chosen else y
    return fused, sr, {
        **info, "status": "fused", "chosen": len(chosen),
        "total_dur": round(float(total), 2),
        "output_dur": round(len(fused) / sr, 2),
    }
