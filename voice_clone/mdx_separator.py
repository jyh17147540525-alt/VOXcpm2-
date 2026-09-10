"""
MDX-NET 神经网络人声分离引擎 (Neural Vocal Separation)
=======================================================
用 UVR（Ultimate Vocal Remover）的 MDX-NET ONNX 权重做真正的模型级人声分离，
替代/增强纯 DSP 的 REPET-lite 与 HPSS。

与 DSP 方法的本质区别：
  REPET/HPSS 是"假设驱动"—— 它们假设背景在时频域上是可预测/可分离的，
  遇到与语音重叠的旋律与和声就会连人声一起削掉（听感：发闷、像隔棉被）。
  MDX-NET 是"数据驱动"—— 在数万小时 (人声, 伴奏) 配对上训练过，
  学到的是人声的统计特征，能在人声与伴奏重叠时把人声"捞"回来。

模型：UVR-MDX-NET 系列（dim_f=3072 规格，见 _MODELS）
  - ONNX Runtime 推理，CUDA 优先，CPU 回退
  - 推理参数严格对齐 UVR / audio-separator 官方实现（n_fft=6144, hop=1024,
    chunk=dim_t*hop, overlap=0.25, 前 3 个频点置零, 负谱平均去噪）

用法：
  is_available()                       —— 是否有可用模型
  separate_vocals(y, sr, model=...)    —— 返回人声波形（float32）
  separate(y, sr, model=...)           —— 返回 (人声, 伴奏)

无模型时所有函数返回 None，由上层回退到 DSP 链。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_MODEL_ROOT: Path | None = None
_SESSIONS: dict[str, object] = {}

# UVR MDX-NET 模型规格表
#   dim_f / dim_t / n_fft 必须与权重训练时一致，否则输出会变成噪声
#   dim_t 是 2 的指数（segment = dim_t * hop_length / sr 秒）
_MODELS: dict[str, dict] = {
    "UVR-MDX-NET_Main_340.onnx": {
        "dim_f": 3072, "dim_t_pow": 8, "n_fft": 6144,
        "hop_length": 1024, "overlap": 0.25, "batch_size": 1,
        "label": "Main 340（人声/伴奏通用，推荐）",
    },
    "UVR_MDXNET_9482.onnx": {
        "dim_f": 2048, "dim_t_pow": 8, "n_fft": 4096,
        "hop_length": 1024, "overlap": 0.25, "batch_size": 1,
        "label": "MDXNET 9482（人声）",
    },
}
DEFAULT_MODEL = "UVR-MDX-NET_Main_340.onnx"
# UVR 把 chunk 首尾各留 trim 再送 STFT，使时间维恰好等于 dim_t（2 的幂）
SEG_TOL = 256


def _t_frames(n_samples: int, n_fft: int, hop: int) -> int:
    """center=True 时 torch.stft 的时间帧数。"""
    return n_samples // hop + 1


def init(base_dir) -> None:
    """指定模型根目录（base_dir/models/mdx/）。"""
    global _MODEL_ROOT
    _MODEL_ROOT = Path(base_dir) / "models" / "mdx"
    try:
        _MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _root() -> Path:
    """模型根目录：显式 init 过就用它，否则按包位置推断（<base>/models/mdx）。"""
    global _MODEL_ROOT
    if _MODEL_ROOT is None:
        here = Path(__file__).resolve().parent.parent       # <base>/voice_clone/.. = <base>
        _MODEL_ROOT = here / "models" / "mdx"
    return _MODEL_ROOT


def list_models() -> list:
    """列出本地已就绪的模型文件名。"""
    root = _root()
    if not root.exists():
        return []
    out = []
    for name in _MODELS:
        p = root / name
        if p.exists() and p.stat().st_size > 1 << 20:
            out.append(name)
    return out


def is_available() -> bool:
    """是否有任何可直接使用的模型。"""
    return bool(list_models())


def _pick_model(model: str | None) -> str | None:
    """选择要用的模型：显式指定 -> 默认 -> 第一个可用。"""
    have = list_models()
    if not have:
        return None
    if model and model in have:
        return model
    if DEFAULT_MODEL in have:
        return DEFAULT_MODEL
    return have[0]


def _get_session(name: str):
    """懒加载 ONNX Runtime 会话（CUDA 优先）。"""
    if name in _SESSIONS:
        return _SESSIONS[name]
    try:
        import onnxruntime as ort
    except Exception:
        return None
    path = str(_root() / name)
    if not Path(path).exists():
        return None
    providers = []
    try:
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
    except Exception:
        providers = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(path, providers=providers)
    except Exception:
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception:
            return None
    _SESSIONS[name] = sess
    return sess


def _spectrogram(y: np.ndarray, n_fft: int, hop_length: int, dim_f: int, device,
                 channels: int = 2):
    """UVR 同款 STFT：Hann(periodic) + center=True，返回 [1, C*2, dim_f, T]。

    y 形状 [C, T]；C 不足 channels 时复制补齐（UVR 对单声道也会复制成 2 轨）。
    实部/虚部按 UVR 的通道排布堆叠（每声道先实后虚）。
    """
    import torch
    win = torch.hann_window(n_fft, periodic=True, device=device)
    t = torch.from_numpy(np.ascontiguousarray(y)).float().to(device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.shape[0] < channels:
        t = t.repeat(channels, 1)
    spec = torch.stft(t, n_fft=n_fft, hop_length=hop_length, window=win,
                      center=True, return_complex=False)      # [C, F, T, 2]
    spec = spec.permute(0, 3, 1, 2)                           # [C, 2, F, T]
    c = spec.shape[0]
    spec = spec.reshape(1, c * 2, spec.shape[2], spec.shape[3])
    return spec[..., :dim_f, :]


def _istft(spec: np.ndarray, n_fft: int, hop_length: int, n_ch: int, out_len: int, device):
    """UVR 同款 iSTFT：把 [B, C*2, F, T] 还原为波形 [B, C, T]。"""
    import torch
    n_bins = n_fft // 2 + 1
    x = torch.from_numpy(np.ascontiguousarray(spec)).float().to(device)
    b, c2, f, t = x.shape
    freq_dim = f
    if freq_dim < n_bins:                                     # 补零回完整频点数
        pad = torch.zeros(b, c2, n_bins - freq_dim, t, device=device)
        x = torch.cat([x, pad], dim=-2)
    x = x.reshape(b, c2 // 2, 2, n_bins, t)
    x = x.reshape(-1, 2, n_bins, t).permute(0, 2, 3, 1)
    cx = x[..., 0] + x[..., 1] * 1j
    win = torch.hann_window(n_fft, periodic=True, device=device)
    y = torch.istft(cx, n_fft=n_fft, hop_length=hop_length, window=win,
                    center=True, length=out_len)
    return y.reshape(b, -1, y.shape[-1])                      # [B, C, T]


def _run_model(sess, spec_np: np.ndarray, device):
    """跑一次模型（含 enable_denoise 的负谱平均，与 UVR 一致）。"""
    import torch
    go = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"
    dev = "cuda" if "CUDA" in go and device == "cuda" else "cpu"
    x = torch.from_numpy(spec_np).float().to(dev)
    name_in = sess.get_inputs()[0].name
    name_out = sess.get_outputs()[0].name
    neg = sess.run([name_out], {name_in: (-x).cpu().numpy()})[0]
    pos = sess.run([name_out], {name_in: x.cpu().numpy()})[0]
    pred = (-0.5) * neg + 0.5 * pos
    return np.asarray(pred, dtype=np.float32)


def separate(y: np.ndarray, sr: int, model: str | None = None,
             progress=None):
    """MDX-NET 分离，返回 (vocals, instrumental)；不可用时返回 (None, None)。

    y  —— float32 单声道波形
    sr —— 采样率（模型按 44.1k 训练，但内部按样本推理，任意 sr 均可）

    instrumental 取「原混音 − 人声」的残差（UVR 同款），与人声同长。
    """
    name = _pick_model(model)
    if name is None:
        return None, None
    sess = _get_session(name)
    if sess is None:
        return None, None
    cfg = _MODELS.get(name)
    if cfg is None:
        return None, None

    import torch
    try:
        use_cuda = "CUDA" in (sess.get_providers() or [""])[0]
    except Exception:
        use_cuda = False
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"

    dim_f = cfg["dim_f"]
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    overlap = cfg["overlap"]
    dim_t = 2 ** cfg["dim_t_pow"]
    trim = n_fft // 2
    # UVR 以 dim_t 帧为一段；center=True 的 STFT 在 N 个样本上给出 N//hop+1 帧，
    # 故取 seg = dim_t*hop - hop，使帧数恰好等于 dim_t（与权重期望一致）。
    seg = dim_t * hop - hop
    gen = seg - 2 * trim
    if gen <= 0:
        return None, None

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    n = len(y)
    if n < 1024:
        return None, None

    try:
        # UVR 同款：前置 trim 零填充，让首段 STFT 的边界伪影落在填充区
        pad = gen - (n % gen) if (n % gen) else 0
        padded = np.concatenate([np.zeros(trim, np.float32), y,
                                 np.zeros(pad + trim, np.float32)])
        step = max(1, int((1.0 - overlap) * seg))
        result = np.zeros(padded.shape[-1], np.float64)
        divider = np.zeros(padded.shape[-1], np.float64)
        wfull = np.hanning(seg).astype(np.float64)
        total = len(range(0, padded.shape[-1], step))
        done = 0
        for i in range(0, padded.shape[-1], step):
            chunk = padded[i:i + seg]
            if chunk.shape[0] < seg:
                chunk = np.concatenate([chunk, np.zeros(seg - chunk.shape[0], np.float32)])
            spec = _spectrogram(chunk[None, :], n_fft, hop, dim_f, device, channels=2)
            spec_np = spec.cpu().numpy()
            spec_np[..., :3, :] = 0.0                      # UVR: 前 3 频点置零
            pred = _run_model(sess, spec_np, device)
            wav = _istft(pred, n_fft, hop, 2, seg, device)
            wav = wav.cpu().numpy().reshape(-1, wav.shape[-1])[:, :seg].mean(axis=0)
            # 整段按 Hann 窗叠加（窗口覆盖 [i, i+seg)，与 padded 坐标系一致）
            hi = min(i + seg, result.shape[0])
            if hi <= i:
                done += 1
                continue
            wr = wav[:hi - i]
            ws = wfull[:hi - i]
            result[i:hi] += wr * ws
            divider[i:hi] += ws
            done += 1
            if progress:
                try:
                    progress(int(done * 100 / max(1, total)))
                except Exception:
                    pass
        # 去掉前置 trim 填充，回到原始时间轴
        out = (result / np.maximum(divider, 1e-8))[trim:trim + n].astype(np.float32)
        # 伴奏 = 原混音 - 人声（UVR 同款残差）。模型只预测人声，残差是免费的。
        inst = (y[:n] - out).astype(np.float32)
        return out, inst
    except Exception:
        return None, None
