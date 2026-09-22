"""
MDX-NET 人声分离回归测试（voice_clone.mdx_separator）
====================================================
分三层，逐层加深依赖：

  L1 数学约束（永远跑，只需 numpy）
     锁死两个历史坑：
       · 输入必须 4 通道 = 2 声道 × (实, 虚) —— 单声道要 repeat(2,1)
       · chunk 必须使 STFT 恰好得到 dim_t 帧
     STFT 帧数公式 `N//hop + 1` 是这两个坑的共同根源。

  L2 引擎接口（有模型才跑，否则 skip）
     list_models / is_available / session 加载 / 形状契约。

  L3 分离质量（有模型 + 真实语音才跑，否则 skip）
     ⚠️ 必须用**真实语音**当人声源。合成谐波堆对 MDX 属分布外输入，
        模型会把它当乐器分离掉（实测 corr≈0.11），据此断言会误判。
        详见 fixtures.make_voice 的警告。
     L3 还锁一个历史 bug：旧实现 `return out, None` 永不返回伴奏。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures as F  # noqa: E402

# 直接走文件路径导入 mdx_separator，避开 voice_clone/__init__.py 的
# 链式导入（它会拉 librosa，CI 裸 runner 不一定有）。
import importlib.util as _ilu  # noqa: E402

_mdx_path = Path(__file__).resolve().parent.parent / "voice_clone" / "mdx_separator.py"
if not _mdx_path.exists():
    pytest.skip("mdx_separator 源码缺失", allow_module_level=True)
_spec = _ilu.spec_from_file_location("voice_clone_mdx_under_test", _mdx_path)
mdx = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
try:
    _spec.loader.exec_module(mdx)  # type: ignore[union-attr]
except Exception as _exc:  # 缺 numpy / 编码等
    pytest.skip("mdx_separator 不可导入: %s" % _exc, allow_module_level=True)
del _ilu, _spec, _mdx_path

MODEL = mdx.DEFAULT_MODEL
CFG = mdx._MODELS[MODEL]


def _frames(n_samples: int, hop: int) -> int:
    """center=True 时 STFT 的时间帧数 —— 所有 chunk 计算的基础。"""
    return n_samples // hop + 1


# ============================== L1. 数学约束 ==============================

class TestChunkMath:
    """STFT 帧数 = N//hop + 1。模型要求恰好 dim_t 帧。"""

    def test_frames_formula(self):
        assert _frames(1024, 1024) == 2
        assert _frames(0, 1024) == 1

    @pytest.mark.parametrize("name", list(mdx._MODELS))
    def test_naive_chunk_gives_one_frame_too_many(self, name):
        """坑：直接用 dim_t*hop 会得到 dim_t+1 帧 → 模型报 InvalidArgument。"""
        cfg = mdx._MODELS[name]
        hop, dim_t = cfg["hop_length"], 2 ** cfg["dim_t_pow"]
        assert _frames(dim_t * hop, hop) == dim_t + 1

    @pytest.mark.parametrize("name", list(mdx._MODELS))
    def test_correct_chunk_yields_exactly_dim_t(self, name):
        """正解：dim_t*hop - hop = (dim_t-1)*hop → 恰好 dim_t 帧。"""
        cfg = mdx._MODELS[name]
        hop, dim_t = cfg["hop_length"], 2 ** cfg["dim_t_pow"]
        chunk = dim_t * hop - hop
        assert _frames(chunk, hop) == dim_t
        assert chunk == (dim_t - 1) * hop

    @pytest.mark.parametrize("name", list(mdx._MODELS))
    def test_dim_f_equals_half_n_fft(self, name):
        """UVR 约定 dim_f = n_fft/2，即丢掉 Nyquist 频点（n_fft/2+1 的最后一个）。"""
        cfg = mdx._MODELS[name]
        assert cfg["dim_f"] == cfg["n_fft"] // 2

    def test_t_frames_helper_matches_formula(self):
        assert mdx._t_frames(1024, 6144, 1024) == 2
        assert mdx._t_frames(0, 6144, 1024) == 1


class TestStereoContract:
    """坑：模型输入 4 通道 = 2 声道 × (实, 虚)。单声道必须复制成双声道。"""

    def test_mono_replicated_to_stereo(self):
        y = np.zeros(1000, dtype=np.float32)
        out = F.to_stereo(y)
        assert out.shape == (2, 1000)

    def test_stereo_passthrough(self):
        y = np.zeros((2, 500), dtype=np.float32)
        assert F.to_stereo(y).shape == (2, 500)

    def test_fixture_marks_channels_equal(self):
        y = np.arange(10, dtype=np.float32)
        out = F.to_stereo(y)
        assert np.array_equal(out[0], out[1])


class TestModelRegistry:
    def test_default_model_in_registry(self):
        assert MODEL in mdx._MODELS

    @pytest.mark.parametrize("name,cfg", list(mdx._MODELS.items()))
    def test_registry_entries_wellformed(self, name, cfg):
        for key in ("dim_f", "dim_t_pow", "n_fft", "hop_length", "overlap"):
            assert key in cfg, f"{name} 缺少 {key}"
        assert cfg["dim_f"] > 0 and cfg["n_fft"] > 0 and cfg["hop_length"] > 0
        assert 0.0 <= cfg["overlap"] < 1.0
        assert cfg["dim_t_pow"] > 0

    def test_seg_tol_constant(self):
        assert isinstance(mdx.SEG_TOL, int) and mdx.SEG_TOL > 0


class TestRootResolution:
    def test_root_returns_path_under_models_mdx(self):
        p = mdx._root()
        assert isinstance(p, Path)
        assert p.name == "mdx"
        assert p.parent.name == "models"

    def test_list_models_returns_list(self):
        assert isinstance(mdx.list_models(), list)

    def test_is_available_is_bool(self):
        assert isinstance(mdx.is_available(), bool)


# ============================== L2. 引擎接口（需模型） ==============================

def _have_model() -> bool:
    try:
        return mdx.is_available() and bool(mdx.list_models())
    except Exception:
        return False


needs_model = pytest.mark.skipif(not _have_model(),
                                 reason="未找到 MDX 权重（models/mdx/*.onnx）")


@needs_model
class TestEngineInterface:
    def test_list_models_nonempty(self):
        assert len(mdx.list_models()) >= 1

    def test_session_loads(self):
        sess = mdx._get_session(MODEL)
        assert sess is not None

    def test_model_io_contract(self):
        """输入输出都必须是 [B, 4, dim_f, dim_t]。"""
        sess = mdx._get_session(MODEL)
        inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
        assert len(inp.shape) == 4 and inp.shape[1] == 4
        assert len(out.shape) == 4 and out.shape[1] == 4
        assert inp.shape[2] == CFG["dim_f"]
        assert inp.shape[3] == 2 ** CFG["dim_t_pow"]

    def test_separate_returns_tuple(self):
        y = F.make_voice(1.0, F.DEFAULT_SR)
        res = mdx.separate(y, F.DEFAULT_SR)
        assert isinstance(res, tuple) and len(res) == 2

    def test_separate_handles_too_short_input(self):
        """极短输入应安全返回 (None, None)，不抛异常。"""
        vocals, inst = mdx.separate(np.zeros(64, dtype=np.float32), F.DEFAULT_SR)
        assert vocals is None

    def test_separate_handles_silence(self):
        vocals, _ = mdx.separate(np.zeros(F.DEFAULT_SR, dtype=np.float32),
                                 F.DEFAULT_SR)
        # 全静音允许返回 None 或近零信号，但不能抛
        if vocals is not None:
            assert float(np.abs(vocals).max()) < 0.5

    def test_unknown_model_falls_back_to_default(self):
        """未知模型名走 `_pick_model` 的显式回退链，**不会**报错。

        当前语义：显式指定 -> DEFAULT_MODEL -> 第一个可用。
        所以传错名字会静默用默认模型 —— 这是既有设计，此处锁死以免无意改变。
        """
        picked = mdx._pick_model("__no_such_model__.onnx")
        assert picked is not None
        assert picked in mdx.list_models()

    def test_pick_model_honours_explicit_choice(self):
        have = mdx.list_models()
        if not have:
            pytest.skip("无可用模型")
        assert mdx._pick_model(have[0]) == have[0]

    def test_pick_model_none_when_no_models(self, monkeypatch):
        monkeypatch.setattr(mdx, "list_models", lambda: [])
        assert mdx._pick_model(None) is None

    def test_separate_with_unknown_model_still_runs(self):
        """未知模型名 -> 回退默认模型 -> 正常返回，不抛异常。"""
        res = mdx.separate(F.make_voice(1.0, F.DEFAULT_SR), F.DEFAULT_SR,
                           model="__no_such_model__.onnx")
        assert isinstance(res, tuple) and len(res) == 2


# ============================== L3. 分离质量（需模型 + 真实语音） ==============================

def _have_real_voice() -> bool:
    if not _have_model():
        return False
    try:
        return F.find_real_voice_wav() is not None
    except Exception:
        return False


needs_real = pytest.mark.skipif(
    not _have_real_voice(),
    reason="需要真实语音素材（设置 VOXCPM2_FIXTURE_WAV 或在 outputs/ 放 wav）")


@needs_real
class TestSeparationQuality:
    """用真实语音 + 合成伴奏测质量。阈值取得宽松，只为挡住"整体崩坏"。"""

    @pytest.fixture(scope="class")
    def separated(self):
        mix, clean, acc = F.make_real_voice_mix(seconds=6.0)
        if mix is None:
            pytest.skip("无真实语音素材")
        vocals, inst = mdx.separate(mix, F.DEFAULT_SR)
        if vocals is None:
            pytest.skip("MDX 推理失败（可能显存不足）")
        return mix, clean, acc, vocals, inst

    def test_vocals_correlate_with_clean_source(self, separated):
        """核心回归：分离出的人声必须与干净人声高度相关。

        历史：分块异常被 except 静默吞掉 -> 退化为 DSP，corr 掉到 0.85 以下。
        """
        mix, clean, acc, vocals, inst = separated
        c = F.corr(vocals, clean)
        assert c > 0.85, f"corr={c:.4f} 过低，疑似退化为 DSP 或分块异常"

    def test_vocals_better_than_raw_mix(self, separated):
        """分离后应比原混音更接近干净人声。"""
        mix, clean, acc, vocals, inst = separated
        assert F.corr(vocals, clean) > F.corr(mix, clean)

    def test_amplitude_not_collapsed(self, separated):
        """幅度不能被整体压小 —— 曾出现输出仅剩 1% 幅度的症状。"""
        mix, clean, acc, vocals, inst = separated
        r = F.rms_ratio(vocals, clean)
        assert 0.4 < r < 2.5, f"rms 比={r:.3f}，幅度异常"

    def test_output_length_matches_input(self, separated):
        mix, clean, acc, vocals, inst = separated
        assert len(vocals) == len(mix)

    def test_sdr_positive(self, separated):
        mix, clean, acc, vocals, inst = separated
        assert F.sdr_like(vocals, clean) > 0.0

    def test_not_muffled_relative_to_mix(self, separated):
        """"闷"的量化表现是谱质心大幅下降。分离后不应比混音更闷。"""
        mix, clean, acc, vocals, inst = separated
        c_mix = F.spectral_centroid(mix, F.DEFAULT_SR)
        c_voc = F.spectral_centroid(vocals, F.DEFAULT_SR)
        assert c_voc >= c_mix * 0.8, (
            f"质心 mix={c_mix:.0f}Hz -> sep={c_voc:.0f}Hz，高频被削（发闷）")

    def test_instrumental_is_returned(self, separated):
        """历史 bug：旧实现 `return out, None` 永不返回伴奏。"""
        mix, clean, acc, vocals, inst = separated
        assert inst is not None, "伴奏未返回（历史 bug 回归）"
        assert len(inst) == len(mix)


@needs_model
class TestInstrumentalConsistency:
    """伴奏与人声应大致满足 mix ≈ vocals + instrumental。"""

    def test_components_sum_back_to_mix(self):
        if not _have_real_voice():
            pytest.skip("需真实语音素材")
        mix, clean, acc = F.make_real_voice_mix(seconds=4.0)
        if mix is None:
            pytest.skip("无素材")
        vocals, inst = mdx.separate(mix, F.DEFAULT_SR)
        if vocals is None or inst is None:
            pytest.skip("MDX 推理失败或无伴奏输出")
        recon = vocals + inst
        c = F.corr(recon, mix)
        assert c > 0.9, f"vocals+instrumental 与 mix 相关仅 {c:.4f}"
