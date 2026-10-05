"""声音资产存储层测试（``voice_assets``）
=========================================
把开发期的一次性探针（``_scratch_probes/_test_voice_assets.py``）固化成 pytest 用例，
让这些行为进入常规回归 —— 探针脚本不会被 ``pytest.ini`` 的 ``testpaths = tests``
收集，放在那里等于没有护栏。

为什么这一层必须有测试
----------------------
声音资产与音色包**语义相反**：音色包是"已提纯、可冻结"的代表参考，而资产是
"原样保存、随时可再加工"的原始素材。所以存储层必须守住三条：

  * **导入是非破坏的** —— 立体声降混为单声道，但采样率与内容不被"顺手清洗"。
    任何给资产偷偷加降噪/融合的改动，都该在这里被拦住。
  * **失败的导入不留残渣** —— 半成品 wav 若留在目录里却不在 manifest 中，
    用户会看到一个占着空间、列表里却看不见、也没法删的幽灵文件。
  * **对外列表不泄露内部文件路径** —— 前端不需要 ``file`` / ``preview_file``，
    一旦泄露，调用方就可能拼路径绕过 API。

一个踩过的坑，写在这里防止后人重犯
----------------------------------
判断"降混是否真的取了均值"时，**不要用「峰值应等于某个常数」去断言**：
两个不同频率正弦之和的峰值由相位关系决定，会让**正确实现**被判失败
（本文件第一版就是这么写错的）。现在改成两条更可靠的路子：
  * 与手工 ``arr.mean(axis=1)`` **逐样本比对**（这就是"降混"的域定义）；
  * 外加一个**区分性**用例：右声道给静音，峰值应为左声道的一半
    （"只取左声道"会得到两倍，一眼可判）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import voice_assets as va  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    """把模块全局目录指到临时路径，保证用例之间互不污染。"""
    va.init(str(tmp_path))
    return va


def _write(path: Path, y: np.ndarray, sr: int) -> str:
    sf.write(str(path), y, sr)
    return str(path)


def _tone(seconds: float, sr: int, freq: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- 初始化
def test_init_creates_asset_dir(tmp_path):
    va.init(str(tmp_path))
    assert (tmp_path / "voice_assets").is_dir()
    assert va.list_assets() == []


def test_import_before_init_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "VOICE_ASSET_DIR", None)
    with pytest.raises(RuntimeError):
        va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000))


# ---------------------------------------------------------------- 导入质量
def test_import_downsamples_stereo_to_mono(store, tmp_path):
    sr = 44100
    n = int(sr * 3)
    t = np.arange(n) / sr
    stereo = np.stack([0.4 * np.sin(2 * np.pi * 440 * t),
                       0.2 * np.sin(2 * np.pi * 880 * t)], axis=1).astype(np.float32)
    rec = va.import_asset(_write(tmp_path / "s.wav", stereo, sr), name="立体声")

    wav, prev = va.get_asset_paths(rec["id"])
    y, sr_out = sf.read(str(wav))

    assert y.ndim == 1, "产物必须是单声道"
    assert int(sr_out) == sr, "采样率应原样保留（不做重采样）"
    assert abs(len(y) / sr_out - 3.0) < 0.05
    assert prev is not None and Path(prev).exists(), "试听片段应一并生成"
    # 与手工均值逐样本比对：这才是"降混"的域定义
    expect = stereo.mean(axis=1)
    assert np.allclose(y, expect, atol=2e-3), "降混结果应等于逐样本均值"


def test_right_channel_silent_peak_halves(store, tmp_path):
    """区分性用例：只取左声道会得到 0.5，取均值应为 0.25。"""
    sr = 24000
    left = _tone(2.0, sr, 200, amp=0.5)
    one_sided = np.stack([left, np.zeros_like(left)], axis=1)
    rec = va.import_asset(_write(tmp_path / "l.wav", one_sided, sr), name="单侧")

    wav, _ = va.get_asset_paths(rec["id"])
    y, _ = sf.read(str(wav))
    assert abs(float(np.max(np.abs(y))) - 0.25) < 0.01, "峰值应约为左声道的一半"


def test_preview_is_truncated_but_main_file_is_not(store, tmp_path):
    sr = 24000
    rec = va.import_asset(_write(tmp_path / "long.wav", _tone(12.0, sr, 300), sr),
                          name="长素材")
    wav, prev = va.get_asset_paths(rec["id"])
    yp, sr_p = sf.read(str(prev))

    assert abs(len(yp) / sr_p - va.PREVIEW_SECONDS) < 0.05, "试听应截到 PREVIEW_SECONDS"
    assert abs(rec["duration"] - 12.0) < 0.05, "主文件不应被截断"


def test_duration_guard_uses_constant(store, tmp_path, monkeypatch):
    """超长素材应被拒；用 monkeypatch 调小上限，避免真的造 20 分钟音频。"""
    monkeypatch.setattr(va, "MAX_ASSET_SECONDS", 1.0)
    with pytest.raises(ValueError, match="过长"):
        va.import_asset(_write(tmp_path / "x.wav", _tone(3.0, 24000, 300), 24000))
    assert va.list_assets() == [], "被拒后不应写进 manifest"


def test_empty_audio_rejected(store, tmp_path):
    sf.write(str(tmp_path / "empty.wav"), np.zeros(0, dtype=np.float32), 24000)
    with pytest.raises(ValueError):
        va.import_asset(str(tmp_path / "empty.wav"))


def test_bad_file_leaves_no_residue(store, tmp_path):
    """失败的导入不能在目录里留半成品（那会变成列表外的幽灵文件）。"""
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is definitely not audio" * 100)
    asset_dir = tmp_path / "voice_assets"
    before = sorted(p.name for p in asset_dir.iterdir())

    with pytest.raises(ValueError):
        va.import_asset(str(bad), name="坏素材")

    assert sorted(p.name for p in asset_dir.iterdir()) == before
    assert va.list_assets() == []


# ---------------------------------------------------------------- 元数据
def test_list_hides_internal_path_fields(store, tmp_path):
    va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000), name="A")
    rows = va.list_assets()
    assert len(rows) == 1
    assert "file" not in rows[0] and "preview_file" not in rows[0]
    # 但内部取路径仍要能用
    assert va.get_asset_meta(rows[0]["id"])["file"].endswith(".wav")


def test_import_records_metadata(store, tmp_path):
    rec = va.import_asset(_write(tmp_path / "a.wav", _tone(2.0, 24000, 300), 24000),
                          name="客服原声", source_name="orig.mp4", kind="video", note="备注")
    assert rec["name"] == "客服原声"
    assert rec["source_kind"] == "video"
    assert rec["source_name"] == "orig.mp4"
    assert rec["note"] == "备注"
    assert rec["sample_rate"] == 24000
    assert rec["created_at"]


def test_default_name_when_blank(store, tmp_path):
    rec = va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000), name="   ")
    assert rec["name"].strip(), "名称为空时应给出默认名，不能是空串"


# ---------------------------------------------------------------- 改名
def test_rename_persists(store, tmp_path):
    rec = va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000), name="旧")
    got = va.rename_asset(rec["id"], "新名字")
    assert got is not None and got["name"] == "新名字"
    assert va.get_asset_meta(rec["id"])["name"] == "新名字"


def test_rename_rejects_blank_and_unknown(store, tmp_path):
    rec = va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000), name="A")
    assert va.rename_asset(rec["id"], "   ") is None
    assert va.rename_asset("nosuchid", "x") is None
    assert va.get_asset_meta(rec["id"])["name"] == "A", "被拒的改名不能改到别的记录"


# ---------------------------------------------------------------- 删除
def test_delete_removes_files_and_record(store, tmp_path):
    rec = va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000), name="A")
    wav, prev = va.get_asset_paths(rec["id"])

    assert va.delete_asset(rec["id"]) is True
    assert not Path(wav).exists() and not Path(prev).exists()
    assert va.get_asset_paths(rec["id"]) == (None, None)
    assert va.list_assets() == []
    assert va.delete_asset(rec["id"]) is False, "重复删除应返回 False"


def test_get_paths_of_unknown_returns_none_pair(store):
    assert va.get_asset_paths("nope") == (None, None)
    assert va.get_asset_meta("nope") is None


def test_missing_main_file_is_treated_as_gone(store, tmp_path):
    """manifest 有记录但 wav 被外部删掉时，取路径必须返回 (None, None) 而不是坏路径。"""
    rec = va.import_asset(_write(tmp_path / "a.wav", _tone(1.0, 24000, 300), 24000), name="A")
    wav, _ = va.get_asset_paths(rec["id"])
    Path(wav).unlink()
    assert va.get_asset_paths(rec["id"]) == (None, None)
