"""单个插件「试运行」（``plugins.invoke_one``）的契约测试。

为什么单独测这一层
------------------
试运行面板是用户唯一能"不合成、不占显卡就看见插件行为"的入口，
它必须和真实流水线**说同一套话**。三条语义一旦含混，界面就会撒谎：

  · 插件返回 ``None``  → 真实 ``emit`` 里是 ``continue``（这行不归它管，保持原值）
  · 插件返回非法值      → 真实 ``emit`` 里丢弃并**保持原值**、记一次错误
  · 插件返回合法字符串  → 真正生效

第一版实现把 ``None`` 硬转成空串，于是"没命中"被显示成
「已改写：<一片空白>」，用户以为自己用错了功能 —— 这个 bug 就是
本文件第 2、3 条用例要永久拦住的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from voice_clone import plugin_core as P         # noqa: E402


# --------------------------------------------------------------------------- 脚手架
@pytest.fixture(autouse=True)
def _clean_state():
    P.reset_for_tests()
    P._reset_lifecycle_for_tests()
    P.set_logger(lambda _m: None)
    yield
    P.reset_for_tests()
    P._reset_lifecycle_for_tests()
    P.set_logger(None)


def _mk(root: Path, pid: str, body: str, manifest: dict | None = None) -> Path:
    d = root / pid
    d.mkdir(parents=True, exist_ok=True)
    mf: dict = {"id": pid, "api_version": "1.0", "hooks": {}, "enabled": True}
    if manifest:
        mf.update(manifest)
    (d / "plugin.json").write_text(json.dumps(mf, ensure_ascii=False),
                                   encoding="utf-8")
    (d / "plugin.py").write_text(body, encoding="utf-8")
    return d


def _init(tmp_path: Path):
    cfg = tmp_path / P.CONFIG_NAME
    cfg.write_text(json.dumps({
        "search_paths": [P.PLUGIN_DIR_NAME],
        "autoload": True,
        "disabled": [],
        "settings": {},
    }, ensure_ascii=False), encoding="utf-8")
    return P.init(str(tmp_path), str(cfg))


_TEXT_PRE = {"hooks": {"text.pre": "on_text_pre"}}


def _setup(tmp_path: Path):
    """搭一窝角色齐全的插件，覆盖试运行会遇到的每一类返回。"""
    root = tmp_path / P.PLUGIN_DIR_NAME
    _mk(root, "p_echo",
        "def on_text_pre(p):\n    return 'A:' + p['text']\n", _TEXT_PRE)
    _mk(root, "p_silent",
        "def on_text_pre(p):\n    return None\n", _TEXT_PRE)
    _mk(root, "p_empty",
        "def on_text_pre(p):\n    return ''\n", _TEXT_PRE)
    _mk(root, "p_boom",
        "def on_text_pre(p):\n    raise ValueError('故意炸')\n", _TEXT_PRE)
    _mk(root, "p_audio",
        "def on_output_post(p):\n    return p['audio']\n",
        {"hooks": {"output.post": "on_output_post"}})
    _mk(root, "p_off",
        "def on_text_pre(p):\n    return 'never'\n",
        {"hooks": {"text.pre": "on_text_pre"}, "enabled": False})
    return _init(tmp_path)


def _call(reg, pid, text="原文"):
    return P.invoke_one(pid, hook="text.pre", value=text, registry=reg)


# --------------------------------------------------------------------------- 1. 白名单
def test_invokable_hooks_is_text_only():
    """只放"pipeline 类且携带纯文本"的钩子 —— audio/ndarray 单独调用没有意义。"""
    assert P.invokable_hooks() == ["text.pre"]
    for h in P.invokable_hooks():
        spec = P.HOOK_SPECS[h]
        assert spec.kind == "pipeline" and spec.arg == "text"


# --------------------------------------------------------------------------- 2. 三态
def test_handled_reports_new_text(tmp_path):
    reg = _setup(tmp_path)
    r = _call(reg, "p_echo", "原文")
    assert r["result"] == "handled"
    assert r["applied"] is True and r["valid"] is True
    assert r["output"] == "A:原文"
    assert r["input"] == "原文"
    assert r["changed"] is True
    assert r["raw_output"] == "A:原文"
    assert isinstance(r["duration_ms"], float)


def test_none_means_untouched_not_empty(tmp_path):
    """插件返回 None = "这行不归我管"，必须原样返回，不能变成空串。

    这是第一版的真实 bug：面板显示「已改写：」后面一片空白。
    """
    reg = _setup(tmp_path)
    r = _call(reg, "p_silent", "原文")
    assert r["result"] == "unchanged"
    assert r["applied"] is False
    assert r["changed"] is False
    assert r["output"] == "原文", "None 被当成了「改写成空」"
    assert r["raw_output"] is None
    assert r["valid"] is True, "None 是合法契约，不该记成非法返回"


def test_empty_string_is_invalid_not_untouched(tmp_path):
    """空串走的是**另一条**路径：text.pre 的校验器要求非空。

    在真实流水线里它会被丢弃、保持原值，并记一次错误 ——
    试运行必须照实说，否则用户会以为一个永不生效的效果已经生效。
    """
    reg = _setup(tmp_path)
    ok = P._VALIDATORS["text.pre"]
    assert ok("x") and not ok("")          # 先自证域假设：空串确实非法
    r = _call(reg, "p_empty", "原文")
    assert r["result"] == "invalid"
    assert r["valid"] is False
    assert r["applied"] is False
    assert r["output"] == "原文", "非法返回在真实流水线里会保持原值"
    assert r["raw_output"] == "", "要如实回显插件到底返回了什么"


# --------------------------------------------------------------------------- 3. 只跑指定插件
def test_only_the_named_plugin_runs(tmp_path):
    """试运行必须能归因到单个插件 —— 链上先跑的插件不能替它作答。"""
    reg = _setup(tmp_path)
    assert _call(reg, "p_echo")["output"] == "A:原文"
    assert _call(reg, "p_silent")["output"] == "原文"
    # 若实现里走了 emit()（整链），p_silent 会拿到 p_echo 的产物
    assert _call(reg, "p_silent")["output"] != "A:原文"


def test_does_not_disturb_other_plugins_stats(tmp_path):
    reg = _setup(tmp_path)
    _call(reg, "p_echo")
    assert reg.plugins["p_echo"].hook_calls == 1
    assert reg.plugins["p_silent"].hook_calls == 0


# --------------------------------------------------------------------------- 4. 失败如实抛出
def test_handler_exception_raises_with_status(tmp_path):
    reg = _setup(tmp_path)
    lp = reg.plugins["p_boom"]
    before = lp.total_errors
    with pytest.raises(P.PluginInvokeError) as ei:
        _call(reg, "p_boom")
    assert ei.value.status == 500
    assert "故意炸" in str(ei.value)
    assert lp.total_errors == before + 1, "异常仍要计入插件统计（熔断依赖它）"


def test_unknown_plugin_is_404(tmp_path):
    reg = _setup(tmp_path)
    with pytest.raises(P.PluginInvokeError) as ei:
        _call(reg, "nope")
    assert ei.value.status == 404


def test_hook_not_mounted_is_409(tmp_path):
    reg = _setup(tmp_path)
    with pytest.raises(P.PluginInvokeError) as ei:
        _call(reg, "p_audio")
    assert ei.value.status == 409
    assert "text.pre" in str(ei.value)


def test_non_invokable_hook_is_400(tmp_path):
    reg = _setup(tmp_path)
    with pytest.raises(P.PluginInvokeError) as ei:
        P.invoke_one("p_audio", hook="output.post", value="x", registry=reg)
    assert ei.value.status == 400
    assert "text.pre" in str(ei.value), "报错要顺带告诉用户能用哪些钩子"


def test_unknown_hook_name_is_400(tmp_path):
    reg = _setup(tmp_path)
    with pytest.raises(P.PluginInvokeError) as ei:
        P.invoke_one("p_echo", hook="no.such.hook", value="x", registry=reg)
    assert ei.value.status == 400


def test_disabled_plugin_is_409(tmp_path):
    reg = _setup(tmp_path)
    assert reg.plugins["p_off"].state == "disabled"
    with pytest.raises(P.PluginInvokeError) as ei:
        _call(reg, "p_off")
    assert ei.value.status == 409
    assert "disabled" in str(ei.value)


def test_without_registry_is_503():
    P.reset_for_tests()
    with pytest.raises(P.PluginInvokeError) as ei:
        P.invoke_one("whatever", value="x")
    assert ei.value.status == 503


# --------------------------------------------------------------------------- 5. 结果可 JSON 序列化
def test_report_is_json_serializable(tmp_path):
    """接口直接 JSONResponse(dict)，返回体里不能混进不可序列化的对象。"""
    reg = _setup(tmp_path)
    for pid in ("p_echo", "p_silent", "p_empty"):
        json.dumps(_call(reg, pid), ensure_ascii=False)


def test_safe_repr_degrades_non_string_returns():
    """插件返回 ndarray / 对象时，``raw_output`` 也必须能落成 JSON 文本。

    真实场景：一个误挂 text.pre 的音频插件返回 ndarray —— 试运行不该
    因为"回显不了"而 500，那会把"插件返回值不对"这个真正的问题盖住。
    """
    import numpy as np

    assert P._safe_repr(None) == ""
    assert P._safe_repr("文本") == "文本"
    assert isinstance(P._safe_repr(np.zeros(4, dtype=np.float32)), str)
    assert isinstance(P._safe_repr({"a": 1}), str)
    long = P._safe_repr("x" * 5000)
    assert len(long) < 500, "超长返回值必须截断，否则面板会被撑爆"
    assert long.endswith("…")
