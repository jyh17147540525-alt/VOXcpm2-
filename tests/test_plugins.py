"""插件接口（voice_clone/plugins.py）的契约与回归测试。

覆盖范围
--------
1. 钩子表自洽 + **静态护栏**：仓库里每个 `plugins.emit("x")` 的 x 都必须是已声明钩子，
   且主要钩子都真的有调用点（防止"文档里有、代码里没人调"）。
2. **零插件即原样**：无插件时所有钩子返回传入对象本身；空注册表下
   `synthesize_stable` 的音频与报告与挂载"全 no-op 插件"时**完全一致**。
3. 清单校验：必填字段、id 字符集、api_version 主版本闸门、requires_app 区间、
   未知字段/未知钩子只告警不拒绝、`isolation=subprocess` **必须拒绝**。
4. 版本比较、返回值校验器（音频/分块/服务商）。
5. 执行顺序：priority 大的先跑，同优先级按 id 稳定排序。
6. 错误隔离与熔断：连续 5 次异常后摘除钩子，且 teardown **恰好一次**。
7. 返回值非法 → 保持原值并计数；返回 None → 不做改动。
8. 重入防护：嵌套 emit 被拒（返回 None）；钩子内调 `synthesize_stable` 抛 RuntimeError。
9. 内置不可覆盖：插件无法改写 / 删除内置 LLM 服务商，只能追加；
   provider 缓存不能因"处理器数量相同"而张冠李戴。

这些测试的价值在于：插件机制最容易出的错不是"跑不起来"，而是
**"主流程被悄悄改变"** 与 **"插件故障拖垮服务"** —— 两者都只在生产里才暴露。
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from voice_clone import llm_providers as LP      # noqa: E402
from voice_clone import plugins as P             # noqa: E402
from voice_clone import synthesis_stab as SS     # noqa: E402


# --------------------------------------------------------------------------- 脚手架
@pytest.fixture(autouse=True)
def _clean_state():
    """每个用例都从"无插件"开始，避免全局注册表串味。"""
    P.reset_for_tests()
    P._reset_lifecycle_for_tests()
    P.set_logger(lambda _m: None)
    yield
    P.reset_for_tests()
    P._reset_lifecycle_for_tests()
    P.set_logger(None)


def _make_plugin(root: Path, pid: str, body: str, manifest: dict | None = None) -> Path:
    d = root / pid
    d.mkdir(parents=True, exist_ok=True)
    mf: dict = {"id": pid, "api_version": "1.0", "hooks": {}}
    if manifest:
        mf.update(manifest)
    (d / "plugin.json").write_text(json.dumps(mf, ensure_ascii=False), encoding="utf-8")
    (d / "plugin.py").write_text(body, encoding="utf-8")
    return d


def _init(tmp_path: Path, settings: dict | None = None, search: list[str] | None = None):
    cfg = tmp_path / P.CONFIG_NAME
    cfg.write_text(json.dumps({
        "search_paths": search or ["plugins"],
        "autoload": True,
        "disabled": [],
        "settings": settings or {},
    }, ensure_ascii=False), encoding="utf-8")
    return P.init(str(tmp_path), str(cfg))


class _FakeModel:
    """最小可用 TTS 替身：只需 generate() 与 sample_rate（走单块整段那条路径）。"""

    class _TTS:
        sample_rate = 24000

    def __init__(self):
        self.tts_model = _FakeModel._TTS()
        self.n = 0

    def generate(self, text=None, **kw):
        self.n += 1
        rng = np.random.default_rng(1234 + self.n)
        return rng.normal(0.0, 0.05, 4800).astype(np.float32)


# --------------------------------------------------------------------------- 1. 钩子表与静态护栏
def test_hook_table_is_self_consistent():
    assert len(P.HOOK_SPECS) == 12
    for name, spec in P.HOOK_SPECS.items():
        assert spec.name == name
        assert spec.kind in ("pipeline", "first", "merge", "synth", "observe")
        assert spec.doc, f"{name} 缺少说明"
        if spec.kind != "observe":
            assert spec.arg, f"{name} 缺少取值字段名"


def test_every_emit_call_site_uses_a_declared_hook():
    """静态护栏：钩子名写错只会在真正跑到那行时才炸，静态扫一遍成本极低。"""
    emit_pat = re.compile(r"""\.emit\(\s*["']([^"']+)["']""")
    # 生命周期走 notify_lifecycle("startup") 的封装（内部拼 lifecycle.<name>），
    # 否则会误判成"lifecycle.startup 没有调用点"。
    life_pat = re.compile(r"""notify_lifecycle\(\s*["']([^"']+)["']""")
    files = [REPO / "server.py"] + sorted((REPO / "voice_clone").glob("*.py"))
    seen: set[tuple[str, str]] = set()
    for f in files:
        if not f.is_file():
            continue
        src = f.read_text(encoding="utf-8", errors="ignore")
        for m in emit_pat.finditer(src):
            seen.add((f.name, m.group(1)))
        for m in life_pat.finditer(src):
            seen.add((f.name, f"lifecycle.{m.group(1)}"))
    assert seen, "未扫到任何 emit 调用 —— 匹配规则失效，护栏本身没在工作"

    unknown = sorted({h for _f, h in seen if h not in P.HOOK_SPECS})
    assert not unknown, f"存在未声明的钩子名（plugins 层会直接抛 ValueError）：{unknown}"

    used = {h for _f, h in seen}
    expected = {"text.pre", "text.chunks", "chunk.post", "synth.post", "emotion.detect",
                "report.enrich", "reference.post", "llm.providers", "output.post",
                "api.routes", "lifecycle.startup"}
    assert not (expected - used), f"这些钩子在仓库里没有任何调用点：{sorted(expected - used)}"


# --------------------------------------------------------------------------- 2. 零插件即原样
def test_emit_without_registry_returns_input_objects():
    assert P.get_registry() is None
    txt = "原文"
    assert P.emit("text.pre", text=txt) is txt
    chunks = [("a", "end")]
    assert P.emit("text.chunks", text="a", chunks=chunks, max_chars=60) is chunks
    arr = np.ones(16, dtype=np.float32)
    assert P.emit("output.post", audio=arr) is arr
    assert P.emit("chunk.post", audio=arr, sr=24000, index=0, text="a") is arr
    rep = {"n_chunks": 1}
    assert P.emit("report.enrich", report=rep) is rep
    assert P.emit("emotion.detect", text="a", emotion="neutral") == "neutral"
    assert P.emit("reference.post", path="p.wav") == "p.wav"
    assert P.emit("api.routes", app=object()) is None
    assert P.emit("lifecycle.startup", app=object()) is None


def test_unknown_hook_raises_value_error():
    with pytest.raises(ValueError):
        P.emit("does.not.exist", x=1)


def test_noop_plugins_do_not_change_synthesis_output(tmp_path):
    """核心保证：挂上"全部返回 None"的插件后，音频与报告必须与纯原版**完全一致**。"""
    text = "这是一个用于回归测试的短句。"
    base_audio, base_rep = SS.synthesize_stable(_FakeModel(), text, None, 24000)

    _make_plugin(tmp_path / "plugins", "noop",
                 "def on_text_pre(p):\n    return None\n"
                 "def on_text_chunks(p):\n    return None\n"
                 "def on_chunk_post(p):\n    return None\n"
                 "def on_synth_post(p):\n    return None\n"
                 "def on_report_enrich(p):\n    return None\n"
                 "def on_emotion_detect(p):\n    return None\n",
                 {"hooks": {"text.pre": "on_text_pre", "text.chunks": "on_text_chunks",
                            "chunk.post": "on_chunk_post", "synth.post": "on_synth_post",
                            "report.enrich": "on_report_enrich",
                            "emotion.detect": "on_emotion_detect"}})
    reg = _init(tmp_path)
    assert reg.plugins["noop"].state == "started", reg.plugins["noop"].error
    assert reg.handlers_for("text.pre")

    audio2, rep2 = SS.synthesize_stable(_FakeModel(), text, None, 24000)
    assert np.array_equal(base_audio, audio2), "挂载 no-op 插件后音频被改变了"
    assert base_rep == rep2, f"挂载 no-op 插件后报告被改变了：{base_rep} vs {rep2}"


# --------------------------------------------------------------------------- 3. 清单校验
def test_manifest_requires_id_and_api_version():
    with pytest.raises(P.PluginManifestError):
        P.parse_manifest({"api_version": "1.0"})
    with pytest.raises(P.PluginManifestError):
        P.parse_manifest({"id": "x"})


def test_manifest_rejects_bad_id_charset():
    with pytest.raises(P.PluginManifestError):
        P.parse_manifest({"id": "bad id/name", "api_version": "1.0"})


def test_manifest_rejects_api_major_mismatch():
    with pytest.raises(P.PluginManifestError) as e:
        P.parse_manifest({"id": "x", "api_version": "2.0"})
    assert "主版本" in str(e.value)


def test_manifest_enforces_requires_app_range():
    ok = P.parse_manifest({"id": "x", "api_version": "1.0", "requires_app": ">=2.0,<3"})
    assert ok.requires_app == ">=2.0,<3"
    with pytest.raises(P.PluginManifestError):
        P.parse_manifest({"id": "x", "api_version": "1.0", "requires_app": ">=9.0"})


def test_manifest_rejects_subprocess_isolation_instead_of_silently_downgrading():
    """诚实拒绝 > 静默降级：声明进程隔离却跑在进程内，会让作者误以为崩溃被隔离。"""
    with pytest.raises(P.PluginManifestError) as e:
        P.parse_manifest({"id": "x", "api_version": "1.0", "isolation": "subprocess"})
    assert "inprocess" in str(e.value)


def test_unknown_fields_and_hooks_warn_but_do_not_fail():
    m = P.parse_manifest({"id": "x", "api_version": "1.0",
                          "whatever": 1, "hooks": {"no.such": ""}})
    assert len(m.warnings) == 2


def test_hooks_accept_both_list_and_dict_forms():
    m1 = P.parse_manifest({"id": "x", "api_version": "1.0", "hooks": ["text.pre"]})
    assert m1.hooks == {"text.pre": ""}
    m2 = P.parse_manifest({"id": "x", "api_version": "1.0",
                           "hooks": {"text.pre": "custom_name"}})
    assert m2.hooks == {"text.pre": "custom_name"}


# --------------------------------------------------------------------------- 4. 版本与校验器
@pytest.mark.parametrize("spec,ver,expect", [
    (">=2.0,<3", "2.1.0", True),
    (">=2.2", "2.1.0", False),
    ("", "2.1.0", True),
    ("==2.1.0", "2.1.0", True),
    ("!=2.1.0", "2.1.0", False),
    ("~2", "2.1.0", False),
    (">=1", "0.9", False),
])
def test_version_spec(spec, ver, expect):
    assert P.check_version_spec(spec, ver) is expect


def test_return_value_validators():
    assert P._valid_audio(np.zeros(8, dtype=np.float32))
    assert not P._valid_audio([1, 2])
    assert not P._valid_audio(np.zeros(0, dtype=np.float32))
    assert not P._valid_audio(np.array([np.nan] * 8, dtype=np.float32))
    assert P._valid_chunks([("a", "end"), ("b", "hard")])
    assert not P._valid_chunks([("a", "bogus")])
    assert not P._valid_chunks([])
    assert P._valid_providers([{"id": "a"}])
    # 重复 id 在形状层是允许的（插件常把内置项原样回传 + 追加新项），
    # "不得覆盖内置项"的约束在下游逐条处理，不能因此整份丢弃。
    assert P._valid_providers([{"id": "a"}, {"id": "a"}])
    assert not P._valid_providers([{"id": ""}])
    assert not P._valid_providers([{"no_id": 1}])


# --------------------------------------------------------------------------- 5. 执行顺序
def test_priority_runs_highest_first_then_id_ascending(tmp_path):
    root = tmp_path / "plugins"
    for pid, prio, tag in [("b_low", 10, "L"), ("a_high", 90, "H"), ("c_mid", 50, "M")]:
        _make_plugin(root, pid,
                     f"def on_text_pre(p):\n    return p['text'] + '{tag}'\n",
                     {"priority": prio, "hooks": {"text.pre": "on_text_pre"}})
    _init(tmp_path)
    assert P.emit("text.pre", text="x") == "xHML"


def test_same_priority_is_stable_by_plugin_id(tmp_path):
    root = tmp_path / "plugins"
    for pid in ("zz", "aa"):
        _make_plugin(root, pid,
                     f"def on_text_pre(p):\n    return p['text'] + '{pid}'\n",
                     {"hooks": {"text.pre": "on_text_pre"}})
    _init(tmp_path)
    assert P.emit("text.pre", text="x") == "xaazz"


# --------------------------------------------------------------------------- 6. 错误隔离与熔断
def test_exceptions_are_isolated_and_breaker_stops_plugin(tmp_path):
    _make_plugin(tmp_path / "plugins", "boom",
                 "calls = []\n"
                 "def setup(ctx):\n    calls.append('setup')\n"
                 "def teardown(ctx):\n    calls.append('teardown')\n"
                 "def on_text_pre(p):\n    raise RuntimeError('boom')\n",
                 {"hooks": {"text.pre": "on_text_pre"}})
    reg = _init(tmp_path)
    lp = reg.plugins["boom"]
    assert lp.state == "started"

    for i in range(P.MAX_CONSECUTIVE_ERRORS):
        assert P.emit("text.pre", text="keep") == "keep", f"第 {i+1} 次异常没被隔离"

    assert lp.state == "failed"
    assert lp.handlers == {}
    assert reg.handlers_for("text.pre") == ()
    assert "boom" in lp.error
    assert lp.module.calls.count("teardown") == 1, "熔断必须释放资源（teardown 恰好一次）"

    P.emit("text.pre", text="keep")
    assert lp.module.calls.count("teardown") == 1, "重复触发不得再次 teardown"


def test_one_broken_plugin_does_not_affect_others(tmp_path):
    root = tmp_path / "plugins"
    _make_plugin(root, "broken",
                 "def on_text_pre(p):\n    raise RuntimeError('x')\n",
                 {"priority": 90, "hooks": {"text.pre": "on_text_pre"}})
    _make_plugin(root, "healthy",
                 "def on_text_pre(p):\n    return p['text'] + '!'\n",
                 {"priority": 10, "hooks": {"text.pre": "on_text_pre"}})
    _init(tmp_path)
    assert P.emit("text.pre", text="ok") == "ok!"
    assert P.get_registry().plugins["healthy"].state == "started"


def test_invalid_return_type_keeps_original_and_counts_error(tmp_path):
    _make_plugin(tmp_path / "plugins", "badtype",
                 "def on_output_post(p):\n    return 'not an array'\n",
                 {"hooks": {"output.post": "on_output_post"}})
    reg = _init(tmp_path)
    arr = np.ones(32, dtype=np.float32)
    assert P.emit("output.post", audio=arr) is arr
    assert reg.plugins["badtype"].total_errors == 1


def test_load_failure_is_recorded_not_raised(tmp_path):
    """缺少依赖 / 入口文件缺失 / 处理器签名不符 → 记 failed，不影响服务启动。"""
    _make_plugin(tmp_path / "plugins", "nofile", "", {"entry": "missing.py",
                                                     "hooks": {"text.pre": "on_text_pre"}})
    _make_plugin(tmp_path / "plugins", "badfn",
                 "def on_text_pre(): \n    return None\n",   # 0 个参数
                 {"hooks": {"text.pre": "on_text_pre"}})
    _make_plugin(tmp_path / "plugins", "nohandler",
                 "pass\n", {"hooks": {"text.pre": "on_text_pre"}})
    reg = _init(tmp_path)
    for pid in ("nofile", "badfn", "nohandler"):
        lp = reg.plugins[pid]
        assert lp.state == "failed", f"{pid} 应为 failed，实际 {lp.state}"
        assert lp.error
    assert reg.handlers_for("text.pre") == ()


# --------------------------------------------------------------------------- 7. 重入防护
def test_nested_emit_is_refused_and_returns_none(tmp_path):
    _make_plugin(tmp_path / "plugins", "nest",
                 "import voice_clone.plugins as PL\n"
                 "def on_text_pre(p):\n"
                 "    return PL.emit('text.pre', text='inner')\n",
                 {"hooks": {"text.pre": "on_text_pre"}})
    _init(tmp_path)
    # 内层被拒 → 返回 None → 外层视为"未改动" → 保持 outer
    assert P.emit("text.pre", text="outer") == "outer"


def test_hook_context_helpers(tmp_path):
    _make_plugin(tmp_path / "plugins", "ctx",
                 "import voice_clone.plugins as PL\n"
                 "def on_text_pre(p):\n"
                 "    return PL.current_hook() + '|' + str(PL.in_hook())\n",
                 {"hooks": {"text.pre": "on_text_pre"}})
    _init(tmp_path)
    assert P.in_hook() is False
    assert P.current_hook() == ""
    assert P.emit("text.pre", text="x") == "text.pre|True"
    assert P.in_hook() is False
    assert P.current_hook() == ""


def test_synthesize_stable_refuses_to_run_inside_a_hook(tmp_path):
    """钩子内同步重入生成路径 → 快速失败（否则是非可重入锁上的静默挂死）。"""
    _make_plugin(tmp_path / "plugins", "guard",
                 "import voice_clone.synthesis_stab as SS\n"
                 "outcome = []\n"
                 "def on_output_post(p):\n"
                 "    try:\n"
                 "        SS.synthesize_stable(None, 'x', None, 24000)\n"
                 "        outcome.append('no-raise')\n"
                 "    except RuntimeError:\n"
                 "        outcome.append('RuntimeError')\n"
                 "    return None\n",
                 {"hooks": {"output.post": "on_output_post"}})
    reg = _init(tmp_path)
    arr = np.ones(16, dtype=np.float32)
    assert P.emit("output.post", audio=arr) is arr
    assert reg.plugins["guard"].module.outcome == ["RuntimeError"]


def test_synthesize_stable_runs_normally_outside_hooks():
    audio, rep = SS.synthesize_stable(_FakeModel(), "普通调用。", None, 24000)
    assert isinstance(audio, np.ndarray) and audio.size > 0
    assert rep["n_chunks"] == 1


# --------------------------------------------------------------------------- 8. 生命周期
def test_lifecycle_notifications_are_idempotent(tmp_path):
    _make_plugin(tmp_path / "plugins", "life",
                 "calls = []\n"
                 "def setup(ctx):\n    calls.append('setup')\n"
                 "def teardown(ctx):\n    calls.append('teardown')\n"
                 "def on_lifecycle_startup(p):\n    calls.append('up')\n"
                 "def on_lifecycle_shutdown(p):\n    calls.append('down')\n",
                 {"hooks": {"lifecycle.startup": "on_lifecycle_startup",
                            "lifecycle.shutdown": "on_lifecycle_shutdown"}})
    reg = _init(tmp_path)
    lp = reg.plugins["life"]
    P.notify_lifecycle("startup", app=None)
    P.notify_lifecycle("startup", app=None)
    assert lp.module.calls.count("up") == 1, "启动通知必须幂等"
    P.notify_lifecycle("shutdown")
    P.notify_lifecycle("shutdown")
    assert lp.module.calls.count("down") == 1, "退出通知必须幂等"
    reg.stop_all()
    assert lp.module.calls.count("teardown") == 1
    assert lp.state == "stopped"


def test_settings_defaults_merge_with_config_overrides(tmp_path):
    _make_plugin(tmp_path / "plugins", "cfg",
                 "def on_output_post(p):\n    return None\n",
                 {"hooks": {"output.post": "on_output_post"},
                  "settings_schema": {"properties": {
                      "a": {"default": 1}, "b": {"default": "x"}}}})
    reg = _init(tmp_path, settings={"cfg": {"b": "overridden", "c": True}})
    assert reg.settings_for("cfg") == {"a": 1, "b": "overridden", "c": True}


def test_disable_and_enable_at_runtime(tmp_path):
    _make_plugin(tmp_path / "plugins", "toggle",
                 "def on_text_pre(p):\n    return p['text'] + '*'\n",
                 {"hooks": {"text.pre": "on_text_pre"}})
    reg = _init(tmp_path)
    assert P.emit("text.pre", text="a") == "a*"
    assert reg.set_enabled("toggle", False) is True
    assert P.emit("text.pre", text="a") == "a"
    assert reg.plugins["toggle"].state == "disabled"
    assert reg.set_enabled("toggle", True) is True
    assert P.emit("text.pre", text="a") == "a*"
    # 落盘：disabled 名单写进 plugins_config.json
    saved = json.loads((tmp_path / P.CONFIG_NAME).read_text(encoding="utf-8"))
    assert "toggle" not in saved["disabled"]


def test_hot_reload_picks_up_new_code(tmp_path):
    d = _make_plugin(tmp_path / "plugins", "hot",
                     "def on_text_pre(p):\n    return p['text'] + '-v1'\n",
                     {"hooks": {"text.pre": "on_text_pre"}})
    reg = _init(tmp_path)
    assert P.emit("text.pre", text="a") == "a-v1"
    (d / "plugin.py").write_text("def on_text_pre(p):\n    return p['text'] + '-v2'\n",
                                 encoding="utf-8")
    assert reg.reload("hot") is True
    assert P.emit("text.pre", text="a") == "a-v2"


# --------------------------------------------------------------------------- 9. LLM 服务商
def test_plugin_can_add_provider_but_not_override_builtin(tmp_path):
    _make_plugin(tmp_path / "plugins", "prov",
                 "def on_llm_providers(p):\n"
                 "    return p['providers'] + [\n"
                 "        {'id': 'deepseek', 'base_url': 'http://evil.invalid/v1'},\n"
                 "        {'id': 'my_local', 'base_url': 'http://127.0.0.1:1234/v1',\n"
                 "         'models': ['m1'], 'local': True}]\n",
                 {"hooks": {"llm.providers": "on_llm_providers"}})
    _init(tmp_path)
    LP.refresh_plugin_providers(force=True)
    ids = [p["id"] for p in LP.merged_providers()]
    assert ids.count("deepseek") == 1, "内置项被追加成了重复条目"
    assert LP.get_provider("deepseek")["base_url"] == "https://api.deepseek.com/v1"
    assert "my_local" in ids
    got = LP.get_provider("my_local")
    assert got["base_url"] == "http://127.0.0.1:1234/v1"
    assert got["local"] is True
    # 界面按固定键取值，缺键会 500 —— 缺省键必须被补齐
    for k in ("name_zh", "name_en", "models", "key_hint", "key_url", "note", "note_en"):
        assert k in got, f"插件服务商缺少界面所需字段 {k}"


def test_provider_cache_invalidates_when_plugin_set_changes(tmp_path):
    """缓存签名若只按"处理器数量"计，两套不同插件会张冠李戴。"""
    _make_plugin(tmp_path / "plugins", "pa",
                 "def on_llm_providers(p):\n"
                 "    return p['providers'] + [{'id': 'alpha', 'base_url': 'http://a/v1'}]\n",
                 {"hooks": {"llm.providers": "on_llm_providers"}})
    _init(tmp_path)
    assert "alpha" in [p["id"] for p in LP.merged_providers()]

    P.reset_for_tests()
    shutil.rmtree(tmp_path / "plugins" / "pa")
    _make_plugin(tmp_path / "plugins", "pb",
                 "def on_llm_providers(p):\n"
                 "    return p['providers'] + [{'id': 'beta', 'base_url': 'http://b/v1'}]\n",
                 {"hooks": {"llm.providers": "on_llm_providers"}})
    _init(tmp_path)   # 刻意不 force：若签名只按"处理器数量"算，这里会命中陈旧缓存
    ids = [p["id"] for p in LP.merged_providers()]
    assert "beta" in ids
    assert "alpha" not in ids, "缓存未失效：上一个插件追加的服务商仍在列表中"


# --------------------------------------------------------------------------- 10. 随仓库发布的示例插件
def test_shipped_example_plugin_is_disabled_by_default():
    """示例插件必须默认停用 —— 否则"零插件 = 行为不变"这条保证立刻破功。"""
    src = REPO / "plugins" / "example_gain"
    if not src.is_dir():
        pytest.skip("示例插件不存在")
    reg = P.PluginRegistry(str(REPO))
    manifests = {m.id: m for m in reg.discover()}
    assert "example_gain" in manifests, "示例插件未被发现"
    reg.load_all()
    lp = reg.plugins["example_gain"]
    assert lp.state == "disabled", f"示例插件应默认停用，实际 {lp.state}"
    assert not reg.has_hook("output.post")
    assert not reg.has_hook("report.enrich")


def test_shipped_example_plugin_works_when_enabled(tmp_path):
    src = REPO / "plugins" / "example_gain"
    if not src.is_dir():
        pytest.skip("示例插件不存在")
    dst = tmp_path / "plugins"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst / "example_gain")
    mf = dst / "example_gain" / "plugin.json"
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["enabled"] = True
    mf.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    reg = _init(tmp_path, settings={"example_gain": {"gain_db": 6.0}})
    lp = reg.plugins["example_gain"]
    assert lp.state == "started", lp.error

    arr = np.full(64, 0.1, dtype=np.float32)
    out = P.emit("output.post", audio=arr)
    assert out[0] == pytest.approx(0.1 * 10 ** (6.0 / 20.0), rel=1e-4)

    rep = P.emit("report.enrich", report={"n_chunks": 1})
    assert "n_core" not in rep
    assert "example_gain" in rep
    assert rep["n_chunks"] == 1, "插件不得覆盖核心报告字段"

    reg.stop_all()
    assert lp.state == "stopped"


def test_report_enrich_cannot_overwrite_core_fields(tmp_path):
    _make_plugin(tmp_path / "plugins", "liar",
                 "def on_report_enrich(p):\n"
                 "    return {'n_chunks': 999, 'mine': 1}\n",
                 {"hooks": {"report.enrich": "on_report_enrich"}})
    _init(tmp_path)
    rep = P.emit("report.enrich", report={"n_chunks": 3})
    assert rep["n_chunks"] == 3, "核心字段被插件覆盖了"
    assert rep["mine"] == 1


# --------------------------------------------------------------------------- 11. 内省
def test_snapshot_is_json_serializable_and_informative(tmp_path):
    _make_plugin(tmp_path / "plugins", "snap",
                 "def on_text_pre(p):\n    return p['text'] + '.'\n",
                 {"hooks": {"text.pre": "on_text_pre"}})
    _init(tmp_path)
    snap = P.snapshot()
    assert snap["initialized"] is True
    assert snap["n_plugins"] == 1 and snap["n_active"] == 1
    row = snap["plugins"][0]
    assert row["id"] == "snap" and row["state"] == "started"
    assert row["hooks"] == ["text.pre"]
    assert isinstance(json.dumps(snap), str)
    hook_names = [h["name"] for h in snap["hooks"]]
    assert "output.post" in hook_names
    assert any(h["n_handlers"] == 1 for h in snap["hooks"])
