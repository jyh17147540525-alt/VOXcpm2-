"""
可扩展插件接口 (Plugin Interface)
==================================
在不 fork 核心文件的前提下，让外部模块注册、加载并被调用：文本改写、音频后处理、
情绪启发式、追加 LLM 服务商预设、追加 HTTP 路由、启动/退出资源管理。

设计约束（为什么是这样）
------------------------
1. **零插件 = 与改造前逐字节一致**。`emit()` 在无插件时只是一次 dict 查询 + 提前返回，
   且**返回调用方传入的原值**；所有钩子的异常都被吞掉、只降级不抛出。
2. **绝不允许钩子回调生成路径**。`server.py` 的 `_infer_lock` 是**非可重入**
   `threading.Lock`；钩子在持锁期间回调 `synthesize_stable()` 会同线程自锁 →
   永久挂死且**无任何报错**。因此：
     - `emit()` 拒绝嵌套调用（同线程深度 > 0 直接返回原值）；
     - `synthesize_stable()` 入口检测 `in_hook()`，命中即**抛 RuntimeError**（快速失败优于静默挂死）。
3. **内置能力不可被覆盖**。插件只能*追加*，重复 id 由各模块自行拒绝并记日志。
4. **单体插件故障不影响主流程与其它插件**：异常计数 + 连续超阈值自动熔断停用。

生命周期
--------
  discovered -> validated -> loaded -> started -> (stopped | failed | disabled)
  任一阶段失败都只记录原因，绝不让异常冒泡到服务启动路径。

快速使用（插件作者视角）
------------------------
  # plugins/my_plugin/plugin.json
  {"id":"my_plugin","version":"1.0.0","api_version":"1.0","requires_app":">=2.0,<3",
   "hooks":{"output.post":"on_output_post"}}

  # plugins/my_plugin/plugin.py
  def setup(ctx):        ctx.log("启动"); ctx.settings["gain_db"]
  def on_output_post(p): return p["audio"] * 0.9
  def teardown(ctx):     pass
"""
from __future__ import annotations

import atexit
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# ----------------------------------------------------------------------------- 常量
PLUGIN_API_VERSION = "1.0"
PLUGIN_API_MAJOR = 1
#: 插件契约所面向的应用版本（与 server.py 的 FastAPI(version=...) 保持一致）
APP_VERSION = "2.1.0"
#: 插件搜索目录名。
#: ⚠️ 必须是 ``vox_plugins`` 而不是 ``plugins``：仓库里已有本模块
#:    ``voice_clone/plugins.py``。由于 pytest 会把每个测试文件所在目录
#:    （包括 ``voice_clone/``）插到 sys.path 首位，顶层名 ``plugins``
#:    会被本模块截胡，导致 ``import plugins.<某插件>`` 报
#:    "ModuleNotFoundError: 'plugins' is not a package"（实测，
#:    且只在特定收集顺序下出现 —— 最难查的一类 bug）。
#:    目录改名后，两个概念各占一个名字，互不干扰。
PLUGIN_DIR_NAME = "vox_plugins"
PLUGIN_DATA_DIR_NAME = "plugins_data"
CONFIG_NAME = "plugins_config.json"
CONFIG_EXAMPLE_NAME = "plugins_config.json.example"

#: 连续多少次钩子异常后自动停用该插件（熔断）
MAX_CONSECUTIVE_ERRORS = 5
#: 单次钩子耗时超过该毫秒数即记一条告警（不中断，仅提示）
SLOW_HOOK_WARN_MS = 250.0

#: `text.chunks` 允许的停顿类型（与 synthesis_stab._join_pieces 的分级停顿表一致）
PAUSE_TYPES = ("end", "comma", "line", "hard")

_LOG_PREFIX = "[VoxCPM2][plugin]"

# 这些顶层字段是已知的；出现其它字段只告警（向前兼容，不拒绝）
_KNOWN_MANIFEST_FIELDS = {
    "id", "name", "name_en", "version", "api_version", "requires_app", "entry",
    "enabled", "priority", "isolation", "hooks", "description", "description_en",
    "author", "requires_packages", "settings_schema",
}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PluginManifestError(ValueError):
    """清单解析/校验失败（该插件被拒绝，不影响其它插件）。"""


class PluginLoadError(RuntimeError):
    """插件模块加载或挂钩失败。"""


# ----------------------------------------------------------------------------- 日志
_logger: Callable[[str], None] | None = None


def set_logger(fn: Callable[[str], None] | None) -> None:
    """把插件日志接到服务端错误日志（可选）。默认打到 stdout。"""
    global _logger
    _logger = fn


def _log(msg: str) -> None:
    line = f"{_LOG_PREFIX} {msg}"
    if _logger is not None:
        try:
            _logger(line)
            return
        except Exception:
            pass
    print(line, flush=True)


# ----------------------------------------------------------------------------- 钩子规格
@dataclass(frozen=True)
class HookSpec:
    """一个钩子的契约：调用时机、数据流向、以及携带值的字段名。

    kind:
      - ``pipeline``  值依次流经各插件，后者收到前者的输出（可组合）
      - ``first``     第一个非空返回生效（用于"只能有一个答案"的场景，如情绪判定）
      - ``merge``     字典浅合并（**不允许覆盖既有键**，保护核心报告契约）
      - ``synth``     返回 {"audio":…, "report":…} 的局部覆盖（音频 + 报告联动）
      - ``observe``   返回值被忽略，插件只做副作用（注册路由 / 申请释放资源）
    """
    name: str
    kind: str
    arg: str
    doc: str


HOOK_SPECS: dict[str, HookSpec] = {
    "text.pre": HookSpec(
        "text.pre", "pipeline", "text",
        "合成前文本（已做规范化/多音字处理之后、分块之前）。返回新字符串。"),
    "text.chunks": HookSpec(
        "text.chunks", "pipeline", "chunks",
        "分块结果 [(text, pause_type)]；可调整停顿类型或合并/拆分（长度须与生成块数一致）。"),
    "chunk.post": HookSpec(
        "chunk.post", "pipeline", "audio",
        "单块生成并后处理之后的音频。返回新 ndarray。"),
    "synth.post": HookSpec(
        "synth.post", "synth", "audio",
        "整段合成完成、拼接之后的音频与报告（落盘之前）。"),
    "emotion.detect": HookSpec(
        "emotion.detect", "first", "emotion",
        "对单块文本的自动情绪判定；返回非空字符串即生效（优先级高者先答）。"),
    "report.enrich": HookSpec(
        "report.enrich", "merge", "report",
        "向合成报告追加自定义字段（不可覆盖核心字段）。"),
    "reference.post": HookSpec(
        "reference.post", "pipeline", "path",
        "参考音频准备完成后的文件路径（可改写为自建的处理结果）。"),
    "llm.providers": HookSpec(
        "llm.providers", "pipeline", "providers",
        "LLM 服务商预设列表；可追加条目（内置 id 不可覆盖）。"),
    "output.post": HookSpec(
        "output.post", "pipeline", "audio",
        "服务端后处理链末尾的成品音频（仍在最终限幅之前，越界由既有 declip 兜住）。"),
    "api.routes": HookSpec(
        "api.routes", "observe", "",
        "注册额外 HTTP 路由；payload 携带 app（FastAPI 实例）。"),
    "lifecycle.startup": HookSpec(
        "lifecycle.startup", "observe", "",
        "服务进程启动、监听之前；payload 携带 app/host/port/version。"),
    "lifecycle.shutdown": HookSpec(
        "lifecycle.shutdown", "observe", "",
        "服务进程退出；用于释放资源。"),
}


# ----------------------------------------------------------------------------- 版本比较
def _ver_tuple(s: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", s or "")
    if not nums:
        return ()
    return tuple(int(x) for x in nums[:4])


def check_version_spec(spec: str, version: str) -> bool:
    """校验 ``spec``（如 ``">=2.0,<3"``，逗号 = 逻辑与）是否被 ``version`` 满足。

    只支持 :mod:`packaging` 的一个安全子集（>= <= > < == !=），
    避免为一个可选特性引入额外依赖。空 spec 视为通过。
    """
    spec = (spec or "").strip()
    if not spec:
        return True
    cur = _ver_tuple(version)
    if not cur:
        return False
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(>=|<=|==|!=|>|<)\s*(.+)$", part)
        if not m:
            return False
        op, rhs_raw = m.group(1), m.group(2).strip()
        rhs = _ver_tuple(rhs_raw)
        if not rhs:
            return False
        n = max(len(cur), len(rhs))
        a = cur + (0,) * (n - len(cur))
        b = rhs + (0,) * (n - len(rhs))
        ok = {
            ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b,
            "==": a == b, "!=": a != b,
        }[op]
        if not ok:
            return False
    return True


# ----------------------------------------------------------------------------- 清单
@dataclass
class PluginManifest:
    id: str
    name: str = ""
    name_en: str = ""
    version: str = "0.0.0"
    api_version: str = ""
    requires_app: str = ""
    entry: str = "plugin.py"
    enabled: bool = True
    priority: int = 100
    isolation: str = "inprocess"
    hooks: dict[str, str] = field(default_factory=dict)
    description: str = ""
    description_en: str = ""
    author: str = ""
    requires_packages: list[str] = field(default_factory=list)
    settings_schema: dict = field(default_factory=dict)
    dir: str = ""
    warnings: list[str] = field(default_factory=list)


def parse_manifest(data: dict, plugin_dir: str = "") -> PluginManifest:
    """解析并校验 plugin.json。校验失败抛 :class:`PluginManifestError`。"""
    if not isinstance(data, dict):
        raise PluginManifestError("plugin.json 顶层必须是 JSON 对象")

    pid = str(data.get("id", "") or "").strip()
    if not pid:
        raise PluginManifestError("缺少必填字段 id")
    if not _ID_RE.match(pid):
        raise PluginManifestError(f"id「{pid}」非法：只允许字母/数字/._-，且不超过 64 字符")

    api_version = str(data.get("api_version", "") or "").strip()
    if not api_version:
        raise PluginManifestError("缺少必填字段 api_version")
    api_major = _ver_tuple(api_version)
    if not api_major:
        raise PluginManifestError(f"api_version「{api_version}」不是合法版本号")
    if api_major[0] != PLUGIN_API_MAJOR:
        raise PluginManifestError(
            f"api_version 主版本不兼容：插件声明 {api_version}，"
            f"当前运行时为 {PLUGIN_API_VERSION}（主版本 {PLUGIN_API_MAJOR}）")

    requires_app = str(data.get("requires_app", "") or "").strip()
    if requires_app and not check_version_spec(requires_app, APP_VERSION):
        raise PluginManifestError(
            f"应用版本不满足：插件要求 requires_app={requires_app}，当前应用为 {APP_VERSION}")

    isolation = str(data.get("isolation", "inprocess") or "inprocess").strip().lower()
    if isolation not in ("inprocess", "subprocess"):
        raise PluginManifestError(f"isolation「{isolation}」非法，只支持 inprocess / subprocess")
    if isolation == "subprocess":
        # 诚实降级比静默降级好：本运行时只提供进程内隔离，声明进程隔离的插件必须被拒绝，
        # 否则插件作者会以为自己的崩溃不会影响主进程 —— 那是错误的安全感。
        raise PluginManifestError(
            "本运行时仅支持 isolation=inprocess；声明 subprocess 的插件被拒绝（不静默降级）")

    raw_hooks = data.get("hooks", {}) or {}
    hooks: dict[str, str] = {}
    if isinstance(raw_hooks, list):
        for h in raw_hooks:
            hooks[str(h).strip()] = ""
    elif isinstance(raw_hooks, dict):
        for k, v in raw_hooks.items():
            if v is True or v is None or v == "":
                hooks[str(k).strip()] = ""
            else:
                hooks[str(k).strip()] = str(v).strip()
    else:
        raise PluginManifestError("hooks 必须是数组或对象")

    req_pkgs = data.get("requires_packages", []) or []
    if not isinstance(req_pkgs, list):
        raise PluginManifestError("requires_packages 必须是数组")

    schema = data.get("settings_schema", {}) or {}
    if not isinstance(schema, dict):
        raise PluginManifestError("settings_schema 必须是对象")

    try:
        priority = int(data.get("priority", 100))
    except (TypeError, ValueError):
        raise PluginManifestError("priority 必须是整数")

    m = PluginManifest(
        id=pid,
        name=str(data.get("name", "") or "") or pid,
        name_en=str(data.get("name_en", "") or ""),
        version=str(data.get("version", "0.0.0") or "0.0.0"),
        api_version=api_version,
        requires_app=requires_app,
        entry=str(data.get("entry", "plugin.py") or "plugin.py"),
        enabled=bool(data.get("enabled", True)),
        priority=priority,
        isolation=isolation,
        hooks=hooks,
        description=str(data.get("description", "") or ""),
        description_en=str(data.get("description_en", "") or ""),
        author=str(data.get("author", "") or ""),
        requires_packages=[str(x) for x in req_pkgs],
        settings_schema=schema,
        dir=plugin_dir,
    )
    for k in data:
        if k not in _KNOWN_MANIFEST_FIELDS:
            m.warnings.append(f"未知字段「{k}」（已忽略）")
    for h in hooks:
        if h not in HOOK_SPECS:
            m.warnings.append(f"未知钩子「{h}」（已忽略；可用钩子见 HOOK_SPECS）")
    for w in m.warnings:
        _log(f"{pid}: {w}")
    return m


# ----------------------------------------------------------------------------- 数值校验
def _valid_audio(v: Any) -> bool:
    if not isinstance(v, np.ndarray):
        return False
    if v.ndim not in (1, 2) or v.size == 0:
        return False
    if not np.issubdtype(v.dtype, np.number):
        return False
    head = v.reshape(-1)[:4096]
    try:
        return bool(np.all(np.isfinite(head)))
    except Exception:
        return False


def _valid_chunks(v: Any) -> bool:
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        return False
    for item in v:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        t, ptype = item
        if not isinstance(t, str) or not t.strip():
            return False
        if ptype not in PAUSE_TYPES:
            return False
    return True


def _valid_providers(v: Any) -> bool:
    """只校验形状：列表、非空、每项是带非空 ``id`` 的字典。

    **刻意不校验 id 唯一性**：插件最自然的写法是
    ``return payload["providers"] + [my_entry]``（把收到的内置项一并回传），
    若因为"含重复 id"就整份丢弃，插件等于完全失效。
    内置项的覆盖/去重由 ``llm_providers.refresh_plugin_providers`` **逐条**处理，
    因此"试图覆盖内置项"只会丢掉那一条，其余新增条目照常生效。
    """
    if not isinstance(v, list) or not v:
        return False
    for it in v:
        if not isinstance(it, dict):
            return False
        pid = it.get("id")
        if not isinstance(pid, str) or not pid.strip():
            return False
    return True


#: 每个钩子对"插件返回值"的额外校验（返回 False = 视为非法，记一次错误并保持原值）
_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "text.pre": lambda v: isinstance(v, str) and len(v) > 0,
    "text.chunks": _valid_chunks,
    "chunk.post": _valid_audio,
    "synth.post": lambda v: isinstance(v, (dict, np.ndarray)),
    "emotion.detect": lambda v: isinstance(v, str) and v.strip() != "",
    "report.enrich": lambda v: isinstance(v, dict),
    "reference.post": lambda v: isinstance(v, str) and v != "",
    "llm.providers": _valid_providers,
    "output.post": _valid_audio,
}


# ----------------------------------------------------------------------------- 线程局部状态
_tls = threading.local()


def _depth() -> int:
    return getattr(_tls, "depth", 0)


def in_hook() -> bool:
    """当前线程是否正运行在某个插件钩子内部。

    供 ``synthesize_stable()`` 等生成路径入口做"快速失败"检测。
    ``server.py`` 用**非可重入** ``threading.Lock`` 串行化推理；钩子内同步回调
    生成路径会重新使用同一个模型 / 在同一临界区内重入，结果不是死锁就是
    CUDA 状态错乱，且都**没有任何报错**。故一律拒绝，宁可抛错也不静默损坏。
    """
    return _depth() > 0


def current_hook() -> str:
    """当前线程正在执行的钩子名（不在钩子内时为空串）。

    插件可据此判断自己是否处于**推理临界区**内 ——
    ``text.pre`` / ``text.chunks`` / ``chunk.post`` / ``synth.post`` / ``emotion.detect``
    在 ``_infer_lock`` 持有期间运行；``output.post`` / ``api.routes`` / ``lifecycle.*`` 在其之外。
    """
    return getattr(_tls, "hook", "") or ""


# ----------------------------------------------------------------------------- 运行时
@dataclass
class PluginContext:
    """传给 ``setup(ctx)`` / ``teardown(ctx)`` 的上下文。"""
    id: str
    base_dir: str
    data_dir: str
    settings: dict
    log: Callable[[str], None]
    registry: "PluginRegistry"

    def settings_for(self, pid: str | None = None) -> dict:
        return self.registry.settings_for(pid or self.id)


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    module: Any = None
    handlers: dict[str, Callable] = field(default_factory=dict)
    state: str = "discovered"        # discovered/loaded/started/stopped/failed/disabled
    error: str = ""
    consecutive_errors: int = 0
    total_errors: int = 0
    hook_calls: int = 0
    last_error: str = ""
    last_duration_ms: float = 0.0
    setup_done: bool = False
    teardown_done: bool = False

    @property
    def active(self) -> bool:
        return self.state == "started" and bool(self.handlers)


class PluginRegistry:
    """插件的发现 / 加载 / 启停 / 热重载 / 内省。

    ``_by_hook`` 是"钩子 -> 已排序插件元组"的只读快照；``emit()`` 只读它，
    因此热路径无需加锁（重排时整体替换元组，赋值在 CPython 下是原子的）。
    """

    def __init__(self, base_dir: str, config_path: str | None = None):
        self.base_dir = os.path.abspath(base_dir)
        self.config_path = config_path or os.path.join(self.base_dir, CONFIG_NAME)
        self.config: dict = _load_config(self.config_path)
        self.plugins: dict[str, LoadedPlugin] = {}
        self.discovery_errors: list[str] = []
        self.fatal: str = ""
        self.started = False
        self.shutdown_done = False
        self._by_hook: dict[str, tuple[LoadedPlugin, ...]] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- 配置
    @property
    def search_paths(self) -> list[str]:
        raw = self.config.get("search_paths") or [PLUGIN_DIR_NAME]
        out: list[str] = []
        for p in raw:
            p = str(p).strip()
            if not p:
                continue
            out.append(p if os.path.isabs(p) else os.path.join(self.base_dir, p))
        return out or [os.path.join(self.base_dir, PLUGIN_DIR_NAME)]

    @property
    def disabled(self) -> set[str]:
        return {str(x) for x in (self.config.get("disabled") or [])}

    def settings_for(self, pid: str) -> dict:
        """插件设置 = settings_schema 里的 default + 配置文件的显式覆盖。"""
        out: dict = {}
        lp = self.plugins.get(pid)
        schema = (lp.manifest.settings_schema if lp else {}) or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else None
        if props is None and all(isinstance(v, dict) for v in schema.values()):
            props = schema
        for k, v in (props or {}).items():
            if isinstance(v, dict) and "default" in v:
                out[k] = v["default"]
        overrides = (self.config.get("settings") or {}).get(pid) or {}
        if isinstance(overrides, dict):
            out.update(overrides)
        return out

    def save_config(self) -> bool:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            _log(f"写配置失败 {self.config_path}: {type(e).__name__}: {e}")
            return False

    def set_enabled(self, pid: str, enabled: bool) -> bool:
        """运行期启用/停用并落盘（不影响其它插件）。"""
        with self._lock:
            dis = [str(x) for x in (self.config.get("disabled") or [])]
            if enabled:
                dis = [x for x in dis if x != pid]
            elif pid not in dis:
                dis.append(pid)
            self.config["disabled"] = dis
            lp = self.plugins.get(pid)
            if lp is not None:
                if enabled:
                    if lp.state == "disabled":
                        self._load_one(lp.manifest)
                        if lp.state == "loaded":
                            self._start_one(lp)
                else:
                    if lp.state == "started":
                        self._stop_one(lp, reason="被停用")
                    lp.state = "disabled"
                self._reindex()
            ok = self.save_config()
        return ok

    # ---------------------------------------------------------------- 发现
    def discover(self) -> list[PluginManifest]:
        found: list[PluginManifest] = []
        for root in self.search_paths:
            if not os.path.isdir(root):
                continue
            try:
                entries = sorted(os.listdir(root))
            except OSError as e:
                self.discovery_errors.append(f"无法读取 {root}: {e}")
                continue
            for name in entries:
                pdir = os.path.join(root, name)
                mf = os.path.join(pdir, "plugin.json")
                if not os.path.isdir(pdir) or not os.path.isfile(mf):
                    continue
                try:
                    with open(mf, encoding="utf-8") as f:
                        data = json.load(f)
                    found.append(parse_manifest(data, pdir))
                except PluginManifestError as e:
                    self.discovery_errors.append(f"{name}: {e}")
                    _log(f"清单被拒绝 {name}: {e}")
                except Exception as e:
                    self.discovery_errors.append(f"{name}: JSON 解析失败 {e}")
                    _log(f"清单解析失败 {name}: {type(e).__name__}: {e}")
        found.sort(key=lambda m: m.id)
        return found

    # ---------------------------------------------------------------- 加载
    def load(self, manifest: PluginManifest) -> LoadedPlugin:
        lp = LoadedPlugin(manifest=manifest)
        with self._lock:
            self.plugins[manifest.id] = lp
            self._load_one(manifest)
            if lp.state == "loaded":
                self._start_one(lp)
            self._reindex()
        return lp

    def load_all(self) -> list[PluginManifest]:
        with self._lock:
            self.started = True
            manifests = self.discover()
            auto = bool(self.config.get("autoload", True))
            dis = self.disabled
            for m in manifests:
                if m.id in self.plugins:
                    continue
                lp = LoadedPlugin(manifest=m)
                self.plugins[m.id] = lp
                if m.id in dis or not m.enabled or not auto:
                    lp.state = "disabled"
                    lp.error = "" if (m.enabled and auto) else "清单 enabled=false 或 autoload=false"
                    continue
                self._load_one(m)
                if lp.state == "loaded":
                    self._start_one(lp)
            self._reindex()
        return manifests

    def _load_one(self, manifest: PluginManifest) -> None:
        lp = self.plugins.get(manifest.id)
        if lp is None:
            return
        try:
            missing = [p for p in manifest.requires_packages
                       if importlib.util.find_spec(p.replace("-", "_")) is None]
            if missing:
                raise PluginLoadError(f"缺少依赖包: {', '.join(missing)}")

            path = os.path.join(manifest.dir, manifest.entry)
            if not os.path.isfile(path):
                raise PluginLoadError(f"入口文件不存在: {manifest.entry}")
            module = _import_plugin_module(manifest, path)

            handlers: dict[str, Callable] = {}
            for hook, target in manifest.hooks.items():
                if hook not in HOOK_SPECS:
                    continue  # 已在 parse 阶段告警
                fname = target or ("on_" + hook.replace(".", "_"))
                fn = getattr(module, fname, None)
                if not callable(fn):
                    raise PluginLoadError(f"钩子 {hook} 指向的函数 {fname}() 不存在或不可调用")
                if not _accepts_payload(fn):
                    raise PluginLoadError(
                        f"钩子 {hook} 的处理器 {fname}() 必须接受且仅接受 1 个参数(payload)")
                handlers[hook] = fn

            lp.module = module
            lp.handlers = handlers
            lp.state = "loaded"
            lp.error = ""
            _log(f"已加载 {manifest.id} v{manifest.version}"
                 f"（{len(handlers)} 个钩子: {', '.join(sorted(handlers)) or '无'}）")
        except Exception as e:
            lp.state = "failed"
            lp.error = f"{type(e).__name__}: {e}"
            lp.handlers = {}
            _log(f"加载失败 {manifest.id}: {lp.error}")

    def _start_one(self, lp: LoadedPlugin) -> None:
        setup = getattr(lp.module, "setup", None) if lp.module is not None else None
        if not callable(setup):
            lp.state = "started"
            lp.setup_done = True
            lp.teardown_done = False   # 新一轮生命周期开始
            return
        try:
            ctx = PluginContext(
                id=lp.manifest.id,
                base_dir=self.base_dir,
                data_dir=self._data_dir(lp.manifest.id),
                settings=self.settings_for(lp.manifest.id),
                log=lambda m, _pid=lp.manifest.id: _log(f"{_pid}: {m}"),
                registry=self,
            )
            setup(ctx)
            lp.state = "started"
            lp.setup_done = True
            lp.teardown_done = False   # 新一轮生命周期开始
        except Exception as e:
            lp.state = "failed"
            lp.error = f"setup() 失败: {type(e).__name__}: {e}"
            lp.handlers = {}
            _log(f"启动失败 {lp.manifest.id}: {lp.error}")
            _log(traceback.format_exc())

    def _stop_one(self, lp: LoadedPlugin, reason: str = "", final_state: str = "stopped") -> None:
        """调用 teardown()（**恰好一次**），然后置为 ``final_state``。

        ``setup_done`` / ``teardown_done`` 两个标记保证资源申请与释放严格配对 ——
        否则被熔断的插件（state 直接变 failed）会永远收不到 teardown，泄漏它在
        setup() 里打开的文件句柄 / 线程 / 连接。
        """
        if lp.setup_done and not lp.teardown_done:
            lp.teardown_done = True
            teardown = getattr(lp.module, "teardown", None) if lp.module is not None else None
            if callable(teardown):
                try:
                    ctx = PluginContext(
                        id=lp.manifest.id, base_dir=self.base_dir,
                        data_dir=self._data_dir(lp.manifest.id),
                        settings=self.settings_for(lp.manifest.id),
                        log=lambda m, _pid=lp.manifest.id: _log(f"{_pid}: {m}"),
                        registry=self,
                    )
                    teardown(ctx)
                except Exception as e:
                    _log(f"{lp.manifest.id}: teardown() 异常（已忽略）: {type(e).__name__}: {e}")
        lp.state = final_state
        if reason:
            lp.error = reason

    def _data_dir(self, pid: str) -> str:
        d = os.path.join(self.base_dir, PLUGIN_DATA_DIR_NAME, pid)
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d

    # ---------------------------------------------------------------- 索引
    def _reindex(self) -> None:
        """重建 ``_by_hook`` 快照：priority 降序（大的先跑），同优先级按 id 稳定排序。"""
        by_hook: dict[str, list[LoadedPlugin]] = {}
        active = [lp for lp in self.plugins.values() if lp.active]
        active.sort(key=lambda lp: (-lp.manifest.priority, lp.manifest.id))
        for lp in active:
            for hook in lp.handlers:
                by_hook.setdefault(hook, []).append(lp)
        self._by_hook = {h: tuple(v) for h, v in by_hook.items()}

    def handlers_for(self, hook: str) -> tuple[LoadedPlugin, ...]:
        return self._by_hook.get(hook, ())

    def has_hook(self, hook: str) -> bool:
        return bool(self._by_hook.get(hook))

    # ---------------------------------------------------------------- 热重载
    def reload(self, pid: str) -> bool:
        with self._lock:
            lp = self.plugins.get(pid)
            if lp is None:
                _log(f"热重载失败：未发现插件 {pid}")
                return False
            mf = os.path.join(lp.manifest.dir, "plugin.json")
            try:
                with open(mf, encoding="utf-8") as f:
                    manifest = parse_manifest(json.load(f), lp.manifest.dir)
            except Exception as e:
                lp.state = "failed"
                lp.error = f"重载清单失败: {type(e).__name__}: {e}"
                lp.handlers = {}
                self._reindex()
                _log(f"热重载失败 {pid}: {lp.error}")
                return False
            if lp.state == "started":
                self._stop_one(lp, reason="热重载")
            _forget_module(pid)
            lp.manifest = manifest
            lp.state = "loaded"
            lp.consecutive_errors = 0
            lp.hook_calls = 0
            self._load_one(manifest)
            if lp.state == "loaded" and pid not in self.disabled:
                self._start_one(lp)
            self._reindex()
            return lp.state == "started"

    # ---------------------------------------------------------------- 生命周期
    def start_all(self) -> None:
        """启用全部已加载插件（``load_all`` 之后一般不需要再调）。"""
        with self._lock:
            for lp in list(self.plugins.values()):
                if lp.state == "loaded" and lp.manifest.id not in self.disabled:
                    self._start_one(lp)
            self._reindex()

    def stop_all(self) -> None:
        with self._lock:
            for lp in list(self.plugins.values()):
                if lp.state == "started":
                    self._stop_one(lp, reason="服务退出", final_state="stopped")
                elif lp.state == "failed":
                    # 熔断时已收尾；这里再兜一次确保资源释放，且**不覆盖**熔断原因
                    self._stop_one(lp, final_state="failed")
                elif lp.state == "loaded" and lp.setup_done:
                    self._stop_one(lp, reason="服务退出", final_state="stopped")
            self._reindex()

    # ---------------------------------------------------------------- 内省
    def snapshot(self) -> dict:
        rows = []
        for pid in sorted(self.plugins):
            lp = self.plugins[pid]
            m = lp.manifest
            rows.append({
                "id": pid,
                "name": m.name,
                "name_en": m.name_en,
                "version": m.version,
                "api_version": m.api_version,
                "requires_app": m.requires_app,
                "author": m.author,
                "description": m.description,
                "description_en": m.description_en,
                "state": lp.state,
                "enabled": pid not in self.disabled and lp.state != "disabled",
                "priority": m.priority,
                "isolation": m.isolation,
                "hooks": sorted(lp.handlers),
                "declared_hooks": sorted(m.hooks),
                "settings": self.settings_for(pid),
                "settings_schema": m.settings_schema,
                "error": lp.error,
                "warnings": list(m.warnings),
                "stats": {
                    "hook_calls": lp.hook_calls,
                    "total_errors": lp.total_errors,
                    "consecutive_errors": lp.consecutive_errors,
                    "last_error": lp.last_error,
                    "last_duration_ms": lp.last_duration_ms,
                },
            })
        return {
            "plugin_api_version": PLUGIN_API_VERSION,
            "app_version": APP_VERSION,
            "config_path": self.config_path,
            "search_paths": self.search_paths,
            "autoload": bool(self.config.get("autoload", True)),
            "fatal": self.fatal,
            "n_plugins": len(rows),
            "n_active": sum(1 for r in rows if r["state"] == "started"),
            "discovery_errors": list(self.discovery_errors),
            "hooks": [
                {"name": h.name, "kind": h.kind, "arg": h.arg, "doc": h.doc,
                 "n_handlers": len(self.handlers_for(h.name))}
                for h in HOOK_SPECS.values()
            ],
            "plugins": rows,
        }


# ----------------------------------------------------------------------------- 模块级注册表
_REGISTRY: PluginRegistry | None = None
_INIT_LOCK = threading.RLock()


def _default_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_plugin_module(manifest: PluginManifest, path: str) -> Any:
    mod_name = f"voxcpm_plugin_{manifest.id}"
    _forget_module(manifest.id)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"无法为 {path} 建立导入规格")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _forget_module(pid: str) -> None:
    prefix = f"voxcpm_plugin_{pid}"
    for k in [k for k in list(sys.modules) if k == prefix or k.startswith(prefix + ".")]:
        sys.modules.pop(k, None)


def _accepts_payload(fn: Callable) -> bool:
    try:
        import inspect
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True  # 内建/无法取签名者，放行（调用时异常会被隔离）
    params = list(sig.parameters.values())
    has_var = any(p.kind in (p.VAR_POSITIONAL,) for p in params)
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return has_var or len(positional) == 1


def _load_config(path: str) -> dict:
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            _log(f"配置 {path} 顶层不是对象，已忽略")
    except Exception as e:
        _log(f"读取配置失败 {path}: {type(e).__name__}: {e}")
    return {"search_paths": [PLUGIN_DIR_NAME], "autoload": True,
            "disabled": [], "settings": {}}


def init(base_dir: str | None = None, config_path: str | None = None,
         autoload: bool = True) -> PluginRegistry:
    """初始化插件子系统（幂等）。

    **绝不抛出**：任何失败都降级为"空注册表 + 记录原因"，
    保证主服务启动路径不受插件影响。
    """
    global _REGISTRY
    with _INIT_LOCK:
        if _REGISTRY is not None:
            return _REGISTRY
        base = base_dir or _default_base_dir()
        try:
            reg = PluginRegistry(base, config_path)
            if autoload:
                reg.load_all()
            _REGISTRY = reg
        except Exception as e:
            _log(f"插件子系统初始化失败（已忽略，主流程不受影响）: {type(e).__name__}: {e}")
            _log(traceback.format_exc())
            fallback = PluginRegistry(base, config_path)
            fallback.fatal = f"{type(e).__name__}: {e}"
            _REGISTRY = fallback
        return _REGISTRY


def get_registry() -> PluginRegistry | None:
    return _REGISTRY


def snapshot() -> dict:
    reg = _REGISTRY
    if reg is None:
        return {
            "plugin_api_version": PLUGIN_API_VERSION,
            "app_version": APP_VERSION,
            "initialized": False,
            "n_plugins": 0,
            "n_active": 0,
            "plugins": [],
            "hooks": [
                {"name": h.name, "kind": h.kind, "arg": h.arg, "doc": h.doc,
                 "n_handlers": 0}
                for h in HOOK_SPECS.values()
            ],
            "search_paths": [],
            "config_path": "",
            "autoload": False,
            "fatal": "",
            "discovery_errors": [],
        }
    snap = reg.snapshot()
    snap["initialized"] = True
    return snap


def plugin_settings(pid: str) -> dict:
    reg = _REGISTRY
    return reg.settings_for(pid) if reg is not None else {}


def reset_for_tests() -> None:
    """清空全局注册表（仅供测试）。"""
    global _REGISTRY
    with _INIT_LOCK:
        if _REGISTRY is not None:
            try:
                _REGISTRY.stop_all()
            except Exception:
                pass
            for pid in list(_REGISTRY.plugins):
                _forget_module(pid)
        _REGISTRY = None


# ----------------------------------------------------------------------------- 熔断
def _disable_runtime(lp: LoadedPlugin, reason: str) -> None:
    """熔断：摘除该插件的全部钩子，并**立即收尾**（保证 setup 已申请的资源被释放）。

    保留 ``final_state="failed"`` 以便 ``/api/plugins`` 如实展示熔断原因。
    """
    reg = _REGISTRY
    lp.handlers = {}
    _log(f"熔断停用 {lp.manifest.id}: {reason}")
    if reg is not None:
        try:
            with reg._lock:
                reg._stop_one(lp, reason=reason, final_state="failed")
                reg._reindex()
            return
        except Exception:
            pass
    lp.state = "failed"
    lp.error = reason


# ----------------------------------------------------------------------------- 调用
def _default_for(spec: HookSpec, payload: dict) -> Any:
    if spec.kind == "observe":
        return None
    return payload.get(spec.arg)


def emit(hook: str, **payload: Any) -> Any:
    """触发钩子。

    - 无插件（或该钩子无处理器）时：**立即返回调用方传入的原值**，与改造前完全一致；
    - ``pipeline``：值依次流经各处理器，非法返回值只记错误、保持原值；
    - ``first``：第一个非空结果生效；
    - ``merge``：字典浅合并，**不覆盖既有键**（保护核心报告契约）；
    - ``synth``：接受 ndarray 或 {"audio":…, "report":…}，由调用方读取；
    - ``observe``：返回值忽略。

    任何处理器异常都被隔离：记录 + 计数，连续超阈值自动熔断该插件。
    """
    if hook not in HOOK_SPECS:
        raise ValueError(
            f"未知钩子「{hook}」；可用钩子：{', '.join(sorted(HOOK_SPECS))}")
    spec = HOOK_SPECS[hook]
    reg = _REGISTRY
    if reg is None:
        return _default_for(spec, payload)
    handlers = reg.handlers_for(hook)
    if not handlers:
        return _default_for(spec, payload)

    if _depth() > 0:
        # 嵌套 emit：插件处理器内部又触发了钩子。直接拒绝，避免递归与"回调生成路径"的死锁。
        # 注意返回 None 而不是原值：原值会被上层 pipeline 当成"插件合法改写了数据"，
        # 等于把一次被拒绝的调用伪装成成功。返回 None = 明确的"什么都没发生"。
        _log(f"拒绝嵌套钩子调用 {hook}（当前深度 {_depth()}）—— 插件处理器内不得再次触发钩子")
        return None

    _tls.depth = _depth() + 1
    _prev_hook = getattr(_tls, "hook", "")
    _tls.hook = hook
    try:
        current = _default_for(spec, payload)
        for lp in handlers:
            out = _invoke(lp, hook, payload)
            if out is None:
                continue
            validator = _VALIDATORS.get(hook)
            if validator is not None and not validator(out):
                _note_invalid(lp, hook, out)
                continue
            if spec.kind == "pipeline":
                current = out
                payload = dict(payload)
                payload[spec.arg] = out
            elif spec.kind == "first":
                current = out
                break
            elif spec.kind == "merge":
                if isinstance(current, dict):
                    added, blocked = _safe_merge(current, out)
                    if blocked:
                        _log(f"{lp.manifest.id}: report.enrich 试图覆盖核心字段 "
                             f"{blocked}（已忽略，只能新增）")
                    current = added
                else:
                    current = dict(out)
            else:  # synth / observe
                current = out
        return current
    finally:
        _tls.hook = _prev_hook
        _tls.depth = _depth() - 1


def _note_invalid(lp: LoadedPlugin, hook: str, value: Any) -> None:
    lp.consecutive_errors += 1
    lp.total_errors += 1
    lp.last_error = (f"钩子 {hook} 返回类型非法: {type(value).__name__} "
                     f"（已保持原值）")
    _log(f"{lp.manifest.id}: {lp.last_error}")
    if lp.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        _disable_runtime(lp, f"连续 {lp.consecutive_errors} 次返回类型非法，自动停用"
                             f"（最后一次：{lp.last_error}）")


def _safe_merge(base: dict, extra: dict) -> tuple[dict, list[str]]:
    """浅合并，拒绝覆盖既有键（保护核心字段，插件只能"追加"）。"""
    out = dict(base)
    blocked: list[str] = []
    for k, v in extra.items():
        if k in out:
            blocked.append(str(k))
        else:
            out[k] = v
    return out, blocked


def _invoke(lp: LoadedPlugin, hook: str, payload: dict) -> Any:
    """调用单个处理器，异常/耗时全部隔离。"""
    fn = lp.handlers.get(hook)
    if fn is None:
        return None
    t0 = time.perf_counter()
    try:
        out = fn(payload)
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        lp.consecutive_errors += 1
        lp.total_errors += 1
        lp.last_error = f"钩子 {hook}: {type(e).__name__}: {e}"
        _log(f"{lp.manifest.id}: {lp.last_error}（连续第 {lp.consecutive_errors} 次）")
        _log(traceback.format_exc())
        if lp.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            _disable_runtime(lp, f"连续 {lp.consecutive_errors} 次钩子异常，自动停用"
                                 f"（最后一次：{lp.last_error}）")
        return None
    dt = (time.perf_counter() - t0) * 1000.0
    lp.consecutive_errors = 0
    lp.hook_calls += 1
    lp.last_duration_ms = round(dt, 2)
    if dt > SLOW_HOOK_WARN_MS:
        _log(f"{lp.manifest.id}: 钩子 {hook} 耗时 {dt:.0f}ms"
             f"（超过 {SLOW_HOOK_WARN_MS:.0f}ms，建议异步化或精简）")
    return out


# ----------------------------------------------------------------------------- 生命周期通知
_LIFECYCLE_DONE: set[str] = set()
_LIFECYCLE_LOCK = threading.Lock()


def notify_lifecycle(name: str, **payload: Any) -> None:
    """触发 ``lifecycle.startup`` / ``lifecycle.shutdown``，**每进程只生效一次**。

    FastAPI 的 lifespan 钩子与 ``__main__`` 直跑两条路径都可能调用，
    幂等保证插件不会收到两次启动通知。
    """
    hook = f"lifecycle.{name}"
    if hook not in HOOK_SPECS:
        raise ValueError(f"未知生命周期事件「{name}」")
    with _LIFECYCLE_LOCK:
        if name in _LIFECYCLE_DONE:
            return
        _LIFECYCLE_DONE.add(name)
    emit(hook, **payload)


def _reset_lifecycle_for_tests() -> None:
    with _LIFECYCLE_LOCK:
        _LIFECYCLE_DONE.clear()
        _SHUTDOWN_HOOKED.discard("atexit")


_SHUTDOWN_HOOKED: set[str] = set()


def attach_atexit_shutdown() -> None:
    """注册进程退出时的插件收尾（幂等）。"""
    if "atexit" in _SHUTDOWN_HOOKED:
        return
    _SHUTDOWN_HOOKED.add("atexit")

    def _on_exit():
        try:
            notify_lifecycle("shutdown")
        finally:
            reg = _REGISTRY
            if reg is not None:
                try:
                    reg.stop_all()
                except Exception:
                    pass

    atexit.register(_on_exit)


def describe_hooks() -> list[dict]:
    """钩子清单（供文档 / 接口自省）。"""
    reg = _REGISTRY
    return [
        {"name": h.name, "kind": h.kind, "arg": h.arg, "doc": h.doc,
         "n_handlers": len(reg.handlers_for(h.name)) if reg else 0}
        for h in HOOK_SPECS.values()
    ]
