"""测试辅助：把「插件子区」变成可 import 的包路径。

背景
----
结构重组后插件位于::

    <repo>/plugins/技能插件/clear_vocal/...
    <repo>/plugins/拓展插件/<方言插件>/...

子区名是**中文**（用户要求的结构约定），而 Python 的模块标识符不允许中文，
所以 ``import plugins.技能插件.clear_vocal`` 是**语法错误**，行不通。

替代方案：把 ``plugins`` 包与子区的 ``__path__`` 打通 ——
让 ``plugins`` 成为一个**命名空间包**，其 ``__path__`` 同时包含
``plugins/技能插件`` 与 ``plugins/拓展插件``。这样::

    import plugins.clear_vocal     # ✅ 正常工作，且不含中文标识符

用法（在测试文件里）::

    from _plugin_path import ensure_plugins_importable
    ensure_plugins_importable()
    from plugins.clear_vocal import aligner

为什么不让插件直接裸放在 ``plugins/`` 下、非要加子区：
用户明确要求划出「技能插件 / 拓展插件」两个子区，这是硬需求。
本模块的职责就是**在不违反 Python 标识符规则的前提下**满足该结构。
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

ZONE_SKILL = "技能插件"
ZONE_EXTRA = "拓展插件"
ZONES = (ZONE_SKILL, ZONE_EXTRA)

_STATE = {"done": False}


def repo_root() -> Path:
    """仓库根（本文件在 core/tests/ 下 → 上溯三层是仓库根）。"""
    return Path(__file__).resolve().parent.parent.parent


def plugins_dir() -> Path:
    return repo_root() / "plugins"


def zone_dir(zone: str) -> Path:
    return plugins_dir() / zone


def ensure_plugins_importable(verbose: bool = False) -> Path:
    """让 ``import plugins.<插件id>`` 可用，覆盖两个子区。幂等。

    做法：手工构造 ``plugins`` 顶层命名空间包，把两个子区的目录塞进它的
    ``__path__``。**不依赖 __init__.py**，也不改动磁盘。
    """
    if _STATE["done"]:
        return plugins_dir()

    pdir = plugins_dir()
    zone_paths = [str(zone_dir(z)) for z in ZONES if zone_dir(z).is_dir()]

    if not zone_paths:
        raise RuntimeError(
            f"未找到插件子区。期望存在 {pdir}/{ZONE_SKILL} 或 {pdir}/{ZONE_EXTRA}")

    # 关键：先清掉可能已存在的同名模块，避免拿到残缺对象
    for name in [k for k in list(sys.modules)
                 if k == "plugins" or k.startswith("plugins.")]:
        sys.modules.pop(name, None)

    mod = types.ModuleType("plugins")
    mod.__path__ = zone_paths            # ← 命名空间包的核心
    mod.__package__ = "plugins"
    mod.__doc__ = "VoxCPM2 插件命名空间包（子区：技能插件 / 拓展插件）"
    sys.modules["plugins"] = mod

    # 同时把两个子区登记为 plugins 的子包，便于 importlib 解析
    for z in ZONES:
        zd = zone_dir(z)
        if not zd.is_dir():
            continue
        # 子区本身用 ASCII 别名暴露（技能插件 -> skill，拓展插件 -> extra）
        alias = "skill" if z == ZONE_SKILL else "extra"
        sub = types.ModuleType(f"plugins.{alias}")
        sub.__path__ = [str(zd)]
        sub.__package__ = f"plugins.{alias}"
        sub.__doc__ = f"插件子区：{z}"
        sys.modules[f"plugins.{alias}"] = sub
        setattr(mod, alias, sub)

    _STATE["done"] = True
    if verbose:
        print(f"[plugin_path] plugins.__path__ = {mod.__path__}")
    return pdir


def load_module(zone: str, plugin_id: str, module: str):
    """按**文件路径**加载插件内某个模块（当命名空间 import 不便时使用）。

    例：load_module("技能插件", "clear_vocal", "aligner")
    """
    ensure_plugins_importable()
    path = zone_dir(zone) / plugin_id / f"{module}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    full = f"voxcpm_plugin_{plugin_id}.{module}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 建立导入规格")
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    try:
        spec.loader.exec_module(m)
    except Exception:
        sys.modules.pop(full, None)
        raise
    return m
