"""方言词库共享层：所有方言插件共用的加载/检索/互译引擎。
设计原则：
- 数据与代码分离：每个插件目录放 dialect_data.json（词表），代码只读不写。
- 词条必须带 src（来源）与 conf（置信度 high/mid/low），绝不虚构。
- 互译是「词典驱动」：查表命中才返回，未命中明确报「未收录」。
"""
import json
import os

# 置信度排序，用于检索排序与展示
_CONF_ORDER = {"high": 0, "mid": 1, "low": 2}


def load_data(plugin_dir):
    """读取插件目录下的 dialect_data.json。缺失/损坏时返回空结构（插件降级为空词典）。"""
    path = os.path.join(plugin_dir, "dialect_data.json")
    if not os.path.isfile(path):
        return {"entries": []}
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return {"entries": []}
    if not isinstance(d, dict) or not isinstance(d.get("entries"), list):
        return {"entries": []}
    return d


def build_index(data):
    """建三个索引：方言词→普通话、普通话→方言词列表、词条原文索引。
    同 key 多条时按置信度排序（high 在前）。"""
    d2m = {}   # dialect term -> [entry]
    m2d = {}   # mandarin term -> [entry]
    by_id = {}
    for e in data.get("entries", []):
        if not isinstance(e, dict):
            continue
        dt = (e.get("dialect") or "").strip()
        mn = (e.get("mandarin") or "").strip()
        if dt:
            d2m.setdefault(dt, []).append(e)
        if mn:
            m2d.setdefault(mn, []).append(e)
        eid = e.get("id")
        if eid:
            by_id[eid] = e
    for idx in (d2m, m2d):
        for k in idx:
            idx[k].sort(key=lambda x: _CONF_ORDER.get(x.get("conf"), 3))
    return {"d2m": d2m, "m2d": m2d, "by_id": by_id}


def lookup(index, term):
    """查一个方言词或普通话词，返回 (direction, entries) 或 (None, [])。
    direction: 'd2m'（方言→普通话）或 'm2d'（普通话→方言）。"""
    if term in index["d2m"]:
        return "d2m", index["d2m"][term]
    if term in index["m2d"]:
        return "m2d", index["m2d"][term]
    return None, []


def translate(index, term):
    """词典驱动互译。返回人类可读的多行字符串；未收录时明确说明。"""
    direction, hits = lookup(index, term)
    if not hits:
        return None
    lines = []
    for e in hits[:3]:  # 最多取 3 条，避免刷屏
        conf = e.get("conf", "?")
        src = e.get("src", "未注明来源")
        if direction == "d2m":
            lines.append(f"「{e['dialect']}」→ 普通话「{e.get('mandarin', '?')}」"
                         f"（{e.get('roman', '')}）【置信度:{conf}｜来源:{src}】")
        else:
            lines.append(f"普通话「{term}」→ {e.get('lang_name', '方言')}"
                         f"「{e['dialect']}」（{e.get('roman', '')}）【置信度:{conf}｜来源:{src}】")
    note = e.get("note") or ""
    if note:
        lines.append(f"    注：{note}")
    return "\n".join(lines)


def stats(data):
    """词条统计（按置信度），用于 report.enrich 与自检。"""
    c = {"high": 0, "mid": 0, "low": 0, "total": 0}
    for e in data.get("entries", []):
        if isinstance(e, dict) and (e.get("dialect") or e.get("mandarin")):
            c["total"] += 1
            conf = e.get("conf")
            if conf in c:
                c[conf] += 1
    return c
