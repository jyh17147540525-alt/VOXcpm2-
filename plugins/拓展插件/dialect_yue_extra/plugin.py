"""粤语·俗语歇后语插件（拓展插件样板）
=====================================
演示「拓展插件」形态：俗语/歇后语/童谣/民俗的查询与讲解。

指令格式（行首）：
  俗语：<关键词或条目文本>
  歇后语：<前半句关键词>
  民俗：<关键词>

检索是「词典驱动」：按子串匹配 dialect_data.json 的 text/text_raw 字段，
命中才讲，未命中明确说「未收录」。绝不编造。
"""

import json
import os

_STATE = {"data": None, "max_queries": 3}

_CMD = {"俗语：": "俗语", "歇后语：": "歇后语", "民俗：": "民俗"}


def _load():
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialect_data.json")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"entries": [], "lang_name": "粤语"}


def setup(ctx):
    s = ctx.settings or {}
    try:
        _STATE["max_queries"] = max(1, min(10, int(s.get("max_queries_per_call") or 3)))
    except (TypeError, ValueError):
        _STATE["max_queries"] = 3
    _STATE["data"] = _load()
    n = len(_STATE["data"].get("entries", []))
    ctx.log(f"粤语拓展插件就绪：文化条目 {n} 条")


def teardown(ctx):
    ctx.log("粤语拓展插件已卸载")


def _find(etype, keyword):
    """按类型+子串匹配。返回最多 2 条。"""
    hits = []
    for e in (_STATE["data"] or {}).get("entries", []):
        if not isinstance(e, dict):
            continue
        if etype and e.get("type") != etype:
            continue
        hay = (e.get("text_raw") or e.get("text") or "")
        if keyword in hay or keyword in (e.get("meaning") or ""):
            hits.append(e)
        if len(hits) >= 2:
            break
    return hits


def _explain(e):
    conf = e.get("conf", "?")
    src = e.get("src", "未注明来源")
    body = e.get("text_raw") or e.get("text") or ""
    roman = e.get("roman") or ""
    line = f"{body} —— {e.get('meaning', '')}"
    if roman:
        line += f"\n（粤拼：{roman}）"
    line += f"\n【{e.get('type', '')}｜置信度:{conf}｜来源:{src}】"
    if e.get("note"):
        line += f"\n注：{e['note']}"
    return line


def on_text_pre(payload):
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text:
        return None
    lines = text.splitlines()
    out_lines = []
    answers = []
    budget = _STATE["max_queries"]
    for ln in lines:
        handled = False
        for prefix, etype in _CMD.items():
            if ln.startswith(prefix) or ln.startswith(prefix.replace("：", ":")):
                kw = ln[len(prefix):].strip()
                if kw and budget > 0:
                    hits = _find(etype, kw)
                    if hits:
                        answers.append(f"〔{ln}〕\n" +
                                       "\n---\n".join(_explain(e) for e in hits))
                    else:
                        answers.append(f"〔{ln}〕\n「{kw}」未收录——本插件只讲解可查证的"
                                       "俗语/歇后语/民俗，不编造内容。")
                    budget -= 1
                    handled = True
                break
        if not handled:
            out_lines.append(ln)
    if not answers:
        return None
    new_text = "\n".join(out_lines).rstrip()
    block = "\n\n".join(answers)
    return (new_text + "\n\n" + block) if new_text else block


def on_report_enrich(payload):
    data = _STATE.get("data") or {}
    by_type = {}
    for e in data.get("entries", []):
        if isinstance(e, dict):
            t = e.get("type", "?")
            by_type[t] = by_type.get(t, 0) + 1
    return {
        "dialect_yue_extra": {
            "lang": data.get("lang_name", "粤语"),
            "entries_by_type": by_type,
            "sources": len(data.get("sources", [])),
        }
    }
