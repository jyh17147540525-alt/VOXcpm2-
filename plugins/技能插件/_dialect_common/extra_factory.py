"""方言「拓展插件」共享工厂
==========================
技能插件管「语言」，拓展插件管「文化」：俗语、歇后语、童谣民谣、民俗。
两者共用同一套骨架（扫描行首指令 → 查表 → 讲解 → 剥离指令），差异同样只在数据。

指令（**必须带方言简称**，行首；中英文冒号都接受）
------------------------------------------------
    <lang_short>俗语：<关键词>
    <lang_short>歇后语：<关键词>
    <lang_short>童谣：<关键词>
    <lang_short>民俗：<关键词>

裸指令（`俗语：X`）交给跑在最后的「路由插件」跨方言检索 —— 原因见 commands.py：
13 个拓展插件若都认裸指令，id 最小的那个会把所有查询吃掉。

数据文件（dialect_data.json）顶层约定
------------------------------------
    lang_name      全称
    lang_short     简称（指令里用）
    sources        [str] 可查证来源清单
    coverage_note  覆盖度与置信度说明
    entries        [{id, type, text, text_raw, meaning, roman, conf, src, note}]

    type ∈ {俗语, 歇后语, 童谣, 民俗}
    text      条目的标准写法（用于展示）
    text_raw  原始口传写法（含异体字、口语记音），检索时一并参与匹配
    roman     方言注音（有则展示）
    conf      high / mid / low
    src       来源；note 备注（low 置信度必须说明待核实之处）

搜索是**子串匹配**：用户不必记得完整条目，给关键词即可。
命中才讲，未命中明确说「未收录」—— 绝不编造。
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import commands as C  # noqa: E402


def find_common(plugin_dir):
    """与 skill_factory 同款：技能插件/拓展插件两个位置都探。"""
    parent = os.path.dirname(plugin_dir)
    for cand in (
        os.path.join(parent, "_dialect_common"),
        os.path.join(os.path.dirname(parent), "技能插件", "_dialect_common"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def build(plugin_dir):
    plugin_dir = os.path.abspath(plugin_dir)
    st = {"data": None, "max_queries": 3}

    def _load():
        try:
            with open(os.path.join(plugin_dir, "dialect_data.json"), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = None
        if not isinstance(d, dict):
            d = {"entries": [], "lang_name": "方言"}
        if not isinstance(d.get("entries"), list):
            d["entries"] = []
        st["data"] = d

    # ------------------------------------------------------------------ 入口
    def setup(ctx):
        s = getattr(ctx, "settings", None) or {}
        try:
            st["max_queries"] = max(1, min(10, int(s.get("max_queries_per_call") or 3)))
        except (TypeError, ValueError):
            st["max_queries"] = 3
        _load()
        d = st["data"] or {}
        ctx.log("%s拓展插件就绪：文化条目 %d 条"
                % (d.get("lang_short") or d.get("lang_name") or "方言",
                   len(d.get("entries", []))))

    def teardown(ctx):
        ctx.log("%s拓展插件已卸载" % ((st["data"] or {}).get("lang_name") or "方言"))

    # ------------------------------------------------------------ 检索与讲解
    def _find(etype, keyword):
        """类型 + 子串匹配，最多 2 条。先比正文，再比释义。"""
        hits = []
        for e in (st["data"] or {}).get("entries", []):
            if not isinstance(e, dict):
                continue
            if etype and e.get("type") != etype:
                continue
            hay = "".join(str(e.get(k) or "") for k in ("text_raw", "text"))
            if keyword in hay or keyword in str(e.get("meaning") or ""):
                hits.append(e)
            if len(hits) >= 2:
                break
        return hits

    def _explain(e):
        roman = e.get("roman") or ""
        line = "%s —— %s" % (e.get("text_raw") or e.get("text") or "", e.get("meaning", ""))
        if roman:
            line += "\n（注音：%s）" % roman
        line += "\n【%s｜置信度:%s｜来源:%s】" % (
            e.get("type", ""), e.get("conf", "?"), e.get("src", "未注明来源"))
        if e.get("note"):
            line += "\n注：%s" % e["note"]
        return line

    # --------------------------------------------------------------- 钩子
    def on_text_pre(payload):
        text = payload.get("text") if isinstance(payload, dict) else None
        if not text or ("：" not in text and ":" not in text):
            return None
        d = st["data"] or {}
        short = d.get("lang_short") or d.get("lang_name") or "方言"
        cmds = tuple((C.scoped_prefix(short, v), v) for v in C.EXTRA_VERBS)

        out_lines, answers = [], []
        budget = st["max_queries"]
        for ln in text.splitlines():
            handled = False
            for prefix, etype in cmds:
                kw = C.match_command(ln, prefix)
                if kw is None:
                    continue
                if kw and budget > 0:
                    budget -= 1
                    handled = True
                    hits = _find(etype, kw)
                    if hits:
                        answers.append("〔%s〕\n%s"
                                       % (ln, "\n---\n".join(_explain(e) for e in hits)))
                    else:
                        answers.append("〔%s〕\n「%s」未收录——本插件只讲解可查证的"
                                       "俗语／歇后语／童谣／民俗，不编造内容；"
                                       "或用裸指令「%s：…」跨方言检索。" % (ln, kw, etype))
                break
            if not handled:
                out_lines.append(ln)

        if not answers:
            return None
        new_text = "\n".join(out_lines).rstrip()
        block = "\n\n".join(answers)
        return (new_text + "\n\n" + block) if new_text else block

    def on_report_enrich(payload):
        d = st["data"] or {}
        by_type = {}
        for e in d.get("entries", []):
            if isinstance(e, dict):
                t = e.get("type", "?")
                by_type[t] = by_type.get(t, 0) + 1
        return {
            os.path.basename(plugin_dir): {
                "lang": d.get("lang_name", "方言"),
                "short": d.get("lang_short") or d.get("lang_name") or "方言",
                "entries_by_type": by_type,
                "sources": len(d.get("sources", [])),
            }
        }

    class _Plugin:
        pass

    m = _Plugin()
    m.setup, m.teardown = setup, teardown
    m.on_text_pre, m.on_report_enrich = on_text_pre, on_report_enrich
    return m
