"""方言「技能插件」共享工厂
==========================
所有方言大区的技能插件共用同一套实现；每个方言目录只留
`dialect_data.json`（全部语料与配置）+ 一份几行的 `plugin.py` 薄壳。

为什么要抽工厂
--------------
「技能插件」的**逻辑**是通用的（扫描行首指令 → 查词典 → 生成答案 → 剥离指令），
各地区的差异全在**数据**里。若 13 个方言各抄一份 190 行代码，任何契约修订都要改 13 处，
且极易出现「改了 A 忘了 B」的静默不一致。工厂把差异收敛到数据文件。

指令（**必须带方言简称**，行首；中英文冒号都接受）
------------------------------------------------
    <lang_short>翻译：<词>       词典驱动互译（方言↔普通话）
    <lang_short>怎么说：<词>      普通话 → 方言
    <lang_short>语法：<主题>      语法差异问答

为什么必须带简称：`text.pre` 是链式的，谁先跑谁能改写文本。若 13 个方言插件都认
裸的「翻译：」，id 最小的那个会抢答一切查询、对它不认识的词回「未收录」并剥掉指令，
其余方言永远轮不到。裸指令统一交给跑在最后的「路由插件」跨方言检索（见 commands.py）。

数据文件（dialect_data.json）顶层约定
------------------------------------
    lang_name      全称，如 "吴语（太湖片·苏州）"
    lang_short     指令里用的简称，如 "苏州话"（缺省回落 lang_name）
    roman_scheme   注音方案说明
    sources        [str] 可查证来源清单
    coverage_note  覆盖度与置信度说明
    grammar_qa     {主题: 讲解}  语法问答库（可缺省）
    entries        [{id, dialect, roman, mandarin, conf, src, note}]

契约遵守（血泪版）
------------------
- 处理器只接受 1 个 payload 参数；
- `text.pre` 必须返回 **字符串**（arg 字段值），不是 payload 字典；
- 绝不回调合成接口 —— 推理锁非可重入，会静默死锁；
- 任何异常只降级（返回 None = 不改动），绝不让插件拖垮合成主流程。
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import commands as C  # noqa: E402


def find_common(plugin_dir):
    """定位共享层 _dialect_common。
    技能插件与它同级；拓展插件在 拓展插件/ 下、与 技能插件/ 同级 ——
    两处都探一次，避免「拓展插件」跑去写死「技能插件」的绝对路径。
    """
    parent = os.path.dirname(plugin_dir)
    for cand in (
        os.path.join(parent, "_dialect_common"),
        os.path.join(os.path.dirname(parent), "技能插件", "_dialect_common"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def build(plugin_dir):
    """为一个方言目录构造四个插件入口。每次调用持有独立 state，互不串味。"""
    plugin_dir = os.path.abspath(plugin_dir)
    st = {"data": None, "index": None, "common": None, "max_queries": 3, "keep_echo": False}

    def _load_data():
        try:
            with open(os.path.join(plugin_dir, "dialect_data.json"), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = None
        if not isinstance(d, dict):
            d = {"entries": [], "lang_name": "方言"}
        if not isinstance(d.get("entries"), list):
            d["entries"] = []
        return d

    def _load():
        st["data"] = _load_data()
        common_dir = find_common(plugin_dir)
        if common_dir and common_dir not in sys.path:
            sys.path.insert(0, common_dir)
        try:
            import common as _c            # noqa: F401  (_dialect_common/common.py)
            st["common"] = _c
            st["index"] = _c.build_index(st["data"])
        except Exception:
            st["common"] = None

            class _Mini:
                """共享层缺失时的最小兜底：只保留方言→普通话直查。"""

                def __init__(self, data):
                    self.d2m = {}
                    for e in data.get("entries", []):
                        if isinstance(e, dict) and e.get("dialect"):
                            self.d2m.setdefault(e["dialect"], []).append(e)

                def translate(self, term):
                    hits = self.d2m.get(term)
                    if not hits:
                        return None
                    e = hits[0]
                    return (f"「{e['dialect']}」→ 普通话「{e.get('mandarin', '?')}」"
                            f"（{e.get('roman', '')}）【置信度:{e.get('conf', '?')}】")

            st["index"] = _Mini(st["data"])

    # ------------------------------------------------------------------ 入口
    def setup(ctx):
        s = getattr(ctx, "settings", None) or {}
        try:
            st["max_queries"] = max(1, min(10, int(s.get("max_queries_per_call") or 3)))
        except (TypeError, ValueError):
            st["max_queries"] = 3
        st["keep_echo"] = bool(s.get("keep_query_echo") or False)
        _load()
        d = st["data"] or {}
        ctx.log("%s技能插件就绪：词典 %d 条，单次最多 %d 个查询"
                % (d.get("lang_short") or d.get("lang_name") or "方言",
                   len(d.get("entries", [])), st["max_queries"]))

    def teardown(ctx):
        ctx.log("%s技能插件已卸载" % ((st["data"] or {}).get("lang_name") or "方言"))

    # ------------------------------------------------------------ 答案构造
    def _answer_translate(term):
        mod = st.get("common")
        if mod is not None:
            return mod.translate(st["index"], term)
        idx = st.get("index")
        return idx.translate(term) if idx is not None else None

    def _answer_grammar(topic):
        qa = (st["data"] or {}).get("grammar_qa") or {}
        if not qa:
            return None
        short = (st["data"] or {}).get("lang_short") or "方言"
        for k in qa:
            if k in topic:
                return f"【{short}语法·{k}】{qa[k]}"
        return ("可问的主题：" + "、".join(qa.keys()) + "。用法：" + short + "语法：" + list(qa)[0])

    # --------------------------------------------------------------- 钩子
    def on_text_pre(payload):
        text = payload.get("text") if isinstance(payload, dict) else None
        if not text or ("：" not in text and ":" not in text):
            return None
        d = st["data"] or {}
        short = d.get("lang_short") or d.get("lang_name") or "方言"
        # 只认带本方言简称的指令 —— 裸指令留给路由插件，避免截胡其它方言
        cmds = ((C.scoped_prefix(short, "翻译"), _answer_translate),
                (C.scoped_prefix(short, "怎么说"), _answer_translate),
                (C.scoped_prefix(short, "语法"), _answer_grammar))

        out_lines, answers = [], []
        budget = st["max_queries"]
        for ln in text.splitlines():
            handled = False
            for prefix, fn in cmds:
                term = C.match_command(ln, prefix)
                if term is None:
                    continue
                if term and budget > 0:
                    budget -= 1
                    handled = True
                    ans = fn(term)
                    if ans:
                        answers.append(f"〔{ln}〕\n{ans}")
                    else:
                        answers.append(
                            f"〔{ln}〕\n词库未收录「{term}」——本插件只做词典驱动互译，"
                            "不编造内容；可换用其它词条，或用裸指令「翻译：…」跨方言检索。")
                break
            if not handled or st["keep_echo"]:
                out_lines.append(ln)

        if not answers:
            return None
        new_text = "\n".join(out_lines).rstrip()
        block = "\n\n".join(answers)
        return (new_text + "\n\n" + block) if new_text else block

    def on_report_enrich(payload):
        d = st["data"] or {}
        mod = st.get("common")
        if mod is not None:
            counts = mod.stats(d)
        else:
            counts = {"high": 0, "mid": 0, "low": 0, "total": 0}
            for e in d.get("entries", []):
                if isinstance(e, dict) and (e.get("dialect") or e.get("mandarin")):
                    counts["total"] += 1
                    if e.get("conf") in counts:
                        counts[e["conf"]] += 1
        return {
            os.path.basename(plugin_dir): {
                "lang": d.get("lang_name", "方言"),
                "short": d.get("lang_short") or d.get("lang_name") or "方言",
                "entries": counts,
                "sources": len(d.get("sources", [])),
            }
        }

    class _Plugin:
        pass

    m = _Plugin()
    m.setup, m.teardown = setup, teardown
    m.on_text_pre, m.on_report_enrich = on_text_pre, on_report_enrich
    return m
