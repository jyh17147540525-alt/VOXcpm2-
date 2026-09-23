"""粤语·语言技能插件
==================
演示「技能插件」的完整形态：词典驱动的词汇互译 + 常用语 + 发音要点 + 语法问答。

工作方式（text.pre 钩子）：
  合成文本里出现查询指令时，插件拦截指令、生成「答案文本」，并把指令从朗读文本中剥离。
  支持的指令格式（行首）：
    翻译：<词语>          —— 词典驱动互译（方言↔普通话）
    粤语怎么说：<普通话>   —— 普通话→粤语
    粤语怎么说：<普通话>（拼音） —— 同上，答案附粤拼

契约遵守：
  - 处理器只接受 1 个 payload 参数；
  - 绝不调用合成接口（不重入推理临界区）；
  - text.pre 返回字符串（arg 字段值），不是 payload 字典。

数据：dialect_data.json，全部词条带 src（来源）与 conf（置信度）。
"""

import json
import os

_STATE = {"index": None, "data": None, "max_queries": 3, "keep_echo": False}

# 指令前缀（行首匹配；中英文冒号都接受，匹配时统一处理）
_CMD_TRANSLATE = "翻译："
_CMD_ASK_YUE = "粤语怎么说："
_CMD_GRAMMAR = "粤语语法："

# 内置语法问答（可查证的基础差异；不虚构）
_GRAMMAR_QA = {
    "语序": "粤语基本语序与普通话相同（主谓宾），但双宾语和间接宾语位置常不同："
           "普通话「给我一本书」，粤语多说「畀本书我」（给+宾语+我）。",
    "量词": "粤语量词可单独充当指示成分：「本书好靓」（这本书很好看）"
           "无需「这/那」，量词直接跟名词，普通话无此用法。",
    "体貌": "粤语完成体用「咗」（食咗饭=吃了饭），经历体用「过」，"
           "进行体用「紧」（食紧饭=正在吃饭），与普通话「了/过/着」大体对应但不完全等同。",
    "否定": "普通话「没有」粤语作「冇」（mou5）；「不是」作「唔系」；"
           "「不要/别」作「咪」（mai6）或「唔好」。",
    "疑问": "粤语疑问句末常用「咩」（是非问，带惊讶）、「呀」、「喔」；"
           "特指问用「乜嘢」（什么）、「点解」（为什么）、「边度」（哪里）。",
}


def _load(plugin_dir):
    """加载词典并建索引。任何失败都降级为空词典（插件不致挂掉合成主流程）。"""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "dialect_data.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {"entries": [], "lang_name": "粤语"}
    try:
        import sys
        common_dir = os.path.join(os.path.dirname(here), "_dialect_common")
        if common_dir not in sys.path:
            sys.path.insert(0, common_dir)
        import common  # noqa: E402
        _STATE["common_mod"] = common                       # 模块（函数用）
        _STATE["index"] = common.build_index(data)          # 索引 dict
    except Exception:
        # 共享库缺失时的最小内联兜底：只保留直查能力
        class _Mini:
            def __init__(self, data):
                self.d2m = {}
                for e in data.get("entries", []):
                    if isinstance(e, dict) and e.get("dialect"):
                        self.d2m.setdefault(e["dialect"], []).append(e)

            def lookup(self, term):
                if term in self.d2m:
                    return "d2m", self.d2m[term]
                return None, []

            def translate(self, term):
                _, hits = self.lookup(term)
                if not hits:
                    return None
                e = hits[0]
                return (f"「{e['dialect']}」→ 普通话「{e.get('mandarin', '?')}」"
                        f"（{e.get('roman', '')}）【置信度:{e.get('conf', '?')}】")

        _STATE["common_mod"] = None
        _STATE["index"] = _Mini(data)
    _STATE["data"] = data


def setup(ctx):
    s = ctx.settings or {}
    try:
        _STATE["max_queries"] = max(1, min(10, int(s.get("max_queries_per_call") or 3)))
    except (TypeError, ValueError):
        _STATE["max_queries"] = 3
    _STATE["keep_echo"] = bool(s.get("keep_query_echo") or False)
    _load(ctx.plugin_dir if hasattr(ctx, "plugin_dir") else os.path.dirname(os.path.abspath(__file__)))
    n = len((_STATE["data"] or {}).get("entries", []))
    ctx.log(f"粤语技能插件就绪：词典 {n} 条，单次最多处理 {_STATE['max_queries']} 个查询")


def teardown(ctx):
    ctx.log("粤语技能插件已卸载")


# ---------------------------------------------------------------- 指令处理
def _answer_translate(term):
    """词典驱动互译：优先共享库；共享库缺失时用内联 Mini 索引。"""
    mod = _STATE.get("common_mod")
    if mod is not None:
        return mod.translate(_STATE["index"], term)
    idx = _STATE.get("index")
    return idx.translate(term) if idx is not None else None


def _answer_ask(term):
    return _answer_translate(term)


def _answer_grammar(topic):
    for k in _GRAMMAR_QA:
        if k in topic:
            return f"【粤语语法·{k}】{_GRAMMAR_QA[k]}"
    return ("可问的主题：" + "、".join(_GRAMMAR_QA.keys()) +
            "。用法：粤语语法：量词")


def on_text_pre(payload):
    """扫描文本中的查询指令 → 生成答案 → 从朗读文本中剥离指令（默认）。

    payload 形如 {"text": "..."}；必须返回字符串（text.pre 的 arg 契约）。
    """
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text or "：" not in text and ":" not in text:
        return None  # 无指令，不改动

    lines = text.splitlines()
    out_lines = []
    answers = []
    budget = _STATE["max_queries"]
    for ln in lines:
        handled = False
        for cmd_prefix, fn in ((_CMD_TRANSLATE, _answer_translate),
                               (_CMD_ASK_YUE, _answer_ask),
                               (_CMD_GRAMMAR, _answer_grammar)):
            # 中英文冒号都接受：把行首的「前缀：」或「前缀:」都视为指令
            plain = cmd_prefix[:-1] + ":"
            if ln.startswith(cmd_prefix) or ln.startswith(plain):
                term = ln[len(cmd_prefix):].lstrip(":").strip()
                if term and budget > 0:
                    ans = fn(term)
                    if ans:
                        answers.append(f"〔{ln}〕\n{ans}")
                        budget -= 1
                        handled = True
                    else:
                        answers.append(f"〔{ln}〕\n词库未收录「{term}」——本插件只做词典驱动互译，"
                                       "不编造内容。可尝试其他词条。")
                        budget -= 1
                        handled = True
                break
        if not handled or _STATE["keep_echo"]:
            out_lines.append(ln)

    if not answers:
        return None
    # 答案附在正文之后，用空行分隔（不插入正文中间，避免打断韵律）
    new_text = "\n".join(out_lines).rstrip()
    block = "\n\n".join(answers)
    return (new_text + "\n\n" + block) if new_text else block


def on_report_enrich(payload):
    """向合成报告追加词典统计（只增不改）。"""
    data = _STATE.get("data") or {}
    c = {"high": 0, "mid": 0, "low": 0, "total": 0}
    for e in data.get("entries", []):
        if isinstance(e, dict) and (e.get("dialect") or e.get("mandarin")):
            c["total"] += 1
            if e.get("conf") in c:
                c[e["conf"]] += 1
    return {
        "dialect_yue_skill": {
            "lang": data.get("lang_name", "粤语"),
            "entries": c,
            "sources": len(data.get("sources", [])),
        }
    }
