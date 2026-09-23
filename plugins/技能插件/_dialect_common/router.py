"""跨方言检索 / 兜底路由插件
==========================
跑在 `text.pre` 链的**最后**（priority 低于所有方言插件），承担三件事：

1. **跨方言检索**：处理裸指令 —— `翻译：X`、`怎么说：X`、`俗语：X`、`歇后语：X`、
   `童谣：X`、`民俗：X`、`语法：T`。它读遍全部方言词表，把命中的方言一起列出来。
   例：`翻译：聊天` → 一次给出 唠嗑（北方）/ 谝（中原）/ 摆龙门阵（西南）/ 倾偈（粤）…
2. **兜底**：某个方言插件被停用或启动失败时，它那份「`<简称>翻译：X`」不会被任何插件消费，
   路由插件直接读词表作答，用户不会因为停用插件就查不到东西。
3. **未收录的最终裁决**：全表都查不到才回「未收录」，并且报出实际检索了多少张表
   —— 让「查不到」这件事本身是可核查的，而不是含糊的失败。

为什么需要它：`text.pre` 链式执行，裸指令会被 id 最小的插件截胡（详见 commands.py）。
把裸指令集中到跑在最后的单一插件处理，是让 13 个方言插件共存的唯一干净解法。

数据来源：直接读各插件目录下的 dialect_data.json（只读，不写），
因此**不依赖**那些插件是否启用、是否加载成功。
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import commands as C  # noqa: E402

_CONF_ORDER = {"high": 0, "mid": 1, "low": 2}
_DEFAULT_MAX_HITS = 6


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _collect(plugins_root):
    """扫描插件目录，收集所有方言表。返回 (skill_langs, extra_langs, shorts)。"""
    skill_langs, extra_langs, shorts = [], [], []
    plan = (
        (os.path.join(plugins_root, "技能插件"), "_skill", skill_langs),
        (os.path.join(plugins_root, "拓展插件"), "_extra", extra_langs),
    )
    for root, suffix, sink in plan:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for nm in names:
            if not nm.startswith("dialect_") or not nm.endswith(suffix):
                continue
            d = _read_json(os.path.join(root, nm, "dialect_data.json"))
            if not d:
                continue
            short = d.get("lang_short") or d.get("lang_name") or nm
            sink.append({
                "id": nm,
                "short": short,
                "name": d.get("lang_name") or nm,
                "grammar_qa": d.get("grammar_qa") or {},
                "entries": [e for e in (d.get("entries") or []) if isinstance(e, dict)],
            })
            shorts.append(short)
    return skill_langs, extra_langs, shorts


def build(plugin_dir):
    plugin_dir = os.path.abspath(plugin_dir)
    plugins_root = os.path.dirname(os.path.dirname(plugin_dir))
    st = {"skill": [], "extra": [], "shorts": [], "max_queries": 3,
          "max_hits": _DEFAULT_MAX_HITS, "loaded": False}

    def _load():
        skill, extra, shorts = _collect(plugins_root)
        st["skill"], st["extra"], st["shorts"] = skill, extra, shorts
        st["loaded"] = True

    def _n_tables():
        return (len(st["skill"]), len(st["extra"]))

    # ------------------------------------------------------------ 查询原语
    def _search_skill(langs, term, limit):
        """在若干张技能表里查词，每张表最多取 1 条（置信度最高者）。"""
        out = []
        for L in langs:
            best = None
            for e in L["entries"]:
                if term == (e.get("dialect") or "") or term == (e.get("mandarin") or ""):
                    if best is None or _CONF_ORDER.get(e.get("conf"), 3) < \
                            _CONF_ORDER.get(best.get("conf"), 3):
                        best = e
            if best is not None:
                out.append((L, best))
        out.sort(key=lambda x: _CONF_ORDER.get(x[1].get("conf"), 3))
        return out[:limit]

    def _search_extra(langs, etype, kw, limit):
        out = []
        for L in langs:
            best = None
            for e in L["entries"]:
                if etype and e.get("type") != etype:
                    continue
                hay = "".join(str(e.get(k) or "") for k in ("text_raw", "text"))
                if kw in hay or kw in str(e.get("meaning") or ""):
                    if best is None or _CONF_ORDER.get(e.get("conf"), 3) < \
                            _CONF_ORDER.get(best.get("conf"), 3):
                        best = e
            if best is not None:
                out.append((L, best))
        out.sort(key=lambda x: _CONF_ORDER.get(x[1].get("conf"), 3))
        return out[:limit]

    def _search_grammar(langs, topic, limit):
        out = []
        for L in langs:
            for k, v in (L["grammar_qa"] or {}).items():
                if topic in k:
                    out.append((L, k, v))
                    break
        return out[:limit]

    # ------------------------------------------------------------ 答案渲染
    def _roman(e):
        r = e.get("roman") or ""
        return f"（{r}）" if r else ""

    def _line_skill(L, e, term):
        if term == (e.get("dialect") or ""):
            body = f"「{e.get('dialect')}」→ 普通话「{e.get('mandarin', '?')}」"
        else:
            body = f"普通话「{term}」→「{e.get('dialect')}」"
        return (f"· 【{L['short']}】{body}{_roman(e)}"
                f"｜置信度:{e.get('conf', '?')}｜{e.get('src', '未注明来源')}")

    def _line_extra(L, e):
        return (f"· 【{L['short']}】{e.get('text_raw') or e.get('text') or ''}"
                f" —— {e.get('meaning', '')}｜置信度:{e.get('conf', '?')}｜"
                f"{e.get('src', '未注明来源')}")

    def _answer(ln, verb, term, scope):
        n_skill, n_extra = _n_tables()
        if verb == "语法":
            langs = [L for L in st["skill"] if scope is None or L["short"] == scope]
            rows = _search_grammar(langs, term, st["max_hits"])
            if not rows:
                return (f"〔{ln}〕\n「{term}」不是已收录的语法主题。"
                        f"可换用裸指令试其它主题，或直接问某个方言，"
                        f"如「{st['shorts'][0] if st['shorts'] else '粤语'}语法：语序」。")
            body = "\n".join(f"· 【{L['short']}】{k}：{v}" for L, k, v in rows)
            return (f"〔{ln}〕\n跨方言语法检索（{len(rows)} 个方言命中）：\n{body}\n"
                    f"定向提问可写「{rows[0][0]['short']}语法：{term}」。")

        if verb in ("翻译", "怎么说"):
            langs = [L for L in st["skill"] if scope is None or L["short"] == scope]
            rows = _search_skill(langs, term, st["max_hits"])
            if not rows:
                return (f"〔{ln}〕\n「{term}」在 {len(langs)} 张方言词表中均未收录"
                        "——本插件族只做词典驱动互译，不编造内容。"
                        "可换用其它词条，或用「<方言简称>翻译：…」定向查询。")
            head = ("跨方言检索命中 %d 个方言" % len(rows)) if scope is None \
                else ("【%s】定向检索命中" % scope)
            body = "\n".join(_line_skill(L, e, term) for L, e in rows)
            tail = ""
            if scope is None and len(rows) > 1:
                tail = ("\n定向查询可写「%s翻译：%s」（简称见方括号）。"
                        % (rows[0][0]["short"], term))
            return f"〔{ln}〕\n{head}：\n{body}{tail}"

        # 俗语 / 歇后语 / 童谣 / 民俗
        langs = [L for L in st["extra"] if scope is None or L["short"] == scope]
        rows = _search_extra(langs, verb, term, st["max_hits"])
        if not rows:
            return (f"〔{ln}〕\n「{term}」在 {len(langs)} 张拓展表中均未收录"
                    "——本插件族只讲解可查证的条目，不编造内容。"
                    "注意：部分方言的该类目尚未采录，未收录不等于该方言没有这个说法。")
        head = ("跨方言检索命中 %d 个方言" % len(rows)) if scope is None \
            else ("【%s】定向检索命中" % scope)
        body = "\n".join(_line_extra(L, e) for L, e in rows)
        return f"〔{ln}〕\n{head}：\n{body}"

    # --------------------------------------------------------------- 钩子
    def setup(ctx):
        s = getattr(ctx, "settings", None) or {}
        try:
            st["max_queries"] = max(1, min(10, int(s.get("max_queries_per_call") or 3)))
        except (TypeError, ValueError):
            st["max_queries"] = 3
        try:
            st["max_hits"] = max(1, min(20, int(s.get("max_hits_per_query") or _DEFAULT_MAX_HITS)))
        except (TypeError, ValueError):
            st["max_hits"] = _DEFAULT_MAX_HITS
        _load()
        ctx.log("跨方言路由就绪：技能表 %d 张、拓展表 %d 张、方言简称 %d 个"
                % (len(st["skill"]), len(st["extra"]), len(set(st["shorts"]))))

    def teardown(ctx):
        ctx.log("跨方言路由插件已卸载")

    def _match(ln):
        """返回 (scope, verb, term, prefix_len) 或 None。
        先长后短地匹配，保证「粤语翻译：」优先于任何更短的候选。"""
        cands = []
        for short in set(st["shorts"]):
            for verb in C.BARE_VERBS:
                cands.append((C.scoped_prefix(short, verb), short, verb))
        for verb in C.BARE_VERBS:
            cands.append((verb + "：", None, verb))
        cands.sort(key=lambda x: -len(x[0]))
        for prefix, scope, verb in cands:
            term = C.match_command(ln, prefix)
            if term is not None:
                return scope, verb, term
        return None

    def on_text_pre(payload):
        text = payload.get("text") if isinstance(payload, dict) else None
        if not text or ("：" not in text and ":" not in text):
            return None
        if not st["loaded"]:
            _load()

        out_lines, answers = [], []
        budget = st["max_queries"]
        for ln in text.splitlines():
            got = _match(ln)
            if got is None:
                out_lines.append(ln)
                continue
            scope, verb, term = got
            if not term or budget <= 0:
                out_lines.append(ln)
                continue
            budget -= 1
            answers.append(_answer(ln, verb, term, scope))

        if not answers:
            return None
        new_text = "\n".join(out_lines).rstrip()
        block = "\n\n".join(answers)
        return (new_text + "\n\n" + block) if new_text else block

    def on_report_enrich(payload):
        n_entries = sum(len(L["entries"]) for L in st["skill"] + st["extra"])
        return {
            os.path.basename(plugin_dir): {
                "skill_tables": len(st["skill"]),
                "extra_tables": len(st["extra"]),
                "dialect_shorts": sorted(set(st["shorts"])),
                "entries_scanned": n_entries,
            }
        }

    class _Plugin:
        pass

    m = _Plugin()
    m.setup, m.teardown = setup, teardown
    m.on_text_pre, m.on_report_enrich = on_text_pre, on_report_enrich
    return m
