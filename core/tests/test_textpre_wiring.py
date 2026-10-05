# -*- coding: utf-8 -*-
"""护栏：`text.pre` 必须在**两条**合成路径上都生效。

背景（这是一个真实发生过的可用性缺陷）
--------------------------------------
`text.pre` 的唯一调用点挂在 `synthesis_stable()` 内部。而 `_do_generate()` 里：

    use_stable = _stable or len(text) >= LONG_TEXT_CHARS   # 100
    if use_stable:
        wav = synthesize_stable(...)      # ← text.pre 在里面
    else:
        wav = model.generate(**kwargs)    # ← 曾经没有 text.pre

于是「我写了一句 8 个字的 `粤语翻译：聊天`」——短文本、没勾「长文本稳定合成」——
插件根本不会被触发，指令被原样念出来。文本改写明明与文本长短无关，
却因为走哪条路径而行为不同，属语义分裂。

本文件用 AST 静态断言把「两条路径都接上」钉死，避免将来被静默删掉。
（选择 AST 而非正则：需要精确表达"在哪个分支、第几条语句、谁在谁前面"，
正则容易在注释和字符串上误命中。）
"""
import ast
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _server_path():
    """兼容两种布局：本地树 <repo>/server.py，发布树 <repo>/core/server.py。"""
    for cand in (os.path.join(REPO, "server.py"),
                 os.path.join(REPO, "core", "server.py")):
        if os.path.isfile(cand):
            return cand
    pytest.skip("未找到 server.py")


@pytest.fixture(scope="module")
def server_ast():
    return ast.parse(open(_server_path(), encoding="utf-8").read(), filename="server.py")


def _find_do_generate(tree):
    for st in ast.walk(tree):
        if isinstance(st, ast.FunctionDef) and st.name == "_do_generate":
            return st
    return None


def _find_use_stable_if(fn):
    for st in ast.walk(fn):
        if isinstance(st, ast.With):
            for item in st.items:
                c = item.context_expr
                if isinstance(c, ast.Name) and c.id == "_infer_lock":
                    for inner in st.body:
                        if isinstance(inner, ast.If) and "use_stable" in ast.unparse(inner.test):
                            return inner
    return None


def _flat(stmts):
    out = []
    for st in stmts:
        for sub in ast.walk(st):
            if isinstance(sub, ast.stmt):
                out.append(sub)
    out.sort(key=lambda s: (s.lineno, s.col_offset))
    return out


def _emit_text_pre_nodes(stmts):
    found = []
    for st in stmts:
        if not isinstance(st, ast.Assign):
            continue
        c = st.value
        if not isinstance(c, ast.Call):
            continue
        f = c.func
        if not (isinstance(f, ast.Attribute) and f.attr == "emit"):
            continue
        if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "text.pre":
            found.append(st)
    return found


def test_do_generate_exists(server_ast):
    assert _find_do_generate(server_ast) is not None, "server.py 里找不到 _do_generate"


def test_both_synthesis_paths_emit_text_pre(server_ast):
    """稳定路径经 synthesize_stable，直通路径经 _do_generate 自身 —— 两条都要有。"""
    fn = _find_do_generate(server_ast)
    ifst = _find_use_stable_if(fn)
    assert ifst is not None, "找不到 `if use_stable` 分支结构"

    stable_src = "\n".join(ast.unparse(s) for s in ifst.body)
    assert "synthesize_stable" in stable_src, \
        "稳定路径不再调用 synthesize_stable（它内部挂着 text.pre）"

    # 直通路径必须自己调一次
    hits = _emit_text_pre_nodes(ifst.orelse)
    assert hits, (
        "短文本直通路径（else 分支）没有调用 text.pre —— "
        "会导致插件只在你勾选「长文本稳定合成」时才生效")
    assert len(hits) == 1, "直通路径出现 %d 次 text.pre 调用，应当只有 1 次" % len(hits)


def test_direct_path_emits_before_generate_and_rewrites_text(server_ast):
    """顺序与回写：emit → 守卫 → 回写 kwargs['text'] → model.generate。"""
    fn = _find_do_generate(server_ast)
    ifst = _find_use_stable_if(fn)
    flat = _flat(ifst.orelse)

    i_emit = i_kw = i_gen = None
    for i, st in enumerate(flat):
        if i_emit is None and _emit_text_pre_nodes([st]):
            i_emit = i
        if i_kw is None and isinstance(st, ast.Assign) and any(
                isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Name) and t.value.id == "kwargs"
                and isinstance(t.slice, ast.Constant) and t.slice.value == "text"
                for t in st.targets):
            i_kw = i
        if i_gen is None and isinstance(st, ast.Assign) \
                and isinstance(st.value, ast.Call) \
                and isinstance(st.value.func, ast.Attribute) \
                and st.value.func.attr == "generate":
            i_gen = i

    assert i_emit is not None, "直通路径缺少 emit('text.pre')"
    assert i_kw is not None, "emit 的返回值没有被回写进 kwargs['text']（等于白调）"
    assert i_gen is not None, "直通路径缺少 model.generate"
    assert i_emit < i_kw < i_gen, \
        "顺序错误：emit(%s) / 回写(%s) / generate(%s)" % (i_emit, i_kw, i_gen)

    # 返回值必须有 isinstance 守卫，否则插件给回非字符串会污染 model.generate
    guard = any(
        isinstance(s, ast.If) and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "isinstance"
            for n in ast.walk(s.test))
        for s in flat[i_emit:i_gen])
    assert guard, "emit 返回值缺少 isinstance(str) 守卫"


def test_synthesis_stab_still_emits_text_pre():
    """稳定路径的实现本体里 text.pre 仍存在（防止有人只在 _do_generate 里保留）。"""
    for cand in (os.path.join(REPO, "voice_clone", "synthesis_stab.py"),
                 os.path.join(REPO, "core", "voice_clone", "synthesis_stab.py")):
        if os.path.isfile(cand):
            src = open(cand, encoding="utf-8").read()
            assert 'emit("text.pre"' in src or "emit('text.pre'" in src, \
                "synthesis_stab.py 里的 text.pre 调用点不见了"
            return
    pytest.skip("未找到 synthesis_stab.py")
