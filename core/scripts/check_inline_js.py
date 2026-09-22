#!/usr/bin/env python
"""
校验 server.py 内联前端 <script> 的 JS 语法（用 node --check）
=============================================================
为什么需要这个
--------------
server.py 把整个前端（HTML + CSS + JS）作为字符串内联返回。历史上曾出现：

  · 用 JS 字符串拼接动态按钮：
      '<button onclick="previewPack('' + p.id + '')">…</button>'
    V8/Chrome 解析时直接 SyntaxError -> **整段 <script> 解析失败**
    -> 所有 onclick 失活、tab 切换失灵、init 也跑不到。
    表面症状是"UI 全部交互失效"，但根因只是一个字符串引号问题。

静态检查（python -m compileall）完全发现不了这类问题，因为它们藏在
Python 字符串里面。用 node --check 在 CI 里兜住，比等用户报"页面卡住了"快得多。

⚠️ 必须检查「浏览器真正收到的 HTML」，不是 server.py 的源码文本
------------------------------------------------------------------
本脚本 v1 有一个致命缺陷：它直接 `read_text()` 读 server.py，把**源码文本**
拿去 node --check。但源码里那段 HTML 还带着 Python 的转义，而 Python 会先把它
消费一遍。两层转义叠加，源码与产物并不相同：

    server.py 源码          node 看到          浏览器实际收到
    setDesc:'...layer\\'s'   '...layer\\'s'  ✓   '...layer's'  ✗ SyntaxError

源码里 `\\'` 是**合法的 JS 转义撇号**，于是 node --check 满意放行；
而 Python 解析字符串时把反斜杠吃掉，送到浏览器的是**裸撇号**，
字符串被提前终止 -> 整段 <script> 报废 -> 页面所有交互失效。
（2026-09-14 真实踩到：设置页 i18n 的英文文案 `layer's` 让整个前端死掉。）

修法：用 `ast` 取出模块级 `*_HTML` 常量 —— `ast` 拿到的是 **Python 已经求值过**
的字符串（转义已按 Python 规则处理完），再套上 `render()` 的占位符替换，
就等价于浏览器收到的内容。

另外增加了「所有块拼接后再检查」这一步：浏览器里多个 <script> **共享同一个
全局词法作用域**，所以「同一个 let 在两个块里各声明一次」会让第二个块整体抛错，
而逐块检查永远发现不了。

还拦第二类 bug：**`data-i18n` 与 JS 动态文本打架**（`check_i18n_dynamic_ownership`）
--------------------------------------------------------------------------
`setLang()` 是一刀切的 —— `querySelectorAll('[data-i18n]')` 全部刷成字典值。
于是「既有 data-i18n、文本又由 JS 维护」的元素，切一次语言就被打回静态文案。
2026-09-14 真实事故：切语言后 #modelBadge 从「模型已加载 · 48kHz」变成
「模型未加载」，看起来像模型掉了；#recBtn 从「🎤 重新录制」变回「🎤 开始录制」。
这类 bug 语法上完全合法，node --check 永远放行，只能靠语义检查兜住。

用法：
    python scripts/check_inline_js.py            # 自动定位 server.py
    python scripts/check_inline_js.py path.py
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path


def _force_utf8_stdout() -> None:
    """把 stdout/stderr 切到 UTF-8。

    ⚠️ 不加这段，Windows（含 GitHub Actions 的 windows-latest）会**崩在 print 上**：
    控制台默认 cp1252，而本脚本要打印中文，于是

        UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-1

    后果比"输出乱码"严重得多 —— 进程以**非零码退出**，而调用它的
    test_inline_js.py 把「非零退出」解读成「JS 有语法错误」，
    于是报出一个**完全不存在**的前端故障，真因被彻底掩盖。
    （2026-09-22 实测：CI windows 三个 job 全红，Ubuntu 全绿，
    报错信息还指向前端转义，实际脚本连第一个断言都没走到。）
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 非 TextIOWrapper（被重定向/包装过）时静默跳过


_force_utf8_stdout()

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "server.py"

# 匹配 <script> ... </script>；带 src= 的外链脚本没有内容，跳过
SCRIPT_RE = re.compile(r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
                       re.DOTALL | re.IGNORECASE)

# 与 server.py 的 render() 保持一致
PLACEHOLDER_SUBS = (
    ("PORT_PLACEHOLDER", "8808"),
    ("TOKEN_PLACEHOLDER", "test-token-placeholder"),
)


def _literal(node: ast.AST) -> str | None:
    """把一个 AST 字面量节点还原成 Python 求值后的字符串。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value                      # ast 已按 Python 规则解完转义
    if isinstance(node, ast.JoinedStr):        # f-string
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append("FSTRING_PLACEHOLDER")   # 保持长度非 0，避免误拼
        return "".join(out)
    return None


def rendered_pages(src_path: Path) -> dict[str, str]:
    r"""取出模块级 *_HTML 常量的「运行时取值」，并套用 render() 的替换。

    ⚠️ 这里为什么要吞掉 SyntaxWarning
    ----------------------------------
    APP_HTML 里大量 JS 正则字面量写成 `/\(([^()]*)\)/`、`/\.wav$/`。对 Python 来说
    `\(`、`\.` 是"无效转义"，会报 SyntaxWarning；但 Python 的规则是**原样保留**
    反斜杠，所以 JS 收到的正是正确的 `\(`，运行时完全没问题。

    **不要**为了消这个警告把 APP_HTML 改成 raw string —— 那会让 `\n` 不再是换行，
    页面格式会整体塌掉，是个远比警告严重的事故。所以只在解析时静音。
    """
    src = src_path.read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src)
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.endswith("_HTML"):
                val = _literal(node.value)
                if val is None:
                    continue
                for k, v in PLACEHOLDER_SUBS:
                    val = val.replace(k, v)
                out[t.id] = val
    return out


def check_with_node(code: str, node: str) -> tuple[bool, str]:
    """把 JS 写进临时文件后 node --check。返回 (ok, 错误信息)。"""
    fd = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8")
    try:
        fd.write(code)
        fd.close()
        r = subprocess.run([node, "--check", fd.name],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout).strip()
    finally:
        try:
            Path(fd.name).unlink()
        except OSError:
            pass


def _i18n_tagged_ids(html: str) -> dict[str, str]:
    """HTML 里「带 id 且带 data-i18n」的元素 -> i18n 键名。"""
    out: dict[str, str] = {}
    for m in re.finditer(r"<[a-zA-Z][\w-]*\b([^>]*)>", html, re.S):
        attrs = m.group(1)
        mi = re.search(r'data-i18n\s*=\s*"([^"]*)"', attrs)
        if not mi:
            continue
        moid = re.search(r'id\s*=\s*"([^"]*)"', attrs)
        if moid:
            out[moid.group(1)] = mi.group(1)
    return out


def _function_defs(js: str) -> list[tuple[str, int, int]]:
    """[(函数名, 起点, 终点)] —— 按花括号配对定位每个具名函数的完整跨度。"""
    out: list[tuple[str, int, int]] = []
    for m in re.finditer(r"function\s+(\w+)\s*\([^)]*\)\s*\{", js):
        depth = 0
        for i in range(m.end() - 1, len(js)):
            if js[i] == "{":
                depth += 1
            elif js[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append((m.group(1), m.start(), i + 1))
                    break
    return out


def _function_body(js: str, fname: str) -> str:
    """抠出某个具名函数的完整文本（含 `function ... {...}`）；找不到返回空串。"""
    for name, s, e in _function_defs(js):
        if name == fname:
            return js[s:e]
    return ""


def _scope_texts(js: str) -> list[str]:
    """所有「变量作用域单元」：每个具名函数体 + 去掉函数体后的顶层代码。

    ⚠️ 必须分开统计，不能把整份 JS 拼起来一次性找变量别名：
        const b = document.getElementById('modelBadge');  b.textContent = ...
    这里的 b 是**函数局部**变量，不同函数里的同名 b 指向不同元素。
    拼在一起会让后写覆盖先写，从而漏判 —— 第一版正是因此漏掉了 #modelBadge，
    而用 btn 的 #recBtn 因没撞名侥幸通过。这种"看运气"的检查等于没有。
    """
    defs = _function_defs(js)
    texts = [js[s:e] for _n, s, e in defs]
    cut: list[str] = []
    last = 0
    for _n, s, e in sorted(defs, key=lambda t: t[1]):
        if s >= last:
            cut.append(js[last:s])
            last = e
    cut.append(js[last:])
    texts.append("".join(cut))
    return texts


def _written_ids_in(text: str) -> set[str]:
    """在**单个作用域单元**内，文本被动态写过的元素 id。"""
    written: set[str] = set()
    for m in re.finditer(
            r"getElementById\(\s*'([^']+)'\s*\)\s*\.\s*"
            r"(?:textContent|innerHTML)\s*=", text):
        written.add(m.group(1))
    alias: dict[str, set[str]] = {}
    for m in re.finditer(
            r"(?:const|let|var)\s+(\w+)\s*=\s*"
            r"document\.getElementById\(\s*'([^']+)'\s*\)", text):
        alias.setdefault(m.group(1), set()).add(m.group(2))
    for m in re.finditer(r"\b(\w+)\.(?:textContent|innerHTML)\s*=", text):
        written |= alias.get(m.group(1), set())
    return written


def _js_written_ids(js: str) -> set[str]:
    """JS 里文本被动态写过的元素 id（按作用域单元分别统计后取并集）。

    两种写法都要认：
      document.getElementById('x').textContent = ...
      const b = document.getElementById('x');  b.textContent = ...   <- 别漏了这种
    """
    out: set[str] = set()
    for t in _scope_texts(js):
        out |= _written_ids_in(t)
    return out


# setLang() 的收尾钩子。它**直接或经多层转调**写到的元素，视为「切语言后会重放」。
# 必须做多层展开，真实链路是：
#     repaintDynamicText → repaintModelBadge → setModelBadge
# 只展开一层会把 #modelBadge 误判成「没重放」（第一版就犯了这个错）。
REPLAY_ENTRY = "repaintDynamicText"
REPLAY_MAX_DEPTH = 4
REPLAY_MAX_FUNCS = 60

# 切语言时即使被打回静态文案、也无需重放的元素 —— 逐个给出理由，不要图省事乱加。
TRANSIENT_OK = {
    "statusText": "生成中由定时器高频重写",
    "vpStatusText": "提取中由定时器高频重写",
    "recStatus": "录音中由定时器高频重写",
    "trainStartBtn": "训练轮询每 2s 重写",
    "vpDropHint": "拖拽瞬间的提示，放下/离开即被重设，不可能停在切语言的那一刻",
    "trainAddBtn": "只在一次上传请求期间显示「上传中」，请求结束就写回与静态文案"
                   "相同的值，且那时按钮是 disabled 的",
}


def _functions(js: str) -> dict[str, str]:
    """具名函数 -> 完整定义文本。"""
    return {n: js[s:e] for n, s, e in _function_defs(js)}


def _replay_texts(js: str) -> list[str]:
    """REPLAY_ENTRY 及其（多层）被调函数 —— 以**独立作用域单元**返回。"""
    fns = _functions(js)
    entry = fns.get(REPLAY_ENTRY, "")
    if not entry:
        return []
    out = [entry]
    seen = {REPLAY_ENTRY}
    frontier = [entry]
    for _ in range(REPLAY_MAX_DEPTH):
        nxt = []
        for body in frontier:
            for name in sorted(set(re.findall(r"\b(\w+)\s*\(", body))):
                if name in seen or name not in fns:
                    continue
                seen.add(name)
                if len(seen) > REPLAY_MAX_FUNCS:
                    return out
                nxt.append(fns[name])
                out.append(fns[name])
        if not nxt:
            break
        frontier = nxt
    return out


def check_i18n_dynamic_ownership(html: str, js: str) -> int:
    """拦「data-i18n 与 JS 动态文本打架」这一类 bug。

    setLang() 是一刀切的：`querySelectorAll('[data-i18n]')` 全部刷成字典值。
    所以任何「既有 data-i18n、文本又由 JS 动态维护」的元素，切一次语言就会被
    打回静态文案。真实事故（2026-09-14）：

        #modelBadge  切语言后从「模型已加载 · 48kHz」变成「模型未加载」
        #recBtn      录制后从「🎤 重新录制」变回「🎤 开始录制」（且无定时器兜底）

    规则：冲突元素必须要么在 TRANSIENT_OK 里（附理由），要么由
    repaintDynamicText() 重放，否则报错。
    """
    i18n_ids = _i18n_tagged_ids(html)
    written = _js_written_ids(js)
    risky = sorted(set(i18n_ids) & written)
    if not risky:
        return 0

    replayed: set[str] = set()
    for t in _replay_texts(js):
        replayed |= _written_ids_in(t)
    bad = [i for i in risky if i not in TRANSIENT_OK and i not in replayed]
    ok_ids = [i for i in risky if i in replayed]
    transient = [i for i in risky if i in TRANSIENT_OK]

    if bad:
        print("  i18n/动态文本冲突: FAIL", file=sys.stderr)
        for i in bad:
            print(f"      #{i} 带 data-i18n=\"{i18n_ids[i]}\" 但文本由 JS 写，"
                  f"切语言会被打回静态文案", file=sys.stderr)
        print(f"      -> 修法：在 {REPLAY_ENTRY}() 里加一个 repaint 调用重放它的状态，"
              f"或确认它确实会被高频重写后加进 TRANSIENT_OK（并写清理由）",
              file=sys.stderr)
        return len(bad)

    print(f"  i18n/动态文本冲突: OK（重放 {len(ok_ids)} 个"
          f"{'，瞬时豁免 ' + str(len(transient)) + ' 个' if transient else ''}）")
    return 0


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    if not target.exists():
        print(f"[skip] 找不到 {target}", file=sys.stderr)
        return 0

    node = shutil.which("node")
    if not node:
        print("[skip] 未安装 node，跳过 JS 语法检查", file=sys.stderr)
        return 0

    pages = rendered_pages(target)
    if not pages:
        print(f"[skip] {target.name} 里没有找到 *_HTML 常量", file=sys.stderr)
        return 0

    total_failed = 0
    for name, html in sorted(pages.items()):
        scripts = [m.group(1) for m in SCRIPT_RE.finditer(html)]
        scripts = [s for s in scripts if s.strip()]
        if not scripts:
            continue
        print(f"检查 {target.name} → {name}（渲染后 {len(html)} 字节，"
              f"{len(scripts)} 段内联 JS）…")

        failed = 0
        for i, code in enumerate(scripts, 1):
            ok, err = check_with_node(code, node)
            if ok:
                print(f"  script #{i}: OK ({code.count(chr(10)) + 1} 行)")
            else:
                failed += 1
                print(f"  script #{i}: SYNTAX ERROR", file=sys.stderr)
                for line in err.splitlines()[:25]:
                    print(f"      {line}", file=sys.stderr)

        # 跨块重复声明的护栏：多个 <script> 共享全局词法作用域，
        # 逐块合法 ≠ 整页合法，所以必须拼接后再验一次。
        if len(scripts) > 1:
            ok, err = check_with_node("\n;\n".join(scripts), node)
            if ok:
                print("  拼接全部块: OK（无跨块重复声明）")
            else:
                failed += 1
                print("  拼接全部块: SYNTAX ERROR —— 可能是跨 <script> 的重复声明",
                      file=sys.stderr)
                for line in err.splitlines()[:25]:
                    print(f"      {line}", file=sys.stderr)

        failed += check_i18n_dynamic_ownership(html, "\n".join(scripts))
        total_failed += failed

    if total_failed:
        print(f"\n[fail] {total_failed} 处内联 JS 问题 —— "
              f"页面交互会整体失效或状态显示错误，请修复后再提交。",
              file=sys.stderr)
        return 1
    print("[ok] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
