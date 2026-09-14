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
from pathlib import Path

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
    """取出模块级 *_HTML 常量的「运行时取值」，并套用 render() 的替换。"""
    tree = ast.parse(src_path.read_text(encoding="utf-8", errors="replace"))
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

        total_failed += failed

    if total_failed:
        print(f"\n[fail] {total_failed} 处内联 JS 语法错误 —— "
              f"页面交互会整体失效，请修复后再提交。", file=sys.stderr)
        return 1
    print("[ok] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
