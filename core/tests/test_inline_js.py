"""护栏测试：server.py 内联前端 JS 必须语法合法（针对**渲染后**的 HTML）。

为什么需要这个测试
------------------
server.py 把整个前端作为 Python 字符串内联返回，于是存在"两层转义"耦合：
为 JS 写的转义会被 Python 先消费一遍。真实踩过的两个坑（2026-09-14）：

    server.py 源码          node 看到          浏览器实际收到
    '...layer\\'s'          '...layer\\'s'  ✓   '...layer's'   ✗ 字符串提前终止
    '...head+'\\n''         '...head+'\\n''  ✓   '...head+'<换行>''  ✗ 同上

两处都是「源码里合法、产物里致命」。而且 JS 只报**第一个**语法错，
后面同类错误全被挡住——所以看起来"只有一个 bug"，实际有 7 处。

因此本测试**不检查源码文本**，而是用 `ast` 取出 `*_HTML` 常量的**运行时取值**
（转义已按 Python 规则处理完），再交给 node --check。
这正是旧的 `scripts/check_inline_js.py`（直接读 .py 源码）漏掉这个 bug 的原因。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CHECKER = REPO / "scripts" / "check_inline_js.py"
SERVER = REPO / "server.py"


def test_inline_js_has_no_syntax_error():
    """渲染后的内联 JS 必须能通过 node --check。"""
    if not SERVER.exists():
        pytest.skip("server.py 不存在")
    if not CHECKER.exists():
        pytest.skip("scripts/check_inline_js.py 不存在")
    if not shutil.which("node"):
        pytest.skip("未安装 node，无法校验 JS 语法")

    r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True,
                       text=True, timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, (
        "内联 JS 存在语法错误 —— 页面交互会整体失效。\n"
        "最常见的根因：把 JS 字符串写进了 Python 字符串，转义被 Python 消费掉了。\n"
        "例如源码写 '\\n' 会变成真换行、写 \\' 会变成裸撇号。\n"
        "修法：源码里写 '\\\\n' / '\\\\''，或改用不含该字符的措辞。\n\n"
        f"检查器输出：\n{out}"
    )
    assert "[ok]" in out, f"检查器未正常完成：\n{out}"