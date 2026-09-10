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

用法：
    python scripts/check_inline_js.py            # 自动定位 server.py
    python scripts/check_inline_js.py path.py
"""
from __future__ import annotations

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


def extract_scripts(src_path: Path) -> list[str]:
    text = src_path.read_text(encoding="utf-8", errors="replace")
    return [m.group(1) for m in SCRIPT_RE.finditer(text)]


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

    scripts = extract_scripts(target)
    if not scripts:
        print(f"[skip] {target.name} 里没有内联 <script>")
        return 0

    print(f"检查 {target.name} 的 {len(scripts)} 段内联 JS …")
    failed = 0
    for i, code in enumerate(scripts, 1):
        if not code.strip():
            continue
        ok, err = check_with_node(code, node)
        if ok:
            n_lines = code.count("\n") + 1
            print(f"  script #{i}: OK ({n_lines} 行)")
        else:
            failed += 1
            print(f"  script #{i}: SYNTAX ERROR", file=sys.stderr)
            for line in err.splitlines()[:25]:
                print(f"      {line}", file=sys.stderr)

    if failed:
        print(f"\n[fail] {failed} 段内联 JS 存在语法错误 —— "
              f"页面交互会整体失效，请修复后再提交。", file=sys.stderr)
        return 1
    print("[ok] 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
