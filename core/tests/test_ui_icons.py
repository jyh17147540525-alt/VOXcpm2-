"""护栏测试：前端图标 / 设计令牌 / i18n 动态文本的一致性（针对**渲染后**的 HTML）。

为什么需要这个测试
------------------
`server.py` 的前端同样是内联在 Python 字符串里的，所以"图标没换成 SVG""某页少了
`<symbol>` 定义""某页整块缺 `:root` 令牌""i18n 字典里还留着 emoji"这类问题
**在源码里看不出来**，必须读 `*_HTML` 常量的运行时取值。

2026-09-15 的 UI 重构一次踩了 7 类，全部是"改了但没生效"：

  1. 标记区换成 SVG，**i18n 字典值里还留着 emoji** → 切一次语言就打回界面
  2. 图标被塞进**带 `data-i18n` 的容器** → setLang 刷 textContent 连带抹掉图标
  3. 图标所在按钮**由 JS 写 textContent** → 状态一变图标就没了（静态标记完全正确）
  4. 某页**只加了 `<use>` 没加 `<symbol>`** → 图标空白，只有跨页比对才暴露
  5. 某页**缺整块令牌定义** → `var()` 过渡是无效 CSS，静默失效
  6. 硬编码动画时长残留，绕过令牌体系
  7. **JS 动态拼出来的消息里的 emoji**（❌✅⚠️⚡✓✗）—— 既不在标记区、也不在
     i18n 字典里，所以 1 和 6 的检查同时放过，但它会真的渲染进界面

第 7 类修完后，检查器的第 6 组已从"信息性"升格为硬断言：**渲染产物任何位置
emoji 都必须为 0**。该断言用 6 个突变体自测过（每一类注入都能让它失败）。

对应的独立脚本是 `scripts/check_ui_icons.py`，可与
`tests/test_inline_js.py`（JS 语法 + i18n 动态文本归属）配合使用。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CHECKER = REPO / "scripts" / "check_ui_icons.py"
SERVER = REPO / "server.py"


def test_ui_icons_and_tokens_are_consistent():
    """图标引用/定义、设计令牌、i18n 键与 emoji 归属必须自洽。"""
    if not SERVER.exists():
        pytest.skip("server.py 不存在")
    if not CHECKER.exists():
        pytest.skip("scripts/check_ui_icons.py 不存在")

    # ⚠️ 必须显式 `encoding="utf-8"`：检查器打印的是中文，子进程按 UTF-8 写出，
    #    而 `text=True` 不给 encoding 时按**本机 locale** 解码 —— 在 Windows
    #    （含 CI 的 windows-latest）locale 是 cp1252，解码失败会让 `out` 变成**空串**，
    #    于是下面的 `"[ok]" in out` 报"检查器未正常完成"，把一次**成功**的检查判成失败。
    #    （2026-09-22 实测：CI windows 三个 job 全红、Ubuntu 全绿。）
    r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True,
                       text=True, timeout=180,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, (
        "前端图标 / 令牌 / i18n 一致性检查未通过 —— 页面上会出现图标空白、"
        "过渡失效、或切语言后 emoji 复活。\n"
        "常见根因：\n"
        "  * 图标写进了带 data-i18n 的容器（setLang 刷 textContent 会抹掉它）\n"
        "  * 图标按钮的文字由 JS 写 textContent（状态一变图标就丢）\n"
        "  * 某页只加了 <use> 没加 <symbol>，或整块缺 :root 令牌\n"
        "  * i18n 字典值 / JS 消息字符串里还留着 emoji\n\n"
        f"检查器输出：\n{out}"
    )
    assert "[ok]" in out, f"检查器未正常完成：\n{out}"
