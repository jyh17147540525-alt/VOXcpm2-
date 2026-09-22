"""护栏：前端图标 / 设计令牌 / 内联 JS 动态文本的一致性检查。

为什么需要这个
--------------
2026-09-15 做 UI 重构（emoji → SVG sprite）时，同一类事故在一个会话里踩了 6 次，
而且每一处都是「改了但没生效」，肉眼看不出来：

  1. 标记区的 emoji 换成了 <svg>，但 **i18n 字典值里还留着 emoji**
     → setLang() 一刷就打回界面（trainStop='⏹ 停止'、themeGoDark='🌙 深色' 就是这样复现的）。
  2. 图标塞进了**带 data-i18n 的容器**内 → setLang 刷 textContent 把图标一起抹掉。
  3. 图标所在的按钮**由 JS 写 textContent**（如 setRecBtn / updateThemeBtn / trainStartTxt）
     → 状态一变图标就没了。这类最隐蔽：静态标记完全正确，只有运行时才丢。
  4. LOGIN_HTML **只加了 <use> 没有 <symbol>** → 登录页 logo 是空白的。
     单独看两个页面各自的源码都"没问题"，只有跨页比对才发现。
  5. LOGIN_HTML **没有令牌块**（--ease/--t-fast/--t-base 未定义）→ 过渡是无效 CSS，静默失效。
  6. 硬编码的 `.16s ease` 残留，绕过令牌体系。

第 7 类（第 1~6 类都修完之后才暴露，因为它被前两类"检查盲区"同时漏掉）：
  JS **动态拼出来**的状态消息里还留着 ❌✅⚠️⚡✓✗ —— 生成结果、保存/测试反馈、
  训练报错、音色包提示、alert()。它们 **既不在标记区**（所以第 6 组的"标记区=0"放过），
  **也不在 i18n 字典里**（所以第 4b 组放过），但会真的渲染进界面，跟线性图标语言打架。
  → 因此第 6 组从"信息性"升格为**硬断言：渲染产物任何位置的 emoji 都必须为 0**。

这些检查**必须看渲染后的产物**（用 ast 取 `*_HTML` 常量的运行时取值），
并且要能**按页面分别**核对——因为 4、5 两类只有分页面才暴露。

用法：python scripts/check_ui_icons.py     # 退出码 0=通过，1=有问题
      VX_SRC=<path> python scripts/check_ui_icons.py   # 指向别的文件（护栏自测用）
"""
from __future__ import annotations

import ast
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path


def _force_utf8_stdout() -> None:
    """把 stdout/stderr 切到 UTF-8。

    ⚠️ 不加这段，Windows（含 GitHub Actions 的 windows-latest）会**崩在 print 上**：
    控制台默认 cp1252，而本脚本要打印中文小节标题（如「1. 图标引用/定义（逐页）」），于是

        UnicodeEncodeError: 'charmap' codec can't encode characters in position 3-6

    后果比"输出乱码"严重得多 —— 进程以**非零码退出**，而调用它的
    test_ui_icons.py 把「非零退出」解读成「图标/令牌/i18n 不一致」，
    于是报出一个**完全不存在**的前端故障。（2026-09-22 实测：CI windows 全红、
    Ubuntu 全绿，报错信息指向前端转义，实际脚本连检查都没开始。）
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 非 TextIOWrapper（被重定向/包装过）时静默跳过


_force_utf8_stdout()

REPO = Path(__file__).resolve().parent.parent
SRC = os.environ.get("VX_SRC") or str(REPO / "server.py")

EMOJI = re.compile("[\u23e9-\u23fa\u2600-\u27bf\u2b00-\u2bff\ufe0f\U0001F300-\U0001FAFF]")
CORE_TOKENS = ["--r-md", "--sp-2", "--fs-sm", "--ease", "--t-fast", "--t-base"]
# 图标所在的可见元素（这些位置出现 emoji 一律视为漏改）
VISIBLE_TAGS = {"button", "span", "label", "summary", "option", "div", "p", "a", "h1", "h2", "h3"}
# 故意保留裸时长的连续动画（加载转圈），不参与令牌化
ANIM_WHITELIST = ("animation:sp ", ".001ms")


def load_html_constants(path: str) -> dict[str, str]:
    warnings.simplefilter("ignore", SyntaxWarning)
    src = open(path, encoding="utf-8", newline="").read()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Name) and t.id.endswith("_HTML")
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    out[t.id] = node.value.value
    return out


def i18n_blocks(src: str) -> tuple[str, str]:
    """切出 I18N 的 zh / en 两块。

    ⚠️ 别用一个非贪婪正则去猜块的边界——之前这么写会提前截断，
    导致 zh 块只解析到一小段，于是"字典里没有 emoji"这种假阴性出现过三次。
    """
    m = re.search(r"const I18N=\{(.*?)[\r\n]+\};", src, re.S)
    if not m:
        return "", ""
    body = m.group(1)
    ei = re.search(r"[\r\n]\s*en\s*:\s*\{", body)
    return (body[:ei.start()], body[ei.end():]) if ei else (body, "")


def main() -> int:
    consts = load_html_constants(SRC)
    if not consts:
        print("[fail] 没解析到任何 *_HTML 常量：%s" % SRC)
        return 1
    src = open(SRC, encoding="utf-8", newline="").read()
    zh_body, en_body = i18n_blocks(src)
    bad: list[str] = []
    tag_of: dict[str, str] = {}

    # ---- 1. 图标引用必须有定义（逐页；只加 <use> 不加 <symbol> 就靠这里抓）----
    print("1. 图标引用/定义（逐页）")
    for name, html in consts.items():
        syms = set(re.findall(r'<symbol id="([a-z0-9-]+)"', html))
        used = set(re.findall(r'<use href="#([a-z0-9-]+)"', html))
        undef = sorted(used - syms)
        print("   [%s] symbol=%d 引用=%d%s" % (name, len(syms), len(used),
                                              ("  未定义=" + str(undef)) if undef else ""))
        if undef:
            bad.append("%s 引用了未定义的图标 %s（该处会渲染成空白）" % (name, undef))

    # ---- 2. 核心设计令牌必须在每个页面各自定义 ----
    print("2. 核心设计令牌（逐页）")
    for name, html in consts.items():
        defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", html))
        used = set(re.findall(r"var\((--[a-z0-9-]+)", html))
        unresolved = sorted(used - defined)
        miss = [t for t in CORE_TOKENS if t not in defined]
        print("   [%s] 定义 %d 个，未解析 var()=%s，核心令牌缺失=%s"
              % (name, len(defined), unresolved or "无", miss or "无"))
        if unresolved:
            bad.append("%s 有 %d 个 var() 没有定义（回退成 initial，过渡会静默失效）：%s"
                       % (name, len(unresolved), unresolved[:6]))
        if miss:
            bad.append("%s 缺核心令牌 %s" % (name, miss))

    # ---- 3. 动画/过渡必须走令牌，不留硬编码时长 ----
    print("3. 硬编码动画时长")
    pat = re.compile(r"(transition|animation)[^;{}\"']*?\d*\.?\d+(?:ms|s)\b[^;{}\"']*")
    for name, html in consts.items():
        hits = []
        for blk in re.findall(r"<style>(.*?)</style>", html, re.S):
            for m in pat.finditer(blk):
                txt = m.group(0).strip()
                if any(w in txt for w in ANIM_WHITELIST):
                    continue
                hits.append(txt)
        print("   [%s] %d 处%s" % (name, len(hits), ("  " + "; ".join(sorted(set(hits))[:3])) if hits else ""))
        if hits:
            bad.append("%s 残留 %d 处硬编码时长（应改用 var(--t-*) / var(--ease)）" % (name, len(hits)))

    # ---- 4. i18n 键集对称 + 用到的键都存在 ----
    print("4. i18n 键完整性")
    KEY_RE = r"(?:^|[,{]\s*)([A-Za-z_]\w*)\s*:\s*'"
    kz = set(re.findall(KEY_RE, zh_body, re.M))
    ke = set(re.findall(KEY_RE, en_body, re.M))
    print("   zh=%d en=%d 仅zh=%s 仅en=%s" % (len(kz), len(ke),
                                            sorted(kz - ke)[:5] or "无", sorted(ke - kz)[:5] or "无"))
    if kz - ke or ke - kz:
        bad.append("i18n 的 zh/en 键集不对称")
    for name, html in consts.items():
        used_keys = set(re.findall(r'data-i18n(?:-ph)?="([A-Za-z_]\w*)"', html))
        all_keys = set(re.findall(KEY_RE, src, re.M))
        missk = sorted(used_keys - kz - ke - all_keys)
        # 记录每个键所在元素，供 4b 判定"是否可见"
        for km in re.finditer(r'data-i18n="([A-Za-z_]\w*)"', html):
            tags = re.findall(r"<(\w+)[^>]*$", html[max(0, km.start() - 300):km.start()])
            if tags:
                tag_of[km.group(1)] = tags[-1]
        print("   [%s] 用到 %d 键，缺失=%s" % (name, len(used_keys), missk or "无"))
        if missk:
            bad.append("%s 缺 i18n 键 %s" % (name, missk))

    # ---- 4b. 可见元素的 i18n 文案不得含 emoji（标记区改了、字典没改就会在这里暴露）----
    print("4b. 可见元素的 i18n 文案含 emoji")
    dict_emoji = []
    for lang, blk in (("zh", zh_body), ("en", en_body)):
        for km in re.finditer(KEY_RE, blk, re.M):
            k = km.group(1)
            vm = re.match(r"[A-Za-z_]\w*\s*:\s*'([^']*)'", blk[km.start(1):])
            v = vm.group(1) if vm else ""
            if EMOJI.search(v) and tag_of.get(k) in VISIBLE_TAGS:
                dict_emoji.append((lang, k, tag_of.get(k), v))
    print("   " + ("无" if not dict_emoji else str(dict_emoji)))
    if dict_emoji:
        bad.append("有 %d 条可见元素的 i18n 文案含 emoji（切语言会打回界面）：%s"
                   % (len(dict_emoji), [d[1] for d in dict_emoji]))

    # ---- 5. data-i18n 容器内不得有 svg（setLang 刷 textContent 会连图标一起抹掉）----
    print("5. data-i18n 容器内含 svg")
    risky = []
    for name, html in consts.items():
        for m2 in re.finditer(r"<(\w+)([^>]*\bdata-i18n=\"[^\"]+\"[^>]*)>(.*?)</\1>", html, re.S):
            if "<svg" in m2.group(3):
                risky.append((name, m2.group(1)))
        print("   [%s] %d 处" % (name, len([r for r in risky if r[0] == name])))
    if risky:
        bad.append("有 %d 处 data-i18n 容器内含 svg：%s" % (len(risky), risky))

    # ---- 5b. JS 写 textContent 时不得落在含 svg 的容器上（静态看着对、运行时丢图标）----
    print("5b. JS 往含 svg 的容器写 textContent")
    js_hits = []
    for name, html in consts.items():
        svg_ids = set()
        for m3 in re.finditer(r"<(\w+)([^>]*\bid=\"([^\"]+)\"[^>]*)>(.*?)</\1>", html, re.S):
            if "<svg" in m3.group(4):
                svg_ids.add(m3.group(3))
        lines = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S)).split("\n")
        hit_ids = {}
        for i, ln in enumerate(lines):
            for wm in re.finditer(r"(?:^|[^.\w$])([\w$]+)\s*\.\s*(?:textContent|innerHTML)\s*=", ln):
                var = wm.group(1)
                for k in range(i, max(-1, i - 300), -1):   # 就近向上找声明，别用全局别名表
                    dm = re.search(r"(?:var|let|const)\s+%s\s*=\s*document\.getElementById\(['\"]([^'\"]+)['\"]\)"
                                   % re.escape(var), lines[k])
                    if dm:
                        if dm.group(1) in svg_ids:
                            hit_ids.setdefault(dm.group(1), var)
                        break
            for wm in re.finditer(r"getElementById\(['\"]([^'\"]+)['\"]\)\s*\.\s*(?:textContent|innerHTML)\s*=", ln):
                if wm.group(1) in svg_ids:
                    hit_ids.setdefault(wm.group(1), "(直连)")
        for eid in sorted(hit_ids):
            js_hits.append((name, eid, hit_ids[eid]))
        print("   [%s] 含 svg 的容器 %d 个，被 JS 写 textContent 的 %d 个"
              % (name, len(svg_ids), len(hit_ids)))
    for h in js_hits:
        print("      !! %s #%s (变量 %s) → 状态一变图标就被抹掉，应改为写内层 span" % h)
    if js_hits:
        bad.append("有 %d 处 JS 写 textContent 会抹掉容器内的 svg：%s"
                   % (len(js_hits), [h[1] for h in js_hits]))

    # ---- 6. 渲染产物 emoji 必须为 0（硬断言）----
    # 2026-09-15 升格：原先只断言"标记区为 0"，脚本区当信息看。
    # 但 JS 动态拼出来的状态消息（生成结果 / 保存测试 / 训练报错 / alert）里的
    # ❌✅⚠️⚡ 会渲染进界面，而它们既不在标记区、也不在 i18n 字典里 —— 两道检查同时漏。
    # 全站 emoji 已归零，所以这里改成任何位置出现 emoji 都算失败。
    print("6. 渲染产物 emoji 必须为 0（含 JS 动态消息与注释）")
    for name, html in consts.items():
        ms = list(EMOJI.finditer(html))
        print("   [%s] emoji %d 个%s" % (
            name, len(ms), (" " + str(Counter(m.group(0) for m in ms).most_common(6))) if ms else ""))
        if ms:
            for m in ms[:3]:
                around = html[max(0, m.start() - 45):m.end() + 45].replace("\n", " ").replace("\r", "")
                print("      ... %s" % around)
            bad.append("%s 渲染产物还有 %d 个 emoji（含 JS 动态消息与注释），应换成 sprite 图标：%s"
                       % (name, len(ms), Counter(m.group(0) for m in ms).most_common(6)))

    print("-" * 56)
    if bad:
        print("[fail] %d 项 UI 一致性问题：" % len(bad))
        for b in bad:
            print("   -", b)
        return 1
    print("[ok] 图标 / 设计令牌 / i18n 动态文本一致性检查全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
