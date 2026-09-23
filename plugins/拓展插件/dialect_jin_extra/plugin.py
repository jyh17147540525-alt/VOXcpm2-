"""晋语·俗语歇后语
========================================

实现见 `_dialect_common/extra_factory.py`（全方言共用）；本文件只负责定位共享层并构造入口。
条目全部在 `dialect_data.json` 里 —— 改语料不必改代码。

指令（写在合成文本的行首，中英文冒号都接受；**必须带本方言简称**）：
    晋语俗语：<关键词>
    晋语歇后语：<关键词>
    晋语童谣：<关键词>
    晋语民俗：<关键词>

检索是子串匹配：不必记得完整条目，给关键词即可。命中才讲，未命中明确说「未收录」。
裸指令（`俗语：X`）交给跑在最后的「跨方言路由插件」跨方言检索 —— 原因见上方技能插件说明。
契约：text.pre 返回字符串（arg 字段值）；任何异常只降级为「不改动」。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(os.path.dirname(_HERE), "_dialect_common"),
              os.path.join(os.path.dirname(os.path.dirname(_HERE)), "技能插件", "_dialect_common")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from extra_factory import build  # noqa: E402

_M = build(_HERE)

setup = _M.setup
teardown = _M.teardown
on_text_pre = _M.on_text_pre
on_report_enrich = _M.on_report_enrich
