"""官话·西南与江淮·语言技能
========================================

实现见 `_dialect_common/skill_factory.py`（全方言共用）；本文件只负责定位共享层并构造入口。
语料、指令简称、语法问答全部在 `dialect_data.json` 里 —— 改语料不必改代码。

指令（写在合成文本的行首，中英文冒号都接受；**必须带本方言简称**）：
    西南官话翻译：<词>             词典驱动互译（双向）
    西南官话怎么说：<词>           普通话 → 方言
    西南官话语法：<主题>           语法差异问答

为什么不认裸的「翻译：」：text.pre 是链式执行，13 个方言插件都认裸指令的话，
id 最小的那个会抢答一切查询并剥掉指令，其余方言永远轮不到。
裸指令交给跑在最后的「跨方言路由插件」处理（它会列出命中的方言）。

契约：text.pre 返回字符串（arg 字段值）；任何异常只降级为「不改动」，不拖垮合成主流程。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(os.path.dirname(_HERE), "_dialect_common"),
              os.path.join(os.path.dirname(os.path.dirname(_HERE)), "技能插件", "_dialect_common")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from skill_factory import build  # noqa: E402

_M = build(_HERE)

setup = _M.setup
teardown = _M.teardown
on_text_pre = _M.on_text_pre
on_report_enrich = _M.on_report_enrich
