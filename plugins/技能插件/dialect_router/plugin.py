"""方言·跨方言检索路由插件
========================================

实现见 `_dialect_common/router.py`；本文件只负责定位共享层并构造入口。

本插件**必须跑在所有方言插件之后**（priority 20，方言插件是 60）：
  - 它处理不带方言简称的裸指令：`翻译：X`、`怎么说：X`、`语法：T`、
    `俗语：X`、`歇后语：X`、`童谣：X`、`民俗：X`；
  - 它兜住被停用／加载失败的方言插件留下的作用域指令（如 `粤语翻译：X`）；
  - 全表查不到时由它给出「未收录」，并报出实际检索了多少张表。

指令（行首，中英文冒号都接受）：
    翻译：<词> / 怎么说：<词>      跨方言检索，列出命中的方言
    语法：<主题>                   跨方言列出该主题的说明
    俗语：/歇后语：/童谣：/民俗：<关键词>
    <方言简称>翻译：<词>            定向查询（正常由该方言插件自己应答）

契约：text.pre 返回字符串（arg 字段值）；任何异常只降级为「不改动」。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(os.path.dirname(_HERE), "_dialect_common"),
              os.path.join(os.path.dirname(os.path.dirname(_HERE)), "技能插件", "_dialect_common")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from router import build  # noqa: E402

_M = build(_HERE)

setup = _M.setup
teardown = _M.teardown
on_text_pre = _M.on_text_pre
on_report_enrich = _M.on_report_enrich
