# 技能插件

语言**能力**型插件放置区：听辨识别、词汇、发音、语法问答、互译。
与 [`../拓展插件`](../拓展插件) 平级，二者由同一套发现机制扫描（`core/voice_clone/plugin_core.py`）。

本目录当前内容 = **方言插件族**：13 个方言大区各一个技能插件 + 1 个跨方言路由插件。

## 1. 目录结构

```
技能插件/
  _dialect_common/            # 共享层（不是插件，不会被执行，也不需要 plugin.json）
    commands.py               #   指令词表的唯一来源（三方共用，禁止各写一份）
    common.py                 #   词典加载 / 建索引 / 互译 / 统计
    skill_factory.py          #   技能插件工厂：挖出「扫描指令→查词典→答→剥离」的通用逻辑
    extra_factory.py          #   拓展插件工厂（被 ../拓展插件 的插件复用）
    router.py                 #   跨方言检索 + 兜底路由的实现
  dialect_router/             # 跨方言路由插件（priority 20，跑在最后）
  方言_xxx_skill/  × 13        # 各方言技能插件（priority 60）
```

> **下划线前缀 = 不对发现机制可见**。`plugin_core._iter_plugin_dirs()` 里有
> `if name.startswith((".", "_")): continue` —— 所以 `_dialect_common/` 天生不会被当作插件扫描，
> **不需要**也不应该放 `plugin.json`（放了也是死文件；其 `id` 以 `_` 开头还会被 `_ID_RE` 拒绝）。
> 共享代码靠 `plugin.py` 里的 `sys.path` 注入来 import，不靠目录被发现。

> **默认停用**。本族 27 个插件清单里都是 `"enabled": false`，与 `example_gain` /
> `clear_vocal` 一致。原因是仓库承诺「**零插件 = 行为不变**」：所有插件都挂
> `report.enrich` + `text.pre`，默认启用就会改变默认行为
> （`core/tests/test_plugins.py::test_shipped_example_plugin_is_disabled_by_default` 就是这么护着的）。
> 要使用方言插件，在服务面板的「插件」页逐个启用，或点「**全部启用**」一键打开
> （启用状态落在 `core/plugins_config.json`，不随仓库提交）。

**为什么要有工厂**：技能插件的**逻辑**是通用的（扫描行首指令 → 查词典 → 生成答案 → 剥离指令），
各地区的差异全在**数据**里。13 个方言各抄一份 190 行代码，任何契约修订都要改 13 处，
极易出现「改了 A 忘了 B」的静默不一致。所以实现收敛到 `skill_factory.py`，
每个方言目录只剩 `dialect_data.json` + 一份几行的 `plugin.py` 薄壳。

## 2. 指令（写在合成文本的行首，中英文冒号都接受）

| 指令 | 作用 |
|---|---|
| `<方言简称>翻译：<词>` | 词典驱动互译（方言 ↔ 普通话，自动判方向） |
| `<方言简称>怎么说：<词>` | 普通话 → 方言 |
| `<方言简称>语法：<主题>` | 语法差异问答（主题见该方言的 `grammar_qa` 键） |
| `翻译：<词>` | **跨方言检索**：一次列出所有命中该词的方言 |
| `语法：<主题>` | 跨方言列出该主题在各方言的说明 |

方言简称：`粤语` `北方官话` `中原官话` `西南官话` `晋语` `吴语` `闽南语` `闽东语`
`客家话` `赣语` `湘语` `徽语` `平话`（也可用数据文件里的 `lang_short` 自定义）。

示例：

```
粤语翻译：唔该          → 「唔该」→ 普通话「谢谢/劳驾」（m4 goi1）【置信度:high｜来源:…】
晋语语法：入声          → 【晋语语法·入声】…（入声收喉塞音［-ʔ］，太原话「一、六、七、八、十」…）
翻译：聊天              → 跨方言检索命中 4 个方言：晋语「拉话」/ 北方官话「唠嗑」/
                          西南官话「摆龙门阵」/ 粤语「倾偈」
```

命中后**指令会从朗读文本里被剥离**（答案附在正文之后），不会被念出来。
查不到时明确回「未收录」，并报出实际检索了多少张表 —— **绝不编造**。
`keep_query_echo` 设置项可保留指令用于调试。

### ⚠️ 为什么必须带方言简称

`text.pre` 是**链式**钩子：值依次流经各处理器，谁先跑谁就能改写文本。
插件排序是 `priority 降序，同优先级按 id 升序`。
如果 13 个方言插件都认裸的 `翻译：`，那么 id 最小的那个会**抢答一切查询**、
对它不认识的词回「未收录」**并把指令从文本里剥掉** —— 其余 12 个方言永远轮不到。

这是实测踩到的坑，不是推测：加方言作用域之前，`翻译：圪蹴`（晋语特有词）
返回的是「未收录」，而 `翻译：我` 返回的是**赣语**的答案（因为赣语恰好排第一）。

因此：**各方言插件只认带自己简称的指令**；裸指令集中交给跑在最后的
[`dialect_router`](dialect_router/plugin.json)（`priority 20`）跨表检索。
路由插件同时是**兜底**：某个方言插件被停用或加载失败时，它那份作用域指令
不会被任何插件消费，此时由路由插件直接读词表作答。
（它只读 `dialect_data.json`，不依赖那些插件是否启用。）

## 3. 数据文件 `dialect_data.json`

```jsonc
{
  "lang_name": "吴语（太湖、台州、瓯江、婺州、宣州等片）",   // 全称
  "lang_short": "吴语",                                     // 指令里用的简称
  "roman_scheme": "通行拉丁化近似转写（非严格音位标音；精确记音见来源词典）",
  "sources": ["《苏州方言词典》（叶祥苓，江苏教育出版社 1993，现代汉语方言大词典分卷本）", "…"],
  "coverage_note": "本表共 23 条（置信度 high 15 / mid 7 / low 1）。…",  // 首句统计须与实际一致
  "grammar_qa": { "浊音": "吴语最显著的特征是保留全浊声母…" },
  "entries": [
    { "id": "wu_0001", "dialect": "弗", "roman": "feq", "mandarin": "不",
      "conf": "high", "src": "《苏州方言词典》", "note": "表一般否定，区别于表禁止的「勿」" }
  ]
}
```

字段约束（`coverage_note` 首句的统计由脚本核对，写死的数字一旦与数据脱节会被判错）：

- 每条必须有 `src`（来源）与 `conf ∈ {high, mid, low}`；
- `conf = "low"` 的条目**必须**写 `note` 说明待核实之处 —— 这是「不虚构」的机械保证；
- 来源必须是**真实出版物**（《现代汉语方言大词典》各分卷、《中国语言地图集（第2版）》等），
  不得凭印象编书名。

## 4. 新增一个方言插件的步骤

1. 在 `plugins/技能插件/` 下建目录 `dialect_<key>_skill/`。
2. 放 `dialect_data.json`（照上面 §3 的结构），并在同级的 `plugins/拓展插件/`
   建一个配套的 `dialect_<key>_extra/`（文化延伸，见那份 README）。
3. 放 `plugin.py`，内容固定为薄壳（照抄任一同族插件）：

```python
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(os.path.dirname(_HERE), "_dialect_common"),
              os.path.join(os.path.dirname(os.path.dirname(_HERE)), "技能插件", "_dialect_common")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from skill_factory import build   # noqa: E402
_M = build(_HERE)

setup = _M.setup
teardown = _M.teardown
on_text_pre = _M.on_text_pre
on_report_enrich = _M.on_report_enrich
```

4. 放 `plugin.json`：`id` 与目录名一致、`priority: 60`（**必须大于路由插件的 20**，
   否则会抢在路由之前把裸指令吃掉）、`enabled: false`（守住「零插件 = 行为不变」）、
   `hooks` 为 `{"text.pre": "on_text_pre", "report.enrich": "on_report_enrich"}`；
   `description` 里要写清**适用范围**与**示例用法**。
5. 补一个 `grammar_qa`，至少 3～5 个主题。
6. **不需要**改路由插件：它启动时扫描本目录，自动把新方言并入跨方言检索。

## 5. 扩展内容请放拓展插件

「技能插件」只回答**语言本体**（这个词怎么说、语法怎么变）。
俗语 / 歇后语 / 童谣 / 民俗属于文化延伸，放
[`../拓展插件`](../拓展插件)，复用同一套指令约定（动词换成 `俗语` `歇后语` `童谣` `民俗`）。
