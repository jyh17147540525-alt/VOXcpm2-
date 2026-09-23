# 插件（Plugins）

外部模块通过钩子（hook）扩展 VoxCPM2，**不需要修改核心文件**。
核心实现在 `core/voice_clone/plugin_core.py`，契约以该文件的 `HOOK_SPECS` 为唯一准绳。

> 默认状态：**零插件生效**，所有钩子在无处理器时是常数时间 no-op，
> 行为与引入插件机制之前逐字节一致。

---

## 1. 用途

| 场景 | 用哪个钩子 |
| --- | --- |
| 文本预处理（术语替换、注音、去括号提示语） | `text.pre` |
| 调整分块 / 停顿类型（把逗号改成句末长停顿等） | `text.chunks` |
| 逐块音频处理（重采样、去咔哒、限幅） | `chunk.post` |
| 整段成品处理（EQ、响度、导出副本） | `synth.post` / `output.post` |
| 自定义情绪判定（替换/细化规则引擎） | `emotion.detect` |
| 往合成报告里塞自定义指标 | `report.enrich` |
| 参考音频换成自建的处理结果 | `reference.post` |
| 追加 LLM 服务商预设 | `llm.providers` |
| 追加 HTTP 路由（自建 UI / Webhook） | `api.routes` |
| 启动加载资源、退出释放资源 | `lifecycle.startup` / `lifecycle.shutdown` |

## 2. 目录结构

```
<仓库根>/
  core/                      # 主项目（server.py / voice_clone/ / tests/ ...）
  plugins/                   # ← 插件总目录（本文件所在）
    技能插件/                  # 能力型：听辨、词汇、发音、语法问答、互译
      README.md              #   方言插件族的说明与「新增一个方言」步骤
      _dialect_common/       #   共享层（不是插件）：指令词表 + 词典引擎 + 两个工厂 + 路由
      dialect_router/        #   跨方言检索 / 兜底路由（priority 20）
      dialect_<key>_skill/   #   13 个方言大区的技能插件（priority 60）
      clear_vocal/           #   内置清唱生成（默认停用）
      example_gain/          #   仓库自带的可提交模板
    拓展插件/                  # 延伸型：俗语、歇后语、民谣、民俗
      README.md
      dialect_<key>_extra/   #   13 个方言大区的文化延伸插件
```

> **随仓库发布的插件一律 `enabled: false`**（`example_gain`、`clear_vocal`、
> 方言插件族的 27 个都是）。这是为了守住上面那条「零插件生效」的保证 ——
> 已在 `core/tests/test_plugins.py::test_shipped_example_plugin_is_disabled_by_default`
> 里作为回归护栏固定下来。要用就在面板里启用，或点「全部启用」。

> **下划线前缀的目录不对发现机制可见**（`_iter_plugin_dirs()` 里
> `if name.startswith((".", "_")): continue`），可用来放共享代码 ——
> 例如 `技能插件/_dialect_common/`。所以共享层**不需要**也不应该放 `plugin.json`。

**两个子区是结构约定**（`plugin_core.PLUGIN_ZONE_SKILL` / `PLUGIN_ZONE_EXTRA`）：
发现逻辑本身不依赖它们 —— 只要目录（含子区）里有 `plugin.json` 就会被发现，
往下最多走 `DISCOVER_MAX_DEPTH`（3）层。

### ⚠️ 同一钩子上的多插件：注意「谁先跑」

`priority` **数值大的先执行**，同值按 id 稳定排序；而 `pipeline` 类钩子
（`text.pre` 等）是**链式**的 —— 后者收到的是前者的输出。
所以当多个插件都想处理同一类输入时，**排在前面且「认领」了输入的插件会把机会吃掉**。

方言插件族就是这个坑的实例：13 个方言若都认裸指令 `翻译：`，id 最小的那个会抢答
一切查询、对它不认识的词回「未收录」并把指令从文本里剥掉，其余方言永远轮不到。
解法是给指令加**方言作用域**（`<简称>翻译：X`），并把裸指令集中交给
跑在最后的 `dialect_router`。详见 [`技能插件/README.md`](技能插件/README.md) §2。

**新增插件时请先想清楚**：你的插件会不会认领一段本来属于别人的输入？
如果需要「只有在能处理时才消费」，就**把不认识的输入原样放行**（返回 None 或保持原行），
不要「认领后报错」。

### ⚠️ 子区名是中文，不能直接 `import`

`技能插件` / `拓展插件` 不是合法的 Python 标识符，所以**不能**写
`from plugins.技能插件.clear_vocal import ...`。测试与库函数用
`core/tests/_plugin_path.py` 构造一个 `plugins` 命名空间包（`__path__`
指向两个子区），再按 `plugins.<插件 id>` 导入。

### 为什么机制模块叫 `plugin_core` 而不是 `plugins`

历史坑，别再改回去：顶层名 `plugins` 一旦被**目录**占用，包属性解析会与
`voice_clone/plugins.py`（机制模块）互相干扰，症状是运行期
`module 'plugins' has no attribute 'get_registry'`，且**只在有人 import 过
`plugins` 时才复现**，极难定位。现已把机制模块改名为
`voice_clone/plugin_core.py`，两个概念各占一个名字，目录才能安心叫 `plugins`。

`plugins_config.json`（与 `core/server.py` 同级，**不要提交**）控制启用状态：

```json
{
  "search_paths": ["../plugins"],
  "autoload": true,
  "disabled": ["my_plugin"],
  "settings": { "my_plugin": { "gain_db": 3.0 } }
}
```

字段含义：
- `search_paths`：插件搜索目录（相对**应用根 `core/`** 或绝对路径）。
  默认值 `"../plugins"` 即指向仓库根的插件总目录。若没配，会依次回落到
  `<base_dir>/plugins`、`<base_dir>/../plugins` 以及环境变量 `VOXCPM_PLUGIN_PATH`。
- `autoload`：`false` 时只发现不加载（排查用）。
- `disabled`：停用名单，**优先级最高**（运行期停用也写在这里）。
- `settings`：插件设置，覆盖 `settings_schema` 里的 default。

## 3. 清单字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 稳定标识，`[A-Za-z0-9._-]`，≤64 字符 |
| `api_version` | 是 | 插件接口版本，**主版本必须等于** `PLUGIN_API_VERSION` 的主版本 |
| `version` | 否 | 插件自身版本 |
| `requires_app` | 否 | 应用版本约束，如 `">=2.0,<3"`（逗号 = 逻辑与） |
| `entry` | 否 | 入口文件名，默认 `plugin.py` |
| `enabled` | 否 | 清单级开关，默认 `true`（示例插件为 `false`） |
| `priority` | 否 | 整数，**数值大的先执行**；同值按 id 稳定排序 |
| `isolation` | 否 | `inprocess`（默认）。`subprocess` **会被拒绝**，见 §6 |
| `hooks` | 否 | 钩子 → 处理器函数名；省略函数名则用 `on_<钩子名，点换下划线>` |
| `requires_packages` | 否 | 依赖的 Python 包，缺失则加载失败并给出明确原因 |
| `settings_schema` | 否 | JSON-Schema 风格，`properties.*.default` 作为设置默认值 |

未知顶层字段与未知钩子名**只告警不拒绝**（向前兼容）；非法值（类型错、版本不符）**直接拒绝**。

## 4. 处理器契约

```python
def on_output_post(payload):      # 只接受 1 个参数
    return payload["audio"] * 0.9 # 返回新值；返回 None = 不做改动
```

按钩子种类的返回值语义：

| kind | 语义 | 返回值要求 |
| --- | --- | --- |
| `pipeline` | 值依次流经各插件，后者收到前者的输出 | 与输入同类型（音频=ndarray，文本=str） |
| `first` | 第一个非空结果生效（`emotion.detect`） | 非空字符串 |
| `merge` | 字典浅合并，**不覆盖既有键**（`report.enrich`） | dict |
| `synth` | `synth.post`，可只覆盖 `{"audio":…}` 或 `{"report":…}` | ndarray 或 dict |
| `observe` | 返回值忽略，只做副作用（`api.routes` / `lifecycle.*`） | 无要求 |

返回值类型不合法时**保持原值**并记一次错误，不会污染主流程。

## 5. 生命周期

```
discovered → validated → loaded → started → (stopped | failed | disabled)
```

- `setup(ctx)`：加载成功后调用。`ctx` 提供 `id` / `base_dir` / `data_dir`（可写目录，已 gitignore）/ `settings` / `log()` / `registry`。
- `teardown(ctx)`：停止、被停用、被熔断、服务退出时调用，**保证恰好一次**。
- 热重载：`POST /api/plugins/<id>/reload`（改完插件代码无需重启服务）。
- 启停：`POST /api/plugins/<id>/enabled`（表单字段 `enabled=true|false`，落盘）。
- 单独试运行：`POST /api/plugins/<id>/invoke`（只跑这一个插件，不合成、不占显卡，见 §8）。
- 查看状态：`GET /api/plugins`（需 `x-api-key`）。

## 6. 错误处理与隔离（重要）

- 任何钩子异常都被隔离：**记录 + 计数**，主流程继续，其它插件不受影响。
- 连续 `MAX_CONSECUTIVE_ERRORS`（5）次异常或非法返回 → **自动熔断停用**该插件，
  并调用其 `teardown()` 释放资源（`/api/plugins` 的 `state` 会显示 `failed` 与原因）。
- 单次钩子耗时超过 `SLOW_HOOK_WARN_MS`（250ms）会记一条告警。**钩子是同步执行的，
  它会阻塞这次请求**，所以重活儿请异步化或移到钩子外。
- `isolation: "subprocess"` 声明会被**拒绝加载**：本运行时只提供进程内隔离。
  这是有意为之 —— 静默降级会让插件作者误以为自己崩溃不会影响主进程。

## 7. 红线：钩子内不得回调生成路径

`server.py` 用**非可重入**的 `threading.Lock` 串行化推理。因此：

- ✅ 钩子里做纯数据变换（文本 / 音频 / 报告）。
- ❌ 钩子里同步调用 `synthesize_stable()` 等生成接口 —— 会抛 `RuntimeError`（快速失败优于静默损坏）。
- ❌ 钩子里启动线程去请求本机的生成接口并等待它 —— 会撞上推理锁**永久挂死**，
  这类跨线程等待无法自动检测，属于插件作者的红线。

需要二次合成时，请在钩子之外另行发起。

可用 `plugins.current_hook()` 判断当前处于哪个钩子：
在 `_infer_lock` **之内**运行的是 `text.pre` / `text.chunks` / `chunk.post` /
`synth.post` / `emotion.detect`；`output.post` / `api.routes` / `lifecycle.*` 在其之外。

## 8. 单独试运行单个插件（不合成、不占显卡）

排查"这个插件到底有没有生效"最省事的第一步：把它单独跑一次，不走合成流水线、
不加载模型、不占显存，毫秒级返回。

- **界面**：插件面板里每张已启用的卡片自带一个输入框，填一行文本点「试运行」
  （或直接按回车），结果就显示在输入框下方。
- **接口**：`POST /api/plugins/<id>/invoke`，表单字段 `hook`（默认 `text.pre`）与 `text`。

```bash
curl -X POST http://127.0.0.1:8808/api/plugins/my_plugin/invoke \
     -H "x-api-key: <TOKEN>" \
     -F "hook=text.pre" -F "text=要试的内容"
```

只允许试运行 **pipeline 类且携带纯文本**的钩子（当前即 `text.pre`）：音频 / ndarray
之类的钩子依赖真实合成上下文，单独调用没有意义。

返回的 `result` 是三态之一，它是**"真实流水线会怎么处理"的如实预告**：

| `result` | 含义 | `output` |
| --- | --- | --- |
| `handled` | 插件处理了这行 | 真正会进入合成的新文本 |
| `unchanged` | 插件返回 `None`（"这行不归我管"） | 原样等于 `input` |
| `invalid` | 返回值未通过钩子校验（如 `text.pre` 返回空串） | 原样等于 `input`（真实合成会丢弃它并保持原值） |

另外还有 `raw_output`（插件到底返回了什么，便于自查）、`applied` / `changed` /
`valid` / `duration_ms`。

失败时给出明确状态码：`404` 插件不存在、`409` 插件未启用或没挂该钩子、
`400` 钩子不可试运行、`401` 令牌无效、`500` 处理器抛异常。

⚠️ 与合成时的一个区别：**只跑指定的那一个插件**。整链执行下，链上先跑的插件可能
已经改写了输入，结果无法归因到某一个插件 —— 这正是试运行要解决的问题。

## 9. 方言指令在合成里怎么触发（面向使用者）

指令写在**合成文本框的行首**，中英文冒号都接受。命中后指令行会从朗读文本里被剥离，
答案作为待朗读内容追加在正文之后。例：

```
粤语翻译：聊天          → 普通话「聊天」→ 粤语（广州话）「倾偈」
翻译：聊天              → 跨方言检索：晋语「拉话」/ 北方官话「唠嗑」/
                          西南官话「摆龙门阵」/ 粤语「倾偈」
粤语语法：量词          → 【粤语语法·量词】…
粤语俗语：扮猪食老虎    → 释义 + 出处
```

⚠️ **触发路径**（这里曾有一个真实缺陷，已修）：`text.pre` 的唯一调用点挂在
`synthesis_stable()` 内，而 `_do_generate()` 对**短文本**会走 `model.generate()`
直通路径 —— 于是「只写了一句八个字、又没勾『长文本稳定合成』」时插件根本不会被触发，
指令被原样念出来。实测（真机合成 + CPU ASR 反向转写）：

| 条件 | 时长 | ASR 转写 |
| --- | --- | --- |
| 八个字 + 未勾稳定（修复前） | 1.9 s | `寓语翻译聊天` ← 插件没生效，指令被念了 |
| 八个字 + 勾选稳定 | 11.0 s | 完整答案（含注音/来源） |

现在两条路径都会调用 `text.pre`，`tests/test_textpre_wiring.py` 用 AST 把这条
不变量钉住：**改任何一条路径而漏掉另一条，测试会红**。

| 入口 | 是否触发插件 |
| --- | --- |
| 多人对话 / 对话台本 | ✅ 一定（无条件走稳定路径） |
| 设计 / 克隆，勾选「长文本稳定合成」 | ✅ |
| 设计 / 克隆，文本 ≥ 100 字（`VOXCPM_LONG_TEXT_CHARS`） | ✅ 自动切稳定路径 |
| 设计 / 克隆，短文本且未勾选 | ✅ 修复后 ／ ❌ 修复前 |

⚠️ 方言插件族在**发布树里默认 `enabled: false`**（仓库承诺「零插件 = 行为不变」），
需先在插件面板启用，或把对应 `plugin.json` 的 `enabled` 改成 `true`。
一次合成最多处理 3 条查询（`max_queries_per_call`，可调到 10）；**超出的那几行不会被丢弃**，
而是落到跑在最后的 `dialect_router` 继续处理（措辞略有不同）；若路由插件也被停用，
超出的行就会原样被朗读。

⚠️ **已知体验问题**：答案里的注音、置信度、来源是**书面元信息**，它们同样进入朗读文本，
会被念成「…听改二知信度来源广州方言词典」之类的噪音；同理 `〔原指令〕` 归属标注也会被念。
让答案"给耳朵听"而不是"给眼睛看"是下一步的事（尚未实现）。

## 10. 调试建议

1. 先在插件面板里「试运行」一行文本（见 §8）—— 不用合成、不占显卡，
   能立刻分清"是插件没生效"还是"是合成这条链路的问题"。
2. `GET /api/plugins` 看 `state` / `error` / `stats.last_error` / `discovery_errors`。
3. 插件日志前缀为 `[VoxCPM2][plugin] <id>:`，同时进 stdout 与 `server_error.log`。
4. 先只用 `report.enrich` 这类"只读增强"钩子验证挂载成功，再动音频。
5. 改完代码调用 `/reload` 即可，不必重启整个服务（模型不会重新加载）。

## 11. 内置插件：清唱生成（`clear_vocal`）

从一首歌生成"节奏与旋律同原曲一致"的清唱人声。默认**停用**，
启用后只注册两个**只读**钩子（`report.enrich` 汇报状态、`api.routes` 暴露查询接口），
真正的生成通过库函数在钩子之外发起。

### 数据流

```
人声 stem + 歌词文本 + 目标音色参考
  → analyzer.analyze     节拍 / F0 / 音符序列 / 调式
  → planner.plan         音符 × 歌词 → note_plan.json（人工可校正）
  → singer.sing_plan     逐音符合成（唯一接触模型的步骤，音色锚定同一参考）
  → singer.measure_clips 实测各片段自身音高（防整曲被移调一个八度）
  → aligner.align_all    零漂移对齐 + 拼接
  → clear_vocal.wav + align_diag.json
```

### 调用方式（Python，必须在钩子之外）

```python
from plugins.技能插件.clear_vocal import plugin as CV   # 子区名中文，需 _plugin_path 辅助
# 或：core/tests/_plugin_path.py 的 load_module("技能插件", "clear_vocal", "plugin")

res = CV.run(
    model, sr,
    reference_wav="voice.wav",        # 目标音色
    vocal_audio="stems/vocals.wav",   # ⚠️ 必须是**人声 stem**，不是混音原曲
    lyric_text="春天的花开啦",          # 空则整首哼鸣
    out_dir="outputs/my_song",
    lock=server._infer_lock,          # 传了就只在推理瞬间加锁
    snap=True,                        # 散板素材设 False
)
print(res["out_wav"], res["summary"])
```

### 产物（全部落在 `out_dir`，便于定位问题出在哪一步）

| 文件 | 内容 |
|---|---|
| `analysis.json` | 节拍、F0、音符、调式原始分析结果 |
| `note_plan.json` | 每个音符挂哪个字、目标时刻/时长/音高、实测音高 |
| `clips/*.wav` | 每个音符的合成片段（未对齐），出问题时先听这里 |
| `align_diag.json` | 逐音符"要求什么 / 实际做了什么"+ 汇总 |
| `clear_vocal.wav` | 最终清唱 |

### 三个必须知道的坑

1. **输入必须是分离后的人声轨**。直接喂混音原曲会让节拍与音高估计严重失真
   （实测：纯净 120 BPM 信号叠加一个持续音后，librosa 给出 92.29 BPM，偏差 23%）。
2. **合成只能在钩子外调用**。`singer` 内置守卫会在钩子内立刻抛 `HookContextError`；
   若绕过守卫（例如自行调用 `synthesize_stable`），结果是**永久死锁且无报错**。
3. **实测片段音高不可省**。`aligner` 的搬移量 = 目标音高 − 片段自身音高；
   若用错一个八度的猜测值，整曲会被平移一个八度，
   而单音符 ±12 半音的钳制恰好不会拦下 12 这个值。故测不出时**回落为不搬移**。

### 精度（实测）

| 指标 | 结果 |
|---|---|
| 音符落点误差 | **0 采样**（含 300 音符累计漂移） |
| 整曲长度误差 | **0 采样** |
| 移调误差（±12 半音） | 0.008 – 0.192 半音 |
| 变速不变调 | 0.000 – 0.100 半音 |
| 叠加 8 路满幅限幅后峰值 | 0.969（不削波） |

护栏：`core/tests/test_clear_vocal_plugin.py`（17）、`test_clear_vocal_units.py`（24）、
`test_clear_vocal_aligner.py`（33）；突变自测 `_mut_clear_vocal.py`（8/8 捕获）。
