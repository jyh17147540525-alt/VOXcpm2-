# 插件（Plugins）

外部模块通过钩子（hook）扩展 VoxCPM2，**不需要修改核心文件**。
核心实现在 `voice_clone/plugins.py`，契约以该文件的 `HOOK_SPECS` 为唯一准绳。

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
vox_plugins/
  my_plugin/
    plugin.json      # 清单（必需）
    plugin.py        # 入口模块（默认名，可在 entry 里改）
```

> ⚠️ 目录名是 `vox_plugins`，**不是** `plugins` —— 仓库里已有
> `voice_clone/plugins.py`（插件*机制*本身）。若目录也叫 `plugins`，
> 由于 pytest 会把测试文件所在目录（含 `voice_clone/`）插到 `sys.path`，
> `import plugins.<某插件>` 会命中那个模块文件而报
> `'plugins' is not a package`（且只在特定收集顺序下复现）。
> 目录名与机制模块名分开，两个概念各占一个名字。

`plugins_config.json`（与 `server.py` 同级，**不要提交**）控制启用状态：

```json
{
  "search_paths": ["vox_plugins"],
  "autoload": true,
  "disabled": ["my_plugin"],
  "settings": { "my_plugin": { "gain_db": 3.0 } }
}
```

字段含义：
- `search_paths`：插件搜索目录（相对项目根或绝对路径）。
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

## 8. 调试建议

1. `GET /api/plugins` 看 `state` / `error` / `stats.last_error` / `discovery_errors`。
2. 插件日志前缀为 `[VoxCPM2][plugin] <id>:`，同时进 stdout 与 `server_error.log`。
3. 先只用 `report.enrich` 这类"只读增强"钩子验证挂载成功，再动音频。
4. 改完代码调用 `/reload` 即可，不必重启整个服务（模型不会重新加载）。

## 9. 内置插件：清唱生成（`clear_vocal`）

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
from vox_plugins.clear_vocal import plugin as CV

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

护栏：`tests/test_clear_vocal_plugin.py`（17）、`tests/test_clear_vocal_units.py`（24）、
`tests/test_clear_vocal_aligner.py`（33）；突变自测 `_mut_clear_vocal.py`（8/8 捕获）。
