"""示例插件：输出增益 + 报告追加字段
=====================================
这是插件接口的最小可用示例，演示三件事：
  1. ``setup(ctx)`` 里读取插件设置（settings_schema 的 default 会被 plugins_config.json 覆盖）；
  2. ``output.post`` 钩子里改写**成品音频**；
  3. ``report.enrich`` 钩子里向合成报告追加自定义字段。

默认状态：**停用**（plugin.json 的 enabled=false）。
启用方式（任选其一）：
  - 把 plugin.json 的 enabled 改成 true，然后重启服务；
  - 运行期：POST /api/plugins/example_gain/enabled  带表单字段 enabled=true
  - 或把 id 从 plugins_config.json 的 disabled 列表里移除。

必须遵守的三条契约
------------------
1. **处理器只接受 1 个参数**（payload 字典）。签名不符会在加载期被拒绝。
2. **绝不调用合成接口**（``synthesize_stable`` 等）。钩子运行在推理临界区内，
   同步重入会复用同一个模型 —— 轻则 CUDA 状态错乱，重则永久挂死且无任何报错。
   需要二次合成请放到钩子之外另行发起。
3. **返回类型必须匹配钩子契约**；返回 ``None`` 表示"不做改动"。
   类型不合法只会被丢弃并记一次错误，不会污染主流程。
"""

import numpy as np

# 模块级状态：setup() 里初始化，热重载时会重新执行本模块（状态归零）
_STATE = {"gain_db": 0.0, "report_tag": "example_gain", "applied": 0}


def setup(ctx):
    """插件启动：读设置、准备数据目录。异常会导致本插件被标记为 failed（不影响其它插件）。"""
    s = ctx.settings or {}
    try:
        _STATE["gain_db"] = float(s.get("gain_db") or 0.0)
    except (TypeError, ValueError):
        _STATE["gain_db"] = 0.0
    _STATE["report_tag"] = str(s.get("report_tag") or "example_gain")
    _STATE["applied"] = 0
    ctx.log(f"就绪：增益 {_STATE['gain_db']} dB，数据目录 {ctx.data_dir}")


def teardown(ctx):
    """插件停止/被停用/服务退出时调用（恰好一次），用于释放资源。"""
    ctx.log(f"已释放，本次累计处理 {_STATE['applied']} 次")


def on_output_post(payload):
    """成品音频后处理：按 dB 调整增益。

    不需要自己夹紧幅度 —— 服务端随后还会跑一次最终限幅（declip），
    插件输出越界不会写坏文件。
    """
    gain = _STATE["gain_db"]
    if abs(gain) < 1e-9:
        return None                       # 返回 None = 保持原样
    audio = np.asarray(payload["audio"], dtype=np.float32)
    _STATE["applied"] += 1
    return audio * float(10.0 ** (gain / 20.0))


def on_report_enrich(payload):
    """向报告追加字段。合并策略是"只增不改"：同名核心字段会被拒绝并记录日志。"""
    return {
        _STATE["report_tag"]: {
            "gain_db": _STATE["gain_db"],
            "applied": _STATE["applied"],
        }
    }


def on_startup(payload):
    print(f"[example_gain] 服务启动于 {payload.get('host')}:{payload.get('port')}"
          f"（应用版本 {payload.get('version')}）", flush=True)


def on_shutdown(payload):
    print(f"[example_gain] 服务退出，累计处理 {_STATE['applied']} 次", flush=True)
