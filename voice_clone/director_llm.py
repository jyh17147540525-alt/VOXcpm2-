"""
模块 5：AI 导演层（LLM 内核）
==============================
用大模型替换/增强模块 4 的规则判定：规则版只能识别词表里有的情绪，
LLM 版读得懂语境、潜台词与人物心理，且情绪描述不限于固定枚举。

架构上的关键设计：**LLM 不接触原文**
------------------------------------
上游先用规则分块器把文本切好并编号，只把「全文 + 编号分段」给 LLM，
LLM 返回的每个元素只有一个编号和若干韵律数值 —— **没有 text 字段**。
最终 Segment.text 一律取自分块结果，因此 LLM 在物理上不可能改动原文，
无需事后 diff 校验，也不受模型「爱润色」的习性影响。

情绪如何落地（受 VoxCPM2 接口限制）
-----------------------------------
VoxCPM2 的 generate() 没有 emotion / style / speed 参数，可用的表达通路只有：
  1. 文本修饰      —— 本模块不做（只梳理不改写）
  2. cfg_value     —— 表现力强度，Segment.cfg
  3. 事后 DSP      —— Segment.pace / pitch_st
所以 LLM 的自由情绪描述（emotion / note）用于展示与决策，
真正送进合成的是 pace / pitch_st / cfg / pause_after 这几个数值。

配置
----
读取项目根目录的 llm_config.json；api_key 为空时依次回退到
api_key.txt、环境变量 DEEPSEEK_API_KEY / VOXCPM_LLM_API_KEY。
任一环节缺失或调用失败都会**自动降级到规则版**（director.plan），
合成流程不会因此中断。

换成本地小模型：把 base_url 改成 http://127.0.0.1:11434/v1（Ollama 等
同样提供 OpenAI 兼容接口），其余不用动。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

try:
    from . import llm_providers as _providers
except ImportError:  # pragma: no cover - 无包上下文的直接加载
    import sys as _sys0

    _sys0.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import llm_providers as _providers  # type: ignore

try:
    from .director import PlanResult, Segment
    from .director import plan as rule_plan
    from .director import rechunk as _rechunk
    from .director import verify_text_intact
except ImportError:  # pragma: no cover - 无包上下文的直接加载
    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from director import PlanResult, Segment  # type: ignore
    from director import plan as rule_plan  # type: ignore
    from director import rechunk as _rechunk  # type: ignore
    from director import verify_text_intact  # type: ignore


# --------------------------------------------------------------------- 配置

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(_BASE_DIR, "llm_config.json")
API_KEY_TXT = os.path.join(_BASE_DIR, "api_key.txt")
CACHE_DIR = os.path.join(_BASE_DIR, "_llm_cache")

DEFAULT_CONFIG = {
    "enabled": False,
    "provider": "deepseek",    # 服务商预设 id（见 llm_providers.PROVIDERS）
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key": "",
    # 用户在界面上显式清空过密钥 → 跳过 api_key.txt / 环境变量回退。
    # 否则环境变量里的 DEEPSEEK_API_KEY 会让"清空"看起来没生效。
    "api_key_cleared": False,
    "timeout": 60,
    "temperature": 0.3,
    "proxy": "",               # 留空 = 不走代理（沙箱里的 HTTP_PROXY 常指向失效代理）
    "json_mode": True,         # 服务端不支持 response_format 时自动关掉重试
    "max_segments_per_call": 120,
    "full_text_limit": 6000,
    "cache": True,
}

#: 数值安全区间 —— LLM 偶尔会给出夸张值，超界一律夹回，避免机械感/破音。
_BOUNDS = {
    "intensity": (0.0, 1.0),
    "pace": (0.80, 1.20),
    "pitch_st": (-2.0, 2.0),
    "cfg": (1.8, 3.2),
    "pause_after": (0.0, 1.5),
}


def load_config(path: str | None = None) -> dict:
    """读取配置。缺失字段用默认值补齐；api_key 支持多来源回退。"""
    cfg = dict(DEFAULT_CONFIG)
    p = path or CONFIG_PATH
    _user_set_timeout = False
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                _user_set_timeout = "timeout" in user
                cfg.update({k: v for k, v in user.items() if v is not None})
        except Exception:
            pass

    # 服务商预设兜底：base_url / model 为空时按 provider 补全。
    # 必须在 api_key 回退链**之前**做，因为要不要回退取决于是否本地部署。
    cfg = _providers.apply_preset(cfg)
    _p = _providers.get_provider(cfg.get("provider"))

    # timeout 按服务商给推荐值 —— 仅当用户从未显式设过（配置文件里没有该键）。
    # 一旦用户自己填过，就完全尊重他的选择，不再"智能纠偏"，避免他设了 30s
    # 却发现程序偷偷改成 300s。切服务商时由 save_config 清掉遗留值，
    # 让推荐值重新生效（见 save_config 里对 provider 变化的处理）。
    if not _user_set_timeout:
        cfg["timeout"] = _providers.recommended_timeout(cfg.get("provider"))

    # 用户在界面上显式清空过密钥 → 尊重这个决定，跳过全部回退。
    # 否则环境变量里的 DEEPSEEK_API_KEY 会把 key "变回来"，用户以为没清掉。
    if cfg.get("api_key_cleared") and not (cfg.get("api_key") or "").strip():
        if _p.get("local"):
            cfg["api_key"] = _p["id"]
        return cfg

    # api_key 回退链：配置文件 → api_key.txt → 环境变量
    # ⚠️ 本地部署（Ollama / LM Studio / vLLM）无鉴权：此时空 key 是**正常状态**，
    # 绝不能拿环境变量里的 DEEPSEEK_API_KEY 去填 —— 那会把一个云服务商的密钥
    # 塞进本地请求，既无意义又可能把密钥发给本机之外的地址（若 base_url 被改错）。
    if _p.get("local"):
        if not (cfg.get("api_key") or "").strip():
            cfg["api_key"] = _p["id"]        # 占位符，让 Bearer 头格式统一
        return cfg

    if not (cfg.get("api_key") or "").strip():
        if os.path.isfile(API_KEY_TXT):
            try:
                with open(API_KEY_TXT, encoding="utf-8") as f:
                    cfg["api_key"] = f.read().strip()
            except Exception:
                pass
    if not (cfg.get("api_key") or "").strip():
        for env in ("DEEPSEEK_API_KEY", "VOXCPM_LLM_API_KEY", "OPENAI_API_KEY"):
            v = (os.environ.get(env) or "").strip()
            if v:
                cfg["api_key"] = v
                break
    return cfg


def save_config(cfg: dict, path: str | None = None) -> str:
    """写回配置（保留未知字段）。

    特殊处理 `api_key_cleared`：当 caller 显式把 api_key 写成空串时，打上
    "用户已清空"标记，让 load_config 不再去 api_key.txt / 环境变量捞 key —— 
    否则用户点了清空、界面上却依旧显示"已配置"，属于欺骗性的 UI。
    写入非空 key 时自动清除该标记。
    """
    p = path or CONFIG_PATH
    data = dict(DEFAULT_CONFIG)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                cur = json.load(f)
            if isinstance(cur, dict):
                data.update(cur)
        except Exception:
            pass
    data.update(cfg or {})
    # 调用方想"让 timeout 回到推荐值"时，传 timeout=None 即可 —— 这里把它删掉，
    # 下次 load_config 就会按当前服务商填入推荐值（云 60s / 本地 300s）。
    if (cfg or {}).get("timeout", "keep") is None:
        data.pop("timeout", None)
    # 依据最终 key 值决定标记状态，避免调用方漏传而留下陈旧标记
    if (data.get("api_key") or "").strip():
        data["api_key_cleared"] = False
    elif "api_key" in (cfg or {}):
        data["api_key_cleared"] = True
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def is_ready(cfg: dict | None = None) -> tuple:
    """检查是否具备调用条件，返回 (可用?, 原因)。"""
    c = cfg or load_config()
    if not c.get("enabled"):
        return False, "AI 内核未启用（llm_config.json 里 enabled 为 false，或在界面上打开开关）"
    _p = _providers.get_provider(c.get("provider"))
    if not (c.get("api_key") or "").strip():
        if _p.get("local"):
            # 本地服务通常无鉴权，不该因缺 Key 被拦
            pass
        else:
            return False, "缺少 API Key —— 打开「AI 内核配置」填写，或写入 llm_config.json / api_key.txt"
    if not (c.get("base_url") or "").strip():
        return False, "缺少 base_url（接口地址）"
    if not (c.get("model") or "").strip():
        return False, "缺少 model（模型名）"
    return True, "ok"


# --------------------------------------------------------------------- HTTP

def _opener(cfg: dict):
    """构造 URL opener。proxy 为空时显式禁用代理继承。

    沙箱/部分环境残留的 HTTP_PROXY 会指向已失效的代理，导致
    WinError 10061「拒绝连接」—— 假故障，直连其实通。默认不走代理。
    """
    proxy = (cfg.get("proxy") or "").strip()
    handler = urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {})
    return urllib.request.build_opener(handler)


def _chat(cfg: dict, messages: list) -> str:
    """调用 OpenAI 兼容的 /chat/completions，返回助手文本。"""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"

    def _post(use_json_mode: bool):
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": float(cfg.get("temperature", 0.3)),
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": "Bearer %s" % cfg["api_key"],
            },
            method="POST",
        )
        with _opener(cfg).open(req, timeout=float(cfg.get("timeout", 60))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    try:
        return _post(bool(cfg.get("json_mode", True)))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        # 有些兼容层不认 response_format，回退到纯提示词约束再试一次
        if e.code == 400 and "response_format" in detail.lower() and cfg.get("json_mode", True):
            return _post(False)
        if e.code == 401:
            raise RuntimeError("API Key 无效或被拒绝（HTTP 401）：%s" % detail)
        if e.code == 402:
            raise RuntimeError("账户余额不足（HTTP 402）：%s" % detail)
        if e.code == 429:
            raise RuntimeError("触发限流（HTTP 429），稍后重试：%s" % detail)
        raise RuntimeError("API 调用失败 HTTP %s：%s" % (e.code, detail))
    except urllib.error.URLError as e:
        raise RuntimeError("网络不可达：%s" % e)


# --------------------------------------------------------------------- 诊断
# 供 Web UI 的「测试连接」用。设计原则：**永不抛异常**，一律返回
# {"ok": bool, "stage": str, "message": str, ...}，让前端直接渲染。


def list_models(cfg: dict) -> list:
    """尝试拉取 /models 列表。失败返回空列表（不抛异常）。

    部分服务商不实现 /models（或需要特殊权限），所以拿不到列表**不代表**
    Key 无效 —— 调用方不能据此判定失败。
    """
    url = str(cfg.get("base_url") or "").rstrip("/") + "/models"
    try:
        req = urllib.request.Request(
            url,
            headers={"Authorization": "Bearer %s" % cfg.get("api_key", "")},
            method="GET",
        )
        with _opener(cfg).open(req, timeout=min(float(cfg.get("timeout", 60)), 15)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if isinstance(it, dict):
            mid = it.get("id") or it.get("name")
            if mid:
                out.append(str(mid))
        elif isinstance(it, str):
            out.append(it)
    return sorted(set(out))


def test_connection(cfg: dict | None = None, *, probe_chat: bool = True) -> dict:
    """端到端连通性诊断，给 UI 的「测试连接」按钮用。

    分三段独立判定，便于定位问题到底出在哪一层：
      1. 本地配置完整性（不联网）
      2. GET /models（鉴权 + 网络；拿不到列表不算失败）
      3. POST /chat/completions 极短探针（真正验证能否干活）

    参数 probe_chat=False 时跳过第 3 段，纯看配置与鉴权，用于"省一次调用"。
    """
    from . import llm_providers as LP

    c = LP.apply_preset(dict(cfg or load_config()))
    provider = LP.get_provider(c.get("provider") or LP.detect_provider(c.get("base_url")))

    result = {
        "ok": False,
        "stage": "config",
        "message": "",
        "provider": provider["id"],
        "provider_name": provider["name_zh"],
        "base_url": c.get("base_url", ""),
        "model": c.get("model", ""),
        "models": [],
        "latency_ms": None,
        "chat_ok": False,
    }

    # --- 第 1 段：配置完整性
    if not str(c.get("base_url") or "").strip():
        result["message"] = "还没填 base_url（接口地址）"
        return result
    if not str(c.get("model") or "").strip():
        result["message"] = "还没填模型名（model）"
        return result

    raw_key = c.get("api_key") or ""
    ok_key, tip = LP.key_looks_valid(raw_key, provider["id"])
    if not ok_key:
        result["message"] = tip
        return result
    c["api_key"] = LP.normalize_key(raw_key)
    result["key_hint"] = tip  # 通过时可能带"本地服务"之类的说明

    # --- 第 2 段：/models（鉴权探针）
    t0 = time.time()
    models = list_models(c)
    result["latency_ms"] = int((time.time() - t0) * 1000)
    result["models"] = models
    result["stage"] = "models"

    # --- 第 3 段：真实 chat 探针
    if not probe_chat:
        result["ok"] = True
        result["stage"] = "models"
        result["message"] = (
            f"配置完整，鉴权接口连通（{len(models)} 个模型可见）"
            if models else "配置完整；该服务商未提供模型列表接口，建议做一次实际生成验证"
        )
        return result

    probe = {
        "model": c["model"],
        "messages": [{"role": "user", "content": "回复两个字：收到"}],
        "temperature": 0.0,
    }
    url = str(c["base_url"]).rstrip("/") + "/chat/completions"
    result["stage"] = "chat"
    # 探针超时 = min(配置超时, 上限)。本地模型首次调用要加载权重（实测 27B
    # 冷启 ~30s），所以本地服务的上限放宽，否则用户第一次点"测试连接"必失败。
    _cap = 300 if provider.get("local") else 60
    _probe_timeout = max(5.0, min(float(c.get("timeout", 60)), float(_cap)))
    result["probe_timeout_s"] = _probe_timeout
    t0 = time.time()
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(probe, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": "Bearer %s" % c["api_key"],
            },
            method="POST",
        )
        with _opener(c).open(req, timeout=_probe_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = ""
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            pass
        result["latency_ms"] = int((time.time() - t0) * 1000)
        result["ok"] = True
        result["chat_ok"] = True
        result["stage"] = "chat"
        result["reply"] = str(content)[:40]
        result["message"] = (
            f"连接成功 · {result['latency_ms']}ms"
            + (f" · 可见 {len(models)} 个模型" if models else "")
        )
        return result
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        result["latency_ms"] = int((time.time() - t0) * 1000)
        if e.code == 401:
            msg = "API Key 无效或被拒绝（401）—— 请检查 Key 是否复制完整、是否属于当前服务商"
        elif e.code == 403:
            msg = "无权访问该模型（403）—— 可能 Key 没有开通此模型权限"
        elif e.code == 402:
            msg = "账户余额不足（402）"
        elif e.code == 404:
            msg = ("接口地址或模型名不存在（404）—— 请确认 base_url 是否漏了 /v1 "
                   "或 compatible-mode 之类的路径，以及模型名是否拼错")
        elif e.code == 429:
            msg = "触发限流（429），Key 有效但请求过频，稍后重试"
        else:
            msg = "API 调用失败 HTTP %s" % e.code
        result["message"] = msg
        result["detail"] = detail
        return result
    except urllib.error.URLError as e:
        result["latency_ms"] = int((time.time() - t0) * 1000)
        result["message"] = "网络不可达：%s" % e
        return result
    except Exception as e:
        result["latency_ms"] = int((time.time() - t0) * 1000)
        result["message"] = "未预期的错误：%s" % e
        return result


# --------------------------------------------------------------------- 提示词

_SYSTEM = (
    "你是一位资深配音导演，负责为有声书标注每一句的演播方式。"
    "你的职责是判断「这句话该怎么念」，而不是修改文字 —— 原文一个字都不会变。"
)

_USER_TMPL = """下面是一部作品的片段，以及它已经切好的分段。请为每一段决定演播方式。

【全文】（用于理解语境、人物心理与潜台词）
{full_text}

【待标注分段】共 {n} 段，编号 0 到 {n_max}
{numbered}

只输出一个 JSON 对象，不要任何解释或 markdown 代码块，格式如下：
{{
  "tone": "全文基调，一句话",
  "segments": [
    {{"i": 0, "emotion": "情绪描述", "intensity": 0.0,
      "pace": 1.0, "pitch_st": 0.0, "cfg": 2.0,
      "pause_after": 0.3, "role": "narration", "note": "演播提示"}}
  ]
}}

要求：
1. 必须覆盖全部 {n} 段，i 从 0 到 {n_max}，不重不漏。
2. emotion 用自由中文描述，越具体越好 —— 写「隐忍的怒气，尾音压着不抬」
   而不是笼统的「生气」。要结合上下文推断潜台词，不要只看字面有没有情绪词。
3. **数值必须克制**，这是硬要求：pace ∈ [0.8, 1.2]，pitch_st ∈ [-2, 2] 半音，
   cfg ∈ [1.8, 3.2]，pause_after ∈ [0, 1.5] 秒，intensity ∈ [0, 1]。
   幅度过大会产生机械感或破音，宁可接近中值也不要夸张。
4. 旁白（叙述）比台词克制，intensity 通常不超过 0.5。
5. role 只能是 "speech"（引号内的台词）或 "narration"（叙述/旁白）。
6. note 是给演播者的一句话提示（如「尾音下沉，像咬牙」），不参与合成。"""


def _build_messages(full_text: str, chunks: list, offset: int, limit: int):
    """构造单批请求的消息。offset/limit 决定这批处理哪些分段。"""
    seg = chunks[offset:offset + limit]
    n = len(seg)
    numbered = "\n".join("%d| %s" % (offset + k, t) for k, t in enumerate(seg))
    user = _USER_TMPL.format(
        full_text=full_text,
        n=n,
        n_max=offset + n - 1,
        numbered=numbered,
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------- 解析

def _extract_json(raw: str) -> dict:
    """从模型输出里抠出 JSON 对象（容忍 markdown 包裹与前后废话）。"""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("响应里找不到 JSON 对象")
    return json.loads(s[i:j + 1])


def _clamp(v, key: str, default=None):
    """夹到安全区间；无法转数值时返回 default。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:                       # NaN
        return default
    lo, hi = _BOUNDS[key]
    return max(lo, min(hi, f))


def _clean_str(v, limit: int = 80) -> str:
    if v is None:
        return ""
    s = str(v).strip().replace("\n", " ")
    return s[:limit]


def _clean_role(v, default: str) -> str:
    s = _clean_str(v).lower()
    if s in ("speech", "narration"):
        return s
    return default


def _merge(base: list, parsed: dict) -> list:
    """把 LLM 的标注合并回分块结果。

    关键：Segment.text 一律取自 base（分块结果），LLM 输出里没有、也不接受
    任何文本字段 —— 原文因此在架构上不可能被改动。
    LLM 漏标的段直接沿用规则版结果，保证段数与顺序恒定。
    """
    by_i = {}
    for item in (parsed.get("segments") or []):
        if not isinstance(item, dict):
            continue
        try:
            by_i[int(item.get("i"))] = item
        except (TypeError, ValueError):
            continue

    out = []
    for i, b in enumerate(base):
        s = by_i.get(i)
        if not s:
            out.append(b)                     # 漏标 → 规则版兜底
            continue
        emo = _clean_str(s.get("emotion")) or b.emotion
        out.append(Segment(
            text=b.text,                      # ← 永远来自分块，绝不用 LLM 的文本
            pause_type=b.pause_type,
            emotion=emo,
            intensity=_clamp(s.get("intensity"), "intensity", b.intensity),
            pause_after=_clamp(s.get("pause_after"), "pause_after", b.pause_after),
            role=_clean_role(s.get("role"), b.role),
            reason="LLM 导演：%s" % emo,
            pace=_clamp(s.get("pace"), "pace", None),
            pitch_st=_clamp(s.get("pitch_st"), "pitch_st", None),
            cfg=_clamp(s.get("cfg"), "cfg", None),
            note=_clean_str(s.get("note"), 120),
        ))
    return out


# --------------------------------------------------------------------- 缓存

def _cache_path(text: str, cfg: dict) -> str:
    key = "|".join([text, cfg.get("model", ""), cfg.get("base_url", ""), _PROMPT_VERSION])
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return os.path.join(CACHE_DIR, h + ".json")


_PROMPT_VERSION = "v1"   # 提示词改动时递增，自动让旧缓存失效


def _cache_read(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_write(path: str, obj: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


# --------------------------------------------------------------------- 主入口

def plan_llm(text: str, max_chars: int = 60, *,
             config: dict | None = None,
             fallback: bool = True,
             use_cache: bool | None = None,
             chunks: list | None = None,
             context: str | None = None) -> PlanResult:
    """用 LLM 梳理文本，输出与 director.plan 同构的 PlanResult。

    任何环节失败（未启用 / 无 key / 网络不通 / 解析失败）都会自动降级到
    规则版（除非 fallback=False），降级原因记在 PlanResult.error 里。

    chunks：可选，直接指定切分边界（如多人对话的说话人 turn 列表）。
      - 不传 → 由规则版切分（默认行为，逐句）。
      - 传了 → 按给定边界归组（director.rechunk），**LLM 仍能看到全文**（prompt
        里带 full_text），一次调用即可拿到逐 turn 的韵律，无需事后按字符对齐。
        这正是多人对话需要的粒度：turn 与结果天然 1:1，不会跨说话人串味。
      ⚠️ chunks 必须是 text 的顺序切片（拼接后与 text 一致），否则对齐会错位。

    context：可选，给 LLM 看的**上下文全文**，可以与 text 不同。
      典型用法：text 是各 turn 台词按序拼接（用于精确对齐），context 额外带上
      「(@角色名)」说话人标记，让模型知道每句是谁说的。
      context 只进提示词、不参与切分，因此不影响 Segment.text 的对齐关系。
    """
    raw = (text or "").strip()
    cfg = config or load_config()

    def _fallback(reason: str) -> PlanResult:
        r = rule_plan(raw, max_chars=max_chars)
        r.source = "rule"
        r.error = reason
        return r

    if not raw:
        return _fallback("空文本")

    ok, why = is_ready(cfg)
    if not ok:
        if not fallback:
            raise RuntimeError(why)
        return _fallback(why)

    # 分块交给规则版 —— LLM 只做标注，不参与切分，保证与现有流程一致的边界
    base = rule_plan(raw, max_chars=max_chars)
    if chunks:
        try:
            base = _rechunk(base, [str(c) for c in chunks])
        except Exception as e:
            print("[warn] rechunk 失败，回落规则切分：%s" % e, flush=True)
    chunks = [s.text for s in base.segments]
    if not chunks:
        return _fallback("分块为空")

    do_cache = cfg.get("cache", True) if use_cache is None else use_cache
    cpath = _cache_path(raw, cfg)
    if do_cache:
        cached = _cache_read(cpath)
        if isinstance(cached, dict) and cached.get("segments"):
            r = PlanResult(
                segments=_merge(base.segments, cached),
                global_tone=_clean_str(cached.get("tone")) or base.global_tone,
                global_share=base.global_share,
            )
            r.source = "llm-cache"
            return r

    full_text = (context if context is not None else raw)[: int(cfg.get("full_text_limit", 6000))]
    batch = max(10, int(cfg.get("max_segments_per_call", 120)))
    merged_parsed = {"segments": []}
    tone = ""

    try:
        for off in range(0, len(chunks), batch):
            msgs = _build_messages(full_text, chunks, off, batch)
            content = _chat(cfg, msgs)
            parsed = _extract_json(content)
            merged_parsed["segments"].extend(parsed.get("segments") or [])
            if not tone:
                tone = _clean_str(parsed.get("tone"))
    except Exception as e:
        if not fallback:
            raise
        return _fallback("LLM 调用失败：%s" % e)

    covered = len(merged_parsed["segments"])
    if covered == 0:
        if not fallback:
            raise RuntimeError("LLM 未返回任何标注")
        return _fallback("LLM 返回的标注为空")

    if do_cache:
        _cache_write(cpath, {"tone": tone, "segments": merged_parsed["segments"]})

    r = PlanResult(
        segments=_merge(base.segments, merged_parsed),
        global_tone=tone or base.global_tone,
        global_share=base.global_share,
    )
    r.source = "llm"
    if covered < len(chunks):
        r.error = "LLM 仅覆盖 %d/%d 段，其余沿用规则版" % (covered, len(chunks))
    return r


def verify_text_intact_recheck(text: str, result: PlanResult) -> bool:
    """复检原文完整性。

    LLM 版在架构上已不可能改字（输出里根本没有文本字段），这里作为最后一道
    防线保留 —— 若本函数返回 False，说明分块环节出了问题，应立即排查。
    """
    return verify_text_intact(text, result)


def write_config_template(path: str | None = None, enabled: bool = True) -> str:
    """生成配置模板（保留原有 api_key，避免覆盖已填的 key）。"""
    p = path or CONFIG_PATH
    cur = {}
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                cur = json.load(f) or {}
        except Exception:
            cur = {}
    data = dict(DEFAULT_CONFIG)
    data.update(cur)
    data["enabled"] = enabled
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


# --------------------------------------------------------------------- 自测

if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cfg = load_config()
    ok, why = is_ready(cfg)

    print("配置文件：", CONFIG_PATH, "(%s)" % ("存在" if os.path.isfile(CONFIG_PATH) else "不存在"))
    print("Key 来源：", "已配置 (%d 字符)" % len(cfg.get("api_key") or "")
          if (cfg.get("api_key") or "").strip() else "未配置")
    print("就绪状态：", "可以调用" if ok else "不可用 —— %s" % why)
    print("模型：", cfg.get("model"), "| 端点：", cfg.get("base_url"))
    print("-" * 68)

    DEMO = (
        "他推开门。屋里没人。\n"
        "\u201c你来了。\u201d阿秾的声音很轻。\n"
        "石头没有说话。\n"
        "\u201c我等了你三年！\u201d她忽然吼道。\n"
        "窗外的雪，停了。\n"
    )

    result = plan_llm(DEMO)
    print("来源：", result.source, ("| 提示：%s" % result.error) if result.error else "")
    print(result.summary())
    print("-" * 68)
    print("原文完整性：", "通过" if verify_text_intact(DEMO, result) else "失败")

    if not ok:
        print()
        print("要启用 AI 内核：把 API Key 填进 %s 的 api_key 字段，或写入 %s"
              % (CONFIG_PATH, API_KEY_TXT))
