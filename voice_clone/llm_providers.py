"""LLM 服务商预设表（导演层 AI 内核的"密钥格式适配层"）。

为什么要单独一个模块
--------------------
导演层（voice_clone/director_llm.py）走的是 **OpenAI 兼容的 /chat/completions**。
市面上的服务商虽然都自称"兼容 OpenAI"，但在三件事上各不相同：

    base_url 写法          —— 有的带 /v1 有的不带，有的在路径中间插版本号
    model 命名             —— deepseek-chat / qwen-plus / moonshot-v1-8k ...
    鉴权头与额外要求       —— 绝大多数是 Authorization: Bearer，少数需要别的头

本模块把这些差异收敛成一张表，前端下拉、后端校验、文档生成都从这里取，
避免"UI 里写一套、代码里又写一套"的漂移。

设计取舍
--------
- **不按 key 前缀猜服务商**。中转站/自建网关的 key 格式高度雷同（大量都是
  sk- 开头），猜错会把用户引到错误的 base_url，比不猜更糟。改为"用户选服务商
  → 自动填 base_url + 模型候选"，另外永远保留自定义口子。
- **key 不做格式校验拦截**。只做"是否为空"和"是否含空白/换行"的检查——后者是
  粘贴时最常见的真实错误（尾随换行会导致 HTTP 头非法）。前缀不匹配只给软提示。
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------- 预设表
# 字段说明：
#   id        稳定标识（写进配置文件，不要随意改）
#   name_zh/en 界面显示名
#   base_url  默认 API 根地址（**不带**尾部 /chat/completions）
#   models    常用模型候选（第一个为默认）
#   key_hint  key 长相的软提示，仅用于提示文案
#   key_url   申请 Key 的页面
#   local     是否为本地部署（本地部署不需要真实 key，占位即可）
#   note      额外注意事项

PROVIDERS: list[dict] = [
    {
        "id": "deepseek",
        "name_zh": "DeepSeek 官方",
        "name_en": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "key_hint": "sk- 开头",
        "key_hint_en": "starts with sk-",
        "key_url": "https://platform.deepseek.com/api_keys",
        "local": False,
        "note": "国内直连稳定，性价比高。deepseek-chat 适合本项目的韵律规划任务。",
        "note_en": "Stable direct access from China and good value. deepseek-chat fits this project's prosody planning well.",
    },
    {
        "id": "openai",
        "name_zh": "OpenAI 官方",
        "name_en": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini"],
        "key_hint": "sk- 开头（较长）",
        "key_hint_en": "starts with sk- (quite long)",
        "key_url": "https://platform.openai.com/api-keys",
        "local": False,
        "note": "国内需自备网络环境。建议用 mini 系列，导演任务不需要强模型。",
        "note_en": "Needs your own network route from outside China. A mini model is enough - the director task does not need a strong model.",
    },
    {
        "id": "moonshot",
        "name_zh": "月之暗面 Kimi",
        "name_en": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2-0905-preview"],
        "key_hint": "sk- 开头",
        "key_hint_en": "starts with sk-",
        "key_url": "https://platform.moonshot.cn/console/api-keys",
        "local": False,
        "note": "中文长文本处理好；8k 版足够本项目使用。",
        "note_en": "Handles long Chinese text well; the 8k variant is plenty for this project.",
    },
    {
        "id": "dashscope",
        "name_zh": "阿里云百炼（通义千问）",
        "name_en": "Alibaba DashScope (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-7b-instruct"],
        "key_hint": "sk- 开头",
        "key_hint_en": "starts with sk-",
        "key_url": "https://bailian.console.aliyun.com/",
        "local": False,
        "note": "注意 base_url 里的 compatible-mode 路径，漏掉会 404。",
        "note_en": "Mind the compatible-mode path inside base_url - leaving it out causes a 404.",
    },
    {
        "id": "zhipu",
        "name_zh": "智谱 GLM",
        "name_en": "Zhipu GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4-air", "glm-4-plus"],
        "key_hint": "形如 xxxxxxxx.yyyyyyyy（id.secret）",
        "key_hint_en": "looks like xxxxxxxx.yyyyyyyy (id.secret)",
        "key_url": "https://open.bigmodel.cn/usercenter/apikeys",
        "local": False,
        "note": "glm-4-flash 免费额度充足，适合本项目。",
        "note_en": "glm-4-flash has a generous free tier and fits this project.",
    },
    {
        "id": "siliconflow",
        "name_zh": "硅基流动 SiliconFlow",
        "name_en": "SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "models": ["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3"],
        "key_hint": "sk- 开头",
        "key_hint_en": "starts with sk-",
        "key_url": "https://cloud.siliconflow.cn/account/ak",
        "local": False,
        "note": "聚合多家开源模型，模型名带厂商前缀（如 Qwen/）。",
        "note_en": "Aggregates many open models; model names carry a vendor prefix (for example Qwen/).",
    },
    {
        "id": "openrouter",
        "name_zh": "OpenRouter（聚合）",
        "name_en": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001",
                   "anthropic/claude-3.5-haiku"],
        "key_hint": "sk-or- 开头",
        "key_hint_en": "starts with sk-or-",
        "key_url": "https://openrouter.ai/keys",
        "local": False,
        "note": "一个 Key 打通多家模型；模型名必须带厂商前缀。",
        "note_en": "One key reaches many providers; model names must carry a vendor prefix.",
    },
    {
        "id": "ollama",
        "name_zh": "Ollama（本地）",
        "name_en": "Ollama (local)",
        "base_url": "http://127.0.0.1:11434/v1",
        "models": ["qwen3.8:27b", "qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b"],
        "key_hint": "本地无需真实 Key，填 ollama 即可",
        "key_hint_en": "no real key needed locally - type ollama",
        "key_url": "https://ollama.com/download",
        "local": True,
        "note": "完全离线，不上传任何文本。需先 ollama serve 并 pull 模型；"
                "首次调用要加载权重（27B 冷启约 30s，之后约 1.6s），"
                "所以默认超时给了 300s。建议用 7B~14B 级别模型，规划任务够用且更快。",
        "note_en": "Fully offline and nothing is uploaded. Run ollama serve and pull a model first. The first call loads weights (about 30s cold for a 27B model, about 1.6s afterwards), so the default timeout is 300s. A 7B-14B model is enough for planning and is faster.",
    },
    {
        "id": "lmstudio",
        "name_zh": "LM Studio（本地）",
        "name_en": "LM Studio (local)",
        "base_url": "http://127.0.0.1:1234/v1",
        "models": ["local-model"],
        "key_hint": "本地无需真实 Key，填 lm-studio 即可",
        "key_hint_en": "no real key needed locally - type lm-studio",
        "key_url": "https://lmstudio.ai/",
        "local": True,
        "note": "在 LM Studio 里开启 Local Server 后再填。",
        "note_en": "Turn on Local Server inside LM Studio before filling this in.",
    },
    {
        "id": "vllm",
        "name_zh": "vLLM / 自建网关（本地）",
        "name_en": "vLLM / self-hosted (local)",
        "base_url": "http://127.0.0.1:8000/v1",
        "models": ["default"],
        "key_hint": "自建服务通常无鉴权，留空或任意填",
        "key_hint_en": "self-hosted services are usually unauthenticated - leave blank or type anything",
        "key_url": "",
        "local": True,
        "note": "任何 OpenAI 兼容的自建服务都选这个，或直接用「自定义」。",
        "note_en": "Pick this for any OpenAI-compatible self-hosted service, or just use Custom.",
    },
    {
        "id": "custom",
        "name_zh": "自定义（任意 OpenAI 兼容服务）",
        "name_en": "Custom (any OpenAI-compatible)",
        "base_url": "",
        "models": [],
        "key_hint": "按服务商要求填写",
        "key_hint_en": "whatever your provider requires",
        "key_url": "",
        "local": False,
        "note": "中转站、公司内网网关等都选这个，手动填 base_url 与模型名。",
        "note_en": "Relay stations and corporate gateways go here. Fill in base_url and model by hand.",
    },
]

PROVIDER_BY_ID: dict[str, dict] = {p["id"]: p for p in PROVIDERS}


def get_provider(pid: str | None) -> dict:
    """按 id 取预设；未知 id 返回 custom 预设（不抛异常）。"""
    return PROVIDER_BY_ID.get(str(pid or "").strip(), PROVIDER_BY_ID["custom"])


def detect_provider(base_url: str | None) -> str:
    """由 base_url 反查是哪个预设，用于界面回显。查不到返回 'custom'。"""
    url = str(base_url or "").strip().rstrip("/").lower()
    if not url:
        return "custom"
    for p in PROVIDERS:
        if p["id"] == "custom":
            continue
        if p["base_url"] and p["base_url"].rstrip("/").lower() == url:
            return p["id"]
    # 放宽匹配：域名级相同也认（用户可能自行加了/去了 /v1）
    for p in PROVIDERS:
        if p["id"] == "custom" or not p["base_url"]:
            continue
        host = re.sub(r"^https?://", "", p["base_url"]).split("/")[0]
        if host and host in url:
            return p["id"]
    return "custom"


# --------------------------------------------------------------------- key 处理

_WS_RE = re.compile(r"\s+")


def normalize_key(raw: str | None) -> str:
    """清洗粘贴进来的 Key。

    这是本模块**唯一会改动用户输入**的地方，规则保守：
    - 去掉首尾空白（复制时几乎必然带上的换行/空格）
    - 去掉内部所有空白（部分平台复制出来带折行）
    - 剥掉用户误粘的 "Bearer " 前缀与包裹的引号

    **不做**前缀合法性判断——中转站和自建网关的格式无法穷举，
    拒绝一个合法 Key 比放过一个错误 Key 的代价大得多。
    """
    k = str(raw or "")
    k = k.strip().strip('"').strip("'").strip()
    k = _WS_RE.sub("", k)
    if k.lower().startswith("bearer"):
        k = k[6:].strip()
    return k


def key_looks_valid(key: str, provider_id: str | None = None) -> tuple[bool, str]:
    """软校验。只拦真正会出错的输入，其余交给「测试连接」按钮。

    返回 (是否通过, 提示语)。提示语在通过时也可能非空（表示"能用但有疑虑"）。
    """
    k = normalize_key(key)
    p = get_provider(provider_id)
    # 本地部署（Ollama / LM Studio / vLLM）默认无鉴权，空 Key 合法 —— 必须先于
    # "不能为空" 判断，否则本地用户永远配不通。
    if p.get("local"):
        return True, "本地服务：Key 只作占位，不影响调用" if k else "本地服务可不填 Key"
    if not k:
        return False, "API Key 不能为空"
    # 明显不可能是 key 的输入：含中文、过长过短、明显是网址
    if re.search(r"[\u4e00-\u9fff]", k):
        return False, "Key 里出现中文，请检查是否复制错了内容"
    if len(k) < 8:
        return False, "Key 太短（少于 8 字符），请检查是否复制完整"
    if k.startswith("http://") or k.startswith("https://"):
        return False, "这看起来是网址而不是 API Key，base_url 请填在下方对应输入框"
    if len(k) > 512:
        return False, "Key 异常地长（超过 512 字符），请确认没把整段配置粘进来"
    return True, ""


def describe(provider_id: str | None, base_url: str | None = None) -> dict:
    """给界面用的一站式描述：预设信息 + 当前 base_url 的归属判定。"""
    p = get_provider(provider_id)
    detected = detect_provider(base_url)
    return {
        "provider": p["id"],
        "name_zh": p["name_zh"],
        "name_en": p["name_en"],
        "base_url": p["base_url"],
        "models": list(p["models"]),
        "key_hint": p["key_hint"],
        "key_hint_en": p["key_hint_en"],
        "key_url": p["key_url"],
        "local": bool(p.get("local")),
        "note": p["note"],
        "note_en": p["note_en"],
        "detected_provider": detected,
        "matches_detected": detected == p["id"],
        "chat_url": (str(base_url or "").rstrip("/") + "/chat/completions")
        if base_url else "",
    }


def apply_preset(cfg: dict) -> dict:
    """把配置里的 provider 预设补全到 cfg（仅填空，不覆盖用户已填的值）。

    model 为空 / base_url 为空时才用预设兜底，避免用户手改的地址被悄悄冲掉。
    timeout 特殊：本地服务的首次调用要加载模型（实测 27B 冷启约 30s，
    之后就 1.6s），云端 API 则通常几秒内返回，所以两者默认值不同。
    """
    out = dict(cfg or {})
    pid = out.get("provider") or detect_provider(out.get("base_url"))
    p = get_provider(pid)
    out["provider"] = p["id"]
    # 先把两个键补成字符串，保证调用方可以无条件 out["base_url"] 而不 KeyError
    out["base_url"] = str(out.get("base_url") or "").strip()
    out["model"] = str(out.get("model") or "").strip()
    if not out["base_url"] and p["base_url"]:
        out["base_url"] = p["base_url"]
    if not out["model"] and p["models"]:
        out["model"] = p["models"][0]
    return out


def recommended_timeout(provider_id: str | None) -> int:
    """推荐超时（秒）。

    本地大模型首次调用要把它从磁盘加载进显存 —— 实测 qwen3.8:27b 冷启约 30s，
    之后约 1.6s。若沿用云 API 的短超时，用户第一次配本地模型必然被判"失败"，
    是很差的初体验。所以本地服务给足 300s。
    """
    p = get_provider(provider_id)
    return 300 if p.get("local") else 60


if __name__ == "__main__":  # 自检
    print(f"预设数量：{len(PROVIDERS)}")
    for p in PROVIDERS:
        print(f"  - {p['id']:12s} {p['name_zh']:22s} {p['base_url']}")
    print("\n--- detect_provider 自检 ---")
    cases = [
        "https://api.deepseek.com/v1", "https://api.deepseek.com/v1/",
        "https://api.moonshot.cn/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://open.bigmodel.cn/api/paas/v4", "http://127.0.0.1:11434/v1",
        "https://my-proxy.example.com/v1", "",
    ]
    for c in cases:
        print(f"  {c or '(空)':58s} -> {detect_provider(c)}")
    print("\n--- normalize_key 自检 ---")
    for raw in ["  sk-abc123  ", "sk-abc\n123", '"sk-xyz"', "Bearer sk-777", "  ", "sk-中文"]:
        print(f"  {raw!r:22s} -> {normalize_key(raw)!r}")
    print("\n--- key_looks_valid 自检 ---")
    for k, pid in [("sk-abc123456789", "deepseek"), ("", "deepseek"),
                   ("short", "deepseek"), ("sk-中文key123", "deepseek"),
                   ("https://api.x.com", "deepseek"), ("ollama", "ollama")]:
        print(f"  {k!r:22s} [{pid:10s}] -> {key_looks_valid(k, pid)}")
