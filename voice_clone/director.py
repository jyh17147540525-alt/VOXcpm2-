"""
模块 4：文本梳理 / 导演层 (Director)
=====================================
只梳理，不改写 —— 输出「这段该怎么念」的结构化标注，原文逐字保留。

解决的问题（现有 detect_emotion 的局限）
----------------------------------------
  1. 不辨台词与旁白：引号内的台词和引号外的叙述用同一套规则判情绪。
  2. 无上下文：逐块独立判断，前一句的情绪不影响后一句。
  3. 无情绪惯性：同一场景连续的几句话本应共享情绪，现在是割裂的「点」——
     「你来了。」单独看毫无线索（中性 0.0），但它承接的是三年的等待。
  4. 强度公式粗糙：0.5 + 0.25×命中次数，多数块落在触发阈值 0.6 以下，
     导致「检测到了情绪但最终什么也没施加」。
  5. 单字词误命中：「悲」「怒」「喜」在词表里权重与长词相同，
     「慈悲」会被判成 sad、「息怒」会被判成 angry。本模块对短词降权。
  6. 停顿时长固定：句末一律 0.30s、逗号一律 0.15s，不随情绪变化。

设计原则
--------
  * **只标注，不改写**：Segment.text 是原文切片，与输入逐字相同，可做 diff 校验。
  * **可解释**：每段带 reason，说明判定依据，便于人工核对与调参。
  * **零依赖**：纯规则，不加载模型、不占显存、不联网。
  * **可降级**：接入方拿到的三个属性（chunks_text / chunks_with_pause / emotions）
    与现有 synthesize_stable 的入参完全同构，不接也不影响现有流程。

用法
----
    from voice_clone.director import plan
    r = plan("他推开门。\\n\"你来了。\"阿秾的声音很轻。")
    print(r.summary())
    # 接入时：
    #   chunks_text        -> synthesize_stable 的 chunks_text
    #   chunks_with_pause  -> synthesize_stable 的 chunks_with_pause
    #   emotions           -> 替代 [detect_emotion(ct) for ct in chunks_text]
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 复用现有分块与词表，避免两处维护导致行为分叉。
# 兼容两种加载方式：包内导入（正常）与文件路径直接加载（测试/importlib）。
try:
    from .synthesis_stab import (
        _EMOTION_WORDS as _RAW_EMOTION_WORDS,
        split_with_pauses,
    )
except ImportError:  # pragma: no cover - 仅在无包上下文的直接加载时触发
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from synthesis_stab import (  # type: ignore
        _EMOTION_WORDS as _RAW_EMOTION_WORDS,
        split_with_pauses,
    )

#: 剔除单字情感词后再使用。
#:
#: 原始词表里的「悲」「怒」「喜」三个单字极易误命中 ——「慈悲」被判成悲伤、
#: 「息怒」被判成愤怒、「喜鹊」被判成喜悦。这类错误比漏判严重得多：
#: 漏判只是回落到中性（安全），误判却会给出方向相反的情绪。
#: 只在本模块过滤，不动 synthesis_stab 的原始词表，既有行为不受影响。
_EMOTION_WORDS = {
    label: [w for w in words if len(w) >= 2]
    for label, words in _RAW_EMOTION_WORDS.items()
}


# --------------------------------------------------------------------- 判定词表

#: 全部引号字符（含左右与直引号）。
_QUOTE_CHARS = "\"'「」『』“”‘’"

#: 左右引号分列 —— 用于判断引号开合方向。
#: 分块会从标点处切开，导致引号被孤立：`“你来了。` / `”阿秾的声音很轻。`，
#: 后者本是旁白，若只看「块内有无引号」会被误判成台词。
_LEFT_QUOTES = "「『“‘"
_RIGHT_QUOTES = "」』”’"

#: 引导动词：出现在块尾说明「下一块大概率是台词」。
_SPEECH_LEAD = (
    "说", "道", "问", "答", "喊", "叫", "吼", "骂", "念", "读",
    "低语", "呢喃", "嘟囔", "咆哮", "笑道", "叹道", "冷笑", "回答",
)

#: 加强语：出现则情绪强度上调。含「吼道/喊道」这类后置修饰，
#: 它们常落在下一块（`“我等了你三年！”她忽然吼道。`），故检测时用邻域文本。
_INTENSIFIERS = (
    "狠狠", "死死", "拼命", "大声", "用力", "疯狂", "歇斯底里",
    "嘶哑", "尖叫", "厉声", "猛然", "骤然", "吼道", "喊道", "叫道",
)

#: 减弱语：出现则情绪强度下调（压抑、克制、低语）。
_DIMINISHERS = (
    "轻声", "小声", "低低", "淡淡", "轻轻", "微微", "缓慢",
    "几乎", "若有若无", "喃喃", "幽幽", "极轻",
)

#: 各停顿类型的基准秒数 —— 与 synthesis_stab._join_pieces 的默认值对齐
#: (pause=0.15 时：end 0.30 / comma 0.15 / line 0.2025 / hard 0.05)。
_PAUSE_BASE = {
    "end": 0.30,
    "comma": 0.15,
    "line": 0.20,
    "hard": 0.05,
}

#: 情绪对停顿时长的倍率。>1 停顿拉长（沉重/迟疑），<1 停顿缩短（急促/轻快）。
_PAUSE_FACTOR = {
    "neutral": 1.00,
    "sad": 1.35,        # 沉重，呼吸间隔更长
    "fear": 1.15,       # 迟疑
    "surprise": 1.10,   # 需要反应时间
    "question": 1.10,   # 留出被回答的空隙
    "happy": 0.88,      # 轻快
    "exclamation": 0.85,
    "angry": 0.82,      # 急促
}

#: 情绪标签中英对照，仅用于 summary 展示。
_LABEL_ZH = {
    "neutral": "中性", "happy": "喜悦", "sad": "悲伤", "angry": "愤怒",
    "fear": "恐惧", "surprise": "惊讶", "question": "疑问",
    "exclamation": "激昂",
}

_ROLE_ZH = {"speech": "台词", "narration": "旁白"}


# --------------------------------------------------------------------- 数据结构

@dataclass
class Segment:
    """一段待合成文本及其「怎么念」标注。text 与输入原文逐字相同。"""

    text: str
    pause_type: str      # end / comma / line / hard（沿用 synthesis_stab 的语义）
    emotion: str         # 规则版为固定标签；LLM 版为自由描述（不受枚举限制）
    intensity: float     # 0.0 ~ 1.0
    pause_after: float   # 建议停顿秒数
    role: str            # speech 台词 / narration 旁白
    reason: str          # 判定依据（可解释性）

    # —— 以下为 LLM 导演层扩展：直接给韵律数值，绕过「标签→参数」的映射损失 ——
    # 规则版一律留 None，由 intensity 走原有的 _apply_emotion 路径，行为不变。
    pace: float | None = None      # 语速倍率（1.0 = 原速），建议区间 0.8 ~ 1.2
    pitch_st: float | None = None  # 音调偏移（半音），建议区间 -2.0 ~ 2.0
    cfg: float | None = None       # 表现力 cfg_value，建议区间 1.8 ~ 3.2
    note: str = ""                 # 给演播者的提示，仅供展示，不参与合成


@dataclass
class PlanResult:
    """梳理结果。三个 property 与 synthesize_stable 的入参同构，可直接喂进去。"""

    segments: list = field(default_factory=list)
    global_tone: str = "neutral"
    global_share: float = 0.0
    source: str = "rule"        # rule / llm / llm-cache —— 本次梳理用的哪条通路
    error: str = ""             # 降级或部分失败的原因（仅用于展示，不影响结果可用性）

    @property
    def chunks_text(self) -> list:
        return [s.text for s in self.segments]

    @property
    def chunks_with_pause(self) -> list:
        return [(s.text, s.pause_type) for s in self.segments]

    @property
    def emotions(self) -> list:
        return [(s.emotion, s.intensity) for s in self.segments]

    @property
    def pause_plan(self) -> list:
        """建议的停顿秒数序列（按段）。供接入方覆盖固定停顿使用。"""
        return [s.pause_after for s in self.segments]

    def summary(self) -> str:
        """人类可读的梳理结果，用于人工核对与调参。"""
        if not self.segments:
            return "(空文本)"
        head = "全局基调：%s (%.2f) · 共 %d 段 · 来源：%s" % (
            _LABEL_ZH.get(self.global_tone, self.global_tone),
            self.global_share,
            len(self.segments),
            self.source,
        )
        lines = [head]
        if self.error:
            lines.append("提示：%s" % self.error)
        lines.append("-" * 68)
        for i, s in enumerate(self.segments, 1):
            lines.append(
                "[%02d] %s %s %.2f 停顿%.2fs  %s" % (
                    i,
                    _ROLE_ZH.get(s.role, s.role),
                    _LABEL_ZH.get(s.emotion, s.emotion),
                    s.intensity,
                    s.pause_after,
                    s.text,
                )
            )
            # 韵律数值只有 LLM 版才填，规则版留空即不显示这几行
            nums = []
            if s.pace is not None:
                nums.append("语速×%.2f" % s.pace)
            if s.pitch_st is not None:
                nums.append("音调%+.1f半音" % s.pitch_st)
            if s.cfg is not None:
                nums.append("表现力%.1f" % s.cfg)
            if nums:
                lines.append("      韵律：%s" % " | ".join(nums))
            if s.note:
                lines.append("      提示：%s" % s.note)
            lines.append("      依据：%s" % s.reason)
        return "\n".join(lines)


# --------------------------------------------------------------------- 内部判定

def _word_weight(word: str) -> float:
    """情感词权重：长词更具体，权重高；单字词极易误命中，降权。

    现有实现里「悲」「怒」「喜」与「歇斯底里」等权，导致
    「慈悲」「息怒」被误判 —— 这里按词长分级削弱单字影响。
    """
    n = len(word)
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.8
    return 0.4


def _scan_speech(chunk: str, in_speech: bool):
    """顺序扫描块内引号，返回 (台词字符占比, 扫描后的引号状态)。

    为什么必须逐字符扫、而不能静态计数：直引号 `"` 开闭同形、没有方向，
    只看「块内有没有引号」或「引号在不在块首」都会误判 ——
    `"你来了。` 是台词开头，`"阿秾的声音很轻。` 则是引号在此闭合后的叙述，
    两者块首字符完全相同。只有沿途翻转引号状态才能区分。

    顺带得到扫描前后的状态差异，用于识别「引号跨块」（分块把整句话切开了）。
    """
    n = len(chunk)
    if n == 0:
        return 0.0, in_speech
    speech_chars = 0
    for ch in chunk:
        if ch in _LEFT_QUOTES:
            in_speech = True
        elif ch in _RIGHT_QUOTES:
            in_speech = False
        elif ch == '"':
            in_speech = not in_speech
        if in_speech:
            speech_chars += 1
    return speech_chars / n, in_speech


def _detect_role(speech_ratio: float, in_speech: bool, pending_speech: bool) -> str:
    """判定本段是台词还是旁白。以下任一成立即为台词：

    * **扫描结束时仍处于引号内** —— 台词尚未闭合（`他说：“好。` 被从句号
      切开，占比恰好 0.5 卡在阈值上，只看占比会漏判）；
    * **台词字符占比 > 0.5** —— 本段主体在引号内；
    * **上一块有引导动词且本段无引号** —— `他说：好。` 这类不用引号的写法。
    """
    if in_speech or speech_ratio > 0.5:
        return "speech"
    if pending_speech:
        return "speech"
    return "narration"


def _leads_speech(chunk: str) -> bool:
    """块尾出现引导动词（「……他说」）→ 下一块**可能**是台词。

    若本块已含引号则不作预测：中文里引导动词可前置（`他说：“好。”`）
    也可后置（`“好。”他说。`），仅凭块尾无法区分方向，贸然预测会把
    后续旁白误判成台词（`”她忽然吼道。` 之后本是叙述）。
    """
    if any(q in chunk for q in _QUOTE_CHARS):
        return False
    tail = chunk[-8:]
    return any(v in tail for v in _SPEECH_LEAD)


def _global_tone(text: str):
    """全文情绪基调：统计各情绪词加权命中，取占比最高者。

    返回 (label, share)。share 为占比，用于判断基调是否明确。
    """
    counts = {}
    for label, words in _EMOTION_WORDS.items():
        s = 0.0
        for w in words:
            c = text.count(w)
            if c:
                s += c * _word_weight(w)
        if s > 0:
            counts[label] = s
    if not counts:
        return "neutral", 0.0
    label = max(counts, key=counts.get)
    total = sum(counts.values())
    share = counts[label] / total if total else 0.0
    return label, round(min(1.0, share), 3)


def _score(chunk: str, role: str, tone: str, tone_share: float,
           context: str = ""):
    """对单段打分，返回 (emotion, intensity, reason)。

    优先级：情感词 > 多重感叹 > 单感叹 > 疑问 > 省略号 > 中性。
    随后叠加加强语 / 减弱语 / 旁白收敛 / 全局基调契合度调制。

    context 为「前一块 + 本块 + 后一块」的邻域文本，**仅用于检测加强语**：
    像 `“我等了你三年！”她忽然吼道。` 这种，修饰语落在了下一块，
    只看本块会漏掉「吼道」这个明显的强度信号。情绪主体仍由本块决定。
    """
    ctx = context or chunk
    # 1) 情感词加权命中
    hits = {}
    for label, words in _EMOTION_WORDS.items():
        s = 0.0
        for w in words:
            c = chunk.count(w)
            if c:
                s += c * _word_weight(w)
        if s > 0:
            hits[label] = round(s, 2)

    if hits:
        label = max(hits, key=hits.get)
        score = hits[label]
        intensity = min(1.0, 0.35 + 0.20 * score)
        reason = "情感词命中 %s(权重%.1f)" % (_LABEL_ZH.get(label, label), score)
    else:
        n_excl = chunk.count("！") + chunk.count("!")
        n_ques = chunk.count("？") + chunk.count("?")
        n_ell = chunk.count("…") + chunk.count("...")
        if n_excl >= 2:
            label = "exclamation"
            intensity = min(1.0, 0.45 + 0.15 * n_excl)
            reason = "多重感叹 ×%d" % n_excl
        elif n_excl == 1:
            label, intensity, reason = "exclamation", 0.55, "单感叹"
        elif n_ques:
            label, intensity, reason = "question", 0.50, "疑问句"
        elif n_ell:
            label, intensity, reason = "sad", 0.45, "省略号（余韵）"
        else:
            return "neutral", 0.0, "无线索（保持中性）"

    # 2) 加强语 / 减弱语（在邻域文本上检测，修饰语可能落在相邻块）
    for w in _INTENSIFIERS:
        if w in ctx:
            intensity = min(1.0, intensity + 0.12)
            reason += " · 加强语「%s」" % w
            break
    for w in _DIMINISHERS:
        if w in ctx:
            intensity = max(0.05, intensity - 0.10)
            reason += " · 减弱语「%s」" % w
            break

    # 3) 旁白比台词克制
    if role == "narration":
        intensity *= 0.85
        reason += " · 旁白收敛"

    # 4) 与全文基调一致 → 略微增强（让整段情绪更连贯）
    if label == tone and tone_share > 0:
        intensity = min(1.0, intensity * 1.15)
        reason += " · 契合全局基调"

    return label, intensity, reason


def _pause_for(ptype: str, emotion: str, intensity: float, scale: float) -> float:
    """按停顿类型 + 情绪 + 强度给出建议停顿时长（秒）。

    情绪越强，偏离中性基准的幅度越大；中性段保持基准值不变。
    """
    base = _PAUSE_BASE.get(ptype, 0.15) * scale
    factor = _PAUSE_FACTOR.get(emotion, 1.0)
    w = max(0.35, min(1.0, intensity))       # 强度下限 0.35，避免情绪弱时完全不动
    factor = 1.0 + (factor - 1.0) * w
    return round(base * factor, 3)


# --------------------------------------------------------------------- 主入口

def plan(text: str, max_chars: int = 60, *,
         pause_scale: float = 1.0,
         context_decay: float = 0.55,
         context_floor: float = 0.22,
         emotion: str = "") -> PlanResult:
    """梳理文本，输出分段标注。**原文逐字不改**。

    参数
    ----
    text          : 待合成文本（可含换行，诗歌/多段皆可）。
    max_chars     : 无标点超长串的硬切长度，与 synthesis_stab 默认一致。
    pause_scale   : 停顿时长整体缩放（1.0 = 与现有 _join_pieces 基准一致）。
    context_decay : 情绪惯性衰减系数 —— 前一段情绪传递到后一段的保留比例。
    context_floor : 惯性生效下限，低于此强度不承接（避免弱情绪无限蔓延）。
    emotion       : 若显式指定情绪（如 UI 选了「悲伤」），则全篇统一，跳过自动判断。

    返回
    ----
    PlanResult，其 chunks_text / chunks_with_pause / emotions 可直接喂给
    synthesize_stable，pause_plan 可用于覆盖固定停顿。
    """
    raw = (text or "").strip()
    if not raw:
        return PlanResult(segments=[], global_tone="neutral", global_share=0.0)

    chunks = split_with_pauses(raw, max_chars=max_chars)
    if not chunks:
        return PlanResult(segments=[], global_tone="neutral", global_share=0.0)

    # 显式情绪：全篇统一为指定值，不做逐段自动判断（与 _apply_emotion_uniform 对应）。
    explicit = (emotion or "").strip()

    tone, tone_share = _global_tone(raw)

    segments = []
    prev_emotion, prev_intensity = "neutral", 0.0
    pending_speech = False
    in_speech = False          # 引号状态机：当前是否处于台词（引号）内部
    n_chunks = len(chunks)

    for i, (chunk, ptype) in enumerate(chunks):
        # 引号状态机：扫描后状态发生变化 → 引号被分块切断（跨块）
        prev_state = in_speech
        speech_ratio, in_speech = _scan_speech(chunk, in_speech)
        role = _detect_role(speech_ratio, in_speech, pending_speech)

        # 邻域文本（前块 + 本块 + 后块）：仅用于加强语检测
        context = (
            (chunks[i - 1][0] if i > 0 else "")
            + chunk
            + (chunks[i + 1][0] if i + 1 < n_chunks else "")
        )

        # 引号跨块 → 块尾句号并非真正句末，后面紧跟同句剩余部分，降级为短停顿
        dangling = ptype == "end" and in_speech != prev_state
        if dangling:
            ptype = "comma"

        if explicit:
            label, intensity = explicit, 1.0
            reason = "显式指定情绪（全篇统一）"
        else:
            label, intensity, reason = _score(
                chunk, role, tone, tone_share, context)

            # 情绪惯性：本段无线索，但上一段有明确情绪 → 承接（解决「割裂的点」）。
            if label == "neutral" and prev_emotion != "neutral":
                carried = prev_intensity * context_decay
                if carried >= context_floor:
                    label = prev_emotion
                    intensity = round(carried, 3)
                    reason = "承接上句情绪（%s 衰减后）" % _LABEL_ZH.get(
                        prev_emotion, prev_emotion)

        if dangling:
            reason += " · 引号跨块（降为短停顿）"

        pause_after = _pause_for(ptype, label, intensity, pause_scale)

        segments.append(Segment(
            text=chunk,               # 原文切片，逐字保留
            pause_type=ptype,
            emotion=label,
            intensity=round(float(intensity), 3),
            pause_after=pause_after,
            role=role,
            reason=reason,
        ))

        pending_speech = _leads_speech(chunk)
        prev_emotion, prev_intensity = label, float(intensity)

    return PlanResult(segments=segments, global_tone=tone, global_share=tone_share)


def verify_text_intact(text: str, result: PlanResult) -> bool:
    """自检：确认梳理没有改动原文（拼接所有段应还原为原文的精简形式）。

    用于接入前的安全校验 —— 第 1 档只标注，本函数必须永远返回 True。
    """
    join = "".join(s.text for s in result.segments)
    norm = lambda s: "".join(s.split())  # noqa: E731  仅忽略空白差异
    return norm(join) == norm(text)


# --------------------------------------------------------------------- 自测

if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    DEMO = (
        "他推开门。屋里没人。\n"
        "\u201c你来了。\u201d阿秾的声音很轻。\n"
        "石头没有说话。\n"
        "\u201c我等了你三年！\u201d她忽然吼道。\n"
        "窗外的雪，停了。\n"
    )

    result = plan(DEMO)
    print(result.summary())
    print("-" * 68)
    print("原文完整性校验：", "通过" if verify_text_intact(DEMO, result) else "失败")
    print("喂给 synthesize_stable 的 emotions：")
    print(" ", result.emotions)
    print("喂给 synthesize_stable 的 pauses：")
    print(" ", result.pause_plan)
