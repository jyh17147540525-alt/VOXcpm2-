"""清唱生成插件 · 音节切分
============================
把一句歌词切成**可唱的单元**（中文=字，英文=音节），供 planner 分配给音符。

为什么必须自己做，不能用 whisper 的词级时间戳（实测结论）
------------------------------------------------------------
我在本机实测了 faster-whisper small 的 ``word_timestamps=True``：

* **真实语音**：文字准确（"您好啊"/"是啊"/"你好啊" 全对），边界可用；
* **歌唱/哼鸣**：文字退化为 ``'嗯'``（无语义），**但分段边界仍然可用**
  （4 个音 → 4 段，时间戳与音符边界吻合）。
* ⚠️ **中文词切分不稳定**：同样说"你好啊"，一次切成 ``['你','好','啊']``（3 词），
  另一次切成 ``['你好','啊']``（2 词）。

第三条是致命的：词数会随音频内容漂移，拿它去对齐音符会在不同素材上时好时坏。
所以本模块的设计是——

    **整句文本照抄 whisper 的结果（它的文字是准的），
      音节切分与分配完全自己做（可控、可测、不随素材漂移）。**

这样"歌词"这条输入是 100% 确定性的：给定文本，切分结果永远一致。

中英分治（这不是过度设计，是必须的）
--------------------------------------
* **中文**：一字一音节（汉字天然是音节单位）。多音字不处理（唱错音的代价远小于
  引入词典的复杂度，且用户可手工改 ``note_plan.json``）。
* **英文**：一个词可能有多个音节（``beau-ti-ful``）。若整词塞给一个音符，
  长词会严重超出音符时长，被迫做 4 倍以上拉伸 → 音质崩坏。
  故用**元音核启发式**切分（见 ``split_english_word``）。
* **混排**：按字符类型分段处理，中英各自走自己的规则。

⚠️ 英文音节切分是**启发式**，不追求语言学正确
------------------------------------------------
完整做法需要 CMUdict 之类的发音词典（本机没有，且会引入外部依赖）。
元音核启发式在本场景的目标不是"切得对"，而是**切得"不过长"** ——
保证每个单元都能在合理倍率内被拉伸。切错了顶多让某个音的咬字怪一点，
比整词硬拉 4 倍的崩坏好得多。用户可在 ``note_plan.json`` 里手工改。
"""
from __future__ import annotations

import re

# 英文元音字母（含 y 作元音的情形，交给启发式处理）
_VOWELS = "aeiouyAEIOUY"

# 中文判断（CJK 统一表意文字 + 扩展 A）
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# 英文单词（含撇号，如 don't）
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

#: 标点（不发声，不占音符；遇到时并入前一个单元或忽略）
_PUNCT = set("，。、！？；：,.!?;:…—～~“”\"'()（）《》〈〉[]【】 \t\n\r")


def is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def split_english_word(word: str) -> list[str]:
    """把一个英文单词切成音节（元音核启发式）。

    规则（按优先级）：
      1. 找出所有**元音组**（连续元音算一个核）；
      2. 每个核与其前面的辅音群构成一个音节（前导辅音归本音节）；
      3. 核之间的辅音群：若 >1 个，最后一个辅音留给下一个音节
         （即 V-C|CV 分界，符合 "beauti-ful" 类直觉）。
    收尾：保证每个音节至少含 1 个元音，否则并入邻接音节。

    示例：``beautiful`` → ``['beau', 'ti', 'ful']``
          ``hello`` → ``['hel', 'lo']``
          ``strange`` → ``['strange']``
    """
    w = word or ""
    if not w:
        return []
    if len(w) <= 3:
        return [w]

    # 元音核的起止
    n = len(w)
    is_v = [c in _VOWELS for c in w]

    # ⚠️ 处理"哑音尾 e"（silent final e）：strange / make / time / love 等，
    #    词尾的 e 不发音、不构成音节核。若不排除，``strange`` 会被切成
    #    ``stran|ge``；排除后只剩一个核 → 整词一个音节（正确）。
    #    判据：末位是 e/E 且它前面是辅音，且词中存在**更靠前**的元音核。
    if n >= 3 and w[-1] in "eE" and (w[-2] not in _VOWELS):
        if any(is_v[i] for i in range(n - 2)):
            is_v[n - 1] = False

    cores: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if is_v[i]:
            j = i
            while j + 1 < n and is_v[j + 1]:
                j += 1
            cores.append((i, j))
            i = j + 1
        else:
            i += 1

    if len(cores) <= 1:
        return [w]

    # 切点：取"第一个元音核结束之后"为基准，再看核间辅音群如何分配。
    #
    # ⚠️ 这里极易写错（我第一版就错了）：直觉上会想用**下一个核的起点**
    #    当切点，但那会把核间辅音全留在前一个音节里 ——
    #    ``beautiful`` 变成 ``beaut|i|ful``、``music`` 变成 ``mus|ic``。
    #    正确基准是 ``核尾 + 1``（即第一个核之后立刻切），
    #    再把核间辅音按"英语常见的最大起始原则"分配：
    #
    #      核间辅音数 == 0  → 直接在核尾切（元音相邻）
    #      核间辅音数 == 1  → 辅音**归后**（V-CV，如 mu|sic、beau|ti|ful）
    #      核间辅音数 >= 2  → 最后一个辅音归后，其余留在前面
    #                        （VC-CV，如 stran|ge 的 ng 归属、hap|py）
    cuts: list[int] = []
    for k in range(len(cores) - 1):
        c_end = cores[k][1]                     # 第一个核的末位
        n_start = cores[k + 1][0]               # 下一个核的起位
        cons = n_start - c_end - 1              # 核间辅音个数
        if cons <= 0:
            cut = c_end + 1                     # 元音相邻
        elif cons == 1:
            cut = c_end + 1                     # 单辅音归后：mu|sic
        else:
            cut = n_start - 1                   # 多辅音：最后一个归后
        if 0 < cut < n and (not cuts or cut > cuts[-1]):
            cuts.append(cut)

    out: list[str] = []
    prev = 0
    for c in cuts:
        out.append(w[prev:c])
        prev = c
    out.append(w[prev:])

    # 收尾：丢掉空串；把不含元音的碎片并入前一个
    cleaned: list[str] = []
    for s in out:
        if not s:
            continue
        if cleaned and not any(ch in _VOWELS for ch in s):
            cleaned[-1] += s
        else:
            cleaned.append(s)
    # 首个若不含元音也并给后面
    if len(cleaned) >= 2 and not any(ch in _VOWELS for ch in cleaned[0]):
        cleaned[1] = cleaned[0] + cleaned[1]
        cleaned.pop(0)
    return cleaned or [w]


def split_lyric(text: str) -> list[dict]:
    """把歌词切成可唱单元序列。

    返回 ``[{'text': 单元, 'kind': 'cjk'|'en', 'src_word': 原词 or None}, ...]``

    * 标点与空白**不产生单元**（它们是"不发声"的，不该占用音符）
    * 中文：一个汉字 = 一个单元
    * 英文：按词切，再按音节切
    * 混排：按字符类型顺序处理，保持语序
    """
    if not text:
        return []

    units: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _PUNCT:
            i += 1
            continue
        if is_cjk(ch):
            units.append({"text": ch, "kind": "cjk", "src_word": None})
            i += 1
            continue
        # 英文/拉丁词
        m = _WORD_RE.match(text, i)
        if m:
            word = m.group(0)
            for syl in split_english_word(word):
                units.append({"text": syl, "kind": "en", "src_word": word})
            i = m.end()
            continue
        # 数字/其它字母：单字符成单元（保守，不丢内容）
        if ch.isalnum():
            units.append({"text": ch, "kind": "other", "src_word": None})
        i += 1
    return units


def count_units(text: str) -> int:
    """只数单元个数（给 planner 判断是否需要拉伸/拆分）。"""
    return len(split_lyric(text))
