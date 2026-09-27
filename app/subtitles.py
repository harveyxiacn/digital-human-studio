# -*- coding: utf-8 -*-
"""字幕：读取 / 平移 / 合并 SRT；没有字幕时调用声音工坊的 SenseVoice 识别（普通话 / 粤语 / 英语，逐字时间戳）。

逐字时间戳同时用来找「句子边界」，工作流按句子切换镜头，不会在半句话中间切。
"""
import json
import os
import re
import subprocess
import time

import config
import gpu

PUNCT = "，。！？、；：,.!?;:…"
SENT_END = "。！？!?.…；;"
LINE_MAX = 16  # 每行字幕最多字数（与声音工坊一致，剪映竖屏 / 横屏都合适）
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------- SRT

def _ts(t):
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _parse_ts(s):
    h, m, rest = s.strip().replace(".", ",").split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def read_srt(path):
    """[(开始, 结束, 文字)]"""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        for i, l in enumerate(lines):
            m = re.match(r"(\S+)\s*-->\s*(\S+)", l)
            if m:
                out.append((_parse_ts(m[1]), _parse_ts(m[2]), " ".join(lines[i + 1:]).strip()))
                break
    return out


def write_srt(items, path):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(f"{i + 1}\n{_ts(a)} --> {_ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(items) if t))
    return path


# ---------------------------------------------------------------- 识别

def _vs_asr():
    vs = config.voice_studio_dir()
    if not vs:
        return None
    py = os.path.join(vs, "engines", "voxcpm_env", "Scripts", "python.exe")
    script = os.path.join(vs, "engines", "sensevoice_asr.py")
    model = os.path.join(vs, "engines", "models", "SenseVoiceSmall")
    return (py, script, vs) if all(os.path.exists(p) for p in (py, script, model)) else None


def asr_available():
    return _vs_asr() is not None


def asr_chars(wav, lang="auto", on_wait=None):
    """[[字, 开始秒, 结束秒], ...]（含标点）；声音工坊不可用或识别失败时返回 None。"""
    a = _vs_asr()
    if not a:
        return None
    py, script, vs = a
    out = os.path.join(config.TMP, f"asr_{time.strftime('%H%M%S')}_{os.getpid()}.json")
    with gpu.use("语音识别", on_wait=on_wait):
        r = subprocess.run([py, script, wav, out, "--lang", lang, "--timestamps"], cwd=os.path.join(vs, "engines"),
                           capture_output=True, creationflags=NOWIN,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
    if r.returncode != 0 or not os.path.exists(out):
        return None
    with open(out, encoding="utf-8") as f:
        chars = json.load(f).get("chars") or None
    os.remove(out)
    return chars


def lines_from_chars(chars):
    """逐字时间戳 → 字幕行：先按标点分句；超过 LINE_MAX 字的分句平均切成几行（不会切出「欢 / 迎你」这种一两个字的尾巴）；
    太短的相邻分句合并成一行。去掉行尾标点，行内逗号换成空格（剪映习惯）。"""
    import math
    clauses, cur = [], []
    for c in chars:
        if c[0] in PUNCT:
            if cur:
                cur[-1] = [cur[-1][0] + c[0], cur[-1][1], cur[-1][2]]  # 标点挂在前一个字上
            if c[0] in SENT_END + "，,":
                if cur:
                    clauses.append(cur)
                cur = []
            continue
        cur.append(list(c))
    if cur:
        clauses.append(cur)

    def text(ws):
        t = "".join(w[0] for w in ws).strip().rstrip("，,、；;：:。…")
        return re.sub(r"[，,、；;：:]\s*", " ", t).strip()

    lines = []
    for cl in clauses:
        k = math.ceil(len(cl) / LINE_MAX)
        n = math.ceil(len(cl) / k)
        for i in range(0, len(cl), n):
            lines.append(cl[i:i + n])
    merged = []
    for ln in lines:  # 很短的分句（例如「系啊」）和下一句合并，间隔太大则不合并
        if merged and len(merged[-1]) + len(ln) <= LINE_MAX and len(merged[-1]) <= 4 and ln[0][1] - merged[-1][-1][2] < 0.6:
            merged[-1] = merged[-1] + ln
        else:
            merged.append(ln)
    return [(ln[0][1], ln[-1][2], text(ln)) for ln in merged if text(ln)]


def sentence_cuts(chars):
    """句子边界的时间点（前一句最后一个字结束和下一句第一个字开始的中点）。"""
    cuts, words = [], [c for c in chars if c[0] not in PUNCT]
    idx = {id(c): i for i, c in enumerate(words)}
    last = None
    for c in chars:
        if c[0] in PUNCT:
            if c[0] in SENT_END + "，," and last is not None:
                i = idx[id(last)]
                if i + 1 < len(words):
                    cuts.append((last[2] + words[i + 1][1]) / 2)
        else:
            last = c
    return sorted(set(round(t, 3) for t in cuts))
