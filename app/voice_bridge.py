# -*- coding: utf-8 -*-
"""与「声音工坊」联动：读取它生成的作品（整段 wav + SRT + 逐句情感），或直接调用它把文字变成语音。"""
import json
import os
import subprocess
import tempfile

import config

NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _vs():
    return config.voice_studio_dir()


def available():
    return _vs() is not None


def takes(limit=40):
    """声音工坊的作品，最新在前：[{id, label, wav, srt, total, sentences, voice_name, engine}]"""
    vs = _vs()
    if not vs:
        return []
    d = os.path.join(vs, "outputs", "_takes")
    out = []
    if not os.path.isdir(d):
        return out
    for tid in sorted(os.listdir(d), reverse=True)[:limit]:
        p = os.path.join(d, tid, "meta.json")
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        if not os.path.exists(m.get("wav", "")):
            continue
        first = m["sentences"][0]["text"][:16] if m.get("sentences") else ""
        out.append({"id": tid, "wav": m["wav"], "srt": m.get("srt") if os.path.exists(m.get("srt", "")) else None,
                    "total": m.get("total", 0), "sentences": m.get("sentences", []),
                    "voice_name": m.get("voice_name", ""), "engine": m.get("engine", ""),
                    "label": f"{m.get('created', '')[5:16]} · {m.get('voice_name', '')} · {m.get('engine', '')} · "
                             f"{m.get('total', 0):.0f}秒 · {first}…"})
    return out


def take(tid):
    return next((t for t in takes(200) if t["id"] == tid), None)


def _run(args, on_log=None, timeout=3600):
    vs = _vs()
    if not vs:
        raise RuntimeError("没有找到声音工坊，请在「设置」里填写它的目录")
    py = os.path.join(vs, "GPT-SoVITS", "runtime", "python.exe")
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    p = subprocess.Popen([py, "-s", "-u", os.path.join(config.APP_DIR, "vs_runner.py"), vs] + args,
                         cwd=os.path.join(vs, "GPT-SoVITS"), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, creationflags=NOWIN)
    logs = []
    for raw in iter(p.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").strip()
        if "@@RESULT@@" in line:
            p.wait()
            r = json.loads(line.split("@@RESULT@@", 1)[1])
            if not r.get("ok"):
                raise RuntimeError("声音工坊出错：" + r.get("error", ""))
            return r
        if line.startswith("@@LOG@@") and on_log:
            on_log(line[7:].strip())
        elif line:
            logs.append(line)
    p.wait()
    raise RuntimeError("声音工坊进程异常退出：\n" + "\n".join(logs[-10:]))


def voices():
    """[(显示名, 语音包 id)]"""
    try:
        return [(v["label"], v["id"]) for v in _run(["voices"])["voices"]]
    except Exception:
        return []


ENGINES = ["VoxCPM2", "CosyVoice3", "Qwen3-TTS", "GPT-SoVITS"]


def synth(vid, text, lang="auto", engine="VoxCPM2", emotion="平静", auto_emo=True, speed=1.0, seed=-1, on_log=None):
    """文字 → 声音工坊作品（整段 wav + SRT）。返回 take(作品 id) 的字典。"""
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=config.TMP)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"vid": vid, "text": text, "lang": lang, "engine": engine, "emotion": emotion,
                   "auto_emo": auto_emo, "speed": speed, "seed": seed}, f, ensure_ascii=False)
    try:
        r = _run(["synth", tmp], on_log)
    finally:
        os.remove(tmp)
    return take(r["take"]) or {"id": r["take"], "wav": r["wav"], "srt": r["srt"], "sentences": []}
