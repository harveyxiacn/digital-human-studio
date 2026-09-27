# -*- coding: utf-8 -*-
"""路径与设置。设置保存在 workspace/settings.json，可在网页「设置」页修改。"""
import json
import os
import subprocess

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
ENG_DIR = os.path.join(ROOT, "engines")
MODELS = os.path.join(ROOT, "models")
WORK = os.path.join(ROOT, "workspace")
AVATARS = os.path.join(WORK, "avatars")
RENDERS = os.path.join(WORK, "renders")
TMP = os.path.join(WORK, "tmp")
OUTPUTS = os.path.join(ROOT, "outputs")
ENGINE_PY = os.path.join(ENG_DIR, "avatar_env", "Scripts", "python.exe")
for d in (WORK, AVATARS, RENDERS, TMP, OUTPUTS):
    os.makedirs(d, exist_ok=True)

SETTINGS_FILE = os.path.join(WORK, "settings.json")
DEFAULTS = {
    # 声音工坊目录：默认找同级的 voice_clone
    "voice_studio": os.path.join(os.path.dirname(ROOT), "voice_clone"),
    # 剪映草稿目录（剪映 → 设置 → 草稿位置）
    "jianying_drafts": os.path.join(os.environ.get("LOCALAPPDATA", ""), "JianyingPro", "User Data", "Projects",
                                    "com.lveditor.draft"),
    "max_side": 1080,  # 形象视频最长边（像素）
    "crf": 18,         # 成片画质（越小越清晰，文件越大）
}


def settings():
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            s.update(json.load(f))
    except (OSError, ValueError):
        pass
    return s


def save_settings(**kw):
    s = settings()
    s.update({k: v for k, v in kw.items() if v is not None})
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    return s


def voice_studio_dir():
    d = settings()["voice_studio"]
    return d if d and os.path.exists(os.path.join(d, "app", "takes.py")) else None


def ffmpeg():
    """优先用 engines/bin/ffmpeg.exe（安装时放入，带 libass 字幕），否则用 imageio-ffmpeg 自带的。"""
    p = os.path.join(ENG_DIR, "bin", "ffmpeg.exe")
    if os.path.exists(p):
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def gpu_info():
    """(显卡名, 总显存 MB, 已用 MB)；没有 NVIDIA 显卡时返回 ("", 0, 0)。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()[0]
        name, total, used = [x.strip() for x in out.split(",")]
        return name, int(total), int(used)
    except Exception:
        return "", 0, 0
