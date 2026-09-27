# -*- coding: utf-8 -*-
"""ffmpeg 工具：探测、标准化、音频转换、画幅、字幕烧录、背景合成。"""
import os
import re
import subprocess

import config

NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ff(args, check=True):
    r = subprocess.run([config.ffmpeg(), "-hide_banner", "-y"] + args, capture_output=True, creationflags=NOWIN)
    if check and r.returncode != 0:
        raise RuntimeError("ffmpeg 失败：" + r.stderr.decode("utf-8", "replace")[-1000:])
    return r


def probe(path):
    """{"duration", "width", "height", "fps", "has_video", "has_audio", "rotation"}，不依赖 ffprobe。"""
    r = subprocess.run([config.ffmpeg(), "-hide_banner", "-i", path], capture_output=True, creationflags=NOWIN)
    s = r.stderr.decode("utf-8", "replace")
    info = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_video": False, "has_audio": False, "rotation": 0}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", s)
    if m:
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    for line in s.splitlines():
        if "Stream #" in line and "Video:" in line and not info["has_video"]:
            if "attached pic" in line:
                continue
            info["has_video"] = True
            m = re.search(r", (\d{2,5})x(\d{2,5})", line)
            if m:
                info["width"], info["height"] = int(m[1]), int(m[2])
            m = re.search(r"([\d.]+) fps", line) or re.search(r"([\d.]+) tbr", line)
            if m:
                info["fps"] = float(m[1])
        elif "Stream #" in line and "Audio:" in line:
            info["has_audio"] = True
    m = re.search(r"rotate\s*:\s*(-?\d+)", s) or re.search(r"rotation of (-?[\d.]+) degrees", s)
    if m:
        info["rotation"] = int(float(m[1]))
    return info


def _even(v):
    return int(v) // 2 * 2


def scale_filter(w, h, max_side):
    """等比缩到最长边不超过 max_side，宽高为偶数。"""
    k = min(1.0, max_side / max(w, h)) if max(w, h) else 1.0
    return f"scale={_even(w * k)}:{_even(h * k)}:flags=lanczos"


def normalize_video(src, dst, max_side=1080, start=None, end=None):
    """形象底片：25fps、H.264、最长边 ≤ max_side、去掉音频。ffmpeg 会自动按手机视频的旋转信息转正。"""
    info = probe(src)
    if not info["has_video"]:
        raise ValueError("这个文件里没有视频画面")
    w, h = info["width"], info["height"]
    if abs(info["rotation"]) in (90, 270):
        w, h = h, w
    args = []
    if start:
        args += ["-ss", str(start)]
    args += ["-i", src]
    if end:
        args += ["-t", str(float(end) - float(start or 0))]
    args += ["-an", "-vf", f"fps=25,{scale_filter(w, h, max_side)}", "-c:v", "libx264", "-crf", "15",
             "-preset", "medium", "-pix_fmt", "yuv420p", dst]
    ff(args)
    return probe(dst)


def to_wav(src, dst, sr=None, mono=False):
    args = ["-i", src, "-vn"]
    if sr:
        args += ["-ar", str(sr)]
    if mono:
        args += ["-ac", "1"]
    ff(args + ["-c:a", "pcm_s16le", dst])
    return dst


def image_frame(src, dst, max_side=1920):
    """照片：转正（EXIF 方向）、限制尺寸、宽高偶数，存成 PNG。"""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    k = min(1.0, max_side / max(im.size))
    im = im.resize((_even(im.width * k), _even(im.height * k)), Image.LANCZOS)
    im.save(dst)
    return im.size


def thumbnail(video_or_image, dst, t=0.5):
    if os.path.splitext(video_or_image)[1].lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        ff(["-i", video_or_image, "-vf", "scale=320:-2", dst])
    else:
        ff(["-ss", str(t), "-i", video_or_image, "-frames:v", "1", "-vf", "scale=320:-2", dst])
    return dst


# ---------------------------------------------------------------- 后期

ASPECTS = {"原始": None, "竖屏 9:16": (9, 16), "横屏 16:9": (16, 9), "方形 1:1": (1, 1)}


def _ass_path(p):
    """subtitles 滤镜里的 Windows 路径需要转义冒号。"""
    return p.replace("\\", "/").replace(":", "\\:")


def post(src, dst, aspect="原始", srt=None, font_size=0, bg=None, crf=18, face_cx=None):
    """画幅裁切 + 字幕烧录。aspect 见 ASPECTS；face_cx：人脸中心 x（0~1），裁切时尽量让人在画面中间。"""
    info = probe(src)
    w, h = info["width"], info["height"]
    vf = []
    ratio = ASPECTS.get(aspect)
    if ratio:
        rw, rh = ratio
        if w / h > rw / rh:  # 太宽：裁左右
            cw, ch = _even(h * rw / rh), h
        else:
            cw, ch = w, _even(w * rh / rw)
        cx = (face_cx if face_cx is not None else 0.5) * w
        x = int(min(max(0, cx - cw / 2), w - cw))
        y = int((h - ch) / 3) if ch < h else 0  # 竖裁时偏上，保留头部
        vf.append(f"crop={cw}:{ch}:{x}:{y}")
        w, h = cw, ch
    if srt:
        size = font_size or max(12, round(h / 22 if h > w else h / 16))
        # libass 的字号以 PlayResY=288 为基准，这里换算成像素
        fs = round(size * 288 / h)
        style = (f"FontName=Microsoft YaHei,FontSize={fs},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                 f"BorderStyle=1,Outline=1.6,Shadow=0,Bold=1,MarginV={round(28 if h > w else 18)}")
        vf.append(f"subtitles='{_ass_path(srt)}':force_style='{style}'")
    if not vf:
        if src != dst:
            import shutil
            shutil.copyfile(src, dst)
        return dst
    ff(["-i", src, "-vf", ",".join(vf), "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", dst])
    return dst


def srt_shift(src, dst, offset):
    """把 SRT 整体平移 offset 秒（负数提前）。"""
    def fmt(t):
        t = max(0.0, t)
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    def parse(s):
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    with open(src, encoding="utf-8-sig") as f:
        text = f.read()
    text = re.sub(r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)",
                  lambda m: f"{fmt(parse(m[1]) + offset)} --> {fmt(parse(m[2]) + offset)}", text)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    return dst
