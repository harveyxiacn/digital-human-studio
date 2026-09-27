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
        # 竖屏按宽度算（一行 16 个字要放得下），横屏按高度算
        size = font_size or max(12, round(min(h / 22 if h > w else h / 16, w * 0.9 / 17)))
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


# ---------------------------------------------------------------- 工作流用

SR = 48000  # 拼接后的统一采样率


def pcm16k(path):
    """解码成 16kHz 单声道 float32（用于找停顿）。"""
    import numpy as np
    r = subprocess.run([config.ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1",
                        "-ar", "16000", "-"], capture_output=True, creationflags=NOWIN)
    if r.returncode != 0:
        raise RuntimeError("读取音频失败：" + r.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768


def concat_audio(paths, dst, gap=0.3):
    """按顺序拼接多段音频（统一成 48kHz 单声道），段间插入 gap 秒静音。返回每段的 (开始, 结束) 秒。"""
    import numpy as np
    pieces, spans, t = [], [], 0.0
    for i, p in enumerate(paths):
        r = subprocess.run([config.ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", p, "-f", "s16le", "-ac", "1",
                            "-ar", str(SR), "-"], capture_output=True, creationflags=NOWIN)
        if r.returncode != 0:
            raise RuntimeError(f"读取音频失败：{os.path.basename(p)}")
        y = np.frombuffer(r.stdout, np.int16)
        spans.append((t, t + len(y) / SR))
        pieces.append(y)
        t += len(y) / SR
        if gap and i < len(paths) - 1:
            pieces.append(np.zeros(int(SR * gap), np.int16))
            t += gap
    import wave
    with wave.open(dst, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.concatenate(pieces).tobytes())
    return spans


def cut_audio(src, dst, start, end):
    ff(["-ss", f"{start:.3f}", "-i", src, "-t", f"{end - start:.3f}", "-c:a", "pcm_s16le", dst])
    return dst


def silence_points(y16k, min_gap=0.25, thresh_db=-40.0):
    """停顿的中点（秒）：音量低于阈值且持续 ≥ min_gap 秒的区间。"""
    import numpy as np
    hop = 160  # 10ms
    n = len(y16k) // hop
    if n == 0:
        return []
    rms = np.sqrt(np.mean(y16k[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms / (np.percentile(rms, 95) + 1e-9) + 1e-10)
    quiet = db < thresh_db
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.01 >= min_gap:
                out.append((i + j) / 2 * 0.01)
            i = j
        else:
            i += 1
    return out


def fit_frames(src, dst, w, h, frames, face=None, fps=25):
    """把一段视频按「铺满后裁切」缩放到 w×h，并精确输出 frames 帧（不够时重复最后一帧），不带声音。
    face：人脸中心 (x, y)，相对源画面 0~1；裁切时让脸落在水平居中、距顶部约 35% 的位置。"""
    info = probe(src)
    sw, sh = info["width"], info["height"]
    k = max(w / sw, h / sh)
    tw, th = _even(sw * k + 1), _even(sh * k + 1)
    fx, fy = face if face else (0.5, 0.4)
    x = int(min(max(0, fx * tw - w / 2), tw - w))
    y = int(min(max(0, fy * th - h * 0.35), th - h))
    ff(["-i", src, "-an", "-vf", f"fps={fps},scale={tw}:{th}:flags=lanczos,crop={w}:{h}:{x}:{y},"
        f"tpad=stop_mode=clone:stop_duration=2", "-frames:v", str(frames), "-c:v", "libx264", "-crf", "16",
        "-preset", "fast", "-pix_fmt", "yuv420p", dst])
    return dst


def concat_videos(paths, audio, dst, crf=18):
    """拼接（同尺寸同编码的）无声视频片段，再混入整段音频。"""
    lst = dst + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in paths:
            f.write("file '" + p.replace("\\", "/").replace("'", "'\''") + "'\n")
    ff(["-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
        "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", dst])
    os.remove(lst)
    return dst
