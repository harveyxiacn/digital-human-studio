# -*- coding: utf-8 -*-
"""背景处理工作进程：MediaPipe 人像分割 → 绿幕 / 纯色 / 图片背景 / 透明背景（ProRes 4444 .mov，剪映可直接叠加）。

分割结果做时间平滑（指数滑动平均），减少边缘闪烁。
"""
import subprocess

import cv2
import numpy as np

from common import ffmpeg_exe, imread, progress, serve

_seg = {}


def _segmenter():
    if "s" not in _seg:
        import mediapipe as mp
        _seg["s"] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    return _seg["s"]


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], np.float32)  # BGR


def render(video, out, mode="green", color="#00FF00", image=None, smooth=0.6, feather=5):
    """mode：green（绿幕）/ color（纯色）/ image（图片背景）/ alpha（透明 .mov）"""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    bg = None
    if mode == "image" and image:
        bg = imread(image)
        bh, bw = bg.shape[:2]
        k = max(w / bw, h / bh)  # 铺满后居中裁切
        bg = cv2.resize(bg, (int(bw * k + 1), int(bh * k + 1)))
        y0, x0 = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
        bg = bg[y0:y0 + h, x0:x0 + w].astype(np.float32)
    elif mode != "alpha":
        bg = np.empty((h, w, 3), np.float32)
        bg[:] = _hex("#00FF00" if mode == "green" else color)

    alpha_out = mode == "alpha"
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "bgra" if alpha_out else "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-i", video, "-map", "0:v", "-map", "1:a?"]
    if alpha_out:
        args += ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-c:a", "pcm_s16le", out]
    else:
        args += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", out]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    seg = _segmenter()
    prev = None
    i = 0
    k = feather * 2 + 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask.astype(np.float32)
        prev = m if prev is None else prev * smooth + m * (1 - smooth)
        a = np.clip((prev - 0.3) / 0.4, 0, 1)  # 拉开对比，边缘更干净
        a = cv2.GaussianBlur(a, (k, k), 0)
        if alpha_out:
            rgba = np.dstack([frame, (a * 255).astype(np.uint8)])
            p.stdin.write(rgba.tobytes())
        else:
            a3 = a[..., None]
            comp = frame.astype(np.float32) * a3 + bg * (1 - a3)
            p.stdin.write(comp.astype(np.uint8).tobytes())
        i += 1
        if i % 10 == 0:
            progress(i / n, f"背景处理 {i}/{n}")
    cap.release()
    p.stdin.close()
    err = p.stderr.read().decode("utf-8", "replace")
    if p.wait() != 0:
        raise RuntimeError("编码失败：" + err[-600:])
    return {"out": out, "frames": i}


if __name__ == "__main__":
    serve({"render": render, "ping": lambda: {"engine": "matting"}})
