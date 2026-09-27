# -*- coding: utf-8 -*-
"""引擎工作进程的公共部分：JSON 行协议、进度输出、ffmpeg、视频读写、人脸关键点。

协议：主进程每次写一行 JSON 命令到 stdin；工作进程输出
  @@PROGRESS@@ {"p": 0~1, "msg": "..."}   进度
  @@RESULT@@ {...}                         结果（每条命令恰好一条）
其他输出都当作日志。
"""
import json
import os
import subprocess
import sys
import time
import traceback

ENG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ENG_DIR)
MODELS = os.path.join(ROOT, "models")
os.environ.setdefault("TORCH_HOME", os.path.join(MODELS, "torch_hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def progress(p, msg=""):
    print("@@PROGRESS@@ " + json.dumps({"p": round(float(p), 4), "msg": msg}, ensure_ascii=False), flush=True)


def result(obj):
    print("\n@@RESULT@@ " + json.dumps(obj, ensure_ascii=False), flush=True)


def serve(handlers):
    """命令循环：{"cmd": 名称, ...其他参数} → handlers[名称](**参数) 的返回值（dict）。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        cmd = req.pop("cmd", "")
        if cmd == "quit":
            break
        t0 = time.time()
        try:
            if cmd not in handlers:
                raise ValueError(f"未知命令：{cmd}")
            out = handlers[cmd](**req) or {}
            out.setdefault("ok", True)
            out["sec"] = round(time.time() - t0, 1)
        except Exception as e:
            traceback.print_exc()
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            import torch
            if torch.cuda.is_available():
                out["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / 2 ** 20)
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        result(out)


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 失败：" + r.stderr.decode("utf-8", "replace")[-800:])


class VideoWriter:
    """把 BGR 帧通过管道写给 ffmpeg 编码成 H.264；可同时混入音频。"""

    def __init__(self, path, w, h, fps=25, audio=None, crf=18):
        args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if audio:
            args += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", path]
        self.args = args
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.size = (w, h)
        self.done = False

    def write(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"帧尺寸 {w}x{h} 与视频尺寸 {self.size[0]}x{self.size[1]} 不一致")
        if self.done:
            return
        try:
            self.p.stdin.write(frame.tobytes())
        except OSError:
            # 有音频时用了 -shortest：音频结束后 ffmpeg 会正常退出（返回码 0），多出的几帧直接丢弃
            if self.p.wait() == 0:
                self.done = True
                return
            err = self.p.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"视频编码进程退出（返回码 {self.p.wait()}）：{err[-800:]}\n参数：{self.args}")

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        err = self.p.stderr.read().decode("utf-8", "replace")
        if self.p.wait() != 0:
            raise RuntimeError("视频编码失败：" + err[-800:])


def load_wav16k(path):
    import librosa
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y


def speech_weights(wav16k, n_frames, fps=25, thresh_db=-38.0, min_pause=0.3, fade=3):
    """逐帧「是否在说话」权重 0~1：音量低于阈值且持续超过 min_pause 秒算停顿；边缘淡入淡出 fade 帧。"""
    import numpy as np
    hop = 16000 // fps
    rms = np.array([np.sqrt(np.mean(wav16k[i * hop:(i + 1) * hop] ** 2) + 1e-10) for i in range(n_frames)])
    peak = np.percentile(rms, 95) if len(rms) else 1.0
    db = 20 * np.log10(rms / max(peak, 1e-6) + 1e-10)
    speak = db > thresh_db
    w = speak.astype(np.float32)
    # 短停顿（换气、字间）不算停顿，嘴保持由模型生成
    i, n = 0, len(w)
    min_len = int(min_pause * fps)
    while i < n:
        if not speak[i]:
            j = i
            while j < n and not speak[j]:
                j += 1
            if j - i < min_len:
                w[i:j] = 1
            i = j
        else:
            i += 1
    if fade > 0 and n:
        k = np.ones(2 * fade + 1) / (2 * fade + 1)
        w = np.clip(np.convolve(np.pad(w, fade, mode="edge"), k, mode="valid"), 0, 1)
    return w


def imread(path, flags=None):
    """支持中文路径的 cv2.imread（Windows 上 cv2.imread 读不了非 ASCII 路径）。"""
    import cv2
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR if flags is None else flags)


def imwrite(path, img, params=None):
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img, params or [])
    if not ok:
        raise IOError(f"图片编码失败：{path}")
    buf.tofile(path)


class FaceTracker:
    """逐帧 68 点关键点，带跟踪：每 redetect 帧做一次完整人脸检测（最慢的一步），
    中间帧沿用上次的检测框、按关键点重心的位移平移过去。实测比逐帧检测快约 3 倍，
    脸移动太快（重心位移超过脸宽 15%）时立即重新检测。"""

    def __init__(self, redetect=8):
        self.redetect = redetect
        self.n = 0
        self.box = None       # 上次完整检测得到的人脸框（原图坐标）
        self.box_c = None     # 检测时关键点的重心
        self.prev_c = None

    def __call__(self, img_bgr):
        import numpy as np
        need = self.box is None or self.n % self.redetect == 0
        self.n += 1
        if not need:
            shift = self.prev_c - self.box_c
            box = self.box + np.array([shift[0], shift[1], shift[0], shift[1]])
            lm = _landmarks_in_box(img_bgr, box)
            if lm is not None:
                c = lm.mean(0)
                if np.linalg.norm(c - self.prev_c) < 0.15 * (box[2] - box[0]):
                    self.prev_c = c
                    return lm, box
        lm, box = face_landmarks(img_bgr)
        if lm is None:
            self.box = None
            return None, None
        self.box, self.box_c = box, lm.mean(0)
        self.prev_c = self.box_c
        return lm, box


def _landmarks_in_box(img_bgr, box):
    import numpy as np
    _init_fa()
    lms = _fa.get_landmarks_from_image(img_bgr[:, :, ::-1].copy(), detected_faces=[list(box)])
    return np.asarray(lms[0][:, :2]) if lms else None


_fa = None


def _init_fa():
    global _fa
    if _fa is None:
        import face_alignment
        lt = getattr(face_alignment.LandmarksType, "TWO_D", None) or face_alignment.LandmarksType._2D
        _fa = face_alignment.FaceAlignment(lt, flip_input=False, device="cuda", face_detector="sfd")


def face_landmarks(img_bgr):
    """68 点人脸关键点（iBUG 顺序，原图坐标）和人脸框；检测不到返回 (None, None)。多人时取最大的脸。"""
    import cv2
    import numpy as np
    _init_fa()
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 960 / max(h, w))  # 检测在缩小图上做，速度快很多
    small = img_bgr if scale == 1.0 else cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    dets = _fa.face_detector.detect_from_image(small[:, :, ::-1].copy())
    dets = [d for d in dets if d[4] > 0.8]
    if not dets:
        return None, None
    d = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
    box = np.asarray(d[:4]) / scale
    lm = _landmarks_in_box(img_bgr, box)
    return (lm, box) if lm is not None else (None, None)
