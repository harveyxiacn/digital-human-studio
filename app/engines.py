# -*- coding: utf-8 -*-
"""引擎工作进程管理：每个引擎一个子进程（主进程不占显存），同一时间只保留一个；
空闲 IDLE_SEC 秒后自动退出，把显存还给声音工坊等其他软件。"""
import json
import os
import subprocess
import threading
import time

import config
import gpu

IDLE_SEC = 90
WORKERS = {
    "musetalk": "musetalk_worker.py",
    "latentsync": "latentsync_worker.py",
    "joyvasa": "joyvasa_worker.py",
    "matting": "matting_worker.py",
}
GPU_TASK = {"musetalk": "MuseTalk", "latentsync": "LatentSync", "joyvasa": "JoyVASA", "matting": "背景处理"}
MODEL_FILES = {
    "musetalk": ["musetalk/musetalkV15/unet.pth", "musetalk/sd-vae/diffusion_pytorch_model.safetensors",
                 "musetalk/whisper/pytorch_model.bin", "musetalk/face-parse-bisent/79999_iter.pth",
                 "torch_hub/hub/checkpoints/s3fd-619a316812.pth", "torch_hub/hub/checkpoints/2DFAN4-cd938726ad.zip"],
    "latentsync": ["latentsync/latentsync_unet.pt", "latentsync/whisper/tiny.pt",
                   "latentsync/auxiliary/models/buffalo_l/det_10g.onnx"],
    "joyvasa": ["joyvasa/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt",
                "joyvasa/chinese-hubert-base/pytorch_model.bin", "joyvasa/liveportrait/base_models/spade_generator.pth"],
    "matting": [],  # MediaPipe 自带模型
}


def available(name):
    return os.path.exists(config.ENGINE_PY) and all(os.path.exists(os.path.join(config.MODELS, f))
                                                    for f in MODEL_FILES[name])


class Worker:
    def __init__(self, name):
        self.name = name
        self.p = None
        self.lock = threading.Lock()
        self.logs = []
        self.last_used = 0

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def start(self):
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false", "TORCH_HOME": os.path.join(config.MODELS, "torch_hub"),
                    "PYTHONUNBUFFERED": "1"})
        env["PATH"] = os.path.join(config.ENG_DIR, "bin") + os.pathsep + env.get("PATH", "")
        self.p = subprocess.Popen([config.ENGINE_PY, "-u", os.path.join(config.ENG_DIR, WORKERS[self.name])],
                                  cwd=config.ENG_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def call(self, cmd, on_progress=None, timeout=6 * 3600):
        with self.lock:
            if not self.alive():
                self.start()
            self.p.stdin.write((json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
            self.p.stdin.flush()
            t0 = time.time()
            buf = b""
            while True:
                ch = self.p.stdout.read1(65536) if hasattr(self.p.stdout, "read1") else self.p.stdout.readline()
                if not ch:
                    break
                buf += ch
                # 按 \n 和 \r 拆行（tqdm 进度条用 \r 刷新，会把结果标记粘在后面）
                *lines, buf = buf.replace(b"\r", b"\n").split(b"\n")
                for raw in lines:
                    line = raw.decode("utf-8", "replace").strip()
                    if "@@RESULT@@" in line:
                        self.last_used = time.time()
                        return json.loads(line.split("@@RESULT@@", 1)[1])
                    if "@@PROGRESS@@" in line:
                        if on_progress:
                            try:
                                d = json.loads(line.split("@@PROGRESS@@", 1)[1])
                                on_progress(d["p"], d["msg"])
                            except ValueError:
                                pass
                    elif line:
                        self.logs = (self.logs + [line])[-300:]
                if time.time() - t0 > timeout:
                    break
            tail = "\n".join(self.logs[-12:])
            self.stop()
            raise RuntimeError(f"{self.name} 进程异常退出：\n{tail}")

    def stop(self):
        if self.p is None:
            return
        try:
            if self.alive():
                self.p.stdin.write(b'{"cmd":"quit"}\n')
                self.p.stdin.flush()
                self.p.wait(timeout=10)
        except Exception:
            pass
        if self.alive():
            subprocess.call(f"taskkill /t /f /pid {self.p.pid}", shell=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.p = None


_workers = {n: Worker(n) for n in WORKERS}


def release_all(keep=None):
    for n, w in _workers.items():
        if n != keep:
            w.stop()


def run(name, cmd, on_progress=None, gpu_task=None, on_wait=None):
    """在显卡锁内执行一条引擎命令；失败抛出 RuntimeError。"""
    if not available(name):
        raise RuntimeError(f"{GPU_TASK[name]} 引擎或模型未安装，请先运行 安装.bat")
    task = gpu_task or GPU_TASK[name]
    with gpu.use(task, on_wait=on_wait):
        release_all(keep=name)
        w = _workers[name]
        gpu.check_free(task, already_loaded=w.alive())
        r = w.call(cmd, on_progress)
    if not r.get("ok"):
        # 第一行给用户看，后面附上工作进程日志尾部便于排查
        raise RuntimeError((r.get("error") or "引擎出错") + "\n" + logs(name, 15))
    return r


def logs(name, n=40):
    return "\n".join(_workers[name].logs[-n:])


def _reaper():
    """空闲超过 IDLE_SEC 秒，或其他进程请求释放显存时，关掉空闲的引擎进程。"""
    while True:
        time.sleep(2)
        for w in _workers.values():
            if not w.alive() or w.lock.locked():
                continue
            if time.time() - w.last_used > IDLE_SEC or gpu.release_requested_since(w.last_used):
                w.stop()


threading.Thread(target=_reaper, daemon=True).start()
