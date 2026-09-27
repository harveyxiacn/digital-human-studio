# -*- coding: utf-8 -*-
"""显卡守卫：同一时间只允许一个显卡任务，开始前检查显存。

与「声音工坊」使用同一个文件锁和排队目录（找到声音工坊时用它的 workspace），格式完全一致，
所以两个软件的任务会自动排队，不会同时占显存。
"""
import json
import os
import threading
import time
from contextlib import contextmanager

import config


def _lock_dir():
    vs = config.voice_studio_dir()
    return os.path.join(vs, "workspace") if vs else config.WORK


LOCK_FILE = os.path.join(_lock_dir(), "_gpu.lock")
QUEUE_DIR = os.path.join(_lock_dir(), "_gpu_queue")
# 显存不够时写这个文件：本软件其他进程里空闲的引擎看到后会立即退出、释放显存
RELEASE_REQ = os.path.join(_lock_dir(), "_gpu_release.req")
os.makedirs(QUEUE_DIR, exist_ok=True)
_local = threading.Lock()

HEAVY_APPS = {
    "jianyingpro.exe": "剪映", "capcut.exe": "CapCut", "adobe premiere pro.exe": "Premiere",
    "afterfx.exe": "After Effects", "resolve.exe": "达芬奇", "obs64.exe": "OBS",
    "photoshop.exe": "Photoshop", "blender.exe": "Blender",
}
# 各任务大约需要的空闲显存（MB）：RTX 3070 Laptop 8GB 实测峰值 + 余量，见设计说明
NEED_MB = {"形象预处理": 3000, "MuseTalk": 4000, "LatentSync": 7000, "JoyVASA": 4000, "背景处理": 1500, "人脸检测": 1500}


class GpuBusy(RuntimeError):
    pass


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True


def holder():
    try:
        with open(LOCK_FILE, encoding="utf-8") as f:
            h = json.load(f)
    except Exception:
        return None
    if not _pid_alive(h.get("pid", 0)):
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass
        return None
    return h


def heavy_apps():
    try:
        import psutil
        names = {p.info["name"].lower() for p in psutil.process_iter(["name"]) if p.info.get("name")}
    except Exception:
        return []
    return sorted({v for k, v in HEAVY_APPS.items() if k in names})


def request_release():
    with open(RELEASE_REQ, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "since": time.time()}, f)


def release_requested_since(t):
    """在时间 t 之后，是否有其他进程请求释放显存。"""
    try:
        with open(RELEASE_REQ, encoding="utf-8") as f:
            r = json.load(f)
        return r.get("pid") != os.getpid() and r.get("since", 0) > t
    except (OSError, ValueError):
        return False


def check_free(task, already_loaded=False, wait=20):
    need = 0 if already_loaded else NEED_MB.get(task, 0)
    _, total, used = config.gpu_info()
    if not need or not total:
        return
    free = total - used
    if free >= need:
        return
    # 可能是本软件另一个进程（网页 / 命令行）里空闲的引擎占着：请求它释放，等一会儿再看
    request_release()
    t0 = time.time()
    while time.time() - t0 < wait:
        time.sleep(1)
        _, total, used = config.gpu_info()
        free = total - used
        if free >= need:
            return
    apps = heavy_apps()
    msg = f"显存不足：{task} 约需 {need / 1024:.1f}GB，当前只剩 {free / 1024:.1f}GB（已用 {used / 1024:.1f}/{total / 1024:.0f}GB）。"
    msg += f"检测到正在运行：{'、'.join(apps)}，请先关闭。" if apps else "请关闭占用显卡的软件后重试。"
    raise GpuBusy(msg)


def _keep_awake(on):
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def queue():
    out = []
    for fn in sorted(os.listdir(QUEUE_DIR)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(QUEUE_DIR, fn)
        try:
            with open(p, encoding="utf-8") as f:
                q = json.load(f)
        except Exception:
            continue
        if not _pid_alive(q.get("pid", 0)):
            try:
                os.remove(p)
            except OSError:
                pass
            continue
        q["id"] = fn
        out.append(q)
    return out


def cancel(qid=None):
    for q in queue():
        if qid in (None, q["id"]) and (qid or q["pid"] == os.getpid()):
            open(os.path.join(QUEUE_DIR, q["id"] + ".cancel"), "w").close()


@contextmanager
def use(task, wait=True, on_wait=None):
    """独占显卡；忙时排队（与声音工坊共用队列，先来先到）。任务期间阻止系统睡眠。"""
    def busy():
        h = holder()
        return h and h.get("pid") != os.getpid()

    if not wait and busy():
        raise GpuBusy(f"显卡正忙：{(holder() or {}).get('task')}，请等它完成后再试。")
    qid = f"{time.time():.3f}_{os.getpid()}_{threading.get_ident()}.json"
    qpath = os.path.join(QUEUE_DIR, qid)
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "task": "数字人·" + task, "since": time.time()}, f, ensure_ascii=False)
    try:
        while True:
            if os.path.exists(qpath + ".cancel"):
                raise GpuBusy("已取消排队")
            q = queue()
            pos = next((i for i, x in enumerate(q) if x["id"] == qid), 0) + 1
            if pos == 1 and not busy() and _local.acquire(blocking=False):
                break
            if on_wait:
                on_wait(pos, (holder() or {}).get("task") or (q[0]["task"] if q else ""))
            time.sleep(2)
    finally:
        for p in (qpath, qpath + ".cancel"):
            try:
                os.remove(p)
            except OSError:
                pass
    try:
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "task": "数字人·" + task, "since": time.time()}, f, ensure_ascii=False)
        _keep_awake(True)
        yield
    finally:
        _keep_awake(False)
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass
        _local.release()


def status():
    name, total, used = config.gpu_info()
    h = holder()
    return {"name": name, "used": used, "total": total, "task": h.get("task") if h else None,
            "apps": heavy_apps(), "queue": [q["task"] for q in queue()]}
