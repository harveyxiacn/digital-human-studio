# -*- coding: utf-8 -*-
"""下载各引擎需要的模型权重到 models/。已存在且大小正确的文件会跳过，可重复运行（断点续传）。

用法：python app/download_models.py [musetalk] [joyvasa] [latentsync]   （不带参数 = 全部）
网络：先试 HuggingFace（可用环境变量 HF_ENDPOINT 指定镜像），失败再试 ModelScope。
"""
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

# 引擎 → [(HF 仓库, 仓库内文件, 本地相对路径, ModelScope 仓库或 None)]
FILES = {
    "musetalk": [
        ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "musetalk/musetalkV15/unet.pth", "AI-ModelScope/MuseTalk"),
        ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "musetalk/musetalkV15/musetalk.json", "AI-ModelScope/MuseTalk"),
        ("stabilityai/sd-vae-ft-mse", "config.json", "musetalk/sd-vae/config.json", "stabilityai/sd-vae-ft-mse"),
        ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.safetensors", "musetalk/sd-vae/diffusion_pytorch_model.safetensors", "stabilityai/sd-vae-ft-mse"),
        ("openai/whisper-tiny", "config.json", "musetalk/whisper/config.json", "openai-mirror/whisper-tiny"),
        ("openai/whisper-tiny", "pytorch_model.bin", "musetalk/whisper/pytorch_model.bin", "openai-mirror/whisper-tiny"),
        ("openai/whisper-tiny", "preprocessor_config.json", "musetalk/whisper/preprocessor_config.json", "openai-mirror/whisper-tiny"),
        ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", "musetalk/face-parse-bisent/79999_iter.pth", None),
        ("ManyOtherFunctions/face-parse-bisent", "resnet18-5c106cde.pth", "musetalk/face-parse-bisent/resnet18-5c106cde.pth", None),
        # face_alignment 1.4.1（人脸检测 + 68 点关键点）的权重，放在 TORCH_HOME/hub/checkpoints
        ("ByteDance/LatentSync-1.5", "auxiliary/s3fd-619a316812.pth", "torch_hub/hub/checkpoints/s3fd-619a316812.pth", None),
        ("ByteDance/LatentSync-1.5", "auxiliary/2DFAN4-cd938726ad.zip", "torch_hub/hub/checkpoints/2DFAN4-cd938726ad.zip", None),
    ],
    "joyvasa": [
        ("jdh-algo/JoyVASA", "motion_generator/motion_generator_hubert_chinese.pt",
         "joyvasa/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt", None),
        ("jdh-algo/JoyVASA", "motion_template/motion_template.pkl", "joyvasa/JoyVASA/motion_template/motion_template.pkl", None),
        ("TencentGameMate/chinese-hubert-base", "config.json", "joyvasa/chinese-hubert-base/config.json", "innnky/chinese-hubert-base-tencent"),
        ("TencentGameMate/chinese-hubert-base", "preprocessor_config.json", "joyvasa/chinese-hubert-base/preprocessor_config.json", "innnky/chinese-hubert-base-tencent"),
        ("TencentGameMate/chinese-hubert-base", "pytorch_model.bin", "joyvasa/chinese-hubert-base/pytorch_model.bin", "innnky/chinese-hubert-base-tencent"),
    ] + [
        ("KlingTeam/LivePortrait", f, "joyvasa/" + f, "AI-ModelScope/LivePortrait") for f in (
            "insightface/models/buffalo_l/2d106det.onnx", "insightface/models/buffalo_l/det_10g.onnx",
            "liveportrait/base_models/appearance_feature_extractor.pth", "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/spade_generator.pth", "liveportrait/base_models/warping_module.pth",
            "liveportrait/landmark.onnx", "liveportrait/retargeting_models/stitching_retargeting_module.pth")
    ],
    "latentsync": [
        ("ByteDance/LatentSync-1.5", "latentsync_unet.pt", "latentsync/latentsync_unet.pt", "bytedance-community/LatentSync-1.5"),
        ("ByteDance/LatentSync-1.5", "whisper/tiny.pt", "latentsync/whisper/tiny.pt", "bytedance-community/LatentSync-1.5"),
        # 官方代码用 InsightFace 检测人脸，模型与 LivePortrait 的相同
        ("KlingTeam/LivePortrait", "insightface/models/buffalo_l/det_10g.onnx",
         "latentsync/auxiliary/models/buffalo_l/det_10g.onnx", "AI-ModelScope/LivePortrait"),
        ("KlingTeam/LivePortrait", "insightface/models/buffalo_l/2d106det.onnx",
         "latentsync/auxiliary/models/buffalo_l/2d106det.onnx", "AI-ModelScope/LivePortrait"),
        ("stabilityai/sd-vae-ft-mse", "config.json", "musetalk/sd-vae/config.json", "stabilityai/sd-vae-ft-mse"),
        ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.safetensors", "musetalk/sd-vae/diffusion_pytorch_model.safetensors", "stabilityai/sd-vae-ft-mse"),
    ],
}


def _urls(repo, path, ms_repo):
    hf = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    yield f"{hf}/{repo}/resolve/main/{path}"
    if hf != "https://hf-mirror.com":
        yield f"https://hf-mirror.com/{repo}/resolve/main/{path}"
    if ms_repo:
        yield f"https://www.modelscope.cn/models/{ms_repo}/resolve/master/{path}"


def _remote_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "digital-human-studio"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length") or 0)


class _Slow(IOError):
    pass


def _fetch(url, dst, log):
    """下载到 dst.part（断点续传），完成后改名；速度过慢时自动重连。"""
    for _ in range(20):
        try:
            return _fetch_once(url, dst, log)
        except _Slow:
            log("    速度过慢，重新连接…")
    raise IOError("多次重连后仍然很慢")


def _fetch_once(url, dst, log):
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "digital-human-studio"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        if have and r.status != 206:  # 服务器不支持续传：从头下载
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        t0, last = time.time(), 0
        win_t, win_b = time.time(), have
        with open(part, "ab" if have else "wb") as f:
            while True:
                buf = r.read1(1 << 16)
                if not buf:
                    break
                f.write(buf)
                have += len(buf)
                if time.time() - last > 3 and total:
                    last = time.time()
                    speed = (have / 1e6) / max(0.1, time.time() - t0)
                    log(f"    {have / 1e6:.0f}/{total / 1e6:.0f} MB（{speed:.1f} MB/s）")
                # 连接变慢（实测 HuggingFace 偶尔会掉到几十 KB/s）：断开重连，从断点继续
                if time.time() - win_t > 30:
                    if (have - win_b) / (time.time() - win_t) < 200e3:
                        raise _Slow()
                    win_t, win_b = time.time(), have
    if total and os.path.getsize(part) < total:
        raise IOError("下载不完整")
    os.replace(part, dst)


def download(engines=None, log=print):
    engines = engines or list(FILES)
    failed = []
    for eng in engines:
        log(f"== {eng}")
        for repo, path, rel, ms_repo in FILES[eng]:
            dst = os.path.join(MODELS, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                log(f"  ✓ {rel}")
                continue
            for url in _urls(repo, path, ms_repo):
                try:
                    log(f"  ↓ {rel}  ← {url.split('/')[2]}")
                    _fetch(url, dst, log)
                    break
                except Exception as e:
                    log(f"    失败：{e}")
            else:
                failed.append(rel)
    if failed:
        log("以下文件下载失败，请检查网络后重试（会从断点继续）：\n  " + "\n  ".join(failed))
        return False
    log("全部模型就绪")
    return True


if __name__ == "__main__":
    ok = download([a for a in sys.argv[1:] if a in FILES] or None)
    sys.exit(0 if ok else 1)
