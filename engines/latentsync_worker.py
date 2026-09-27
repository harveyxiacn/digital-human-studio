# -*- coding: utf-8 -*-
"""LatentSync 1.5 工作进程：视频 + 音频 → 高质量口型（扩散模型，比 MuseTalk 慢，口型更准）。

权重在 models/latentsync（运行时重定向官方写死的 checkpoints/ 路径）；VAE 与 MuseTalk 共用 models/musetalk/sd-vae。工作目录必须是 latentsync_src（官方代码用相对路径）。
"""
import os
import shutil
import sys
import tempfile

from common import ENG_DIR, MODELS, progress, run_ffmpeg, serve

SRC = os.path.join(ENG_DIR, "latentsync_src")
sys.path.insert(0, SRC)
os.chdir(SRC)

_pipe = {}


def _load(deepcache):
    if _pipe:
        return _pipe["p"]
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    # 官方人脸检测写死 root="checkpoints/auxiliary"：改到 models/latentsync/auxiliary
    import latentsync.utils.face_detector as fd
    _FA = fd.FaceAnalysis

    def FaceAnalysis(*a, **kw):
        kw["root"] = os.path.join(MODELS, "latentsync", "auxiliary")
        return _FA(*a, **kw)

    fd.FaceAnalysis = FaceAnalysis
    progress(0.02, "加载 LatentSync 模型…")
    cfg = OmegaConf.load("configs/unet/stage2.yaml")
    dtype = torch.float16
    scheduler = DDIMScheduler.from_pretrained("configs")
    audio_encoder = Audio2Feature(model_path=os.path.join(MODELS, "latentsync", "whisper", "tiny.pt"), device="cuda",
                                  num_frames=cfg.data.num_frames, audio_feat_length=cfg.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained(os.path.join(MODELS, "musetalk", "sd-vae"), torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model),
                                                   os.path.join(MODELS, "latentsync", "latentsync_unet.pt"), device="cpu")
    unet = unet.to(dtype=dtype)
    p = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to("cuda")
    if deepcache:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=p)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    _pipe.update(p=p, cfg=cfg)
    return p


def render(video, audio, out, steps=20, guidance=1.5, seed=1247, deepcache=True, audio_mux=None, crf=18):
    import torch
    p = _load(deepcache)
    cfg = _pipe["cfg"]
    torch.manual_seed(int(seed))
    tmp = tempfile.mkdtemp(prefix="ls_")
    wav = os.path.join(tmp, "a16k.wav")
    run_ffmpeg(["-i", audio, "-ac", "1", "-ar", "16000", wav])

    total = [0]

    def cb(step, t, latents):  # 每个 16 帧片段内的去噪步数回调
        total[0] += 1
        progress(min(0.97, 0.1 + total[0] * 0.002), f"去噪 第 {total[0]} 步")

    raw = os.path.join(tmp, "raw.mp4")
    progress(0.05, "检测人脸并对齐…")
    p(video_path=video, audio_path=wav, video_out_path=raw, num_frames=cfg.data.num_frames,
      num_inference_steps=int(steps), guidance_scale=float(guidance), weight_dtype=torch.float16,
      width=cfg.data.resolution, height=cfg.data.resolution, mask_image_path=cfg.data.mask_image_path,
      temp_dir=os.path.join(tmp, "t"), callback=cb, callback_steps=1)
    # 换成原始音频（官方输出的是 16kHz 单声道）
    run_ffmpeg(["-i", raw, "-i", audio_mux or audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
    shutil.rmtree(tmp, ignore_errors=True)
    progress(1, "完成")
    return {"out": out}


if __name__ == "__main__":
    serve({"render": render, "ping": lambda: {"engine": "latentsync"}})
