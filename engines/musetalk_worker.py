# -*- coding: utf-8 -*-
"""MuseTalk 1.5 工作进程：形象预处理（prepare）+ 按音频重做口型（render）。

与官方脚本的差别：
- 不依赖 mmpose/mmcv（Windows 上很难装）：人脸框和 68 点关键点改用 face_alignment（与官方的 S3FD 检测器相同），
  按官方规则由关键点算出嘴部区域，并做时间平滑，减少画面里嘴部区域抖动。
- 预处理结果（人脸框、融合遮罩、VAE 潜变量）缓存到形象目录，之后每次生成只跑 UNet + VAE 解码。
- 帧不全部放内存：按需读取，长视频也不会占满内存。
- 停顿时可以换回原画面（speech_weights 淡入淡出），嘴不会在不说话时乱动。
"""
import json
import math
import os
import sys
import threading
import queue

import cv2
import numpy as np

from common import (MODELS, ENG_DIR, FaceTracker, VideoWriter, imread, imwrite, load_wav16k, progress, serve,
                    speech_weights)

sys.path.insert(0, os.path.join(ENG_DIR, "musetalk_src"))
M = os.path.join(MODELS, "musetalk")
PREP_VERSION = 2

_models = {}


def _load(full=True):
    """预处理只需要 VAE 和人脸解析；UNet（3.4GB）和 Whisper 到生成口型时才加载。"""
    import torch
    if "vae" not in _models:
        from diffusers import AutoencoderKL
        from musetalk.utils.face_parsing import FaceParsing
        progress(0.02, "加载模型…")
        dev = torch.device("cuda")
        vae = AutoencoderKL.from_pretrained(os.path.join(M, "sd-vae")).half().to(dev).eval()
        fp_dir = os.path.join(M, "face-parse-bisent")
        FaceParsing.model_init.__defaults__ = (os.path.join(fp_dir, "resnet18-5c106cde.pth"),
                                               os.path.join(fp_dir, "79999_iter.pth"))
        FaceParsing.__call__ = _face_parse
        _models.update(dev=dev, vae=vae, FaceParsing=FaceParsing)
    if full and "unet" not in _models:
        from diffusers import UNet2DConditionModel
        from transformers import WhisperModel
        from musetalk.models.unet import PositionalEncoding
        from musetalk.utils.audio_processor import AudioProcessor
        progress(0.02, "加载 MuseTalk 模型…")
        dev = _models["dev"]
        with open(os.path.join(M, "musetalkV15", "musetalk.json"), encoding="utf-8") as f:
            unet = UNet2DConditionModel(**json.load(f))
        sd = torch.load(os.path.join(M, "musetalkV15", "unet.pth"), map_location="cpu")
        unet.load_state_dict(sd)
        del sd
        _models.update(unet=unet.half().to(dev).eval(),
                       whisper=WhisperModel.from_pretrained(os.path.join(M, "whisper")).half().to(dev).eval(),
                       ap=AudioProcessor(feature_extractor_path=os.path.join(M, "whisper")),
                       pe=PositionalEncoding(d_model=384).half().to(dev))
    return _models


def _face_parse(self, image, size=(512, 512), mode="raw"):
    """替换官方 FaceParsing.__call__：逻辑相同，只是把 19 类得分的 argmax 放在显卡上做。
    官方先把 19×512×512 的浮点得分拷回内存再 argmax，实测占这一步 80% 的时间。"""
    import torch
    from PIL import Image
    if isinstance(image, str):
        image = Image.open(image)
    with torch.no_grad():
        img = self.preprocess(image.resize(size, Image.BILINEAR)).unsqueeze(0).cuda()
        parsing = self.net(img)[0].squeeze(0).argmax(0).byte().cpu().numpy().astype(np.int64)
    if mode == "neck":
        parsing[np.isin(parsing, [1, 11, 12, 13, 14])] = 255
        parsing[parsing != 255] = 0
    elif mode == "jaw":
        face_region = (np.isin(parsing, [1]) * 255).astype(np.uint8)
        original_dilated = cv2.dilate(face_region, self.kernel, iterations=1)
        eroded = cv2.erode(original_dilated, self.cheek_kernel, iterations=2)
        face_region = cv2.bitwise_and(eroded, self.cheek_mask)
        face_region = cv2.bitwise_or(face_region, cv2.bitwise_and(original_dilated, ~self.cheek_mask))
        parsing[(face_region == 255) & (~np.isin(parsing, [10]))] = 255
        parsing[np.isin(parsing, [11, 12, 13])] = 255
        parsing[parsing != 255] = 0
    else:
        parsing[np.isin(parsing, [1, 11, 12, 13])] = 255
        parsing[parsing != 255] = 0
    return Image.fromarray(parsing.astype(np.uint8))


# ---------------------------------------------------------------- 预处理

def _mouth_box(lm, extra_margin):
    """按 MuseTalk 官方规则：上边界 = 鼻梁点(29) 往上「半张脸」的距离，下边界 = 下巴最低点 + extra_margin。"""
    half_face_y = lm[29, 1]
    half_face_dist = np.max(lm[:, 1]) - half_face_y
    y1 = max(0.0, half_face_y - half_face_dist)
    return np.array([np.min(lm[:, 0]), y1, np.max(lm[:, 0]), np.max(lm[:, 1]) + extra_margin], np.float64)


def _smooth(boxes, valid, win=5):
    """对有效帧的人脸框做居中滑动平均（缺失帧用最近的有效帧补）。"""
    n = len(boxes)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return boxes
    filled = boxes.copy()
    for i in range(n):
        if not valid[i]:
            filled[i] = boxes[idx[np.argmin(np.abs(idx - i))]]
    out = filled.copy()
    h = win // 2
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        out[i] = filled[a:b].mean(0)
    return out


def _lip_open(lm):
    """内唇张开度 / 脸高，用来判断底片里本人是不是在说话。"""
    face_h = max(1.0, np.max(lm[:, 1]) - lm[27, 1])
    return float(np.linalg.norm(lm[66] - lm[62]) / face_h)


def prepare(video, out_dir, extra_margin=10, parsing_mode="jaw", left_cheek_width=90, right_cheek_width=90):
    """video：已标准化为 25fps 的视频。输出 out_dir/frames/*.jpg、prep.npz（人脸框/遮罩框）、masks/*.png、latents.pt。"""
    import torch
    from PIL import Image
    from musetalk.utils.blending import get_image_prepare_material
    m = _load(full=False)
    fp = m["FaceParsing"](left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
    frames_dir = os.path.join(out_dir, "frames")
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    track = FaceTracker()
    boxes, valid, opens, n = [], [], [], 0
    h = w = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        imwrite(os.path.join(frames_dir, f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lm, _ = track(frame)
        if lm is None:
            boxes.append(np.zeros(4))
            valid.append(False)
        else:
            boxes.append(_mouth_box(lm, extra_margin))
            valid.append(True)
            opens.append(_lip_open(lm))
        n += 1
        if n % 10 == 0:
            progress(0.05 + 0.45 * n / total, f"检测人脸 {n}/{total}")
    cap.release()
    if n == 0:
        raise ValueError("读不到视频帧")
    valid = np.array(valid)
    if valid.mean() < 0.5:
        raise ValueError(f"只有 {valid.mean():.0%} 的画面检测到人脸，请换一段正脸清晰的视频")
    boxes = _smooth(np.array(boxes), valid)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    boxes = boxes.round().astype(np.int32)

    # 潜变量：每 4 帧一批做 VAE 编码；融合遮罩（人脸解析 + 大核模糊，主要耗 CPU）放到线程池里并行
    from concurrent.futures import ThreadPoolExecutor
    vae, dev = m["vae"], m["dev"]
    lat_list, crop_boxes = [], [None] * n
    mask_t = torch.zeros((256, 256))
    mask_t[:128] = 1
    norm = lambda x: (x - 0.5) / 0.5

    def mask_job(i, frame):
        mask, crop_box = get_image_prepare_material(frame, [int(v) for v in boxes[i]], fp=fp, mode=parsing_mode)
        imwrite(os.path.join(masks_dir, f"{i:06d}.png"), mask)
        crop_boxes[i] = crop_box

    B = 4  # 每批 4 帧（8 张图）：更大的批次提速不明显，显存却会涨到 5GB 以上
    with ThreadPoolExecutor(4) as pool:
        futs = []
        for s0 in range(0, n, B):
            ids = list(range(s0, min(n, s0 + B)))
            frames = [imread(os.path.join(frames_dir, f"{i:06d}.jpg")) for i in ids]
            ins = []
            for i, frame in zip(ids, frames):
                x1, y1, x2, y2 = boxes[i]
                crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                rgb = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0
                ins += [norm(rgb * mask_t), norm(rgb)]
                futs.append(pool.submit(mask_job, i, frame))
            with torch.no_grad():
                lat = vae.encode(torch.stack(ins).to(dev).half()).latent_dist.sample() * vae.config.scaling_factor
            lat = lat.view(len(ids), 2, *lat.shape[1:])
            lat_list.append(torch.cat([lat[:, 0], lat[:, 1]], dim=1).cpu())  # [k, 8, 32, 32]：半遮罩 + 参考
            done = sum(f.done() for f in futs)
            progress(0.5 + 0.5 * done / n, f"计算融合遮罩 {done}/{n}")
        for f in futs:
            f.result()  # 线程里的异常在这里抛出
    torch.save(torch.cat(lat_list), os.path.join(out_dir, "latents.pt"))
    np.savez(os.path.join(out_dir, "prep.npz"), boxes=boxes, crop_boxes=np.array(crop_boxes), valid=valid)
    lip = np.array(opens)
    info = {"frames": n, "width": w, "height": h, "face_ratio": round(float(valid.mean()), 3),
            "face_size": int(np.median(boxes[:, 2] - boxes[:, 0])),
            "lip_open_mean": round(float(lip.mean()), 4), "lip_open_std": round(float(lip.std()), 4),
            "talking_in_source": bool(lip.std() > 0.035), "prep_version": PREP_VERSION,
            "extra_margin": extra_margin, "parsing_mode": parsing_mode}
    with open(os.path.join(out_dir, "prep.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    progress(1, "预处理完成")
    return info


# ---------------------------------------------------------------- 生成

def _order(n_src, n_out, start=0):
    """往返循环（正放→倒放）的源帧序号，衔接处不跳帧。"""
    cycle = list(range(n_src)) + list(range(n_src - 2, 0, -1)) if n_src > 2 else list(range(n_src))
    return [cycle[(start + i) % len(cycle)] for i in range(n_out)]


def render(avatar_dir, audio, out, batch_size=8, idle_original=None, fps=25, start_frame=0, crf=18,
           audio_mux=None):
    """audio：任意格式音频（内部转 16kHz）。out：输出 mp4（混入 audio_mux 或 audio）。
    idle_original：停顿时是否换回原画面；None = 底片里本人没说话时自动开启。"""
    import torch
    from musetalk.utils.blending import get_image_blending
    m = _load()
    with open(os.path.join(avatar_dir, "prep.json"), encoding="utf-8") as f:
        info = json.load(f)
    prep = np.load(os.path.join(avatar_dir, "prep.npz"))
    boxes, crop_boxes, valid = prep["boxes"], prep["crop_boxes"], prep["valid"]
    latents = torch.load(os.path.join(avatar_dir, "latents.pt"))
    frames_dir = os.path.join(avatar_dir, "frames")
    masks_dir = os.path.join(avatar_dir, "masks")
    if idle_original is None:
        idle_original = not info.get("talking_in_source", False)

    progress(0.03, "提取音频特征…")
    wav = load_wav16k(audio)
    dev, dtype = m["dev"], torch.float16
    feats = m["ap"].feature_extractor
    segs = [wav[i:i + 30 * 16000] for i in range(0, len(wav), 30 * 16000)]
    inputs = [feats(s, return_tensors="pt", sampling_rate=16000).input_features.to(dtype) for s in segs]
    with torch.no_grad():
        chunks = m["ap"].get_whisper_chunk(inputs, dev, dtype, m["whisper"], len(wav), fps=fps,
                                           audio_padding_length_left=2, audio_padding_length_right=2)
    n_out = len(chunks)
    order = _order(info["frames"], n_out, start_frame)
    weights = speech_weights(wav, n_out, fps) if idle_original else np.ones(n_out, np.float32)

    writer = VideoWriter(out, info["width"], info["height"], fps, audio=audio_mux or audio, crf=crf)
    q = queue.Queue(maxsize=64)
    err = []

    def compose():
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                i, face = item
                src = order[i]
                frame = imread(os.path.join(frames_dir, f"{src:06d}.jpg"))
                wgt = float(weights[i])
                if face is not None and valid[src] and wgt > 0.01:
                    x1, y1, x2, y2 = [int(v) for v in boxes[src]]
                    face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
                    mask = imread(os.path.join(masks_dir, f"{src:06d}.png"), cv2.IMREAD_GRAYSCALE)
                    if wgt < 0.99:
                        mask = (mask.astype(np.float32) * wgt).astype(np.uint8)
                    frame = get_image_blending(frame, face, [x1, y1, x2, y2], mask, [int(v) for v in crop_boxes[src]])
                writer.write(np.ascontiguousarray(frame))
        except Exception as e:
            err.append(e)

    th = threading.Thread(target=compose, daemon=True)
    th.start()
    timesteps = torch.tensor([0], device=dev)
    unet, vae, pe = m["unet"], m["vae"], m["pe"]
    try:
        for b in range(0, n_out, batch_size):
            if err:
                break
            ids = list(range(b, min(n_out, b + batch_size)))
            need = [i for i in ids if weights[i] > 0.01 and valid[order[i]]]
            faces = {}
            if need:
                with torch.no_grad():
                    aud = pe(chunks[need].to(dev))
                    lat = torch.stack([latents[order[i]] for i in need]).to(dev, dtype)
                    pred = unet(lat, timesteps, encoder_hidden_states=aud).sample
                    img = vae.decode(pred / vae.config.scaling_factor).sample
                    img = ((img / 2 + 0.5).clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                for i, im in zip(need, img):
                    faces[i] = im[..., ::-1].copy()
            for i in ids:
                q.put((i, faces.get(i)))
            progress(0.05 + 0.93 * min(n_out, b + batch_size) / n_out, f"生成口型 {min(n_out, b + batch_size)}/{n_out} 帧")
    finally:
        q.put(None)
        th.join()
        writer.close()
    if err:
        raise err[0]
    progress(1, "完成")
    return {"out": out, "frames": n_out, "idle_original": idle_original,
            "speech_ratio": round(float((weights > 0.5).mean()), 3), "next_frame": (start_frame + n_out)}


if __name__ == "__main__":
    serve({"prepare": prepare, "render": render, "ping": lambda: {"engine": "musetalk"}})
