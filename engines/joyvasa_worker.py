# -*- coding: utf-8 -*-
"""JoyVASA 工作进程：一张照片 + 音频 → 会说话、会点头眨眼的视频（LivePortrait 渲染，贴回原图）。

与官方脚本的差别：
- 使用 InferenceConfig 的默认值（裁脸 + 贴回原图 + 拼接），官方命令行默认会关掉这些，只适合正方形大头照。
- 帧直接写入编码器，不在内存里攒整段视频。
- 可调：表情幅度（driving_multiplier）、头部动作幅度（按角度 / 位移缩放）、cfg_scale（动作丰富度）。
权重在 models/joyvasa（运行时把官方写死的 pretrained_weights 路径重定向过去）。
"""
import os
import sys

import numpy as np

from common import ENG_DIR, MODELS, VideoWriter, progress, serve

SRC = os.path.join(ENG_DIR, "joyvasa_src")
sys.path.insert(0, SRC)

_pipe = {}


def _load(max_dim):
    if _pipe.get("max_dim") == max_dim:
        return _pipe["p"]
    import platform
    import pathlib
    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath  # 官方权重里有 PosixPath 对象
    # 官方代码把权重路径写死为 joyvasa_src/pretrained_weights：在导入其他模块之前改到 models/joyvasa
    # （不用目录联接，因为 exFAT 等分区不支持）
    import src.config.base_config as bc
    orig = bc.make_abs_path

    def make_abs_path(fn):
        fn = fn.replace("\\", "/")
        if "pretrained_weights/" in fn:
            return os.path.join(MODELS, "joyvasa", fn.split("pretrained_weights/", 1)[1])
        return orig(fn)

    bc.make_abs_path = make_abs_path
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_wmg_pipeline import LivePortraitPipeline
    progress(0.02, "加载 JoyVASA / LivePortrait 模型…")
    inf = InferenceConfig()
    inf.source_max_dim = max_dim
    inf.flag_use_half_precision = True
    p = LivePortraitPipeline(inference_cfg=inf, crop_cfg=CropConfig())
    _pipe.update(max_dim=max_dim, p=p)
    return p


def faces(image, max_dim=1280):
    """检测照片里的所有人脸（按面积从大到小），返回原图坐标的人脸框和置信度，供用户选择。"""
    import cv2
    p = _load(int(max_dim))  # 先加载（替换权重路径），再导入其他上游模块
    from src.utils.io import load_image_rgb
    img = load_image_rgb(image)
    fs = p.cropper.face_analysis_wrapper.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), flag_do_landmark_2d_106=False,
                                             direction="large-small")
    # 过滤掉低置信度和太小的框（人群里的远处小脸）
    fs = [f for f in fs if f.det_score >= 0.5 and f.bbox[2] - f.bbox[0] >= 40]
    return {"faces": [{"box": [round(float(v), 1) for v in f.bbox[:4]], "score": round(float(f.det_score), 3)}
                      for f in fs]}


def _pick_face(p, face_box):
    """让 LivePortrait 裁脸时优先选用户指定的人脸（官方默认选最大的脸，合影里可能选中别人或雕像）。"""
    w = p.cropper.face_analysis_wrapper
    if not hasattr(w, "_orig_get"):
        w._orig_get = w.get
    if not face_box:
        w.get = w._orig_get
        return
    cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2

    def get(img_bgr, **kw):
        fs = w._orig_get(img_bgr, **kw)
        return sorted(fs, key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2)

    w.get = get


class _Args:
    def __init__(self, audio, cfg_scale):
        self.audio = audio
        self.cfg_mode = "incremental"
        self.cfg_cond = None
        self.cfg_scale = cfg_scale
        self.is_smooth_motion = True


def render(image, audio, out, cfg_scale=2.8, expression=1.0, head_motion=1.0, max_dim=1280, crf=18, audio_mux=None,
           face_box=None, keep_expression=False):
    """face_box：要驱动的人脸（原图坐标，来自 faces 命令）；None = 最大的脸。
    keep_expression：保留照片原有表情（例如笑脸）。官方做法是先把嘴合上、再用生成的嘴型绝对值驱动，
    对笑着露齿的照片会变成「嘟嘴」；开启后嘴型改为相对原表情变化。"""
    import torch
    p = _load(int(max_dim))  # 先加载（替换权重路径），再导入其他上游模块
    from src.utils.camera import get_rotation_matrix
    from src.utils.crop import prepare_paste_back, paste_back
    from src.utils.helper import dct2device
    from src.utils.io import load_image_rgb, resize_to_limit
    lpw, inf = p.live_portrait_wrapper, p.live_portrait_wrapper.inference_cfg
    dev = lpw.device

    src = load_image_rgb(image)
    img = resize_to_limit(src, inf.source_max_dim, inf.source_division)
    if face_box:  # 人脸框换算到缩放后的坐标
        k = img.shape[1] / src.shape[1]
        face_box = [v * k for v in face_box]
    _pick_face(p, face_box)
    crop = p.cropper.crop_source_image(img, p.cropper.crop_cfg)
    if crop is None:
        raise ValueError("照片里检测不到人脸，请换一张正脸清晰的照片")
    I_s = lpw.prepare_source(crop["img_crop_256x256"])
    x_s_info = lpw.get_kp_info(I_s)
    x_c_s, R_s = x_s_info["kp"], get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    f_s = lpw.extract_feature_3d(I_s)
    x_s = lpw.transform_keypoint(x_s_info)
    lip_delta0 = None
    if inf.flag_normalize_lip and not keep_expression and crop.get("lmk_crop") is not None:
        r = lpw.calc_combined_lip_ratio([0.0], crop["lmk_crop"])
        if r[0][0] >= inf.lip_normalize_threshold:
            lip_delta0 = lpw.retarget_lip(x_s, r)
    mask = prepare_paste_back(inf.mask_crop, crop["M_c2o"], dsize=(img.shape[1], img.shape[0]))

    progress(0.08, "由声音生成表情和头部动作…")
    tpl = lpw.gen_motion_sequence(_Args(audio, float(cfg_scale)))
    n = tpl["n_frames"]
    motions = tpl["motion"]
    m0 = motions[0]
    # 头部动作幅度：相对第一帧的角度和位移按比例缩放
    k = float(head_motion)
    for mi in motions:
        for key in ("pitch", "yaw", "roll"):
            mi[key] = m0[key] + (mi[key] - m0[key]) * k
        mi["t"] = m0["t"] + (mi["t"] - m0["t"]) * k
        R = get_rotation_matrix(torch.from_numpy(mi["pitch"]), torch.from_numpy(mi["yaw"]), torch.from_numpy(mi["roll"]))
        mi["R"] = R.reshape(1, 3, 3).cpu().numpy().astype(np.float32)

    h, w = img.shape[:2]
    writer = VideoWriter(out, w - w % 2, h - h % 2, 25, audio=audio_mux or audio, crf=crf)
    try:
        R_d0 = d0 = x_d0_new = None
        for i in range(n):
            d = dct2device(motions[i], dev)
            R_d = d["R"]
            if i == 0:
                R_d0, d0 = R_d, d.copy()
            R_new = (R_d @ R_d0.permute(0, 2, 1)) @ R_s
            delta = x_s_info["exp"] + (d["exp"] - d0["exp"])
            if not keep_expression:
                for li in (6, 12, 14, 17, 19, 20):  # 嘴唇相关的关键点用绝对值
                    delta[:, li, :] = d["exp"][:, li, :]
            scale = x_s_info["scale"] * (d["scale"] / d0["scale"])
            t = x_s_info["t"] + (d["t"] - d0["t"])
            t[..., 2].fill_(0)
            x_d = scale * (x_c_s @ R_new + delta) + t
            x_d = lpw.stitching(x_s, x_d)
            if lip_delta0 is not None:
                x_d = x_d + lip_delta0
            x_d = x_s + (x_d - x_s) * float(expression)
            frame = lpw.parse_output(lpw.warp_decode(f_s, x_s, x_d)["out"])[0]
            frame = paste_back(frame, crop["M_c2o"], img, mask)
            writer.write(np.ascontiguousarray(frame[: h - h % 2, : w - w % 2, ::-1]))
            if i % 10 == 0:
                progress(0.15 + 0.84 * i / n, f"渲染 {i}/{n} 帧")
    finally:
        writer.close()
    progress(1, "完成")
    return {"out": out, "frames": n, "width": w - w % 2, "height": h - h % 2}


if __name__ == "__main__":
    serve({"render": render, "faces": faces, "ping": lambda: {"engine": "joyvasa"}})
