# -*- coding: utf-8 -*-
"""形象库：workspace/avatars/<id>/

视频形象：source.mp4（25fps 标准化底片）+ MuseTalk 预处理缓存（frames/ masks/ latents.pt prep.npz prep.json）
照片形象：photo.png
都有 meta.json 和 cover.jpg。
"""
import json
import os
import random
import re
import shutil
import time

import config
import engines
import media

KINDS = {"video": "视频形象", "photo": "照片形象"}


def _dir(aid):
    return os.path.join(config.AVATARS, aid)


def load(aid):
    with open(os.path.join(_dir(aid), "meta.json"), encoding="utf-8") as f:
        m = json.load(f)
    m["dir"] = _dir(aid)
    return m


def _save(meta):
    m = {k: v for k, v in meta.items() if k != "dir"}
    with open(os.path.join(_dir(meta["id"]), "meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)


def all_avatars():
    out = []
    for aid in sorted(os.listdir(config.AVATARS), reverse=True):
        if os.path.exists(os.path.join(_dir(aid), "meta.json")):
            try:
                out.append(load(aid))
            except (OSError, ValueError):
                pass
    return out


def choices():
    """[(显示名, id)]"""
    return [(f"{a['name']} · {KINDS[a['kind']]}{'' if a.get('ready') else '（未完成预处理）'}", a["id"])
            for a in all_avatars()]


def _new_id(name):
    """只用 ASCII：OpenCV 和部分上游代码在 Windows 上读写不了中文路径。名称另存在 meta.json。"""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"


def create_video(name, src, start=None, end=None, on_progress=None, on_wait=None):
    """从视频建立形象：标准化底片 → MuseTalk 预处理（检测人脸、计算遮罩与潜变量）。"""
    name = (name or "").strip() or "我的形象"
    aid = _new_id(name)
    d = _dir(aid)
    os.makedirs(d)
    try:
        if on_progress:
            on_progress(0.01, "转换视频（25fps）…")
        info = media.normalize_video(src, os.path.join(d, "source.mp4"), config.settings()["max_side"], start, end)
        if info["duration"] < 1:
            raise ValueError("视频太短，至少需要 1 秒（推荐 10–60 秒）")
        if info["duration"] > 180:
            raise ValueError("视频太长，请截取 3 分钟以内（推荐 10–60 秒）")
        media.thumbnail(os.path.join(d, "source.mp4"), os.path.join(d, "cover.jpg"), min(1.0, info["duration"] / 2))
        meta = {"id": aid, "name": name, "kind": "video", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_name": os.path.basename(src), "width": info["width"], "height": info["height"],
                "duration": round(info["duration"], 2), "ready": False}
        _save(meta)
        prepare(aid, on_progress, on_wait)
        return load(aid)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def prepare(aid, on_progress=None, on_wait=None, extra_margin=10, parsing_mode="jaw"):
    meta = load(aid)
    d = meta["dir"]
    for sub in ("frames", "masks"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    r = engines.run("musetalk", {"cmd": "prepare", "video": os.path.join(d, "source.mp4"), "out_dir": d,
                                 "extra_margin": extra_margin, "parsing_mode": parsing_mode},
                    on_progress, gpu_task="形象预处理", on_wait=on_wait)
    meta.update(ready=True, prep=r, prep_sec=r.get("sec"), prep_vram_mb=r.get("peak_vram_mb"))
    _save(meta)
    return meta


def create_photo(name, src, on_progress=None, on_wait=None):
    """照片形象：转正、限制尺寸后检测所有人脸；默认驱动最大的真人脸，合影时可在形象库里改选。"""
    name = (name or "").strip() or "我的照片形象"
    aid = _new_id(name)
    d = _dir(aid)
    os.makedirs(d)
    try:
        photo = os.path.join(d, "photo.png")
        w, h = media.image_frame(src, photo)
        if on_progress:
            on_progress(0.3, "检测人脸…")
        faces = engines.run("joyvasa", {"cmd": "faces", "image": photo}, gpu_task="人脸检测", on_wait=on_wait)["faces"]
        if not faces:
            raise ValueError("照片里检测不到人脸，请换一张正脸清晰的照片")
        meta = {"id": aid, "name": name, "kind": "photo", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_name": os.path.basename(src), "width": w, "height": h, "ready": True,
                "faces": faces, "face": 0}
        _save(meta)
        _photo_cover(load(aid))
        return load(aid)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def _face_crop_box(box, w, h, k=2.2):
    """以人脸为中心、按脸宽放大 k 倍的方形区域（用作封面）。"""
    x1, y1, x2, y2 = box
    s = max(x2 - x1, y2 - y1) * k
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(w, cx + s / 2)), int(min(h, cy + s / 2)))


def _photo_cover(meta):
    from PIL import Image
    im = Image.open(os.path.join(meta["dir"], "photo.png"))
    f = meta["faces"][meta.get("face", 0)]["box"]
    im.crop(_face_crop_box(f, im.width, im.height)).resize((320, 320)).convert("RGB").save(
        os.path.join(meta["dir"], "cover.jpg"), quality=90)


def faces_image(aid):
    """照片上画出所有检测到的人脸框和编号（当前选中的用绿色），给用户挑选。"""
    from PIL import Image, ImageDraw, ImageFont
    m = load(aid)
    im = Image.open(os.path.join(m["dir"], "photo.png")).convert("RGB")
    dr = ImageDraw.Draw(im)
    lw = max(2, im.width // 400)
    try:
        font = ImageFont.truetype("msyh.ttc", max(18, im.width // 40))
    except OSError:
        font = ImageFont.load_default()
    for i, f in enumerate(m.get("faces", [])):
        col = (0, 220, 90) if i == m.get("face", 0) else (255, 200, 0)
        x1, y1, x2, y2 = f["box"]
        dr.rectangle([x1, y1, x2, y2], outline=col, width=lw * (2 if i == m.get("face", 0) else 1))
        dr.text((x1 + lw, max(0, y1 - font.size - 4)), f"{i + 1}", fill=col, font=font, stroke_width=2,
                stroke_fill=(0, 0, 0))
    out = os.path.join(config.TMP, f"faces_{aid}.jpg")
    im.save(out, quality=88)
    return out


def face_choices(aid):
    m = load(aid)
    return [(f"{i + 1} 号（置信度 {f['score']:.2f}）", i) for i, f in enumerate(m.get("faces", []))]


def set_face(aid, idx):
    m = load(aid)
    if not 0 <= int(idx) < len(m.get("faces", [])):
        raise ValueError("没有这个人脸")
    m["face"] = int(idx)
    _save(m)
    _photo_cover(m)
    return m


def face_box(m):
    fs = m.get("faces") or []
    return fs[m.get("face", 0)]["box"] if fs else None


def rename(aid, name):
    meta = load(aid)
    meta["name"] = name.strip() or meta["name"]
    _save(meta)
    return meta


def delete(aid):
    shutil.rmtree(_dir(aid), ignore_errors=True)


def cover(aid):
    p = os.path.join(_dir(aid), "cover.jpg")
    return p if os.path.exists(p) else None


def preview(aid):
    m = load(aid)
    return os.path.join(m["dir"], "source.mp4" if m["kind"] == "video" else "photo.png")


def describe(m):
    """形象详情（给网页显示）。"""
    lines = [f"**{m['name']}** · {KINDS[m['kind']]} · {m['width']}×{m['height']}"]
    if m["kind"] == "photo" and len(m.get("faces", [])) > 1:
        lines.append(f"检测到 {len(m['faces'])} 张脸，当前驱动 {m.get('face', 0) + 1} 号（绿框）；不对的话在「形象库」里改选")
    if m["kind"] == "video":
        lines.append(f"时长 {m.get('duration', 0)} 秒")
        p = m.get("prep") or {}
        if p:
            lines.append(f"人脸检出 {p.get('face_ratio', 0):.0%}，嘴部区域约 {p.get('face_size', 0)} 像素宽")
            lines.append("底片里本人在说话：停顿时由模型生成闭嘴" if p.get("talking_in_source")
                         else "底片里本人没说话：停顿时自动换回原画面（最自然）")
            if p.get("face_size", 999) < 160:
                lines.append("⚠️ 脸在画面里偏小，嘴部可能不够清晰；建议离镜头近一些重拍")
        elif not m.get("ready"):
            lines.append("⚠️ 预处理没有完成，请点「重新预处理」")
    return "\n\n".join(lines)
