# -*- coding: utf-8 -*-
"""一键工作流：上传一段或多段视频 / 图片（素材）+ 一段或多段音频 → 素材里的人作为数字人说出这些声音。

流程：
  1. 素材 → 形象：按文件内容指纹缓存，同一个视频只预处理一次；图片自动检测人脸（可改选）。
  2. 音频按顺序拼接（段间留 0.3 秒）；字幕优先用上传的同名 SRT，否则调用声音工坊 SenseVoice 自动识别。
  3. 分镜：多个素材时按句子边界（没有识别结果时按停顿）切成若干镜头，素材轮流出镜；
     多段音频也可以「每段音频一个镜头」。
  4. 同一个引擎的镜头集中生成（避免来回加载模型）；同一个视频素材的镜头接着往下放，动作不重复。
  5. 每个镜头按人脸位置铺满裁切到统一画幅，并精确到帧（避免多段拼接后音画逐渐错位），拼接后混入整段音频。
  6. 背景处理、烧录字幕，输出到 outputs/，记录进作品历史（可导出剪映草稿）。

输出方式：合成一条 / 每个素材各一条 / 每段音频各一条。
"""
import json
import os
import random
import re
import shutil
import time

import avatars
import config
import engines
import media
import render as R
import subtitles as SUB

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

MODES = {"合成一条（多个素材轮流出镜）": "mix", "每个素材各生成一条": "per_material", "每段音频各生成一条": "per_audio"}
QUALITY = {"快速（推荐）": "fast", "高质量（视频素材用 LatentSync，慢约 6 倍）": "hq"}
SUBS = {"自动识别（调用声音工坊）": "auto", "只用上传的 SRT": "srt", "不要字幕": "none"}
SIZES = {"竖屏 9:16": (1080, 1920), "横屏 16:9": (1920, 1080), "方形 1:1": (1080, 1080)}
FPS = 25
GAP = 0.3
SPEED = {"musetalk": 2.5, "latentsync": 15.0, "joyvasa": 4.5}  # 每秒视频大约要算多少秒（RTX 3070 实测）


def classify(paths):
    """把上传的文件按扩展名分成素材（视频 / 图片）、音频、字幕。视频既可以当素材也可以提供声音：只当素材。"""
    mats, auds, srts, bad = [], [], [], []
    for p in paths or []:
        ext = os.path.splitext(p)[1].lower()
        if ext in VIDEO_EXT:
            mats.append((p, "video"))
        elif ext in IMAGE_EXT:
            mats.append((p, "photo"))
        elif ext in AUDIO_EXT:
            auds.append(p)
        elif ext == ".srt":
            srts.append(p)
        else:
            bad.append(os.path.basename(p))
    return mats, auds, srts, bad


def _stem(p):
    return os.path.splitext(os.path.basename(p))[0].lower()


# ---------------------------------------------------------------- 素材

def analyze(materials, on_progress=None, on_wait=None):
    """素材预检：图片马上建立形象并检测人脸（几秒）；视频只看有没有缓存（预处理在生成时做）。
    返回 [{path, kind, name, avatar, faces, face, preview, note}]"""
    out = []
    for i, (p, kind) in enumerate(materials):
        if on_progress:
            on_progress(i / max(1, len(materials)), f"检查素材 {i + 1}/{len(materials)}：{os.path.basename(p)}")
        item = {"path": p, "kind": kind, "name": os.path.basename(p), "avatar": None, "faces": [], "face": 0}
        if kind == "photo":
            m, _ = avatars.ensure_photo(p, on_wait=on_wait)
            item.update(avatar=m["id"], faces=m["faces"], face=m.get("face", 0), preview=avatars.faces_image(m["id"]),
                        note=f"{len(m['faces'])} 张脸，默认 {m.get('face', 0) + 1} 号")
        else:
            info = media.probe(p)
            m = avatars.find_by_hash(avatars.file_hash(p), "video")
            prev = os.path.join(config.TMP, f"pv_{avatars.file_hash(p)}.jpg")
            if not os.path.exists(prev):
                media.thumbnail(p, prev, min(1.0, info["duration"] / 2))
            note = f"{info['duration']:.0f} 秒 · {info['width']}×{info['height']}"
            if info["duration"] > avatars.MAX_AUTO_SEC:
                note += f"（只取前 {avatars.MAX_AUTO_SEC} 秒作底片）"
            note += " · 已预处理，可直接用" if m else " · 首次使用需要预处理"
            item.update(avatar=m["id"] if m else None, preview=prev, note=note)
        out.append(item)
    return out


# ---------------------------------------------------------------- 分镜

def plan_cuts(cands, dur, shot_len, min_len=2.0):
    """在候选切点（句子边界 / 停顿）里挑镜头切点：每个镜头尽量接近 shot_len 秒，最短 min_len 秒；
    长时间没有候选切点时在目标位置硬切。"""
    cands = sorted(c for c in cands if min_len <= c <= dur - min_len)
    cuts, last = [], 0.0
    while dur - last > shot_len * 1.4:
        target = last + shot_len
        ok = [c for c in cands if c - last >= min_len and c <= last + shot_len * 1.6 and dur - c >= min_len]
        c = min(ok, key=lambda c: abs(c - target)) if ok else target
        cuts.append(c)
        last = c
    return cuts


def _grid(t):
    return round(t * FPS) / FPS


def _face_center(av):
    """人脸中心（相对画面 0~1），裁切画幅时让脸在合适的位置。"""
    try:
        if av["kind"] == "photo":
            x1, y1, x2, y2 = avatars.face_box(av)
            return (x1 + x2) / 2 / av["width"], (y1 + y2) / 2 / av["height"]
        import numpy as np
        b = np.load(os.path.join(av["dir"], "prep.npz"))["boxes"]
        cx = float(np.median((b[:, 0] + b[:, 2]) / 2)) / av["width"]
        cy = float(np.median(b[:, 1])) / av["height"]  # 嘴部区域上沿约在鼻梁处，接近脸的中心
        return cx, cy
    except Exception:
        return None


def _out_size(aspect, avs):
    if aspect in SIZES:
        return SIZES[aspect]
    a = avs[0]
    w, h = a["width"], a["height"]
    if a["kind"] == "photo":  # JoyVASA 输出的最长边不超过 1920
        k = min(1.0, 1920 / max(w, h))
        w, h = w * k, h * k
    return int(w) // 2 * 2, int(h) // 2 * 2


# ---------------------------------------------------------------- 生成

def run(material_paths, audio_paths, srt_paths=(), mode="mix", quality="fast", shot_len=6.0, per_audio_shot=True,
        subs="auto", burn_subs=True, aspect="原始", bg_mode="原样", bg_color="#00B140", bg_image=None,
        face_choice=None, opts=None, on_progress=None, on_wait=None):
    """返回生成记录列表（每条和 render.render 的记录格式相同，可以导出剪映草稿）。
    material_paths：[(路径, "video"|"photo")]；face_choice：{图片路径: 人脸序号}。"""
    if not material_paths:
        raise ValueError("请上传至少一个视频或图片素材")
    if not audio_paths:
        raise ValueError("请上传至少一段音频")
    opts = dict(opts or {})
    face_choice = face_choice or {}
    cb = on_progress or (lambda p, m="": None)

    # ---- 1. 素材 → 形象
    new_videos = [p for p, k in material_paths if k == "video"
                  and not avatars.find_by_hash(avatars.file_hash(p), "video")]
    prep_w = sum(min(media.probe(p)["duration"], avatars.MAX_AUTO_SEC) * 3 for p in new_videos)
    audio_dur = sum(media.probe(a)["duration"] for a in audio_paths)
    n_jobs = len(material_paths) if mode == "per_material" else len(audio_paths) if mode == "per_audio" else 1
    gen_w = audio_dur * (len(material_paths) if mode == "per_material" else 1) * 4
    total_w = prep_w + gen_w + 1
    done_w = [0.0]

    def sub(weight):
        base = done_w[0]
        done_w[0] += weight

        def f(p, m=""):
            cb(min(0.999, (base + weight * p) / total_w), m)
        return f

    avs = []
    for p, kind in material_paths:
        if kind == "video":
            vid_w = min(media.probe(p)["duration"], avatars.MAX_AUTO_SEC) * 3 if p in new_videos else 0
            sp = sub(vid_w)
            m, _ = avatars.ensure_video(p, lambda q, msg, n=os.path.basename(p): sp(q, f"预处理 {n}：{msg}"), on_wait)
        else:
            m, _ = avatars.ensure_photo(p, on_wait=on_wait)
            if p in face_choice and face_choice[p] is not None:
                m = avatars.set_face(m["id"], int(face_choice[p]))
        avs.append(avatars.load(m["id"]))

    # ---- 2. 任务
    if mode == "per_material":
        jobs = [([a], audio_paths) for a in avs]
    elif mode == "per_audio":
        jobs = [(avs, [a]) for a in audio_paths]
    else:
        jobs = [(avs, audio_paths)]
    results = []
    for i, (javs, jauds) in enumerate(jobs):
        w = gen_w / len(jobs)
        sp = sub(w)
        tag = f"（{i + 1}/{len(jobs)}）" if len(jobs) > 1 else ""
        results.append(_job(javs, jauds, srt_paths, quality, shot_len, per_audio_shot, subs, burn_subs, aspect,
                            bg_mode, bg_color, bg_image, opts, lambda q, msg: sp(q, tag + msg), on_wait))
    cb(1.0, "完成")
    return results


def _job(avs, audio_paths, srt_paths, quality, shot_len, per_audio_shot, subs, burn_subs, aspect, bg_mode, bg_color,
         bg_image, opts, cb, on_wait):
    t0 = time.time()
    rid = time.strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 999):03d}"
    d = os.path.join(config.RENDERS, rid)
    os.makedirs(d)
    crf = config.settings()["crf"]
    try:
        # ---- 音频
        cb(0.01, "拼接音频…")
        wav = os.path.join(d, "audio.wav")
        spans = media.concat_audio(audio_paths, wav, GAP if len(audio_paths) > 1 else 0)
        dur = spans[-1][1]
        if dur < 0.5:
            raise ValueError("音频太短")

        # ---- 字幕：上传的同名 SRT（只有一段音频时任何 SRT 都算）> 自动识别
        items, chars = [], None
        srt_by = {_stem(s): s for s in srt_paths}
        for k, (a, (s0, _)) in enumerate(zip(audio_paths, spans)):
            sp = srt_by.get(_stem(a)) or (srt_paths[0] if len(audio_paths) == 1 and srt_paths else None)
            if sp and subs != "none":
                items += [(x + s0, y + s0, t) for x, y, t in SUB.read_srt(sp)]
        if not items and subs == "auto" and SUB.asr_available():
            cb(0.03, "识别字幕（声音工坊 SenseVoice）…")
            chars = SUB.asr_chars(wav, on_wait=on_wait)
            if chars:
                items = SUB.lines_from_chars(chars)
        srt = SUB.write_srt(items, os.path.join(d, "subs.srt")) if items else None

        # ---- 分镜
        if len(avs) == 1:
            bounds = [0.0, dur]
        else:
            if per_audio_shot and len(audio_paths) > 1:
                cuts = [(spans[i][1] + spans[i + 1][0]) / 2 for i in range(len(spans) - 1)]
            else:
                if chars:
                    cands = SUB.sentence_cuts(chars)
                elif items:
                    cands = [(items[i][1] + items[i + 1][0]) / 2 for i in range(len(items) - 1)]
                else:
                    cands = media.silence_points(media.pcm16k(wav))
                cuts = plan_cuts(cands, dur, float(shot_len))
            bounds = [0.0] + [_grid(c) for c in cuts] + [dur]
        shots = [{"i": i, "start": bounds[i], "end": bounds[i + 1], "av": avs[i % len(avs)]}
                 for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 0.2]
        W, H = _out_size(aspect, avs)
        for s in shots:
            s["engine"] = ("latentsync" if quality == "hq" else "musetalk") if s["av"]["kind"] == "video" else "joyvasa"
            s["frames"] = int(round(s["end"] * FPS)) - int(round(s["start"] * FPS))

        # ---- 逐镜头生成（同一引擎集中做；同一视频素材接着往下放）
        work = sum((s["end"] - s["start"]) * SPEED[s["engine"]] for s in shots) or 1
        done = 0.0
        next_frame = {}
        order = sorted(shots, key=lambda s: (s["engine"], s["i"]))
        jopts = dict(opts, max_dim=1920 if max(W, H) > 1280 else 1280)
        for s in order:
            w = (s["end"] - s["start"]) * SPEED[s["engine"]]
            base = 0.05 + 0.85 * done / work
            span = 0.85 * w / work
            label = f"镜头 {s['i'] + 1}/{len(shots)}（{s['av']['name']}）"
            seg = media.cut_audio(wav, os.path.join(d, f"a{s['i']:03d}.wav"), s["start"], s["end"])
            raw = os.path.join(d, f"r{s['i']:03d}.mp4")
            res = R.generate(s["av"], seg, raw, s["engine"], jopts, crf,
                             on_progress=lambda p, m, b=base, sp=span, l=label: cb(b + sp * p, f"{l}：{m}"),
                             on_wait=on_wait, start_frame=next_frame.get(s["av"]["id"], 0))
            if res.get("next_frame") is not None:
                next_frame[s["av"]["id"]] = res["next_frame"]
            s["fit"] = media.fit_frames(raw, os.path.join(d, f"f{s['i']:03d}.mp4"), W, H, s["frames"],
                                        _face_center(s["av"]))
            s["sec"] = res.get("sec")
            s["vram"] = res.get("peak_vram_mb")
            done += w

        # ---- 拼接 + 背景 + 字幕
        cb(0.91, "拼接镜头…")
        cur = media.concat_videos([s["fit"] for s in shots], wav, os.path.join(d, "joined.mp4"), crf)
        bg = R.BG_MODES.get(bg_mode)
        if bg:
            ext = ".mov" if bg == "alpha" else ".mp4"
            out_bg = os.path.join(d, "bg" + ext)
            engines.run("matting", {"cmd": "render", "video": cur, "out": out_bg, "mode": bg, "color": bg_color,
                                    "image": bg_image}, lambda p, m: cb(0.92 + 0.05 * p, m), on_wait=on_wait)
            cur = out_bg
        burned = bool(srt and burn_subs and bg != "alpha")
        if burned:
            cb(0.97, "烧录字幕…")
            cur = media.post(cur, os.path.join(d, "final.mp4"), srt=srt, crf=crf)

        # ---- 输出 + 记录
        names = []
        for a in avs:
            if a["name"] not in names:
                names.append(a["name"])
        title = names[0] + (f"等{len(names)}个素材" if len(names) > 1 else "")
        stem = os.path.join(config.OUTPUTS, re.sub(r'[\\/:*?"<>|\s]', "", title)[:30] + f"_数字人_{rid[:15]}")
        out = stem + os.path.splitext(cur)[1]
        shutil.copyfile(cur, out)
        out_srt = None
        if srt:
            out_srt = stem + ".srt"
            shutil.copyfile(srt, out_srt)
        vram = max([s["vram"] or 0 for s in shots] or [0])
        meta = {"id": rid, "avatar": avs[0]["id"], "avatar_name": "、".join(names), "engine": "workflow",
                "engines": sorted({s["engine"] for s in shots}), "audio_src": "、".join(os.path.basename(a)
                                                                                      for a in audio_paths),
                "take": None, "duration": round(dur, 2), "out": out, "srt": out_srt, "aspect": aspect,
                "size": [W, H], "bg": bg_mode, "subtitles": burned, "refine": False, "opts": opts,
                "shots": [{"start": s["start"], "end": s["end"], "avatar": s["av"]["name"], "engine": s["engine"],
                           "sec": s["sec"]} for s in shots],
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "sec": round(time.time() - t0, 1),
                "engine_sec": round(sum(s["sec"] or 0 for s in shots), 1), "peak_vram_mb": vram or None}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        for fn in os.listdir(d):
            if fn != "meta.json":
                p = os.path.join(d, fn)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        return meta
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def summary(results):
    lines = []
    for m in results:
        speed = m["sec"] / m["duration"] if m.get("duration") else 0
        shots = m.get("shots") or []
        lines.append(f"**{m['avatar_name']}** · {m['duration']:.1f} 秒 · {len(shots)} 个镜头 · "
                     f"{'/'.join(R.ENGINE_LABELS.get(e, e).split('（')[0] for e in m.get('engines', []))} · "
                     f"用时 {m['sec']:.0f} 秒（每秒视频约 {speed:.1f} 秒）"
                     + (f" · 显存峰值 {m['peak_vram_mb'] / 1024:.1f}GB" if m.get("peak_vram_mb") else "")
                     + f"\n\n`{m['out']}`" + (f"　字幕：`{os.path.basename(m['srt'])}`" if m.get("srt") else ""))
    return "\n\n---\n\n".join(lines)
