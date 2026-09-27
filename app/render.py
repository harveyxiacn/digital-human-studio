# -*- coding: utf-8 -*-
"""生成：形象 + 音频 → 数字人视频。

流程：准备音频（上传 / 声音工坊作品）→ 引擎生成口型 →（可选）背景处理 → 画幅裁切 + 字幕烧录 → outputs/
每次生成的参数和耗时记在 workspace/renders/<id>/meta.json。
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
import voice_bridge

ENGINE_LABELS = {
    "musetalk": "MuseTalk（快速，推荐）",
    "latentsync": "LatentSync（口型更准，较慢）",
    "joyvasa": "JoyVASA（照片：会点头眨眼）",
}
BG_MODES = {"原样": None, "绿幕": "green", "纯色": "color", "图片背景": "image", "透明背景（.mov）": "alpha"}


def engines_for(kind):
    return ["joyvasa"] if kind == "photo" else ["musetalk", "latentsync"]


class Stage:
    """把各阶段的进度折算成总进度。"""

    def __init__(self, cb, plan):
        self.cb, self.plan, self.base = cb, plan, 0.0
        self.total = sum(plan.values()) or 1

    def __call__(self, name):
        w = self.plan[name] / self.total
        base = self.base
        self.base += w

        def f(p, msg=""):
            if self.cb:
                self.cb(min(0.999, base + w * p), msg)
        return f


def _face_cx(meta):
    try:
        import numpy as np
        b = np.load(os.path.join(meta["dir"], "prep.npz"))["boxes"]
        return float(np.median((b[:, 0] + b[:, 2]) / 2) / meta["width"])
    except Exception:
        return None


def generate(av, wav, out, engine, opts=None, crf=18, refine=False, on_progress=None, on_refine=None, on_wait=None,
             start_frame=0):
    """形象 + 一段音频 → 口型视频 out（带这段音频）。返回引擎结果（sec、peak_vram_mb、next_frame 等）。
    start_frame：视频形象从底片第几帧开始（工作流里同一个素材的多个镜头接着往下放，动作不重复）。"""
    opts = opts or {}
    if engine == "musetalk":
        return engines.run("musetalk", {"cmd": "render", "avatar_dir": av["dir"], "audio": wav, "out": out,
                                        "batch_size": int(opts.get("batch_size", 8)), "crf": crf,
                                        "idle_original": opts.get("idle_original"), "start_frame": int(start_frame)},
                           on_progress, on_wait=on_wait)
    if engine == "latentsync":
        return engines.run("latentsync", {"cmd": "render", "video": os.path.join(av["dir"], "source.mp4"),
                                          "audio": wav, "out": out, "steps": int(opts.get("steps", 20)),
                                          "guidance": float(opts.get("guidance", 1.5)),
                                          "seed": int(opts.get("seed", 1247)), "crf": crf},
                           on_progress, on_wait=on_wait)
    raw = out if not refine else out + ".joyvasa.mp4"
    res = engines.run("joyvasa", {"cmd": "render", "image": os.path.join(av["dir"], "photo.png"), "audio": wav,
                                  "out": raw, "cfg_scale": float(opts.get("cfg_scale", 2.8)),
                                  "expression": float(opts.get("expression", 1.0)),
                                  "head_motion": float(opts.get("head_motion", 1.0)), "crf": crf,
                                  "face_box": avatars.face_box(av),
                                  "keep_expression": bool(opts.get("keep_expression", True)),
                                  "max_dim": int(opts.get("max_dim", 1280))},
                      on_progress, on_wait=on_wait)
    if refine:  # 把 JoyVASA 的结果当作底片，再用 MuseTalk 重做嘴型
        sp = on_refine or (lambda p, m="": None)
        tmp_av = out + "_refine"
        os.makedirs(tmp_av, exist_ok=True)
        engines.run("musetalk", {"cmd": "prepare", "video": raw, "out_dir": tmp_av},
                    lambda p, m: sp(p * 0.5, "口型精修：" + m), gpu_task="形象预处理", on_wait=on_wait)
        engines.run("musetalk", {"cmd": "render", "avatar_dir": tmp_av, "audio": wav, "out": out,
                                 "idle_original": False, "crf": crf},
                    lambda p, m: sp(0.5 + p * 0.5, "口型精修：" + m), on_wait=on_wait)
        shutil.rmtree(tmp_av, ignore_errors=True)
        os.remove(raw)
    return res


def render(avatar_id, audio=None, take_id=None, engine=None, srt=None, subtitles=True, aspect="原始",
           bg_mode="原样", bg_color="#00B140", bg_image=None, refine=False, opts=None,
           on_progress=None, on_wait=None):
    """audio：音频文件路径（或 None 用 take_id 的声音工坊作品）。返回生成记录 meta。"""
    opts = opts or {}
    av = avatars.load(avatar_id)
    if not av.get("ready"):
        raise ValueError("这个形象还没有完成预处理")
    engine = engine or engines_for(av["kind"])[0]
    if engine not in engines_for(av["kind"]):
        raise ValueError(f"{avatars.KINDS[av['kind']]}不能用 {ENGINE_LABELS[engine]}")

    rid = time.strftime("%Y%m%d_%H%M%S_") + f"{random.randint(0, 999):03d}"
    d = os.path.join(config.RENDERS, rid)
    os.makedirs(d)
    t0 = time.time()
    try:
        # ---- 音频
        src_name = ""
        if take_id:
            tk = voice_bridge.take(take_id)
            if not tk:
                raise ValueError("找不到这个声音工坊作品")
            audio, srt = tk["wav"], srt or tk.get("srt")
            src_name = f"声音工坊 · {tk['voice_name']}"
        if not audio or not os.path.exists(audio):
            raise ValueError("请先选择音频")
        wav = media.to_wav(audio, os.path.join(d, "audio.wav"))
        dur = media.probe(wav)["duration"]
        if dur < 0.5:
            raise ValueError("音频太短")
        if srt and os.path.exists(srt):
            shutil.copyfile(srt, os.path.join(d, "subs.srt"))
            srt = os.path.join(d, "subs.srt")
        else:
            srt = None

        bg = BG_MODES.get(bg_mode)
        plan = {"gen": 10, "refine": 5 if (refine and engine == "joyvasa") else 0, "bg": 3 if bg else 0,
                "post": 1 if (aspect != "原始" or (subtitles and srt)) and bg != "alpha" else 0}
        stage = Stage(on_progress, plan)
        raw = os.path.join(d, "raw.mp4")
        crf = config.settings()["crf"]
        res = {}

        # ---- 口型
        res = generate(av, wav, raw, engine, opts, crf, refine, stage("gen"),
                       stage("refine") if plan["refine"] else None, on_wait)

        # ---- 背景
        cur = raw
        if bg:
            ext = ".mov" if bg == "alpha" else ".mp4"
            out_bg = os.path.join(d, "bg" + ext)
            engines.run("matting", {"cmd": "render", "video": cur, "out": out_bg, "mode": bg, "color": bg_color,
                                    "image": bg_image}, stage("bg"), on_wait=on_wait)
            cur = out_bg

        # ---- 画幅 + 字幕
        if plan["post"]:
            sp = stage("post")
            sp(0.1, "画幅 / 字幕…")
            final_tmp = os.path.join(d, "final.mp4")
            media.post(cur, final_tmp, aspect=aspect, srt=srt if subtitles else None, crf=crf,
                       face_cx=_face_cx(av) if av["kind"] == "video" else None)
            cur = final_tmp

        # ---- 输出
        name = re.sub(r'[\\/:*?"<>|\s]', "", av["name"])[:20]
        stem = os.path.join(config.OUTPUTS, f"{name}_{engine}_{rid[:15]}")
        out = stem + os.path.splitext(cur)[1]
        shutil.copyfile(cur, out)
        out_srt = None
        if srt:
            out_srt = stem + ".srt"
            shutil.copyfile(srt, out_srt)
        meta = {"id": rid, "avatar": avatar_id, "avatar_name": av["name"], "engine": engine, "audio_src": src_name
                or os.path.basename(audio), "take": take_id, "duration": round(dur, 2), "out": out, "srt": out_srt,
                "aspect": aspect, "bg": bg_mode, "subtitles": bool(subtitles and srt), "refine": refine, "opts": opts,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "sec": round(time.time() - t0, 1),
                "engine_sec": res.get("sec"), "peak_vram_mb": res.get("peak_vram_mb")}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        # 中间文件只保留记录
        for fn in os.listdir(d):
            if fn not in ("meta.json",):
                p = os.path.join(d, fn)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        if on_progress:
            on_progress(1.0, "完成")
        return meta
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def history(limit=30):
    out = []
    for rid in sorted(os.listdir(config.RENDERS), reverse=True):
        p = os.path.join(config.RENDERS, rid, "meta.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if os.path.exists(m["out"]):
            out.append(m)
        if len(out) >= limit:
            break
    return out


def summary(m):
    speed = m["sec"] / m["duration"] if m.get("duration") else 0
    s = (f"**{m['avatar_name']} · {ENGINE_LABELS.get(m['engine'], m['engine'])}**　时长 {m['duration']:.1f} 秒，"
         f"用时 {m['sec']:.0f} 秒（每秒视频约 {speed:.1f} 秒）")
    if m.get("peak_vram_mb"):
        s += f"，显存峰值 {m['peak_vram_mb'] / 1024:.1f}GB"
    s += f"\n\n成片：`{m['out']}`"
    if m.get("srt"):
        s += f"\n\n字幕：`{os.path.basename(m['srt'])}`"
    return s
