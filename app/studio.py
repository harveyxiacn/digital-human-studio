# -*- coding: utf-8 -*-
"""数字人工坊 · 网页界面（Gradio）。主进程不加载 torch，所有模型在引擎子进程里运行。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402

import avatars  # noqa: E402
import config  # noqa: E402
import engines  # noqa: E402
import gpu  # noqa: E402
import jianying  # noqa: E402
import render as R  # noqa: E402
import voice_bridge as VB  # noqa: E402

AUDIO_SRC = ["上传音频", "声音工坊作品", "文字 → 声音工坊朗读"]
TIPS = """**拍摄建议（视频形象）**
- 正脸或微侧，脸部清晰、光线均匀，脸在画面里宽度最好超过 250 像素（手机竖拍半身即可）。
- **嘴巴自然闭合、不说话**，可以有轻微点头、眨眼、手势；10–60 秒。停顿时软件会直接用原画面，最自然。
- 头不要转得太大，不要用手挡脸；背景简单一点更适合换背景。

**照片形象**：正脸、不戴墨镜、五官清楚的半身照效果最好；JoyVASA 会根据声音生成点头、眨眼和表情。
"""


# ---------------------------------------------------------------- 通用

def _err(e):
    raise gr.Error(str(e).split("\n")[0][:300])


def _progress_cb(progress):
    def cb(p, msg=""):
        progress(p, desc=msg)
    return cb


def _wait_cb(progress):
    def cb(pos, cur):
        progress(0.0, desc=f"显卡正忙（{cur}），排队中：第 {pos} 位")
    return cb


def gpu_md():
    s = gpu.status()
    if not s["total"]:
        return "⚠️ 没有检测到 NVIDIA 显卡"
    t = f"🖥️ {s['name']}　显存 {s['used'] / 1024:.1f} / {s['total'] / 1024:.0f} GB"
    if s["task"]:
        t += f"　|　正在运行：{s['task']}"
    if s["queue"]:
        t += f"　|　排队 {len(s['queue'])} 个"
    if s["apps"]:
        t += f"　|　⚠️ {'、'.join(s['apps'])} 正在占用显存，生成前请关闭"
    return t


def engines_md():
    rows = []
    for n, label in [("musetalk", "MuseTalk 1.5"), ("latentsync", "LatentSync 1.5"), ("joyvasa", "JoyVASA"),
                     ("matting", "背景处理")]:
        rows.append(f"- {label}：{'✅ 已安装' if engines.available(n) else '❌ 未安装（运行 安装.bat）'}")
    vs = config.voice_studio_dir()
    rows.append(f"- 声音工坊：{'✅ ' + vs if vs else '❌ 未找到（在下方填写目录）'}")
    return "\n".join(rows)


# ---------------------------------------------------------------- 形象库

def avatar_gallery():
    """形象缩略图条（HTML，固定尺寸；Gradio 图库组件在形象少时会把每张图拉得很宽）。"""
    import html
    items = []
    for a in avatars.all_avatars():
        c = avatars.cover(a["id"])
        if not c:
            continue
        src = "/gradio_api/file=" + c.replace("\\", "/")
        items.append(f'<div class="av-card"><img src="{src}"><div>{html.escape(a["name"])}<br>'
                     f'<small>{avatars.KINDS[a["kind"]]}</small></div></div>')
    return '<div class="av-strip">' + ("".join(items) or "还没有形象，先在上面建立一个") + "</div>"


def refresh_avatar_lists(sel=None):
    ch = avatars.choices()
    ids = [c[1] for c in ch]
    v = sel if sel in ids else (ids[0] if ids else None)
    return gr.update(choices=ch, value=v), gr.update(choices=ch, value=v), avatar_gallery()


def on_create_video(name, video, start, end, progress=gr.Progress()):
    if not video:
        raise gr.Error("请先上传视频")
    try:
        m = avatars.create_video(name, video, start or None, end or None, _progress_cb(progress), _wait_cb(progress))
    except Exception as e:
        _err(e)
    a, b, g = refresh_avatar_lists(m["id"])
    return a, b, g, avatars.describe(m)


def on_create_photo(name, image, progress=gr.Progress()):
    if not image:
        raise gr.Error("请先上传照片")
    try:
        m = avatars.create_photo(name, image, _progress_cb(progress), _wait_cb(progress))
    except Exception as e:
        _err(e)
    a, b, g = refresh_avatar_lists(m["id"])
    return a, b, g, avatars.describe(m)


def on_pick_manage(aid):
    if not aid:
        return "", None, None, "", gr.update(visible=False)
    m = avatars.load(aid)
    p = avatars.preview(aid)
    photo = m["kind"] == "photo"
    faces = gr.update(visible=photo and len(m.get("faces", [])) > 1,
                      choices=avatars.face_choices(aid) if photo else [], value=m.get("face", 0))
    return (avatars.describe(m), None if photo else p, avatars.faces_image(aid) if photo else None, m["name"], faces)


def on_set_face(aid, idx):
    if not aid or idx is None:
        return gr.update(), gr.update(), gr.update()
    m = avatars.set_face(aid, idx)
    return avatars.describe(m), avatars.faces_image(aid), avatar_gallery()


def on_rename(aid, name):
    if not aid:
        raise gr.Error("请先选择形象")
    avatars.rename(aid, name)
    a, b, g = refresh_avatar_lists(aid)
    return a, b, g


def on_delete(aid):
    if not aid:
        raise gr.Error("请先选择形象")
    avatars.delete(aid)
    return refresh_avatar_lists()


def on_reprepare(aid, margin, mode, progress=gr.Progress()):
    if not aid:
        raise gr.Error("请先选择形象")
    if avatars.load(aid)["kind"] != "video":
        raise gr.Error("照片形象不需要预处理")
    try:
        m = avatars.prepare(aid, _progress_cb(progress), _wait_cb(progress), int(margin), mode)
    except Exception as e:
        _err(e)
    a, b, _ = refresh_avatar_lists(aid)
    return a, b, avatars.describe(m)


# ---------------------------------------------------------------- 生成

def on_pick_avatar(aid):
    if not aid:
        return None, "", gr.update(), gr.update(visible=False)
    m = avatars.load(aid)
    es = R.engines_for(m["kind"])
    return (avatars.cover(aid), avatars.describe(m),
            gr.update(choices=[(R.ENGINE_LABELS[e], e) for e in es], value=es[0]),
            gr.update(visible=m["kind"] == "photo"))


def on_audio_src(src):
    return (gr.update(visible=src == AUDIO_SRC[0]), gr.update(visible=src == AUDIO_SRC[1]),
            gr.update(visible=src == AUDIO_SRC[2]))


def take_choices():
    return [(t["label"], t["id"]) for t in VB.takes()]


def on_refresh_takes():
    ch = take_choices()
    return gr.update(choices=ch, value=ch[0][1] if ch else None)


def on_pick_take(tid):
    t = VB.take(tid) if tid else None
    if not t:
        return None, ""
    lines = [f"共 {len(t['sentences'])} 句，{t['total']:.1f} 秒" + ("，带字幕" if t.get("srt") else "")]
    lines += [f"{i + 1}.【{s.get('emotion', '')}】{s['text']}" for i, s in enumerate(t["sentences"][:12])]
    return t["wav"], "\n\n".join(lines)


def on_refresh_voices():
    ch = VB.voices()
    return gr.update(choices=ch, value=ch[0][1] if ch else None)


def on_generate(aid, src, up_audio, up_srt, tid, vid, text, lang, v_engine, emo, auto_emo, speed,
                engine, subtitles, aspect, bg_mode, bg_color, bg_image, refine,
                batch, idle, steps, guidance, seed, cfg, expr, head, keep, progress=gr.Progress()):
    if not aid:
        raise gr.Error("请先在「形象库」建立并选择一个形象")
    audio, take_id, srt = None, None, None
    try:
        if src == AUDIO_SRC[0]:
            if not up_audio:
                raise gr.Error("请上传音频")
            audio, srt = up_audio, up_srt
        elif src == AUDIO_SRC[1]:
            if not tid:
                raise gr.Error("请选择声音工坊作品")
            take_id = tid
        else:
            if not vid or not (text or "").strip():
                raise gr.Error("请选择语音包并输入文字")
            progress(0.0, desc="声音工坊正在朗读…")
            t = VB.synth(vid, text, lang={"自动": "auto", "普通话": "zh", "粤语": "yue"}[lang], engine=v_engine,
                         emotion=emo, auto_emo=auto_emo, speed=speed,
                         on_log=lambda m: progress(0.0, desc="声音工坊：" + m))
            take_id = t["id"]
        idle_v = {"自动": None, "换回原画面": True, "模型生成": False}[idle]
        opts = {"batch_size": batch, "idle_original": idle_v, "steps": steps, "guidance": guidance, "seed": seed,
                "cfg_scale": cfg, "expression": expr, "head_motion": head, "keep_expression": keep}
        m = R.render(aid, audio=audio, take_id=take_id, engine=engine, srt=srt, subtitles=subtitles, aspect=aspect,
                     bg_mode=bg_mode, bg_color=bg_color, bg_image=bg_image, refine=refine, opts=opts,
                     on_progress=_progress_cb(progress), on_wait=_wait_cb(progress))
    except gr.Error:
        raise
    except Exception as e:
        _err(e)
    video = m["out"] if m["out"].endswith(".mp4") else None
    return video, R.summary(m), m["id"], history_md(), gr.update(value=m["out"], visible=True)


def history_md():
    hs = R.history(12)
    if not hs:
        return "还没有作品"
    return "\n".join(f"- {h['created'][5:16]} · {h['avatar_name']} · {h['engine']} · {h['duration']:.0f}秒 · "
                     f"用时 {h['sec']:.0f}秒 · `{os.path.basename(h['out'])}`" for h in hs)


def on_export(rid):
    if not rid:
        raise gr.Error("请先生成视频")
    m = next((h for h in R.history(200) if h["id"] == rid), None)
    if not m:
        raise gr.Error("找不到这个作品")
    try:
        p = jianying.export(m)
    except Exception as e:
        _err(e)
    return f"✅ 已导出剪映草稿：`{p}`\n\n打开剪映（如果已经开着，请重启一次）就能在草稿列表里看到。"


# ---------------------------------------------------------------- 设置

def on_save_settings(vs, dr, max_side, crf):
    config.save_settings(voice_studio=vs.strip(), jianying_drafts=dr.strip(), max_side=int(max_side), crf=int(crf))
    return "✅ 已保存（显卡锁位置的变化需要重启软件才生效）", engines_md()


# ---------------------------------------------------------------- 界面

CSS = """
.gradio-container {max-width: 1280px !important}
#gpu-bar {font-size: 0.9em; opacity: 0.85}
.av-strip {display: flex; gap: 12px; flex-wrap: wrap}
.av-card {width: 128px; text-align: center; font-size: 0.85em}
.av-card img {width: 128px; height: 128px; object-fit: cover; border-radius: 8px; display: block; margin-bottom: 4px}
"""


def build():
    s = config.settings()
    with gr.Blocks(title="数字人工坊", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧑‍💼 数字人工坊\n用自己的视频或照片 + 自己的声音，在本机生成口型同步的数字人视频")
        gpu_bar = gr.Markdown(gpu_md(), elem_id="gpu-bar")
        rid_state = gr.State(None)

        with gr.Tab("🎬 生成"):
            with gr.Row():
                with gr.Column(scale=4):
                    g_avatar = gr.Dropdown(label="① 选择形象", choices=avatars.choices(), interactive=True,
                                           value=next(iter(avatars.choices()), (None, None))[1])
                    with gr.Row():
                        g_cover = gr.Image(label="封面", height=200, interactive=False, show_label=False, scale=1)
                        with gr.Column(scale=2):
                            g_info = gr.Markdown()
                    g_src = gr.Radio(AUDIO_SRC, value=AUDIO_SRC[1] if VB.available() else AUDIO_SRC[0],
                                     label="② 声音从哪里来")
                    with gr.Group(visible=not VB.available()) as grp_up:
                        g_audio = gr.Audio(label="上传音频（wav / mp3 / m4a）", type="filepath")
                        g_srt = gr.File(label="字幕 SRT（可选）", file_types=[".srt"], type="filepath")
                    with gr.Group(visible=VB.available()) as grp_take:
                        with gr.Row():
                            g_take = gr.Dropdown(label="声音工坊作品", choices=take_choices(), scale=5,
                                                 value=next(iter(take_choices()), (None, None))[1])
                            g_take_ref = gr.Button("🔄", scale=0, min_width=50)
                        g_take_audio = gr.Audio(label="试听", type="filepath", interactive=False)
                        g_take_info = gr.Markdown()
                    with gr.Group(visible=False) as grp_tts:
                        with gr.Row():
                            g_voice = gr.Dropdown(label="语音包", choices=[], scale=5)
                            g_voice_ref = gr.Button("🔄 读取语音包", scale=0, min_width=120)
                        g_text = gr.Textbox(label="文字（一句一行或用标点分句）", lines=5)
                        with gr.Row():
                            g_lang = gr.Radio(["自动", "普通话", "粤语"], value="自动", label="语言")
                            g_veng = gr.Dropdown(VB.ENGINES, value="VoxCPM2", label="声音引擎")
                        with gr.Row():
                            g_emo = gr.Dropdown(["平静", "开心", "兴奋", "温柔", "悲伤", "生气", "惊讶", "严肃"],
                                                value="平静", label="默认情感")
                            g_auto = gr.Checkbox(value=True, label="按句自动判断情感")
                            g_speed = gr.Slider(0.7, 1.3, 1.0, step=0.05, label="语速")
                with gr.Column(scale=3):
                    g_engine = gr.Radio(label="③ 生成引擎", choices=[])
                    g_refine = gr.Checkbox(label="照片形象：再用 MuseTalk 精修一遍口型（时间约翻倍）", value=False)
                    with gr.Accordion("④ 画面与字幕", open=True):
                        g_aspect = gr.Radio(list(R.media.ASPECTS), value="原始", label="画幅（居中裁切）")
                        g_subs = gr.Checkbox(value=True, label="烧录字幕（有 SRT 时）")
                        g_bg = gr.Radio(list(R.BG_MODES), value="原样", label="背景")
                        with gr.Row():
                            g_bgcolor = gr.ColorPicker(value="#00B140", label="纯色背景颜色")
                            g_bgimg = gr.Image(label="背景图片", type="filepath", height=120)
                    with gr.Accordion("高级参数", open=False):
                        gr.Markdown("**MuseTalk**")
                        g_batch = gr.Slider(2, 16, 8, step=2, label="批大小（显存不够时调小）")
                        g_idle = gr.Radio(["自动", "换回原画面", "模型生成"], value="自动", label="停顿时的嘴")
                        gr.Markdown("**LatentSync**")
                        g_steps = gr.Slider(10, 40, 20, step=1, label="去噪步数（越多越细，越慢）")
                        g_guid = gr.Slider(1.0, 3.0, 1.5, step=0.1, label="引导强度（口型力度）")
                        g_seed = gr.Number(1247, label="随机种子", precision=0)
                        gr.Markdown("**JoyVASA**")
                        g_cfg = gr.Slider(1.0, 4.0, 2.8, step=0.1, label="动作丰富度（cfg）")
                        g_expr = gr.Slider(0.5, 1.5, 1.0, step=0.05, label="表情幅度")
                        g_head = gr.Slider(0.0, 1.5, 1.0, step=0.05, label="头部动作幅度")
                        g_keep = gr.Checkbox(value=True, label="保留照片原有表情（推荐；关闭后先把嘴合上再驱动，笑脸照会显得嘟嘴）")
                    g_go = gr.Button("🎬 生成数字人视频", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=3):
                    g_out = gr.Video(label="成片", height=480)
                with gr.Column(scale=2):
                    g_sum = gr.Markdown()
                    g_file = gr.File(label="下载（透明背景 .mov 在这里）", visible=False)
                    g_exp = gr.Button("📤 导出为剪映草稿")
                    g_exp_msg = gr.Markdown()
            with gr.Accordion("最近的作品", open=False):
                g_hist = gr.Markdown(history_md())

        with gr.Tab("👤 形象库"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 用视频建立形象（推荐）")
                    a_vname = gr.Textbox(label="形象名称", placeholder="例如：正装讲解")
                    a_video = gr.Video(label="上传视频（10–60 秒，不说话、看镜头）", sources=["upload"])
                    with gr.Row():
                        a_start = gr.Number(label="从第几秒开始（可选）", precision=1)
                        a_end = gr.Number(label="到第几秒结束（可选）", precision=1)
                    a_vbtn = gr.Button("建立视频形象", variant="primary")
                with gr.Column():
                    gr.Markdown("### 用照片建立形象")
                    a_pname = gr.Textbox(label="形象名称", placeholder="例如：旅行照")
                    a_photo = gr.Image(label="上传照片", type="filepath", height=300)
                    a_pbtn = gr.Button("建立照片形象", variant="primary")
                    gr.Markdown(TIPS)
            gr.Markdown("### 我的形象")
            a_gallery = gr.HTML(avatar_gallery())
            with gr.Row():
                with gr.Column(scale=2):
                    a_pick = gr.Dropdown(label="选择要管理的形象", choices=avatars.choices(),
                                         value=next(iter(avatars.choices()), (None, None))[1])
                    a_info = gr.Markdown()
                    a_name = gr.Textbox(label="名称")
                    with gr.Row():
                        a_ren = gr.Button("重命名")
                        a_del = gr.Button("🗑️ 删除", variant="stop")
                    with gr.Accordion("重新预处理（嘴部区域不对时）", open=False):
                        a_margin = gr.Slider(0, 40, 10, step=2, label="下巴往下多留（像素）")
                        a_mode = gr.Radio(["jaw", "raw", "neck"], value="jaw", label="融合范围",
                                          info="jaw：只改下半张脸（推荐）；raw：整张脸；neck：带脖子")
                        a_rep = gr.Button("重新预处理")
                with gr.Column(scale=3):
                    a_pv = gr.Video(label="底片", height=360)
                    a_pp = gr.Image(label="照片（绿框 = 要驱动的人脸）", height=360, interactive=False)
                    a_face = gr.Radio(label="照片里有多张脸：选择要让谁说话", choices=[], visible=False)

        with gr.Tab("⚙️ 设置"):
            e_md = gr.Markdown(engines_md())
            st_vs = gr.Textbox(s["voice_studio"], label="声音工坊目录")
            st_dr = gr.Textbox(s["jianying_drafts"], label="剪映草稿目录（剪映 → 设置 → 草稿位置）")
            with gr.Row():
                st_max = gr.Slider(720, 1920, s["max_side"], step=60, label="形象视频最长边（像素，越大越清晰越慢）")
                st_crf = gr.Slider(12, 28, s["crf"], step=1, label="成片画质 CRF（越小越清晰）")
            st_btn = gr.Button("保存设置")
            st_msg = gr.Markdown()

        # ---- 事件
        g_avatar.change(on_pick_avatar, g_avatar, [g_cover, g_info, g_engine, g_refine])
        g_src.change(on_audio_src, g_src, [grp_up, grp_take, grp_tts])
        g_take_ref.click(on_refresh_takes, None, g_take)
        g_take.change(on_pick_take, g_take, [g_take_audio, g_take_info])
        g_voice_ref.click(on_refresh_voices, None, g_voice)
        g_go.click(on_generate, [g_avatar, g_src, g_audio, g_srt, g_take, g_voice, g_text, g_lang, g_veng, g_emo,
                                 g_auto, g_speed, g_engine, g_subs, g_aspect, g_bg, g_bgcolor, g_bgimg, g_refine,
                                 g_batch, g_idle, g_steps, g_guid, g_seed, g_cfg, g_expr, g_head, g_keep],
                   [g_out, g_sum, rid_state, g_hist, g_file], concurrency_limit=1)
        g_exp.click(on_export, rid_state, g_exp_msg)

        a_vbtn.click(on_create_video, [a_vname, a_video, a_start, a_end], [g_avatar, a_pick, a_gallery, a_info],
                     concurrency_limit=1)
        a_pbtn.click(on_create_photo, [a_pname, a_photo], [g_avatar, a_pick, a_gallery, a_info], concurrency_limit=1)
        a_pick.change(on_pick_manage, a_pick, [a_info, a_pv, a_pp, a_name, a_face])
        a_face.input(on_set_face, [a_pick, a_face], [a_info, a_pp, a_gallery])
        a_ren.click(on_rename, [a_pick, a_name], [g_avatar, a_pick, a_gallery])
        a_del.click(on_delete, a_pick, [g_avatar, a_pick, a_gallery])
        a_rep.click(on_reprepare, [a_pick, a_margin, a_mode], [g_avatar, a_pick, a_info], concurrency_limit=1)

        st_btn.click(on_save_settings, [st_vs, st_dr, st_max, st_crf], [st_msg, e_md])
        gr.Timer(5).tick(gpu_md, None, gpu_bar)
        demo.load(on_pick_avatar, g_avatar, [g_cover, g_info, g_engine, g_refine])
        demo.load(on_pick_take, g_take, [g_take_audio, g_take_info])
        demo.load(on_pick_manage, a_pick, [a_info, a_pv, a_pp, a_name, a_face])
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("DH_PORT", "7870"))
    build().queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1", server_port=port, inbrowser="--no-browser" not in sys.argv,
        allowed_paths=[config.OUTPUTS, config.AVATARS, config.voice_studio_dir() or config.OUTPUTS])
