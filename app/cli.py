# -*- coding: utf-8 -*-
"""命令行：适合批量生成和自动化。

  python app/cli.py avatars                               列出形象
  python app/cli.py add-video 名称 视频.mp4 [--start 秒 --end 秒]
  python app/cli.py add-photo 名称 照片.jpg
  python app/cli.py render 形象ID 音频.wav [--engine musetalk|latentsync|joyvasa] [--srt 字幕.srt]
                    [--aspect 竖屏9:16] [--bg 绿幕] [--refine] [--jianying]
  python app/cli.py render 形象ID --take 声音工坊作品ID ...
  python app/cli.py takes                                 列出声音工坊作品
  python app/cli.py workflow 素材1.mp4 照片.jpg 配音1.wav 配音2.wav [配音1.srt]
                    [--mode mix|per_material|per_audio] [--aspect 竖屏9:16] [--subs auto|srt|none] [--hq] [--jianying]
                    一键工作流：素材里的人说出这些音频（文件按扩展名自动归类，按文件名排序）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import avatars  # noqa: E402
import jianying  # noqa: E402
import media  # noqa: E402
import render as R  # noqa: E402
import voice_bridge as VB  # noqa: E402
import workflow as WF  # noqa: E402


def bar(p, msg=""):
    n = int(p * 30)
    sys.stdout.write(f"\r[{'#' * n}{'.' * (30 - n)}] {p * 100:5.1f}% {msg[:60]:<60}")
    sys.stdout.flush()


def wait(pos, cur):
    sys.stdout.write(f"\r显卡正忙（{cur}），排队第 {pos} 位…{' ' * 30}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="数字人工坊 命令行")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("avatars")
    sub.add_parser("takes")
    a = sub.add_parser("add-video")
    a.add_argument("name")
    a.add_argument("video")
    a.add_argument("--start", type=float)
    a.add_argument("--end", type=float)
    a = sub.add_parser("add-photo")
    a.add_argument("name")
    a.add_argument("image")
    a = sub.add_parser("render")
    a.add_argument("avatar")
    a.add_argument("audio", nargs="?")
    a.add_argument("--take")
    a.add_argument("--engine", choices=list(R.ENGINE_LABELS))
    a.add_argument("--srt")
    a.add_argument("--no-subs", action="store_true")
    a.add_argument("--aspect", default="原始", help="原始 / 竖屏9:16 / 横屏16:9 / 方形1:1")
    a.add_argument("--bg", default="原样", help="原样 / 绿幕 / 纯色 / 图片背景 / 透明")
    a.add_argument("--bg-color", default="#00B140")
    a.add_argument("--bg-image")
    a.add_argument("--refine", action="store_true", help="照片形象：再用 MuseTalk 精修口型")
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--cfg", type=float, default=2.8)
    a.add_argument("--head", type=float, default=1.0)
    a.add_argument("--expr", type=float, default=1.0)
    a.add_argument("--official-lip", action="store_true", help="照片形象：不保留原表情（先合嘴再驱动）")
    a.add_argument("--jianying", action="store_true", help="同时导出剪映草稿")
    a = sub.add_parser("workflow")
    a.add_argument("files", nargs="+")
    a.add_argument("--mode", default="mix", choices=["mix", "per_material", "per_audio"])
    a.add_argument("--aspect", default="竖屏9:16", help="原始 / 竖屏9:16 / 横屏16:9 / 方形1:1")
    a.add_argument("--subs", default="auto", choices=["auto", "srt", "none"])
    a.add_argument("--no-burn", action="store_true", help="不烧录字幕，只输出 .srt")
    a.add_argument("--shot", type=float, default=6.0, help="多个素材时每个镜头大约几秒")
    a.add_argument("--hq", action="store_true", help="视频素材用 LatentSync（慢约 6 倍）")
    a.add_argument("--bg", default="原样")
    a.add_argument("--no-sort", action="store_true", help="按命令行顺序，不按文件名排序")
    a.add_argument("--jianying", action="store_true")
    args = ap.parse_args()

    if args.cmd == "avatars":
        for m in avatars.all_avatars():
            print(f"{m['id']}\t{m['name']}\t{avatars.KINDS[m['kind']]}\t{'就绪' if m.get('ready') else '未预处理'}")
    elif args.cmd == "takes":
        for t in VB.takes():
            print(f"{t['id']}\t{t['label']}")
    elif args.cmd == "add-video":
        m = avatars.create_video(args.name, args.video, args.start, args.end, bar, wait)
        print(f"\n已建立：{m['id']}\n{avatars.describe(m)}")
    elif args.cmd == "add-photo":
        m = avatars.create_photo(args.name, args.image)
        print(f"已建立：{m['id']}")
    elif args.cmd == "workflow":
        import re
        mats, auds, srts, bad = WF.classify(args.files)
        if bad:
            print("忽略不支持的文件：" + "、".join(bad))
        if not args.no_sort:
            nat = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p).lower())]
            mats.sort(key=lambda m: nat(m[0]))
            auds.sort(key=nat)
        aspect = next((k for k in ["原始"] + list(WF.SIZES) if k.replace(" ", "") == args.aspect.replace(" ", "")), "原始")
        bg = next((k for k in R.BG_MODES if k.startswith(args.bg)), "原样")
        rs = WF.run(mats, auds, srts, mode=args.mode, quality="hq" if args.hq else "fast", shot_len=args.shot,
                    subs=args.subs, burn_subs=not args.no_burn, aspect=aspect, bg_mode=bg, on_progress=bar, on_wait=wait)
        print("\n" + WF.summary(rs).replace("**", "").replace("`", ""))
        if args.jianying:
            for m in rs:
                print("剪映草稿：" + jianying.export(m))
    elif args.cmd == "render":
        aspect = next((k for k in media.ASPECTS if k.replace(" ", "") == args.aspect.replace(" ", "")), "原始")
        bg = next((k for k in R.BG_MODES if k.startswith(args.bg)), "原样")
        m = R.render(args.avatar, audio=args.audio, take_id=args.take, engine=args.engine, srt=args.srt,
                     subtitles=not args.no_subs, aspect=aspect, bg_mode=bg, bg_color=args.bg_color,
                     bg_image=args.bg_image, refine=args.refine,
                     opts={"steps": args.steps, "cfg_scale": args.cfg, "head_motion": args.head,
                           "expression": args.expr, "keep_expression": not args.official_lip}, on_progress=bar, on_wait=wait)
        print("\n" + R.summary(m).replace("**", "").replace("`", ""))
        if args.jianying:
            print("剪映草稿：" + jianying.export(m))


if __name__ == "__main__":
    main()
