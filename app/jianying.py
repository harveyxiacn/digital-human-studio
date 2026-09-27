# -*- coding: utf-8 -*-
"""导出剪映草稿：数字人视频放在视频轨（自带声音），字幕作为可编辑的文本轨。打开剪映就能在草稿列表里看到。"""
import os
import re
import time

import config
import media


def export(render_meta, draft_name=None, drafts_dir=None, with_subs=True):
    import pyJianYingDraft as draft
    from pyJianYingDraft import SEC, Timerange

    drafts_dir = drafts_dir or config.settings()["jianying_drafts"]
    if not drafts_dir or not os.path.isdir(drafts_dir):
        raise ValueError(f"找不到剪映草稿目录：{drafts_dir}\n请在剪映「设置 → 草稿位置」查看，并填到本软件的「设置」页")
    video = render_meta["out"]
    info = media.probe(video)
    name = draft_name or re.sub(r'[\\/:*?"<>|]', "", f"数字人_{render_meta['avatar_name']}_{time.strftime('%m%d_%H%M%S')}")
    folder = draft.DraftFolder(drafts_dir)
    script = folder.create_draft(name, info["width"], info["height"], 25, allow_replace=True)
    track = script.append_track(draft.TrackSpec(draft.TrackType.video, "数字人"))
    mat = draft.VideoMaterial(video, "数字人")
    dur = min(mat.duration, round(info["duration"] * SEC))
    script.add_segment(draft.VideoSegment(mat, Timerange(0, dur)), track)
    srt = render_meta.get("srt")
    if with_subs and srt and os.path.exists(srt) and not render_meta.get("subtitles"):
        script.import_srt(srt, "字幕")
    script.save()
    return os.path.join(drafts_dir, name)
