# -*- coding: utf-8 -*-
"""在「声音工坊」自己的 Python 环境里运行（由 voice_bridge.py 启动），调用它的语音包和作品模块。
注意：声音工坊用 Python 3.9，这个文件的语法要兼容 3.9。

用法：python vs_runner.py <声音工坊目录> voices
      python vs_runner.py <声音工坊目录> synth <参数 JSON 文件>
结果以一行 @@RESULT@@ {...} 输出。
"""
import json
import os
import sys
import traceback


def main():
    vs, cmd = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.join(vs, "app"))
    import core  # noqa: E402

    if cmd == "voices":
        out = []
        for label, vid in core.list_voices():
            v = core.load_voice(vid)
            out.append({"id": vid, "label": label, "langs": v.get("langs", [])})
        return {"ok": True, "voices": out}

    if cmd == "synth":
        import engines as ENG
        import gpu
        import takes
        with open(sys.argv[3], encoding="utf-8") as f:
            a = json.load(f)
        v = core.load_voice(a["vid"])
        yue = a.get("lang") == "yue" or (a.get("lang") in (None, "auto") and v.get("langs") == ["yue"])
        engine = a.get("engine") or "VoxCPM2"
        params = dict(lang_label="粤语（可夹英文）" if yue else "普通话（可夹英文）",
                      default_emo=a.get("emotion") or "平静", auto_emo=bool(a.get("auto_emo", True)),
                      expr=float(a.get("expr", 0.6)), speed=float(a.get("speed", 1.0)),
                      pause=float(a.get("pause", 0.3)), seed=int(a.get("seed", -1)), multi=True, strong=False,
                      use_lora=True)

        def log(m):
            print("@@LOG@@ " + str(m), flush=True)

        def on_wait(pos, cur):
            log("显卡正忙（%s），排队中：第 %d 位" % (cur, pos))

        try:
            with gpu.use("数字人·生成语音（%s）" % engine, on_wait=on_wait):
                meta = takes.create(engine, a["vid"], a["text"], params, log)
        finally:
            ENG.release_all()  # 释放声音模型占用的显存，接下来数字人引擎要用
        return {"ok": True, "take": meta["id"], "wav": meta["wav"], "srt": meta["srt"]}

    return {"ok": False, "error": "未知命令 " + cmd}


if __name__ == "__main__":
    try:
        r = main()
    except Exception as e:
        traceback.print_exc()
        r = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    print("\n@@RESULT@@ " + json.dumps(r, ensure_ascii=False), flush=True)
