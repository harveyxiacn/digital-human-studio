# 数字人工坊 · Digital Human Studio

本地运行的数字人视频生成工具：用**你自己的视频或照片**，加上**你自己的声音**，生成口型、表情、头部动作同步的数字人口播视频，可以直接导出为剪映草稿继续剪辑。

- 🖥️ **全部在本机运行**：人脸和声音都不上传，不按月付费，形象不会过期
- 🎙️ **和[声音工坊](https://github.com/harveyxiacn/voice-studio)联动**：直接选用它克隆出来的普通话 / 粤语语音和字幕；也能在这里输入文字，调用声音工坊朗读后生成视频；两个软件共用显卡队列，不会同时抢显存
- 🧑 **视频形象**（推荐）：录一段 10–60 秒不说话的视频，按新的声音重做嘴型；身体、手势、眨眼都来自真人视频，最自然
- 🖼️ **照片形象**：一张照片就能说话，带点头、眨眼和表情
- ✂️ **后期**：竖屏 / 横屏裁切、烧录字幕、绿幕 / 纯色 / 图片背景、透明背景 .mov
- 📤 **一键导出剪映草稿**（视频轨 + 可编辑字幕轨）
- 🎯 面向 **8GB 显存**的 NVIDIA 显卡（在 RTX 3070 Laptop 上开发和实测）

## 引擎

| 引擎 | 适用 | 特点 |
|---|---|---|
| [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) | 视频形象 | 快，画质好，默认引擎 |
| [LatentSync 1.5](https://github.com/bytedance/LatentSync) | 视频形象 | 扩散模型，口型更准，较慢 |
| [JoyVASA](https://github.com/jdh-algo/JoyVASA) + [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | 照片形象 | 由声音生成点头、眨眼、表情；可再用 MuseTalk 精修口型 |

这些模型都是按**声音**驱动嘴型（而不是按文字），粤语也能用。选型理由和实测数据见 [设计说明.md](设计说明.md)。

**实测**（RTX 3070 Laptop 8GB）：

| 任务 | 速度 | 显存峰值 |
|---|---|---|
| 视频形象预处理（一次性） | 10 秒视频约 70 秒 | 3.0GB |
| MuseTalk 生成 | 每秒视频约 2.5 秒 | 3.4GB |
| LatentSync 生成 | 每秒视频约 15 秒 | 5.4GB |
| JoyVASA 照片生成 | 每秒视频约 4 秒 | 1.9GB |

## 安装

需要：Windows 10/11、NVIDIA 显卡（建议 8GB 显存以上）、[Git](https://git-scm.com/download/win)。约占 25GB 硬盘（模型约 11GB）。

1. 下载本项目（`git clone` 或下载 ZIP 解压），最好放在声音工坊旁边（例如 `E:\jianying\digital_human` 和 `E:\jianying\voice_clone`），会自动找到它。
2. 双击 **安装.bat**。会自动安装 [uv](https://docs.astral.sh/uv/)、两个 Python 环境、上游模型代码（固定版本）和模型权重。中途断网可以重新运行，会从断点继续。
   - 不需要 LatentSync 可以省 5GB：`powershell -ExecutionPolicy Bypass -File install.ps1 -SkipLatentSync`
3. 双击 **启动数字人工坊.bat**，浏览器会打开 http://127.0.0.1:7870 。

## 使用

1. **形象库**：上传一段视频（或一张照片）建立形象。视频形象会做一次预处理（检测人脸、计算遮罩），之后每次生成都很快。
2. **生成**：选形象 → 选声音（上传音频 / 声音工坊作品 / 输入文字让声音工坊朗读）→ 选引擎和画幅、字幕、背景 → 生成。
3. 成片在 `outputs/`；点「导出为剪映草稿」，打开剪映就能在草稿列表里看到。

详细说明和拍摄建议见 [使用说明.md](使用说明.md)。

### 命令行

```powershell
.venv\Scripts\python app\cli.py add-video 正装讲解 素材.mp4
.venv\Scripts\python app\cli.py avatars
.venv\Scripts\python app\cli.py render <形象ID> 配音.wav --srt 配音.srt --aspect 竖屏9:16 --jianying
.venv\Scripts\python app\cli.py render <形象ID> --take <声音工坊作品ID>
```

## 注意

- 生成时请关闭剪映等占显存的软件（剪映开着时约占 6GB 显存），软件会检测并提示。
- 请只用你本人、或已获得本人授权的肖像和声音。生成的视频请标注「AI 生成」。不要用于冒充他人、诈骗或传播虚假信息。
- 上游模型各有许可：MuseTalk（MIT）、LatentSync（Apache-2.0）、JoyVASA（MIT）、LivePortrait（MIT）；其中人脸检测用到的 InsightFace 模型仅限非商业用途。商用前请自行确认各模型的许可。

## 许可

本项目代码使用 MIT 许可。
