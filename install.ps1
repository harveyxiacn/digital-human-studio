# 数字人工坊 · 一键安装（Windows + NVIDIA 显卡）
# 每一步都会先检查是否已完成，重复运行是安全的（已装好的会跳过，下载会断点续传）。
# 用法：双击 安装.bat，或：powershell -ExecutionPolicy Bypass -File install.ps1
# 可选参数：-SkipLatentSync（不下载 LatentSync 的 5GB 模型）

param([switch]$SkipLatentSync)

$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
$Eng = Join-Path $Root "engines"
$EnvDir = Join-Path $Eng "avatar_env"
$Py = Join-Path $EnvDir "Scripts\python.exe"
$MainPy = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
# 国内镜像（也可以改成官方源）
$PyPI = "https://mirrors.aliyun.com/pypi/simple/"
$TorchCu121 = "https://mirrors.aliyun.com/pytorch-wheels/cu121/"

# 上游项目（固定版本，保证可复现）
$Upstream = @(
    @{Dir = "musetalk_src";   Url = "https://github.com/TMElyralab/MuseTalk";  Sha = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"},
    @{Dir = "joyvasa_src";    Url = "https://github.com/jdh-algo/JoyVASA";     Sha = "916a90f8de490e8648fee460c1200bd5d9a795af"},
    @{Dir = "latentsync_src"; Url = "https://github.com/bytedance/LatentSync";  Sha = "a229c3948406bc2cf6eaf4873e662e70c6a04746"}
)

function Step($msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Done($msg) { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "缺少 $cmd：$hint" }
}

Step "0. 检查环境"
Need "nvidia-smi" "需要 NVIDIA 显卡和驱动（建议显存 ≥ 8GB）"
Need "git" "请先安装 Git：https://git-scm.com/download/win"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  未找到 uv，正在安装…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    Need "uv" "请手动安装 uv：https://docs.astral.sh/uv/"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
Done "环境检查通过"

Step "1. 网页主程序（Python 3.11）"
if (-not (Test-Path $MainPy)) { uv venv (Join-Path $Root ".venv") --python 3.11 }
$env:VIRTUAL_ENV = Join-Path $Root ".venv"
uv pip install -r (Join-Path $Root "requirements.txt") --index-url $PyPI
if ($LASTEXITCODE -ne 0) { throw "主程序依赖安装失败" }
Remove-Item Env:VIRTUAL_ENV
# comtypes（剪映草稿库的依赖）在 uv 环境里缺少 gen 目录，导入时会报错
$gen = Join-Path $Root ".venv\Lib\site-packages\comtypes\gen"
if (-not (Test-Path $gen)) { New-Item -ItemType Directory -Force $gen | Out-Null; New-Item -ItemType File (Join-Path $gen "__init__.py") | Out-Null }
Done "主程序就绪"

Step "2. 上游模型代码"
foreach ($u in $Upstream) {
    $d = Join-Path $Eng $u.Dir
    if (-not (Test-Path (Join-Path $d ".git"))) {
        git clone -q $u.Url $d
        if ($LASTEXITCODE -ne 0) { throw "克隆失败：$($u.Url)" }
    }
    git -c safe.directory='*' -C $d checkout -q $u.Sha
    Done "$($u.Dir) @ $($u.Sha.Substring(0,7))"
}

Step "3. 引擎环境（Python 3.10 + PyTorch 2.3.1 CUDA 12.1）"
if (-not (Test-Path $Py)) { uv venv $EnvDir --python 3.10 }
$env:VIRTUAL_ENV = $EnvDir
$ok = & $Py -c "import torch; print(torch.__version__.startswith('2.3.1') and torch.cuda.is_available())" 2>$null
if ($ok -ne "True") {
    uv pip install "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "torchaudio==2.3.1+cu121" --index-url $PyPI --find-links $TorchCu121 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw "PyTorch 安装失败（下载超时可以重新运行，会继续）" }
}
# 其他依赖（同时写明 torch 版本，防止被某个依赖换成 CPU 版）
uv pip install -r (Join-Path $Eng "requirements.txt") "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "torchaudio==2.3.1+cu121" --index-url $PyPI --find-links $TorchCu121 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw "引擎依赖安装失败" }
Remove-Item Env:VIRTUAL_ENV
$ok = & $Py -c "import torch, diffusers, face_alignment, mediapipe; print(torch.cuda.is_available())"
if ($ok -ne "True") { throw "引擎环境检查失败：PyTorch 无法使用显卡" }
Done "引擎环境就绪"

Step "4. ffmpeg（带字幕渲染）"
$bin = Join-Path $Eng "bin"
New-Item -ItemType Directory -Force $bin | Out-Null
if (-not (Test-Path (Join-Path $bin "ffmpeg.exe"))) {
    $src = & $Py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    Copy-Item $src (Join-Path $bin "ffmpeg.exe")
}
Done "ffmpeg 就绪"

Step "5. 下载模型"
$targets = @("musetalk", "joyvasa")
if (-not $SkipLatentSync) { $targets += "latentsync" }
& $MainPy (Join-Path $Root "app\download_models.py") @targets
if ($LASTEXITCODE -ne 0) { throw "模型下载没有完成，请重新运行本脚本（会从断点继续）" }
Done "模型就绪"

Step "安装完成"
Write-Host "双击「启动数字人工坊.bat」打开网页（http://127.0.0.1:7870）" -ForegroundColor Green
