@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 还没有安装，请先双击「安装.bat」
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app\studio.py %*
pause
