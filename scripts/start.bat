@echo off
chcp 936 >nul
title VoxCPM2 本地推理服务

rem ============================================================
rem  VoxCPM2 一键启动（通用版）
rem  本脚本位于 scripts/ 目录，项目根目录为上一级。
rem  优先使用项目根目录下的虚拟环境 python，否则使用系统 python。
rem ============================================================

rem 定位项目根目录（scripts 的上一级）
set "ROOT=%~dp0.."

rem 端口 / 监听地址（局域网访问改 0.0.0.0）
set "VOXCPM_PORT=8808"
set "VOXCPM_HOST=127.0.0.1"

rem 优先使用项目根目录下的虚拟环境
set "PY="
if exist "%ROOT%\env\python.exe"   set "PY=%ROOT%\env\python.exe"
if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"

if "%PY%"=="" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10-3.12 并安装依赖。
    echo 参考 README.md 的「安装步骤」。
    pause
    exit /b 1
  )
  set "PY=python"
)

if not exist "%ROOT%\server.py" (
  echo [错误] 未找到 server.py，请确认在正确目录运行。
  pause
  exit /b 1
)

rem 离线加载本地权重（若权重已就位）
set "HF_HUB_OFFLINE=1"

echo.
echo 正在启动 VoxCPM2 服务 ...
echo 启动完成后浏览器访问 http://localhost:%VOXCPM_PORT%
echo 访问令牌见项目根目录的 credentials.json（首次启动自动生成）
echo 关闭本窗口即停止服务。
echo.

"%PY%" "%ROOT%\server.py"

echo.
echo 服务已停止。
pause
