@echo off
chcp 65001 >nul
rem 读取项目根目录 credentials.json 的一键登录链接，用默认浏览器打开（免手输令牌）
set "ROOT=%~dp0.."

rem 优先用虚拟环境的 python，否则用系统 python
set "PY="
if exist "%ROOT%\env\python.exe" set "PY=%ROOT%\env\python.exe"
if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"
if "%PY%"=="" set "PY=python"

for /f "usebackq delims=" %%i in (`"%PY%" -c "import json,os;print(json.load(open(os.path.join(r'%ROOT%','credentials.json'),encoding='utf-8'))['quick_login_url'])"`) do set "URL=%%i"

if "%URL%"=="" (
  echo 没找到 credentials.json，请先运行 scripts\start.bat 启动一次服务。
  pause
  exit /b 1
)

echo 正在打开: %URL%
start "" "%URL%"
