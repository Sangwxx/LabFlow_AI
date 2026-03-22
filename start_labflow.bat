@echo off
setlocal
cd /d "%~dp0"

echo [LabFlow] 准备启动项目...

if exist ".venv\Scripts\python.exe" goto VENV_READY

echo [LabFlow] 未检测到本地虚拟环境，开始创建 .venv
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
    echo [LabFlow] 虚拟环境创建失败，请确认本机已安装 Python 3.10+
    pause
    exit /b 1
)

:VENV_READY
echo [LabFlow] 安装运行依赖...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo [LabFlow] 已自动生成 .env，请按需补充 API_KEY
    )
)

echo [LabFlow] 启动完成后请访问 http://127.0.0.1:8501
echo [LabFlow] 关闭当前窗口即可停止服务

".venv\Scripts\python.exe" run_labflow.py
