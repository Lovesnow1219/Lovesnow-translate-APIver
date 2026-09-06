@echo off
setlocal
cd /d "%~dp0"
title Lovesnow-translate
chcp 65001 >nul
set PYTHONNOUSERSITE=1
set PYTHONPATH=%~dp0
set PATH=%~dp0runtime\python;%~dp0runtime\python\Scripts;%~dp0runtime\ffmpeg;%PATH%

if not exist "%~dp0runtime\python\python.exe" (
    echo 找不到內建 Python。請用完整的便攜包，或再跑一次打包腳本。
    pause
    exit /b 1
)
if not exist "%~dp0runtime\ffmpeg\ffmpeg.exe" (
    echo 找不到內建 ffmpeg。請用完整的便攜包。
    pause
    exit /b 1
)

echo 正在啟動 Lovesnow-translate …
"%~dp0runtime\python\python.exe" "%~dp0launch_webui.py"
if errorlevel 1 (
    echo.
    echo 啟動失敗。把上面的紅字截給傳包的人。
    pause
)
endlocal
