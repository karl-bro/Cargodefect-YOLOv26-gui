@echo off
chcp 65001 >nul
title 货件缺陷检测系统

:: ============================================================
:: CargoDefect-YOLOv26-Detect 一键启动脚本
:: 自动创建虚拟环境、安装依赖、启动 GUI
:: ============================================================

cd /d "%~dp0"

:: 提示：不要从其他电脑复制 venv，应在本机重新创建
if exist "venv\Scripts\python.exe" (
    echo [INFO] 检测到已有虚拟环境，直接启动...
) else (
    echo [INFO] 未检测到虚拟环境，开始首次安装...
)

:: 查找可用的 Python 解释器
set "PYTHON_EXE="

:: 1) 检查当前目录的 venv
if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"

:: 2) 检查系统 Python
if "%PYTHON_EXE%"=="" (
    where python >nul 2>&1
    if %errorlevel% equ 0 set "PYTHON_EXE=python"
)

:: 3) 检查 Python3
if "%PYTHON_EXE%"=="" (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 set "PYTHON_EXE=python3"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo         下载地址: https://www.python.org/downloads/
    echo         安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [INFO] 使用 Python: %PYTHON_EXE%

:: ── 如果没有 venv，则创建 ──
if not exist "venv\Scripts\python.exe" (
    echo.
    echo [STEP 1/4] 创建虚拟环境...
    %PYTHON_EXE% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] 创建虚拟环境失败
        pause
        exit /b 1
    )
    set "PYTHON_EXE=venv\Scripts\python.exe"

    echo.
    echo [STEP 2/4] 升级 pip...
    %PYTHON_EXE% -m pip install --upgrade pip -q

    echo.
    echo [STEP 3/4] 安装基础依赖（可能需要几分钟）...
    %PYTHON_EXE% -m pip install -r requirements_gui.txt

    echo.
    echo [STEP 4/4] 安装自定义 Ultralytics（CargoDefect-YOLOv26）...
    %PYTHON_EXE% -m pip install --force-reinstall --no-deps "ultralytics-main"

    echo.
    echo [INFO] 环境初始化完成！
    echo.
)

:: ── 启动应用 ──
echo [INFO] 启动货件缺陷检测系统...
%PYTHON_EXE% app.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 程序异常退出，请检查上方的错误信息。
    pause
)
