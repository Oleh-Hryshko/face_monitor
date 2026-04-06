@echo off
chcp 65001 >nul
title FACE MONITOR GUI

:: Set environment variable for TensorFlow/Keras compatibility
set TF_USE_LEGACY_KERAS=1

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ⚠️  Virtual environment not found!
    echo.
    echo Please run first:
    echo   clear_and_install.bat
    echo.
    pause
    exit /b
)

:: Run the GUI application using venv
echo 🚀 Starting Face Monitor GUI...
echo.
venv\Scripts\python  -m app.launcher

if %errorlevel% neq 0 (
    echo.
    echo ❌ Application exited with error code %errorlevel%
    echo.
    pause
)