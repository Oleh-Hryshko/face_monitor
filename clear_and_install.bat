@echo off
chcp 65001 >nul
title INSTALL - FACE MONITOR GUI (Python 3.12)
echo ========================================
echo    INSTALLATION FOR PYTHON 3.12
echo         with GUI support (PySide6)
echo ========================================
echo.

:: Убеждаемся, что мы в правильной директории
cd /d "%~dp0"
echo Current directory: %CD%
echo.

:: Проверка прав администратора
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  Warning: Not running as administrator
    echo    If you get permission errors, right-click and select
    echo    "Run as administrator"
    echo.
    choice /c YN /m "Continue anyway"
    if errorlevel 2 exit /b
)
echo.

:: Step 1: Remove everything with force
echo [1/6] Cleaning up old installation...

if exist "venv" (
    echo   Removing venv...
    rmdir /s /q venv 2>nul
    if exist "venv" (
        echo   ⚠️  Force removing...
        takeown /f venv /r /d y >nul 2>&1
        icacls venv /grant %USERNAME%:F /t >nul 2>&1
        rmdir /s /q venv 2>nul
    )
    if not exist "venv" echo   ✅ Old venv removed
)

if exist "__pycache__" rmdir /s /q __pycache__ 2>nul
del /s /q *.pyc >nul 2>&1
echo.

:: Step 2: Create fresh virtual environment
echo [2/6] Creating fresh virtual environment...
set "VENV_PATH=%CD%\venv"

python -m venv "%VENV_PATH%" --clear
if errorlevel 1 (
    echo   ❌ Failed to create virtual environment!
    echo   Make sure Python 3.12 is installed and in PATH
    pause
    exit /b
)
echo   ✅ Virtual environment created
echo.

:: Step 3: Upgrade pip and install build tools
echo [3/6] Upgrading pip and installing build tools...
"%VENV_PATH%\Scripts\python" -m pip install --upgrade pip setuptools wheel
echo   ✅ Pip and build tools upgraded
echo.

:: Step 4: Install packages for Python 3.12
echo [4/6] Installing packages for Python 3.12...
echo.

:: Устанавливаем numpy 1.26.4 (совместим с Python 3.12)
echo   Installing numpy 1.26.4...
"%VENV_PATH%\Scripts\pip" install numpy==1.26.4
if errorlevel 1 (
    echo     ❌ NumPy 1.26.4 failed, trying without cache...
    "%VENV_PATH%\Scripts\pip" install --no-cache-dir numpy==1.26.4
    if errorlevel 1 (
        echo     ❌ NumPy installation failed!
        pause
        exit /b
    )
)
echo     ✅ NumPy 1.26.4

echo   Installing PySide6 (Qt for Python)...
"%VENV_PATH%\Scripts\pip" install PySide6==6.6.2
if errorlevel 1 (
    echo     ❌ PySide6 failed
    pause
    exit /b
)
echo     ✅ PySide6 6.6.2

echo   Installing opencv...
"%VENV_PATH%\Scripts\pip" install opencv-python==4.10.0.84
if errorlevel 1 (
    echo     ❌ OpenCV failed
    pause
    exit /b
)
echo     ✅ OpenCV 4.10.0

echo   Installing pandas...
"%VENV_PATH%\Scripts\pip" install pandas==2.2.3
if errorlevel 1 (
    echo     ❌ Pandas failed
    pause
    exit /b
)
echo     ✅ Pandas 2.2.3

:: TensorFlow 2.19.0 совместим с Python 3.12
echo   Installing TensorFlow 2.19.0...
"%VENV_PATH%\Scripts\pip" install tensorflow==2.19.0
if errorlevel 1 (
    echo     ❌ TensorFlow failed
    pause
    exit /b
)
echo     ✅ TensorFlow 2.19.0

:: Устанавливаем tf-keras для совместимости
echo   Installing tf-keras...
"%VENV_PATH%\Scripts\pip" install tf-keras
echo     ✅ tf-keras

echo   Installing mss...
"%VENV_PATH%\Scripts\pip" install mss
if errorlevel 1 (
    echo     ❌ mss install failed, trying without cache...
    "%VENV_PATH%\Scripts\pip" install --no-cache-dir mss
    if errorlevel 1 (
        echo     ❌ mss installation failed!
        pause
        exit /b
    )
)
echo     ✅ mss installed

echo   Installing Pillow (PIL)...
"%VENV_PATH%\Scripts\pip" install Pillow==10.4.0
if errorlevel 1 (
    echo     ❌ Pillow failed, trying without cache...
    "%VENV_PATH%\Scripts\pip" install --no-cache-dir Pillow==10.4.0
    if errorlevel 1 (
        echo     ❌ Pillow installation failed!
        pause
        exit /b
    )
)
echo     ✅ Pillow 10.4.0

:: DeepFace 0.0.79
echo   Installing DeepFace...
"%VENV_PATH%\Scripts\pip" install deepface==0.0.79
if errorlevel 1 (
    echo     ❌ DeepFace failed
    pause
    exit /b
)
echo     ✅ DeepFace 0.0.79

echo   Installing face detectors...
"%VENV_PATH%\Scripts\pip" install mtcnn retina-face
echo     ✅ MTCNN + RetinaFace

echo   Installing utilities...
"%VENV_PATH%\Scripts\pip" install gdown requests tqdm
echo     ✅ Utilities
echo.

:: Step 5: Set environment variable
echo [5/6] Setting compatibility...
setx TF_USE_LEGACY_KERAS 1 >nul 2>&1
echo   ✅ Environment variable set
echo.

:: Step 6: Create launcher scripts
echo [6/6] Creating launcher scripts...

:: diagnostic.bat
(
echo @echo off
echo chcp 65001 ^>nul
echo echo ========================================
echo echo    DIAGNOSTIC TOOL (GUI Version)
echo echo ========================================
echo echo.
echo set "TF_USE_LEGACY_KERAS=1"
echo echo Python version:
echo "%VENV_PATH%\Scripts\python" --version
echo echo.
echo echo Installed packages:
echo "%VENV_PATH%\Scripts\pip" list ^| findstr /i "numpy pandas opencv PySide6 deepface tensorflow tf-keras"
echo echo.
echo echo Testing imports:
echo echo.
echo "%VENV_PATH%\Scripts\python" -c "import numpy; print('✅ numpy', numpy.__version__)"
echo "%VENV_PATH%\Scripts\python" -c "import cv2; print('✅ opencv', cv2.__version__)"
echo "%VENV_PATH%\Scripts\python" -c "import PySide6; print('✅ PySide6', PySide6.__version__)"
echo "%VENV_PATH%\Scripts\python" -c "import tensorflow as tf; print('✅ tensorflow', tf.__version__)"
echo "%VENV_PATH%\Scripts\python" -c "from deepface import DeepFace; print('✅ deepface OK')"
echo echo.
echo pause
) > diagnostic.bat

:: launcher.bat
(
echo @echo off
echo chcp 65001 ^>nul
echo title STORE FACE MONITOR GUI
echo.
echo :: Set environment variable for TensorFlow/Keras compatibility
echo set TF_USE_LEGACY_KERAS=1
echo.
echo :: Run the GUI application using venv
echo echo 🚀 Starting Store Face Monitor GUI...
echo echo.
echo "%VENV_PATH%\Scripts\python" -m app.launcher
echo.
echo if %%errorlevel%% neq 0 (
echo     echo.
echo     echo ❌ Application exited with error code %%errorlevel%%
echo     echo.
echo     pause
echo )
) > launcher.bat

echo   ✅ Launcher scripts created
echo.

:: Final verification
echo Verifying installation...
echo.

"%VENV_PATH%\Scripts\python" -c "
import sys
print('=' * 50)
print('VERIFICATION RESULTS')
print('=' * 50)

try:
    import numpy
    print(f'✅ NumPy {numpy.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')

try:
    import cv2
    print(f'✅ OpenCV {cv2.__version__}')
except Exception as e:
    print(f'❌ OpenCV: {e}')

try:
    import PySide6
    print(f'✅ PySide6 {PySide6.__version__}')
except Exception as e:
    print(f'❌ PySide6: {e}')

try:
    import pandas
    print(f'✅ Pandas {pandas.__version__}')
except Exception as e:
    print(f'❌ Pandas: {e}')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow {tf.__version__}')
except Exception as e:
    print(f'❌ TensorFlow: {e}')

print('\n🎯 Testing DeepFace:')
try:
    from deepface import DeepFace
    print('✅ DeepFace imported successfully!')
except Exception as e:
    print(f'❌ DeepFace: {e}')

print('=' * 50)
"

echo.
echo ========================================
echo ✅ INSTALLATION COMPLETE!
echo ========================================
echo.
echo Available commands:
echo   launcher.bat         - Run the GUI program
echo   diagnostic.bat       - Check installation
echo.
pause