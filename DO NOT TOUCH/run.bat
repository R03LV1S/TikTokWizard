@echo off
cd /d "%~dp0"
title TikTok Wizard

echo =====================================================================
echo                    TikTok Wizard Launcher
echo =====================================================================
echo.

:: -----------------------------------------------------------------------
:: 1. Scan for Python
:: -----------------------------------------------------------------------
set PY_CMD=
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PY_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PY_CMD=py
    )
)

if defined PY_CMD (
    echo [OK] Python is already installed.
) else (
    echo [!] Python not detected. Checking for Windows Package Manager...
    where winget >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [*] Installing Python 3.11 automatically via winget...
        winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
        set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
        set PY_CMD=python
    ) else (
        echo.
        echo =====================================================================
        echo [ERROR] Python 3.10+ is required to run TikTok Wizard.
        echo Please download and install Python from: https://www.python.org/
        echo (Make sure to check "Add Python to PATH" during installation)
        echo =====================================================================
        echo.
        pause
        exit /b 1
    )
)

:: -----------------------------------------------------------------------
:: 2. Scan for FFmpeg
:: -----------------------------------------------------------------------
set FFMPEG_OK=0
if exist "%~dp0bin\ffmpeg.exe" (
    set "PATH=%~dp0bin;%PATH%"
    set FFMPEG_OK=1
) else (
    where ffmpeg >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set FFMPEG_OK=1
    )
)

if "%FFMPEG_OK%"=="1" (
    echo [OK] FFmpeg is already installed.
) else (
    echo [!] FFmpeg not detected. Checking for Windows Package Manager...
    where winget >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [*] Installing FFmpeg automatically via winget...
        winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
        where ffmpeg >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            set FFMPEG_OK=1
            echo [OK] FFmpeg installed successfully.
        )
    ) else (
        echo [!] Notice: FFmpeg was not detected. Video rendering requires FFmpeg.
    )
)

:: -----------------------------------------------------------------------
:: 3. Setup Virtual Environment & Dependencies (Skip if already verified)
:: -----------------------------------------------------------------------
if exist "%~dp0venv\Scripts\python.exe" (
    if not exist "%~dp0venv\.install-complete" (
        "%~dp0venv\Scripts\python.exe" -c "import faster_whisper, audio_separator, torch, torchaudio, transformers, scipy, librosa" >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            echo TikTok Wizard verified existing environment>"%~dp0venv\.install-complete"
            echo [OK] Verified existing Python environment & dependencies.
        )
    ) else (
        echo [OK] Existing Python environment ready.
    )
)

if not exist "%~dp0venv\.install-complete" (
    echo.
    echo [*] First-time setup: Creating isolated Python environment & installing packages...
    echo [*] Existing cached models/stems will be reused automatically.
    echo.
    %PY_CMD% "%~dp0installer.py"
    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] Setup failed. Please check your internet connection and try again.
        pause
        exit /b 1
    )
)
call "%~dp0venv\Scripts\activate.bat"

:: -----------------------------------------------------------------------
:: 4. Launch TikTok Wizard CLI
:: -----------------------------------------------------------------------
python "%~dp0src\cli.py"
if %ERRORLEVEL% neq 0 (
    echo.
    echo TikTok Wizard closed.
    pause
)
