@echo off
cd /d "%~dp0"
title TikTok Wizard

:: Maximize console window if run interactively in cmd.exe
if not "%1"=="maximized" (
    start "" /max cmd.exe /c ""%~f0" maximized"
    exit /b
)

echo =====================================================================
echo                    TikTok Wizard Launcher
echo =====================================================================
echo.

:: -----------------------------------------------------------------------
:: 1. Scan for Python
:: -----------------------------------------------------------------------
set "PY_CMD="
call :find_python
if defined PY_CMD goto :python_ready

echo [!] Python not detected. Attempting automatic installation...
where winget >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [*] Installing Python 3.11 via winget...
    winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements --disable-interactivity
    call :refresh_python_path
    call :find_python
) else (
    echo [!] Windows Package Manager ^(winget^) not found in this environment.
    echo [*] Trying direct Python installer download...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $u='https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'; $o=Join-Path $env:TEMP 'python-3.11.9-amd64.exe'; Invoke-WebRequest -Uri $u -OutFile $o -UseBasicParsing; Start-Process -FilePath $o -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 SimpleInstall=1' -Wait -NoNewWindow } catch { exit 1 }"
    call :refresh_python_path
    call :find_python
)

if defined PY_CMD goto :python_ready

echo.
echo =====================================================================
echo [ERROR] Python 3.10+ is required but could not be installed automatically.
echo - If you are in Windows Sandbox, winget/App Installer may be unavailable.
echo - Install Python manually from: https://www.python.org/downloads/
echo   (check "Add python.exe to PATH"), then run this launcher again.
echo =====================================================================
echo.
pause
exit /b 1

:python_ready
echo [OK] Python is available: %PY_CMD%
goto :after_python

:find_python
set "PY_CMD="
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=python"
        exit /b 0
    )
)
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=py -3"
        exit /b 0
    )
)
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" "%ProgramFiles%\Python311\python.exe" "%ProgramFiles%\Python312\python.exe") do (
    if exist %%~P (
        %%~P -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 (
            set "PY_CMD=%%~P"
            exit /b 0
        )
    )
)
exit /b 1

:refresh_python_path
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;%ProgramFiles%\Python311;%ProgramFiles%\Python311\Scripts;%ProgramFiles%\Python312;%ProgramFiles%\Python312\Scripts;%PATH%"
exit /b 0

:after_python

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
:: 3. Setup Virtual Environment & Dependencies
::    installer.py shows a 0-100% progress bar instead of raw pip output.
:: -----------------------------------------------------------------------
if not exist "%~dp0venv\.install-complete" (
    echo.
    echo [*] First-time setup - this runs once and takes a few minutes.
    echo.
    %PY_CMD% "%~dp0installer.py"
    if %ERRORLEVEL% neq 0 (
        echo.
        echo [ERROR] Setup failed. Please try again.
        pause
        exit /b 1
    )
)

:: 4. Launch TikTok Wizard CLI
:: -----------------------------------------------------------------------
"%~dp0venv\Scripts\python.exe" "%~dp0src\cli.py"
if %ERRORLEVEL% neq 0 (
    echo.
    echo TikTok Wizard closed.
    pause
)
