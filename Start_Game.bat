@echo off
setlocal
cd /d "%~dp0"
set "PYGAME_HIDE_SUPPORT_PROMPT=1"

rem A partially installed local environment must not hide a working one.
call :check_python ".venv\Scripts\python.exe"
if defined GAME_PYTHON goto :run
call :check_python "..\.venv\Scripts\python.exe"
if defined GAME_PYTHON goto :run
call :check_python "python"
if defined GAME_PYTHON goto :run

echo Oyun icin gerekli Python paketleri bulunamadi.
echo Once bu klasordeki Setup.bat dosyasini calistirin.
pause
exit /b 1

:run
"%GAME_PYTHON%" main.py %*
set "GAME_EXIT_CODE=%errorlevel%"
if "%GAME_EXIT_CODE%"=="0" exit /b 0
echo.
echo Oyun baslatilamadi. Yukaridaki hata mesajini kontrol edin.
pause
exit /b %GAME_EXIT_CODE%

:check_python
set "GAME_PYTHON="
"%~1" -c "import numpy, pygame" >nul 2>&1
if not errorlevel 1 set "GAME_PYTHON=%~1"
exit /b 0
