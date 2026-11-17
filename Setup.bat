@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements-lock.txt
if errorlevel 1 goto :failed
echo Setup completed. Double-click Start_Game.bat to launch.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.13 and check your internet connection.
pause
exit /b 1
