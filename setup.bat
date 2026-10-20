@echo off
cd /d "%~dp0"
echo Creating Python environment in .venv ...
python -m venv .venv || (echo Python 3.9-3.12 is required: https://www.python.org/downloads/ & pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Done. Start the demo with run.bat
pause
