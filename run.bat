@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python showroom.py %*
) else (
  python showroom.py %*
)
if errorlevel 1 pause
