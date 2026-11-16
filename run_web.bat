@echo off
chcp 65001 > nul
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=D:\K5-L3B-DAY06-2A202603922-DataIngestion\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python was not found. Create .venv or update PYTHON_EXE in this script.
  pause
  exit /b 1
)
echo ========================================================
echo   VinAI Day 7 - Agentic RAG Web Explorer (Google Chrome)
echo   Sinh vien: Nguyen Tien Dat (2A202603922)
echo ========================================================
echo Dang khoi dong may chu web va mo Google Chrome...
"%PYTHON_EXE%" src/web_server.py
pause
