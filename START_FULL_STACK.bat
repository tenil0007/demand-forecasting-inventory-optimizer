@echo off
title Launch Full Stack (FastAPI + Streamlit)
color 0B

echo ======================================================================
echo   Launching Full Stack (FastAPI Backend + Streamlit Frontend)
echo ======================================================================
echo.

cd /d "%~dp0"

REM Activate virtualenv if available
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Starting FastAPI Backend (Port 8000)...
start "FastAPI Backend" cmd /k "cd /d ""%~dp0"" & python -m uvicorn backend.main:app --reload --port 8000"

echo Starting Streamlit Dashboard (Port 8501)...
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8501"
python -m streamlit run frontend/app.py --server.port 8501 --server.headless true

pause
