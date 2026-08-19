@echo off
title Retail Demand Forecasting & Inventory Optimization Agent
color 0A

echo ======================================================================
echo   Demand Forecasting & Inventory Optimization AI Agent
echo ======================================================================
echo.
echo [1/4] Navigating to project directory...
cd /d "%~dp0"

echo [2/4] Preparing environment...
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo [3/4] Launching FastAPI Backend Server on port 8000...
start "FastAPI Backend (Port 8000)" cmd /k "title FastAPI Backend & python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [4/4] Launching Streamlit Dashboard on port 8501...
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
echo.
echo ======================================================================
echo Note: Keep both terminal windows OPEN while using the application.
echo ======================================================================
echo.

REM Open browser after 3 seconds
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8501"

REM Run Streamlit in the foreground
python -m streamlit run frontend/app.py --server.port 8501 --server.headless true

pause
