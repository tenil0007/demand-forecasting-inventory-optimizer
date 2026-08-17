@echo off
title Retail Demand Forecasting & Inventory Optimization Agent
color 0A

echo ======================================================================
echo   Demand Forecasting & Inventory Optimization AI Agent
echo ======================================================================
echo.
echo [1/3] Navigating to project directory...
cd /d "%~dp0"

echo [2/3] Preparing environment...
REM If virtualenv exists, activate it
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo [3/3] Launching Streamlit Dashboard and opening Chrome/Browser...
echo.
echo Application URL: http://localhost:8501
echo.
echo Note: Keep this terminal window OPEN while using the application.
echo Press Ctrl+C in this window when you want to stop the app.
echo ======================================================================
echo.

REM Open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8501"

REM Run Streamlit
python -m streamlit run frontend/app.py --server.port 8501 --server.headless true

pause
