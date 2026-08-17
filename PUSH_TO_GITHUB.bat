@echo off
title Push to GitHub
color 0B

echo ======================================================================
echo   Pushing Demand Forecasting & Inventory Optimization to GitHub
echo ======================================================================
echo.

cd /d "%~dp0"

echo Repository: https://github.com/tenil0007/demand-forecasting-inventory-optimizer.git
echo Branch: main
echo.
echo If prompted, please authorize GitHub in your browser window...
echo.

git push -u origin main

echo.
echo ======================================================================
echo If you see "Branch 'main' set up to track remote branch 'main'",
echo your repository is successfully uploaded!
echo ======================================================================
echo.
pause
