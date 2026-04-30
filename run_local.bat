@echo off
REM Windows batch script for local development

echo Installing/updating dependencies...
python -m pip install -q -r requirements.txt

echo.
echo Starting Flask development server...
echo Visit: http://127.0.0.1:5000
echo.

python app.py

pause
