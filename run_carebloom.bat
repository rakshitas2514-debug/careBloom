@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python -m streamlit run app.py
goto end
:failed
echo.
echo Could not install dependencies. Check your Python installation and internet connection.
pause
:end
