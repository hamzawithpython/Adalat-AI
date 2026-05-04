@echo off
echo.
echo  ========================================
echo   ADALAT-AI - Legal Assistant
echo   Starting all services...
echo  ========================================
echo.

echo [1/3] Starting FastAPI backend...
start "Adalat-API" cmd /k "cd /d C:\Users\Admin\Desktop\portfolio\adalat-ai && venv\Scripts\activate && python run.py"

echo Waiting for API to load (20 seconds)...
timeout /t 20 /nobreak > nul

echo [2/3] Starting Streamlit UI...
start "Adalat-UI" cmd /k "cd /d C:\Users\Admin\Desktop\portfolio\adalat-ai && venv\Scripts\activate && streamlit run src/ui/app.py --server.port 8501"

timeout /t 5 /nobreak > nul

echo [3/3] Opening browser...
start http://localhost:8501

echo.
echo  ========================================
echo   App is running!
echo   API:  http://localhost:8001
echo   UI:   http://localhost:8501
echo   Docs: http://localhost:8001/docs
echo  ========================================