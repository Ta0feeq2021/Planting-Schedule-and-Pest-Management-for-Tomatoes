@echo off
echo ========================================
echo    Tomato Detection System Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please make sure you're in the correct directory.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Flask dependencies if needed
echo [2/4] Installing Flask dependencies...
cd flask-backend
pip install -r requirements.txt --quiet
cd ..

REM Install Next.js dependencies if needed
echo [3/4] Installing Next.js dependencies...
if not exist "node_modules" (
    npm install
)

REM Start Flask backend in background
echo [4/4] Starting services...
echo.
echo Starting Flask backend (http://127.0.0.1:5000)...
start /B cmd /c "cd flask-backend && python app.py"

REM Wait for Flask to start
timeout /t 3 /nobreak >nul

REM Start Next.js frontend
echo Starting Next.js frontend (http://localhost:3000)...
echo.
echo ========================================
echo   Your Tomato Detection System is ready!
echo   Frontend: http://localhost:3000
echo   Backend:  http://127.0.0.1:5000
echo ========================================
echo.
npm run dev
