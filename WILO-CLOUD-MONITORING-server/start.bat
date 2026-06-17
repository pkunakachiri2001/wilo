@echo off
REM Start both frontend and backend concurrently
REM Usage: start.bat

echo Starting Wilo Cloud Monitoring...
echo.

REM Start backend in a new terminal window
echo Starting Backend (Flask) on port 5001...
start "Backend - Flask" cmd /k "cd /d %~dp0 && python app.py"

REM Wait a moment for backend to initialize
timeout /t 2 /nobreak > nul

REM Start frontend in a new terminal window
echo Starting Frontend (Vite) on port 5173...
start "Frontend - Vite" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are starting...
echo   Backend:  http://localhost:5001
echo   Frontend: http://localhost:5173
echo.
pause
start http://localhost:5173
