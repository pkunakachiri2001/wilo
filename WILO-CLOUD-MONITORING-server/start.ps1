# Start both frontend and backend concurrently
# Usage: .\start.ps1

Write-Host "Starting Wilo Cloud Monitoring..." -ForegroundColor Cyan
Write-Host ""

# Start backend in a new terminal window
Write-Host "Starting Backend (Flask) on port 5001..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\..'; & 'C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe' app.py"

# Wait a moment for backend to initialize
Start-Sleep -Seconds 2

# Start frontend in a new terminal window
Write-Host "Starting Frontend (Vite) on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host ""
Write-Host "Both servers are starting..." -ForegroundColor Green
Write-Host "  Backend:  http://localhost:5001" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to open the dashboard in your browser..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Start-Process "http://localhost:5173"
