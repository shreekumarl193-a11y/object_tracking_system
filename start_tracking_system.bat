@echo off
setlocal
cd /d "%~dp0"

if exist "backend\venv\Scripts\python.exe" (
    set "PYTHON=%~dp0backend\venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

start "Object Tracker Backend" cmd /k "cd /d "%~dp0backend" && "%PYTHON%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
start "Object Tracker Frontend" cmd /k "cd /d "%~dp0" && %PYTHON% -m http.server 3000"
start "" "http://localhost:3000"

endlocal
