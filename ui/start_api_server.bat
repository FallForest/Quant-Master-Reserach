@echo off
REM Kill any existing process on port 5174
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5174 ^| findstr LISTENING') do (
    taskkill /f /pid %%a >nul 2>&1
)
python -m ui.server --port 5174
