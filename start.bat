@echo off

if not exist venv (
    echo venv not found. Run install.bat first.
    pause
    exit /b
)

call venv\Scripts\activate

start "" pythonw main.py

if errorlevel 1 (
    echo Program crashed!
) else (
    echo Program exited normally.
)

exit