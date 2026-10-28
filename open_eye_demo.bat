@echo off
setlocal
cd /d "%~dp0"

set "PYTHONW=%LocalAppData%\Programs\Python\Python313\pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=pythonw"

if not exist "models\eye_state_cnn.pt" (
    echo Model file not found: models\eye_state_cnn.pt
    echo Make sure the project files are in the same folder.
    pause
    exit /b 1
)

start "" "%PYTHONW%" "data\train_model.py" --choose-image