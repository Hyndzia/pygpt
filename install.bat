@echo off
setlocal enabledelayedexpansion

title Token Setup Wizard

:: ==============================
:: UI HEADER
:: ==============================
echo ==========================================
echo           TOKEN SETUP WIZARD
echo ==========================================
echo.

set "TOKENS_DIR=user_tokens"

if not exist "%TOKENS_DIR%" (
    mkdir "%TOKENS_DIR%"
    echo [INFO] Created folder: %TOKENS_DIR%
    echo.
)

set TOKENS=hf_token nagaai_token

echo This wizard will configure your API tokens.
echo ------------------------------------------
echo.

:: ==============================
:: TOKEN LOOP
:: ==============================
for %%T in (%TOKENS%) do (

    set "FILE=%TOKENS_DIR%\%%T"

    echo ------------------------------------------
    echo Checking: %%T
    echo ------------------------------------------

    if not exist "!FILE!" (
        echo [CREATE] %%T file not found. Creating...
        type nul > "!FILE!"
    )

    set "content="
    for /f "usebackq delims=" %%A in ("!FILE!") do (
        set "content=%%A"
    )

    if "!content!"=="" (
        echo.
        echo [INPUT REQUIRED] Enter token for %%T:
        echo ------------------------------------------
        set /p content=^> 

        echo.
        echo [OK] Token saved successfully.
        echo ------------------------------------------

        echo !content! > "!FILE!"
        echo.
    ) else (
        echo [SKIP] %%T already configured
        echo.
    )
)

:: ==============================
:: DONE SCREEN
:: ==============================
echo ==========================================
echo        ALL TOKENS ARE READY
echo ==========================================
echo.

timeout /t 2 >nul
cls


title PyGPT Installer

:: ==============================
:: HEADER
:: ==============================
echo ==========================================
echo              PyGPT INSTALLER
echo ==========================================
echo.

:: ==============================
:: PYTHON CHECK
:: ==============================
echo [CHECK] Detecting Python 3.11...

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found!
    echo Please install Python 3.11 and try again.
    pause
    exit /b
)

echo [OK] Python detected
echo.

:: ==============================
:: VENV SETUP
:: ==============================
if exist venv\pyvenv.cfg (
    echo [SKIP] Virtual environment already exists
) else (
    echo [SETUP] Creating virtual environment...
    py -3.11 -m venv venv
    echo [OK] venv created
)

echo.

call venv\Scripts\activate.bat

:: ==============================
:: PIP UPGRADE
:: ==============================
echo [UPGRADE] Updating pip tools...
python -m pip install --upgrade pip setuptools wheel >nul
echo [OK] pip ready
echo.

:: ==============================
:: CHECKPOINTS
:: ==============================
set "CHECKPOINT_DIR=checkpoints"

if not exist "%CHECKPOINT_DIR%" (
    mkdir "%CHECKPOINT_DIR%"
    echo [CREATE] checkpoints folder
)

echo.
echo [CHECK] Model checkpoint

if not exist "checkpoints\vallex-checkpoint.pt" (
    echo [DOWNLOAD] Fetching VALL-E-X checkpoint...
    echo Please wait, this may take a while...

    powershell -NoProfile -ExecutionPolicy Bypass ^
    -Command "Invoke-WebRequest -Uri 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt' -OutFile 'checkpoints\vallex-checkpoint.pt'"

    if errorlevel 1 (
      echo [ERROR] Download failed!
      echo Check your internet connection.
      pause
      exit /b
	)

    echo [OK] Checkpoint downloaded
) else (
    echo [SKIP] Checkpoint already exists
)

echo.


:: ==============================
:: TORCH CHECK
:: ==============================
echo [CHECK] PyTorch CUDA...

python -c "import torch" >nul 2>&1

if not "%errorlevel%"=="0" (
    echo [INSTALL] Installing PyTorch CUDA...

    call pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    if not "!errorlevel!"=="0" (
        echo [ERROR] PyTorch install failed
        pause
        exit /b 1
    )

    echo [OK] PyTorch installed
) else (
    echo [OK] PyTorch already installed
)

echo.

:: ==============================
:: REQUIREMENTS
:: ==============================
if exist requirements.txt (
    echo [INSTALL] Installing requirements...
    echo.
    pip install -r requirements.txt
    echo. 
    echo [OK] Dependencies installed
) else (
    echo [WARN] requirements.txt not found
)

echo.

:: ==============================
:: HF CACHE SETUP
:: ==============================
echo [SETUP] HuggingFace cache directories...

set "HF_HOME=cache_dir"
set "TRANSFORMERS_CACHE=cache_dir"

if not exist cache_dir (
    mkdir cache_dir
    echo [CREATE] cache_dir created
)

echo.

:: ==============================
:: MODEL PRELOAD
:: ==============================
echo [PRELOAD] Loading Speech-to-Text models...
echo This may take a few minutes...
echo.
echo.

python preload_models.py

echo [OK] Models cached
echo.

:: ==============================
:: FINISH
:: ==============================
echo ==========================================
echo          INSTALLATION COMPLETE
echo ==========================================
echo.
echo You can now run PyGPT.
echo.

pause