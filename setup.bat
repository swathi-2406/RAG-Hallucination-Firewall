@echo off
echo ============================================================
echo  RAG Hallucination Firewall -- Windows Setup
echo ============================================================

echo.
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo.
echo [2/4] Activating venv and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo.
echo [3/4] Setting up .env file...
if not exist .env (
    copy .env.example .env
    echo Created .env -- EDIT IT NOW and add your Groq API key!
    echo Get a free key at: https://console.groq.com
) else (
    echo .env already exists, skipping.
)

echo.
echo [4/4] Done!
echo.
echo NEXT STEPS:
echo   1. Edit .env and add your GROQ_API_KEY
echo   2. Run:  python scripts/ingest_docs.py
echo   3. Run:  streamlit run app.py
echo.
pause
