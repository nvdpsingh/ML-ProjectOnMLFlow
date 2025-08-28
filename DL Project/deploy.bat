@echo off
REM Wine Quality Predictor - Streamlit App Deployment Script (Windows)
REM This script helps deploy and run the Streamlit application on Windows

echo 🍷 Wine Quality Predictor - Deployment Script
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found: 
pip --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📥 Installing requirements...
if exist "requirements_streamlit.txt" (
    pip install -r requirements_streamlit.txt
    echo ✅ Requirements installed
) else (
    echo ❌ requirements_streamlit.txt not found
    pause
    exit /b 1
)

REM Check if MLflow data exists
if not exist "mlruns" (
    echo ⚠️  MLflow data not found. Creating demo data...
    if exist "demo_data.py" (
        python demo_data.py
        echo ✅ Demo data created
    ) else (
        echo ⚠️  demo_data.py not found. You may need to run experiments first.
    )
) else (
    echo ✅ MLflow data found
)

REM Test the setup
echo 🧪 Testing setup...
if exist "test_streamlit.py" (
    python test_streamlit.py
) else (
    echo ⚠️  test_streamlit.py not found. Skipping tests.
)

echo.
echo 🚀 Setup complete! To run the Streamlit app:
echo   1. Open Command Prompt in this directory
echo   2. Run: venv\Scripts\activate.bat
echo   3. Run: streamlit run streamlit_app.py
echo.
echo 🌐 The app will open at: http://localhost:8501
echo.
echo 💡 Tips:
echo    - Keep the Command Prompt open while using the app
echo    - Use Ctrl+C to stop the app
echo    - Check the sidebar for navigation options
echo.
pause
