#!/bin/bash

# Wine Quality Predictor - Streamlit App Deployment Script
# This script helps deploy and run the Streamlit application

echo "🍷 Wine Quality Predictor - Deployment Script"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
if [ -f "requirements_streamlit.txt" ]; then
    pip install -r requirements_streamlit.txt
    echo "✅ Requirements installed"
else
    echo "❌ requirements_streamlit.txt not found"
    exit 1
fi

# Check if MLflow data exists
if [ ! -d "mlruns" ]; then
    echo "⚠️  MLflow data not found. Creating demo data..."
    if [ -f "demo_data.py" ]; then
        python demo_data.py
        echo "✅ Demo data created"
    else
        echo "⚠️  demo_data.py not found. You may need to run experiments first."
    fi
else
    echo "✅ MLflow data found"
fi

# Test the setup
echo "🧪 Testing setup..."
if [ -f "test_streamlit.py" ]; then
    python test_streamlit.py
else
    echo "⚠️  test_streamlit.py not found. Skipping tests."
fi

# Test wine predictions specifically
echo "🍷 Testing wine quality predictions..."
if [ -f "test_predictions.py" ]; then
    python test_predictions.py
else
    echo "⚠️  test_predictions.py not found. Skipping prediction tests."
fi

echo ""
echo "🚀 Setup complete! To run the Streamlit app:"
echo "   source venv/bin/activate"
echo "   streamlit run streamlit_app.py"
echo ""
echo "🌐 The app will open at: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Keep the terminal open while using the app"
echo "   - Use Ctrl+C to stop the app"
echo "   - Check the sidebar for navigation options"
echo "   - Try the sample wines in the Model Predictor page"
echo "   - Use the Quick Testing section on the Home page"
