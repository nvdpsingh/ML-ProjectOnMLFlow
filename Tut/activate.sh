#!/bin/bash
# Activate the .venv virtual environment

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "🐍 Python path: $(which python)"
echo "📦 Pip path: $(which pip)"

echo ""
echo "💡 To deactivate, run: deactivate"
echo "💡 To check installed packages: pip list"
