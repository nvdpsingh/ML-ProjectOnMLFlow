#!/bin/bash
# Script to activate the MLFlow virtual environment

echo "Activating MLFlow virtual environment..."
source mlflow_env/bin/activate

echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

echo ""
echo "To start Jupyter notebook with this environment:"
echo "jupyter notebook mlflow-tracking-server.ipynb"
echo ""
echo "To deactivate, run: deactivate"
