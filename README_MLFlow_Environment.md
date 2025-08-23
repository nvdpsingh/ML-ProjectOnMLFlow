# MLFlow Environment Setup

This directory contains a dedicated virtual environment for MLFlow projects with all necessary dependencies.

## Quick Start

### 1. Activate the Environment
```bash
# Option 1: Use the activation script
./activate_mlflow_env.sh

# Option 2: Manual activation
source mlflow_env/bin/activate
```

### 2. Verify Installation
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check Jupyter kernels
jupyter kernelspec list
```

### 3. Run Jupyter Notebook
```bash
# Start Jupyter notebook
jupyter notebook mlflow-tracking-server.ipynb

# Or start Jupyter Lab
jupyter lab
```

## Available Kernels

When you open your notebook, you can select from these kernels:

1. **MLFlow Environment** - The main kernel for this project
2. **venv** - The previous conda environment
3. **python3** - System Python

## Installed Packages

The environment includes:
- **Core ML**: MLFlow, scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Jupyter**: ipykernel, jupyter, notebook
- **Web Framework**: FastAPI, Flask
- **Data Processing**: pyarrow, SQLAlchemy

## File Structure

```
MLFlowTut/
├── mlflow_env/                 # Virtual environment
├── activate_mlflow_env.sh      # Activation script
├── requirements_mlflow.txt     # Package requirements
├── mlflow-tracking-server.ipynb # Main notebook
└── README_MLFlow_Environment.md # This file
```

## Troubleshooting

### VS Code/Cursor Issues
If the kernel doesn't appear in VS Code:
1. Restart VS Code completely
2. Use Command Palette: `Python: Select Interpreter`
3. Choose: `/Users/navdeepsingh/MLOPS Tut/MLFlowTut/mlflow_env/bin/python`
4. Then: `Python: Select Kernel` → Choose "MLFlow Environment"

### Kernel Not Found
```bash
# Reinstall the kernel
source mlflow_env/bin/activate
python -m ipykernel install --user --name=mlflow_env --display-name="MLFlow Environment"
```

## Deactivation

To deactivate the virtual environment:
```bash
deactivate
```

## Updating Packages

To update packages in the environment:
```bash
source mlflow_env/bin/activate
pip install --upgrade -r requirements_mlflow.txt
```
