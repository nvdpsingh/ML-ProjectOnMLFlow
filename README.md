# MLflow Tutorials & Projects

This folder contains comprehensive tutorials and practical projects demonstrating MLflow (Machine Learning Lifecycle Management) concepts, from basic setup to advanced model deployment.

## 🚀 **Quick Start**

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements.txt

# Or install individually
pip install mlflow scikit-learn numpy pandas matplotlib seaborn
```

### Start MLflow UI
```bash
# Start the MLflow tracking server
mlflow ui

# Access the UI at: http://localhost:5000
```

## 📁 **Folder Structure**

### 🎓 **Tutorials**

#### **`Tut/` - Basic MLflow Fundamentals**
- **`get-started.ipynb`** - Introduction to MLflow experiments and metrics
- **`test_environment.py`** - Environment testing script
- **`activate.sh`** - Virtual environment activation script
- **`requirements.txt`** - Minimal dependencies for basic tutorials

**What you'll learn:**
- Setting up MLflow experiments
- Basic metrics logging
- Localhost connection testing
- Experiment management

#### **`gettingstarted.ipynb` - Main MLflow Tutorial**
Complete MLflow workflow with Iris dataset classification:
- **Data Loading** - Sklearn Iris dataset
- **Model Training** - Logistic Regression with hyperparameters
- **MLflow Tracking** - Parameters, metrics, and artifacts
- **Model Signatures** - Input/output specifications
- **Artifact Storage** - Saving trained models

**Key Concepts:**
- `mlflow.set_tracking_uri()` - Configure tracking server
- `mlflow.start_run()` - Start experiment runs
- `mlflow.log_param()` - Log hyperparameters
- `mlflow.log_metric()` - Log performance metrics
- `mlflow.log_model()` - Save trained models

### 🏗️ **Projects**

#### **`ML_Project/` - Basic ML Project Setup**
- Basic ML project structure
- MLflow integration examples
- Simple model training workflows

#### **`ML Project 2 house prediction/` - House Price Prediction**
- Real-world ML project with MLflow
- House price prediction using regression
- Complete MLflow workflow implementation

#### **`DL Project/` - Deep Learning with MLflow**
- Neural network projects with MLflow integration
- Advanced model tracking and versioning
- Streamlit deployment examples
- **Key Files:**
  - `dlproject1.ipynb` - Deep learning notebook
  - `streamlit_app.py` - Web application deployment
  - `deploy.sh` / `deploy.bat` - Deployment scripts
  - `DEPLOYMENT_GUIDE.md` - Deployment instructions

### 🔧 **Configuration & Setup**

#### **`mlflow-tracking-server.ipynb`**
- MLflow tracking server configuration
- Server setup and management
- Connection troubleshooting

#### **`mlruns/`**
- MLflow experiment tracking data
- Run history and metadata
- Model registry information

## 📚 **Learning Path**

### **Beginner Level**
1. **Start with `Tut/get-started.ipynb`**
   - Learn basic MLflow concepts
   - Understand experiments and runs
   - Practice simple metrics logging

2. **Progress to `gettingstarted.ipynb`**
   - Complete ML workflow with Iris dataset
   - Learn parameter and metric logging
   - Understand model artifacts

### **Intermediate Level**
3. **Explore `ML_Project/`**
   - Basic ML project structure
   - MLflow integration patterns

4. **Study `ML Project 2 house prediction/`**
   - Real-world ML project
   - Complete MLflow workflow

### **Advanced Level**
5. **Dive into `DL Project/`**
   - Deep learning with MLflow
   - Model deployment with Streamlit
   - Production deployment practices

## 🎯 **Key MLflow Concepts Covered**

### **Experiment Tracking**
- Creating and managing experiments
- Logging parameters and hyperparameters
- Recording metrics and performance data
- Storing model artifacts

### **Model Registry**
- Model versioning and storage
- Model signatures and metadata
- Artifact management
- Model deployment tracking

### **MLflow UI**
- Experiment visualization
- Run comparison
- Model registry management
- Performance tracking

## 🔧 **Common Commands**

```python
# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create/switch experiment
mlflow.set_experiment("My Experiment")

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    
    # Log model
    mlflow.log_model(model, "model")
```

## 🚨 **Troubleshooting**

### **Common Issues:**
1. **Connection Error**: Ensure MLflow server is running (`mlflow ui`)
2. **Port Conflicts**: Change port with `mlflow ui --port 5001`
3. **Dependencies**: Install all requirements from `requirements.txt`

### **Getting Help:**
- Check MLflow documentation: https://mlflow.org/docs/
- Verify environment setup with `test_environment.py`
- Use `activate.sh` for virtual environment setup

## 📖 **Additional Resources**

- **MLflow Documentation**: https://mlflow.org/docs/
- **MLflow GitHub**: https://github.com/mlflow/mlflow
- **MLflow Examples**: https://github.com/mlflow/mlflow-examples

---

*This tutorial collection provides hands-on experience with MLflow, covering the complete ML lifecycle from experimentation to deployment. Start with the basics and progressively build your MLOps skills!*
