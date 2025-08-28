# 🍷 Wine Quality Predictor - MLflow Dashboard

A comprehensive machine learning project that demonstrates end-to-end ML pipeline development with deep learning, hyperparameter optimization, and experiment tracking.

## 🎯 Project Overview

This project showcases modern machine learning development practices including:
- **Deep Learning**: Neural network architecture using Keras/TensorFlow
- **Hyperparameter Optimization**: Bayesian optimization with Hyperopt
- **Experiment Tracking**: Complete MLflow integration for reproducibility
- **Production Ready**: Streamlit interface for easy interaction
- **Performance Analysis**: Comprehensive model evaluation and comparison

## 🏗️ Architecture

The project follows a modular architecture with clear separation of concerns:

- **Data Layer**: Wine quality dataset with 11 features
- **Model Layer**: Neural network with Keras/TensorFlow
- **Optimization Layer**: Hyperparameter tuning with Hyperopt
- **Tracking Layer**: MLflow experiment management
- **Interface Layer**: Streamlit web application
- **Deployment Layer**: Model serving and prediction API

## 🛠️ Technology Stack

### Core ML Libraries
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Data preprocessing and evaluation
- **Hyperopt**: Bayesian hyperparameter optimization
- **MLflow**: Experiment tracking and model management

### Web Application
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Development Tools
- **Git**: Version control
- **Python**: Programming language
- **Jupyter**: Development environment

## 📊 Dataset Information

**Wine Quality Dataset:**
- **Source**: UCI Machine Learning Repository
- **Size**: 4,898 samples
- **Features**: 11 wine characteristics
- **Target**: Quality score (0-10)
- **Task**: Regression

**Features:**
1. Fixed Acidity (g/dm³)
2. Volatile Acidity (g/dm³)
3. Citric Acid (g/dm³)
4. Residual Sugar (g/dm³)
5. Chlorides (g/dm³)
6. Free Sulfur Dioxide (mg/dm³)
7. Total Sulfur Dioxide (mg/dm³)
8. Density (g/cm³)
9. pH
10. Sulphates (g/dm³)
11. Alcohol (% by volume)

## 🧠 Model Architecture

**Neural Network Design:**
- **Input Layer**: 11 neurons (one per feature)
- **Normalization Layer**: Feature standardization
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 1 neuron (regression output)
- **Optimizer**: SGD with momentum
- **Loss Function**: Mean Squared Error
- **Metrics**: Root Mean Squared Error

## 🔍 Hyperparameter Optimization

**Optimization Strategy:**
- **Algorithm**: Tree-structured Parzen Estimator (TPE)
- **Parameters**: Learning rate, momentum
- **Search Space**: Log-uniform distributions
- **Objective**: Minimize validation RMSE
- **Trials**: Configurable number of evaluations

## 📈 Experiment Tracking

**MLflow Integration:**
- **Experiment Management**: Organize runs by experiment
- **Parameter Logging**: Track all hyperparameters
- **Metric Logging**: Monitor performance metrics
- **Model Versioning**: Save and load model artifacts
- **Reproducibility**: Ensure experiment reproducibility

## 🚀 Key Features

1. **Interactive Prediction**: Real-time wine quality predictions
2. **Experiment Dashboard**: Comprehensive MLflow integration
3. **Performance Analysis**: Detailed model evaluation
4. **Visualization**: Interactive charts and graphs
5. **Data Export**: Download experiment results
6. **Responsive Design**: Mobile-friendly interface

## 📁 Project Structure

```
DL Project/
├── dlproject1.ipynb          # Main ML training notebook
├── streamlit_app.py          # Streamlit web application
├── requirements_streamlit.txt # Streamlit app dependencies
├── README.md                 # Project documentation
└── mlruns/                  # MLflow experiment data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "MLFlowTut/DL Project"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run MLflow experiments**
   ```bash
   # First, run your notebook to generate MLflow data
   jupyter notebook dlproject1.ipynb
   ```

4. **Launch Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open in browser**
   ```
   http://localhost:8501
   ```

## 📊 Usage

### MLflow Dashboard
- View all experiment runs and metrics
- Compare different hyperparameter combinations
- Download experiment results
- Analyze performance trends

### Model Predictor
- Input wine characteristics
- Get real-time quality predictions
- Visualize input features
- View prediction confidence

### Performance Analysis
- Analyze model performance across runs
- Identify parameter correlations
- Generate performance reports
- Export analysis data

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Deep learning model design and training
- Hyperparameter optimization techniques
- Experiment tracking and reproducibility
- Web application development
- Professional project documentation

## 🔮 Future Enhancements

**Planned Features:**
- Model comparison across algorithms
- Automated hyperparameter tuning
- Real-time model monitoring
- API endpoint for predictions
- Docker containerization
- Cloud deployment options

## 👨‍💻 Developer Information

**Skills Demonstrated:**
- Machine Learning & Deep Learning
- Data Science & Analytics
- Software Engineering
- Web Development
- DevOps & MLOps
- Project Management

## 📚 Resources & References

- **MLflow Documentation**: https://mlflow.org/
- **Streamlit Documentation**: https://streamlit.io/
- **TensorFlow Documentation**: https://tensorflow.org/
- **Hyperopt Documentation**: https://hyperopt.github.io/hyperopt/

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you have any questions or need help, please open an issue in the repository.

---

**Built with ❤️ for demonstrating modern ML development practices**
