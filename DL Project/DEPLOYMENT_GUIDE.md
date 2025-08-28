# 🚀 Deployment Guide - Wine Quality Predictor

## 📦 Requirements Files Overview

This project includes multiple requirements files for different deployment scenarios:

### **1. `requirements.txt` - Complete Development Environment**
- **Use Case**: Local development, full ML training, complete functionality
- **Includes**: All ML libraries, development tools, optional cloud packages
- **Size**: Larger, comprehensive package set
- **Best For**: Developers, researchers, full project setup

### **2. `requirements_streamlit.txt` - Streamlit App Dependencies**
- **Use Case**: Local Streamlit app deployment, focused functionality
- **Includes**: Core ML libraries, Streamlit, essential visualization
- **Size**: Medium, focused package set
- **Best For**: Local app testing, development demonstrations

### **3. `requirements_streamlit_cloud.txt` - Streamlit Cloud Deployment**
- **Use Case**: Streamlit Cloud hosting, minimal deployment
- **Includes**: Essential packages only, optimized for cloud
- **Size**: Small, minimal package set
- **Best For**: Production deployment, cloud hosting

## 🌐 Deployment Options

### **Option 1: Local Development (Recommended for Testing)**

#### **Prerequisites**
- Python 3.8+
- pip or conda
- Git

#### **Quick Start**
```bash
# Clone and navigate
cd "MLFlowTut/DL Project"

# Use deployment script
./deploy.sh  # Mac/Linux
# OR
deploy.bat   # Windows

# Or manual setup
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate.bat  # Windows

pip install -r requirements_streamlit.txt
```

#### **Run Locally**
```bash
streamlit run streamlit_app.py
```

#### **Access**
```
http://localhost:8501
```

### **Option 2: Streamlit Cloud Deployment**

#### **Prerequisites**
- GitHub repository with your code
- Streamlit Cloud account (free)

#### **Deployment Steps**
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "🚀 Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Set requirements file: `requirements_streamlit_cloud.txt`

3. **Deploy**
   - Click "Deploy"
   - Wait for build (usually 2-5 minutes)
   - Get your public URL

#### **Benefits**
- ✅ **Free hosting**
- ✅ **Public URL** for sharing
- ✅ **Automatic updates** on git push
- ✅ **No server management**

### **Option 3: Docker Deployment**

#### **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **Build and Run**
```bash
docker build -t wine-quality-predictor .
docker run -p 8501:8501 wine-quality-predictor
```

### **Option 4: Cloud Platform Deployment**

#### **AWS/Azure/GCP**
- Use `requirements.txt` (includes cloud packages)
- Deploy to EC2, App Service, or Compute Engine
- Set up environment variables for MLflow tracking

#### **Heroku**
- Use `requirements_streamlit_cloud.txt`
- Add `Procfile`:
  ```
  web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
  ```

## 🔧 Configuration Files

### **Streamlit Configuration (`.streamlit/config.toml`)**
```toml
[server]
headless = true
port = 8501

[theme]
primaryColor = "#8B0000"
backgroundColor = "#FFFFFF"
```

### **Environment Variables**
```bash
# MLflow tracking
export MLFLOW_TRACKING_URI="file:./mlruns"

# Streamlit settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
```

## 📊 Package Comparison

| Package | Dev | Streamlit | Cloud |
|---------|-----|-----------|-------|
| **tensorflow** | ✅ | ✅ | ✅ |
| **keras** | ✅ | ✅ | ✅ |
| **scikit-learn** | ✅ | ✅ | ✅ |
| **mlflow** | ✅ | ✅ | ✅ |
| **hyperopt** | ✅ | ✅ | ❌ |
| **streamlit** | ✅ | ✅ | ✅ |
| **plotly** | ✅ | ✅ | ✅ |
| **matplotlib** | ✅ | ❌ | ❌ |
| **jupyter** | ✅ | ❌ | ❌ |
| **pytest** | ✅ | ❌ | ❌ |

## 🚨 Common Deployment Issues

### **Issue 1: Package Conflicts**
```bash
# Solution: Use virtual environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_streamlit.txt
```

### **Issue 2: MLflow Model Loading**
```bash
# Solution: Ensure mlruns directory exists
python demo_data.py  # Creates sample data
```

### **Issue 3: Port Already in Use**
```bash
# Solution: Change port
streamlit run streamlit_app.py --server.port 8502
```

### **Issue 4: Memory Issues on Cloud**
```bash
# Solution: Use minimal requirements
pip install -r requirements_streamlit_cloud.txt
```

## 🎯 Recommended Deployment Strategy

### **For Development & Testing**
1. Use `requirements_streamlit.txt`
2. Deploy locally with `deploy.sh` or `deploy.bat`
3. Test thoroughly with sample data

### **For Production & Sharing**
1. Use `requirements_streamlit_cloud.txt`
2. Deploy to Streamlit Cloud
3. Share public URL with recruiters

### **For Enterprise Deployment**
1. Use `requirements.txt`
2. Deploy to cloud platform (AWS/Azure/GCP)
3. Set up proper MLflow tracking server

## 📱 Testing Your Deployment

### **Local Testing**
```bash
# Test setup
python test_streamlit.py

# Test predictions
python test_predictions.py

# Test app
streamlit run streamlit_app.py
```

### **Cloud Testing**
1. Deploy to Streamlit Cloud
2. Test all features on public URL
3. Verify MLflow integration works
4. Check mobile responsiveness

## 🔮 Future Deployment Enhancements

### **Planned Features**
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline integration
- [ ] Automated testing on deployment
- [ ] Performance monitoring
- [ ] Load balancing

### **Advanced Options**
- [ ] Multi-region deployment
- [ ] CDN integration
- [ ] Database integration
- [ ] User authentication
- [ ] API endpoints

---

**🎉 Your Streamlit app is now ready for deployment on any platform! Choose the requirements file that best fits your deployment scenario.**
