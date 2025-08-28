"""
Test script to verify Streamlit app functionality.
This script tests the core functions without running the full Streamlit interface.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Streamlit: {e}")
        return False
    
    try:
        import mlflow
        print("✅ MLflow imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MLflow: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Plotly: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pandas/NumPy: {e}")
        return False
    
    return True

def test_mlflow_connection():
    """Test MLflow connection and data access"""
    try:
        import mlflow
        
        # Set tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Check if mlruns directory exists
        if not os.path.exists("./mlruns"):
            print("⚠️  mlruns directory not found. Run demo_data.py first to create sample data.")
            return False
        
        # Try to search experiments
        experiments = mlflow.search_experiments()
        print(f"✅ Found {len(experiments)} MLflow experiments")
        
        for exp in experiments:
            print(f"   - {exp.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow connection test failed: {e}")
        return False

def test_streamlit_app_structure():
    """Test if the Streamlit app file exists and has required functions"""
    try:
        # Check if streamlit app exists
        if not os.path.exists("streamlit_app.py"):
            print("❌ streamlit_app.py not found")
            return False
        
        print("✅ streamlit_app.py found")
        
        # Read the file to check for required functions
        with open("streamlit_app.py", "r") as f:
            content = f.read()
        
        required_functions = [
            "show_home",
            "show_mlflow_dashboard", 
            "show_model_predictor",
            "show_model_performance",
            "show_about",
            "main"
        ]
        
        missing_functions = []
        for func in required_functions:
            if func not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing required functions: {missing_functions}")
            return False
        
        print("✅ All required functions found in streamlit_app.py")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app structure test failed: {e}")
        return False

def test_requirements():
    """Test if requirements file exists and has required packages"""
    try:
        if not os.path.exists("requirements_streamlit.txt"):
            print("❌ requirements_streamlit.txt not found")
            return False
        
        print("✅ requirements_streamlit.txt found")
        
        with open("requirements_streamlit.txt", "r") as f:
            requirements = f.read()
        
        required_packages = [
            "streamlit",
            "mlflow", 
            "tensorflow",
            "keras",
            "pandas",
            "numpy",
            "plotly",
            "scikit-learn",
            "hyperopt"
        ]
        
        missing_packages = []
        for pkg in required_packages:
            if pkg not in requirements:
                missing_packages.append(pkg)
        
        if missing_packages:
            print(f"⚠️  Missing packages in requirements: {missing_packages}")
        else:
            print("✅ All required packages listed in requirements")
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit App Setup...")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("MLflow Connection", test_mlflow_connection),
        ("Streamlit App Structure", test_streamlit_app_structure),
        ("Requirements File", test_requirements)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Streamlit app is ready to run.")
        print("\n🚀 To run the app:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the app.")
        
        if passed < total - 1:
            print("\n💡 Suggestions:")
            print("   1. Install missing packages: pip install -r requirements_streamlit.txt")
            print("   2. Run demo_data.py to create sample MLflow data")
            print("   3. Check file permissions and paths")

if __name__ == "__main__":
    main()
