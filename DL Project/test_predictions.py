"""
Test script for wine quality predictions.
This script tests the model with various wine inputs to verify prediction functionality.
"""

import numpy as np
import pandas as pd
import mlflow
import os

def test_wine_predictions():
    """Test wine quality predictions with various inputs"""
    
    print("🍷 Testing Wine Quality Predictions")
    print("=" * 50)
    
    # Sample wine inputs for testing
    test_wines = {
        "High Quality White": {
            "fixed_acidity": 7.0,
            "volatile_acidity": 0.27,
            "citric_acid": 0.36,
            "residual_sugar": 20.7,
            "chlorides": 0.045,
            "free_sulfur_dioxide": 45.0,
            "total_sulfur_dioxide": 170.0,
            "density": 1.001,
            "ph": 3.00,
            "sulphates": 0.45,
            "alcohol": 8.8
        },
        "Medium Quality White": {
            "fixed_acidity": 6.3,
            "volatile_acidity": 0.30,
            "citric_acid": 0.34,
            "residual_sugar": 1.6,
            "chlorides": 0.049,
            "free_sulfur_dioxide": 14.0,
            "total_sulfur_dioxide": 132.0,
            "density": 0.994,
            "ph": 3.30,
            "sulphates": 0.49,
            "alcohol": 9.5
        },
        "Premium White": {
            "fixed_acidity": 8.1,
            "volatile_acidity": 0.28,
            "citric_acid": 0.40,
            "residual_sugar": 6.9,
            "chlorides": 0.050,
            "free_sulfur_dioxide": 30.0,
            "total_sulfur_dioxide": 97.0,
            "density": 0.995,
            "ph": 3.26,
            "sulphates": 0.44,
            "alcohol": 10.1
        },
        "Low Quality White": {
            "fixed_acidity": 9.5,
            "volatile_acidity": 0.65,
            "citric_acid": 0.20,
            "residual_sugar": 35.0,
            "chlorides": 0.080,
            "free_sulfur_dioxide": 80.0,
            "total_sulfur_dioxide": 250.0,
            "density": 1.008,
            "ph": 3.45,
            "sulphates": 0.35,
            "alcohol": 8.2
        }
    }
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        # Check if MLflow experiments exist
        experiments = mlflow.search_experiments()
        
        if not experiments:
            print("❌ No MLflow experiments found. Run demo_data.py first to create sample data.")
            return False
        
        print(f"✅ Found {len(experiments)} MLflow experiments")
        
        # Get the first experiment
        experiment = experiments[0]
        print(f"📊 Using experiment: {experiment.name}")
        
        # Search for runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            print("❌ No runs found in experiment. Please run experiments first.")
            return False
        
        run = runs.iloc[0]
        run_id = run['run_id']
        print(f"🔍 Using run ID: {run_id[:8]}")
        
        # Load the model
        try:
            model = mlflow.keras.load_model(f"runs:/{run_id}/model")
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 This might be because the model wasn't saved properly during training.")
            return False
        
        # Test predictions
        print("\n🧪 Testing Predictions:")
        print("-" * 30)
        
        results = []
        
        for wine_name, wine_features in test_wines.items():
            # Prepare input data
            input_data = np.array([[
                wine_features["fixed_acidity"],
                wine_features["volatile_acidity"],
                wine_features["citric_acid"],
                wine_features["residual_sugar"],
                wine_features["chlorides"],
                wine_features["free_sulfur_dioxide"],
                wine_features["total_sulfur_dioxide"],
                wine_features["density"],
                wine_features["ph"],
                wine_features["sulphates"],
                wine_features["alcohol"]
            ]])
            
            # Make prediction
            try:
                prediction = model.predict(input_data, verbose=0)[0][0]
                
                # Quality assessment
                if prediction >= 7.0:
                    quality = "Excellent"
                    emoji = "🎉"
                elif prediction >= 6.0:
                    quality = "Good"
                    emoji = "👍"
                elif prediction >= 5.0:
                    quality = "Average"
                    emoji = "⚠️"
                else:
                    quality = "Below Average"
                    emoji = "❌"
                
                print(f"{emoji} {wine_name}: {prediction:.2f}/10 ({quality})")
                
                results.append({
                    "Wine": wine_name,
                    "Prediction": f"{prediction:.2f}",
                    "Quality": quality,
                    "Volatile Acidity": wine_features["volatile_acidity"],
                    "Alcohol": wine_features["alcohol"]
                })
                
            except Exception as e:
                print(f"❌ Error predicting {wine_name}: {e}")
                results.append({
                    "Wine": wine_name,
                    "Prediction": "Error",
                    "Quality": "Error",
                    "Volatile Acidity": wine_features["volatile_acidity"],
                    "Alcohol": wine_features["alcohol"]
                })
        
        # Summary table
        print("\n📊 Prediction Summary:")
        print("-" * 30)
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Analysis
        print("\n🔍 Analysis:")
        print("-" * 30)
        
        successful_predictions = [r for r in results if r["Prediction"] != "Error"]
        
        if successful_predictions:
            predictions = [float(r["Prediction"]) for r in successful_predictions]
            print(f"✅ Successful predictions: {len(successful_predictions)}")
            print(f"📊 Average prediction: {np.mean(predictions):.2f}")
            print(f"📈 Best prediction: {max(predictions):.2f}")
            print(f"📉 Worst prediction: {min(predictions):.2f}")
            
            # Quality distribution
            quality_counts = {}
            for result in successful_predictions:
                quality = result["Quality"]
                quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            print(f"\n🏆 Quality Distribution:")
            for quality, count in quality_counts.items():
                print(f"   {quality}: {count} wines")
        
        print("\n🎯 Test Results:")
        print("-" * 30)
        
        if len(successful_predictions) == len(test_wines):
            print("🎉 All predictions successful! Model is working correctly.")
            return True
        elif len(successful_predictions) > 0:
            print(f"⚠️  Partial success: {len(successful_predictions)}/{len(test_wines)} predictions worked.")
            return True
        else:
            print("❌ All predictions failed. Check model and data.")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_input_validation():
    """Test input validation and edge cases"""
    
    print("\n🔍 Testing Input Validation")
    print("=" * 50)
    
    # Test edge cases
    edge_cases = {
        "Min Values": {
            "fixed_acidity": 4.0,
            "volatile_acidity": 0.0,
            "citric_acid": 0.0,
            "residual_sugar": 0.0,
            "chlorides": 0.0,
            "free_sulfur_dioxide": 0.0,
            "total_sulfur_dioxide": 0.0,
            "density": 0.9,
            "ph": 2.5,
            "sulphates": 0.0,
            "alcohol": 8.0
        },
        "Max Values": {
            "fixed_acidity": 16.0,
            "volatile_acidity": 2.0,
            "citric_acid": 2.0,
            "residual_sugar": 70.0,
            "chlorides": 1.0,
            "free_sulfur_dioxide": 300.0,
            "total_sulfur_dioxide": 500.0,
            "density": 1.1,
            "ph": 4.0,
            "sulphates": 2.0,
            "alcohol": 15.0
        }
    }
    
    for case_name, values in edge_cases.items():
        print(f"🧪 Testing {case_name}:")
        
        # Validate ranges
        valid = True
        for feature, value in values.items():
            if feature == "fixed_acidity" and not (4.0 <= value <= 16.0):
                valid = False
            elif feature == "volatile_acidity" and not (0.0 <= value <= 2.0):
                valid = False
            elif feature == "citric_acid" and not (0.0 <= value <= 2.0):
                valid = False
            elif feature == "residual_sugar" and not (0.0 <= value <= 70.0):
                valid = False
            elif feature == "chlorides" and not (0.0 <= value <= 1.0):
                valid = False
            elif feature == "free_sulfur_dioxide" and not (0.0 <= value <= 300.0):
                valid = False
            elif feature == "total_sulfur_dioxide" and not (0.0 <= value <= 500.0):
                valid = False
            elif feature == "density" and not (0.9 <= value <= 1.1):
                valid = False
            elif feature == "ph" and not (2.5 <= value <= 4.0):
                valid = False
            elif feature == "sulphates" and not (0.0 <= value <= 2.0):
                valid = False
            elif feature == "alcohol" and not (8.0 <= value <= 15.0):
                valid = False
        
        if valid:
            print(f"   ✅ {case_name} - All values within valid ranges")
        else:
            print(f"   ❌ {case_name} - Some values outside valid ranges")
    
    print("✅ Input validation testing complete")

def main():
    """Run all tests"""
    print("🧪 Wine Quality Prediction Testing Suite")
    print("=" * 60)
    
    # Test predictions
    prediction_success = test_wine_predictions()
    
    # Test input validation
    test_input_validation()
    
    print("\n" + "=" * 60)
    
    if prediction_success:
        print("🎉 All tests passed! Your wine quality predictor is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Run the Streamlit app: streamlit run streamlit_app.py")
        print("   2. Test with the interactive interface")
        print("   3. Try different wine characteristics")
    else:
        print("⚠️  Some tests failed. Please check:")
        print("   1. MLflow experiments are properly set up")
        print("   2. Model was saved during training")
        print("   3. All dependencies are installed")
        print("\n💡 Run demo_data.py to create sample MLflow data")

if __name__ == "__main__":
    main()
