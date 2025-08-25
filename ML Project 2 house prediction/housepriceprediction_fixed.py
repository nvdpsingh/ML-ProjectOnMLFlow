import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from urllib.parse import urlparse

# Set MLflow tracking URI FIRST (before starting any runs)
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Fetch data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["Price"] = housing.target

print("Dataset shape:", data.shape)
print("Features:", list(data.columns[:-1]))
print("Target:", data.columns[-1])

# Prepare data
y = data['Price']
X = data.drop(columns=['Price'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Hyperparameter tuning function
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2, 
        scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)
    return grid_search

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create signature for model
from mlflow.models import infer_signature
signature = infer_signature(X_train, y_train)

print("Starting MLflow experiment...")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Start the MLflow experiment
with mlflow.start_run():
    print("MLflow run started successfully!")
    
    # Perform hyperparameter tuning
    print("Starting hyperparameter tuning...")
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Log best parameters and metrics
    mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
    mlflow.log_metric("mse", mse)
    
    # Check tracking URI type
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    if tracking_url_type_store != 'file':
        print("Logging model to MLflow server...")
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            registered_model_name="Best Randomforest Model"
        )
    else:
        print("Logging model locally...")
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            signature=signature
        )
    
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse:.4f}")
    print("MLflow experiment completed successfully!")

print("Script completed!")
