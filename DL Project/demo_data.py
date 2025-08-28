"""
Demo script to generate sample MLflow data for testing the Streamlit app.
This creates mock experiment runs to demonstrate the dashboard functionality.
"""

import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def create_demo_experiments():
    """Create demo MLflow experiments with sample data"""
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create experiment
    experiment_name = "wine-quality-demo"
    mlflow.set_experiment(experiment_name)
    
    # Sample hyperparameter ranges
    learning_rates = [0.001, 0.01, 0.1, 0.05, 0.02]
    momentums = [0.8, 0.9, 0.7, 0.85, 0.95]
    
    # Sample metrics
    rmse_values = [0.85, 0.78, 0.92, 0.76, 0.81]
    mse_values = [0.72, 0.61, 0.85, 0.58, 0.66]
    
    # Create multiple runs
    for i in range(5):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("lr", learning_rates[i])
            mlflow.log_param("momentum", momentums[i])
            mlflow.log_param("epochs", 50)
            mlflow.log_param("batch_size", 64)
            mlflow.log_param("hidden_units", 64)
            
            # Log metrics
            mlflow.log_metric("eval_rmse", rmse_values[i])
            mlflow.log_metric("eval_mse", mse_values[i])
            mlflow.log_metric("train_loss", rmse_values[i] * 1.1)
            mlflow.log_metric("val_loss", rmse_values[i] * 0.95)
            
            # Log tags
            mlflow.set_tag("model_type", "neural_network")
            mlflow.set_tag("dataset", "wine-quality")
            mlflow.set_tag("optimizer", "sgd")
            
            # Simulate some delay between runs
            if i > 0:
                mlflow.set_tag("parent_run", "demo_experiment")
    
    print(f"Created demo experiment: {experiment_name}")
    print(f"Generated {len(learning_rates)} sample runs")

def create_additional_experiments():
    """Create additional experiments for comparison"""
    
    # Experiment 2: Different model architectures
    mlflow.set_experiment("model-architecture-comparison")
    
    architectures = ["small", "medium", "large"]
    hidden_units = [32, 64, 128]
    
    for i, (arch, units) in enumerate(zip(architectures, hidden_units)):
        with mlflow.start_run():
            mlflow.log_param("architecture", arch)
            mlflow.log_param("hidden_units", units)
            mlflow.log_param("lr", 0.01)
            mlflow.log_param("momentum", 0.9)
            
            # Simulate different performance for different architectures
            base_rmse = 0.8
            performance_factor = 1.0 + (i * 0.1)  # Larger models perform slightly worse initially
            
            mlflow.log_metric("eval_rmse", base_rmse * performance_factor)
            mlflow.log_metric("eval_mse", (base_rmse * performance_factor) ** 2)
            mlflow.log_metric("training_time", 120 + (i * 30))  # Larger models take longer
            
            mlflow.set_tag("model_type", "neural_network")
            mlflow.set_tag("experiment_type", "architecture_comparison")
    
    print("Created model architecture comparison experiment")
    
    # Experiment 3: Hyperparameter tuning
    mlflow.set_experiment("hyperparameter-optimization")
    
    # Grid search simulation
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    momentum_values = [0.7, 0.8, 0.9, 0.95]
    
    run_count = 0
    for lr in lr_values:
        for momentum in momentum_values:
            with mlflow.start_run():
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("momentum", momentum)
                mlflow.log_param("epochs", 100)
                mlflow.log_param("batch_size", 32)
                
                # Simulate performance based on hyperparameters
                # Optimal values around lr=0.01, momentum=0.9
                lr_factor = 1.0 + abs(np.log10(lr) - np.log10(0.01)) * 0.5
                momentum_factor = 1.0 + abs(momentum - 0.9) * 0.3
                
                base_performance = 0.75
                performance = base_performance * lr_factor * momentum_factor
                
                mlflow.log_metric("eval_rmse", performance)
                mlflow.log_metric("eval_mse", performance ** 2)
                mlflow.log_metric("convergence_epochs", 50 + int(performance * 100))
                
                mlflow.set_tag("optimization_round", run_count)
                mlflow.set_tag("search_method", "grid_search")
                
                run_count += 1
    
    print(f"Created hyperparameter optimization experiment with {run_count} runs")

if __name__ == "__main__":
    print("Creating demo MLflow data for Streamlit app...")
    
    # Create demo experiments
    create_demo_experiments()
    create_additional_experiments()
    
    print("\nDemo data creation complete!")
    print("You can now run the Streamlit app to view the dashboard:")
    print("streamlit run streamlit_app.py")
    
    # Show created experiments
    mlflow.set_tracking_uri("file:./mlruns")
    experiments = mlflow.search_experiments()
    
    print(f"\nCreated {len(experiments)} experiments:")
    for exp in experiments:
        print(f"- {exp.name} (ID: {exp.experiment_id})")
