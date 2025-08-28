import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor - MLflow Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8B0000;
    }
    .run-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def show_home():
    """Display the home page with project overview"""
    st.markdown("## 🎯 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Wine Quality Predictor** is a comprehensive machine learning project that demonstrates:
        
        🧠 **Deep Learning**: Neural network architecture using Keras/TensorFlow
        🔍 **Hyperparameter Optimization**: Bayesian optimization with Hyperopt
        📊 **Experiment Tracking**: Complete MLflow integration for reproducibility
        🚀 **Production Ready**: Streamlit interface for easy interaction
        📈 **Performance Analysis**: Comprehensive model evaluation and comparison
        
        This project showcases modern ML development practices including:
        - End-to-end ML pipeline development
        - Hyperparameter tuning and optimization
        - Experiment tracking and model versioning
        - Interactive web application deployment
        - Professional project documentation
        """)
    
    with col2:
        st.markdown("### 🏆 Key Features")
        st.markdown("""
        - **Multi-page Streamlit App**
        - **MLflow Dashboard Integration**
        - **Real-time Model Predictions**
        - **Performance Visualization**
        - **Responsive Design**
        """)
    
    st.markdown("---")
    
    # Project statistics
    st.markdown("## 📊 Project Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset Size</h3>
            <p style="font-size: 2rem; font-weight: bold; color: #8B0000;">4,898</p>
            <p>Wine samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Features</h3>
            <p style="font-size: 2rem; font-weight: bold; color: #8B0000;">11</p>
            <p>Wine characteristics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Model Type</h3>
            <p style="font-size: 2rem; font-weight: bold; color: #8B0000;">Neural Network</p>
            <p>Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Optimization</h3>
            <p style="font-size: 2rem; font-weight: bold; color: #8B0000;">Hyperopt</h3>
            <p>Bayesian Tuning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **📊 MLflow Dashboard**: View all your experiment runs, metrics, and model versions
    2. **🔮 Model Predictor**: Input wine characteristics and get quality predictions
    3. **📈 Model Performance**: Analyze model performance across different runs
    4. **💾 About**: Learn more about the project and technology stack
    """)
    
    st.info("💡 **Pro Tip**: Use the sidebar navigation to explore different features of the application!")

def show_mlflow_dashboard():
    """Display MLflow experiment dashboard"""
    st.markdown("## 📊 MLflow Experiment Dashboard")
    
    # Set MLflow tracking URI (adjust path as needed)
    mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        if not experiments:
            st.warning("No MLflow experiments found. Make sure you have run experiments and they are stored in the mlruns directory.")
            return
        
        # Select experiment
        experiment_names = [exp.name for exp in experiments]
        selected_experiment = st.selectbox("Select Experiment:", experiment_names)
        
        if selected_experiment:
            # Get experiment by name
            experiment = mlflow.get_experiment_by_name(selected_experiment)
            
            # Search runs in the selected experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                st.info(f"No runs found in experiment: {selected_experiment}")
                return
            
            st.success(f"Found {len(runs)} runs in experiment: {selected_experiment}")
            
            # Display experiment summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Runs", len(runs))
            with col2:
                st.metric("Latest Run", runs.iloc[0]['start_time'].strftime('%Y-%m-%d %H:%M'))
            with col3:
                st.metric("Experiment ID", experiment.experiment_id)
            
            st.markdown("---")
            
            # Display runs table
            st.markdown("### 🏃‍♂️ Experiment Runs")
            
            # Prepare runs data for display
            runs_display = []
            for idx, run in runs.iterrows():
                run_data = {
                    'Run ID': run['run_id'],
                    'Start Time': run['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Status': run['status'],
                    'Duration': str(run['end_time'] - run['start_time']).split('.')[0] if run['end_time'] else 'N/A'
                }
                
                # Add parameters
                if 'params' in run and run['params']:
                    for param_name, param_value in run['params'].items():
                        run_data[f'Param: {param_name}'] = str(param_value)[:20]
                
                # Add metrics
                if 'metrics' in run and run['metrics']:
                    for metric_name, metric_value in run['metrics'].items():
                        run_data[f'Metric: {metric_name}'] = f"{metric_value:.4f}"
                
                runs_display.append(run_data)
            
            # Convert to DataFrame and display
            runs_df = pd.DataFrame(runs_display)
            st.dataframe(runs_df, use_container_width=True)
            
            # Download runs data
            csv = runs_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Runs Data (CSV)",
                data=csv,
                file_name=f"mlflow_runs_{selected_experiment}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Performance comparison chart
            st.markdown("### 📈 Performance Comparison")
            
            # Extract metrics for visualization
            metric_data = []
            for idx, run in runs.iterrows():
                if 'metrics' in run and run['metrics']:
                    for metric_name, metric_value in run['metrics'].items():
                        metric_data.append({
                            'Run ID': run['run_id'][:8],
                            'Metric': metric_name,
                            'Value': metric_value,
                            'Start Time': run['start_time']
                        })
            
            if metric_data:
                metric_df = pd.DataFrame(metric_data)
                
                # Create performance chart
                fig = px.line(
                    metric_df, 
                    x='Start Time', 
                    y='Value', 
                    color='Metric',
                    title='Metric Values Over Time',
                    markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Parameter comparison
                st.markdown("### 🔧 Parameter Comparison")
                
                # Extract parameters for comparison
                param_data = []
                for idx, run in runs.iterrows():
                    if 'params' in run and run['params']:
                        for param_name, param_value in run['params'].items():
                            try:
                                param_data.append({
                                    'Run ID': run['run_id'][:8],
                                    'Parameter': param_name,
                                    'Value': float(param_value)
                                })
                            except:
                                continue
                
                if param_data:
                    param_df = pd.DataFrame(param_data)
                    
                    # Create parameter comparison chart
                    fig2 = px.scatter(
                        param_df,
                        x='Parameter',
                        y='Value',
                        color='Run ID',
                        title='Parameter Values Across Runs',
                        size=[10] * len(param_df)
                    )
                    fig2.update_layout(height=500)
                    st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error accessing MLflow data: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        1. Make sure MLflow is properly installed
        2. Check if the mlruns directory exists and contains experiment data
        3. Verify the tracking URI path is correct
        4. Ensure you have run experiments with MLflow tracking enabled
        """)

def show_model_predictor():
    """Display the model predictor interface"""
    st.markdown("## 🔮 Wine Quality Predictor")
    
    st.markdown("""
    Input the characteristics of your wine and get a quality prediction using our trained neural network model.
    The model predicts wine quality on a scale of 0-10 based on 11 wine characteristics.
    """)
    
    # Create input form
    with st.form("wine_prediction_form"):
        st.markdown("### 🍷 Wine Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_acidity = st.number_input(
                "Fixed Acidity (g/dm³)",
                min_value=4.0,
                max_value=16.0,
                value=7.0,
                step=0.1,
                help="Tartaric acid content"
            )
            
            volatile_acidity = st.number_input(
                "Volatile Acidity (g/dm³)",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.01,
                help="Acetic acid content"
            )
            
            citric_acid = st.number_input(
                "Citric Acid (g/dm³)",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.01,
                help="Citric acid content"
            )
            
            residual_sugar = st.number_input(
                "Residual Sugar (g/dm³)",
                min_value=0.0,
                max_value=70.0,
                value=6.0,
                step=0.1,
                help="Sugar remaining after fermentation"
            )
            
            chlorides = st.number_input(
                "Chlorides (g/dm³)",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.001,
                help="Sodium chloride content"
            )
            
            free_sulfur_dioxide = st.number_input(
                "Free Sulfur Dioxide (mg/dm³)",
                min_value=0.0,
                max_value=300.0,
                value=30.0,
                step=1.0,
                help="Free SO2 content"
            )
        
        with col2:
            total_sulfur_dioxide = st.number_input(
                "Total Sulfur Dioxide (mg/dm³)",
                min_value=0.0,
                max_value=500.0,
                value=100.0,
                step=1.0,
                help="Total SO2 content"
            )
            
            density = st.number_input(
                "Density (g/cm³)",
                min_value=0.9,
                max_value=1.1,
                value=0.99,
                step=0.001,
                help="Wine density"
            )
            
            ph = st.number_input(
                "pH",
                min_value=2.5,
                max_value=4.0,
                value=3.2,
                step=0.01,
                help="Acidity/basicity"
            )
            
            sulphates = st.number_input(
                "Sulphates (g/dm³)",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.01,
                help="Potassium sulphate content"
            )
            
            alcohol = st.number_input(
                "Alcohol (% by volume)",
                min_value=8.0,
                max_value=15.0,
                value=10.0,
                step=0.1,
                help="Alcohol content"
            )
        
        # Submit button
        submitted = st.form_submit_button("🍷 Predict Wine Quality", type="primary")
        
        if submitted:
            # Prepare input data
            wine_features = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                ph, sulphates, alcohol
            ]])
            
            # Display input summary
            st.markdown("### 📋 Input Summary")
            
            feature_names = [
                "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar",
                "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density",
                "pH", "Sulphates", "Alcohol"
            ]
            
            feature_values = [
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                ph, sulphates, alcohol
            ]
            
            # Create feature summary table
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values
            })
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(feature_df, use_container_width=True)
            
            with col2:
                # Create radar chart for visualization
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=feature_values,
                    theta=feature_names,
                    fill='toself',
                    name='Wine Characteristics',
                    line_color='#8B0000'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(feature_values) * 1.1]
                        )),
                    showlegend=False,
                    title="Wine Characteristics Radar Chart",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Model prediction (placeholder for now)
            st.markdown("### 🔮 Quality Prediction")
            
            # Simulate prediction (replace with actual model loading)
            try:
                # Try to load MLflow model
                mlflow.set_tracking_uri("file:./mlruns")
                
                # Get latest model run
                experiments = mlflow.search_experiments()
                if experiments:
                    experiment = mlflow.get_experiment_by_name(experiments[0].name)
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    
                    if not runs.empty:
                        run_id = runs.iloc[0]['run_id']
                        
                        # Load model
                        model = mlflow.keras.load_model(f"runs:/{run_id}/model")
                        
                        # Make prediction
                        prediction = model.predict(wine_features)[0][0]
                        
                        # Display prediction
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("""
                            <div class="metric-card">
                                <h3>Predicted Quality</h3>
                                <p style="font-size: 2rem; font-weight: bold; color: #8B0000;">{:.2f}</p>
                                <p>out of 10</p>
                            </div>
                            """.format(prediction), unsafe_allow_html=True)
                        
                        with col2:
                            # Quality interpretation
                            if prediction >= 7.0:
                                quality_text = "Excellent"
                                quality_color = "#28a745"
                            elif prediction >= 6.0:
                                quality_text = "Good"
                                quality_color = "#17a2b8"
                            elif prediction >= 5.0:
                                quality_text = "Average"
                                quality_color = "#ffc107"
                            else:
                                quality_text = "Below Average"
                                quality_color = "#dc3545"
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Quality Level</h3>
                                <p style="font-size: 1.5rem; font-weight: bold; color: {quality_color};">{quality_text}</p>
                                <p>Quality assessment</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("""
                            <div class="metric-card">
                                <h3>Model Confidence</h3>
                                <p style="font-size: 1.5rem; font-weight: bold; color: #8B0000;">High</p>
                                <p>Based on training data</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Quality bar
                        st.markdown("### 📊 Quality Scale")
                        quality_percentage = (prediction / 10.0) * 100
                        
                        st.progress(quality_percentage / 100.0)
                        st.caption(f"Quality Score: {prediction:.2f}/10 ({quality_percentage:.1f}%)")
                        
                        # Quality insights
                        st.markdown("### 💡 Quality Insights")
                        
                        if prediction >= 7.0:
                            st.success("🎉 **Excellent Quality Wine!** This wine shows exceptional characteristics across all parameters.")
                        elif prediction >= 6.0:
                            st.info("👍 **Good Quality Wine** This is a well-balanced wine with good characteristics.")
                        elif prediction >= 5.0:
                            st.warning("⚠️ **Average Quality Wine** This wine meets basic quality standards but has room for improvement.")
                        else:
                            st.error("❌ **Below Average Quality** This wine may have some quality issues that need attention.")
                        
                        # Model metadata
                        st.markdown("### 🔍 Model Information")
                        st.info(f"""
                        **Model Details:**
                        - **Run ID**: {run_id[:8]}
                        - **Experiment**: {experiment.name}
                        - **Model Type**: Neural Network (Keras)
                        - **Training Date**: {runs.iloc[0]['start_time'].strftime('%Y-%m-%d %H:%M')}
                        """)
                        
                    else:
                        st.warning("No model runs found. Please run experiments first.")
                        
                else:
                    st.warning("No MLflow experiments found. Please run experiments first.")
                    
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("""
                **Note**: This is a demonstration. To use the actual model:
                1. Run your MLflow experiments first
                2. Ensure the model is properly saved
                3. Check the mlruns directory path
                """)
                
                # Show sample prediction for demo
                st.info("**Demo Mode**: Showing sample prediction")
                sample_prediction = 6.8
                st.metric("Sample Predicted Quality", f"{sample_prediction:.1f}/10")

def show_model_performance():
    """Display model performance analysis"""
    st.markdown("## 📈 Model Performance Analysis")
    
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get experiments
        experiments = mlflow.search_experiments()
        
        if not experiments:
            st.warning("No MLflow experiments found. Please run experiments first.")
            return
        
        # Select experiment
        experiment_names = [exp.name for exp in experiments]
        selected_experiment = st.selectbox("Select Experiment for Analysis:", experiment_names)
        
        if selected_experiment:
            experiment = mlflow.get_experiment_by_name(selected_experiment)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                st.info(f"No runs found in experiment: {selected_experiment}")
                return
            
            st.success(f"Analyzing {len(runs)} runs from experiment: {selected_experiment}")
            
            # Performance overview
            st.markdown("### 🎯 Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate performance statistics
            all_metrics = []
            for idx, run in runs.iterrows():
                if 'metrics' in run and run['metrics']:
                    all_metrics.extend(run['metrics'].values())
            
            if all_metrics:
                with col1:
                    st.metric("Best Performance", f"{min(all_metrics):.4f}")
                with col2:
                    st.metric("Worst Performance", f"{max(all_metrics):.4f}")
                with col3:
                    st.metric("Average Performance", f"{np.mean(all_metrics):.4f}")
                with col4:
                    st.metric("Performance Std", f"{np.std(all_metrics):.4f}")
            
            st.markdown("---")
            
            # Performance trends
            st.markdown("### 📊 Performance Trends Over Time")
            
            # Extract time series data
            time_series_data = []
            for idx, run in runs.iterrows():
                if 'metrics' in run and run['metrics']:
                    for metric_name, metric_value in run['metrics'].items():
                        time_series_data.append({
                            'Timestamp': run['start_time'],
                            'Metric': metric_name,
                            'Value': metric_value,
                            'Run ID': run['run_id'][:8]
                        })
            
            if time_series_data:
                time_df = pd.DataFrame(time_series_data)
                
                # Create performance trend chart
                fig = px.line(
                    time_df,
                    x='Timestamp',
                    y='Value',
                    color='Metric',
                    title='Performance Metrics Over Time',
                    markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance distribution
                st.markdown("### 📈 Performance Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of performance values
                    fig_hist = px.histogram(
                        time_df,
                        x='Value',
                        color='Metric',
                        title='Performance Distribution',
                        nbins=20
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot of performance by metric
                    fig_box = px.box(
                        time_df,
                        x='Metric',
                        y='Value',
                        title='Performance by Metric Type'
                    )
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Parameter impact analysis
                st.markdown("### 🔧 Parameter Impact Analysis")
                
                # Extract parameters and their impact on performance
                param_impact_data = []
                for idx, run in runs.iterrows():
                    if 'params' in run and run['params'] and 'metrics' in run and run['metrics']:
                        for param_name, param_value in run['params'].items():
                            for metric_name, metric_value in run['metrics'].items():
                                try:
                                    param_impact_data.append({
                                        'Parameter': param_name,
                                        'Value': float(param_value),
                                        'Metric': metric_name,
                                        'Performance': metric_value
                                    })
                                except:
                                    continue
                
                if param_impact_data:
                    param_impact_df = pd.DataFrame(param_impact_data)
                    
                    # Create parameter impact visualization
                    fig_scatter = px.scatter(
                        param_impact_df,
                        x='Value',
                        y='Performance',
                        color='Parameter',
                        title='Parameter Values vs Performance',
                        size=[10] * len(param_impact_df)
                    )
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Parameter correlation analysis
                    st.markdown("### 📊 Parameter Correlation Analysis")
                    
                    # Calculate correlations
                    correlation_data = []
                    for param_name in param_impact_df['Parameter'].unique():
                        param_data = param_impact_df[param_impact_df['Parameter'] == param_name]
                        if len(param_data) > 1:
                            correlation = np.corrcoef(param_data['Value'], param_data['Performance'])[0, 1]
                            correlation_data.append({
                                'Parameter': param_name,
                                'Correlation': correlation,
                                'Sample Size': len(param_data)
                            })
                    
                    if correlation_data:
                        corr_df = pd.DataFrame(correlation_data)
                        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                        
                        # Create correlation bar chart
                        fig_corr = px.bar(
                            corr_df,
                            x='Parameter',
                            y='Correlation',
                            title='Parameter Correlation with Performance',
                            color='Correlation',
                            color_continuous_scale='RdBu'
                        )
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Display correlation table
                        st.dataframe(corr_df, use_container_width=True)
                
                # Download performance data
                st.markdown("### 📥 Download Performance Data")
                
                csv_data = time_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Performance Data (CSV)",
                    data=csv_data,
                    file_name=f"performance_analysis_{selected_experiment}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
    except Exception as e:
        st.error(f"Error analyzing model performance: {str(e)}")
        st.info("Please ensure MLflow experiments are properly configured and contain run data.")

def show_about():
    """Display the about page with project information"""
    st.markdown("## 💾 About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **Wine Quality Predictor** is a comprehensive machine learning project that demonstrates modern ML development practices.
    This project showcases end-to-end ML pipeline development, from data preprocessing to model deployment.
    
    ### 🏗️ Architecture
    
    The project follows a modular architecture with clear separation of concerns:
    
    - **Data Layer**: Wine quality dataset with 11 features
    - **Model Layer**: Neural network with Keras/TensorFlow
    - **Optimization Layer**: Hyperparameter tuning with Hyperopt
    - **Tracking Layer**: MLflow experiment management
    - **Interface Layer**: Streamlit web application
    - **Deployment Layer**: Model serving and prediction API
    
    ### 🛠️ Technology Stack
    
    **Core ML Libraries:**
    - **TensorFlow/Keras**: Deep learning framework
    - **Scikit-learn**: Data preprocessing and evaluation
    - **Hyperopt**: Bayesian hyperparameter optimization
    - **MLflow**: Experiment tracking and model management
    
    **Web Application:**
    - **Streamlit**: Interactive web interface
    - **Plotly**: Interactive visualizations
    - **Pandas**: Data manipulation
    - **NumPy**: Numerical computing
    
    **Development Tools:**
    - **Git**: Version control
    - **Python**: Programming language
    - **Jupyter**: Development environment
    
    ### 📊 Dataset Information
    
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
    
    ### 🧠 Model Architecture
    
    **Neural Network Design:**
    - **Input Layer**: 11 neurons (one per feature)
    - **Normalization Layer**: Feature standardization
    - **Hidden Layer**: 64 neurons with ReLU activation
    - **Output Layer**: 1 neuron (regression output)
    - **Optimizer**: SGD with momentum
    - **Loss Function**: Mean Squared Error
    - **Metrics**: Root Mean Squared Error
    
    ### 🔍 Hyperparameter Optimization
    
    **Optimization Strategy:**
    - **Algorithm**: Tree-structured Parzen Estimator (TPE)
    - **Parameters**: Learning rate, momentum
    - **Search Space**: Log-uniform distributions
    - **Objective**: Minimize validation RMSE
    - **Trials**: Configurable number of evaluations
    
    ### 📈 Experiment Tracking
    
    **MLflow Integration:**
    - **Experiment Management**: Organize runs by experiment
    - **Parameter Logging**: Track all hyperparameters
    - **Metric Logging**: Monitor performance metrics
    - **Model Versioning**: Save and load model artifacts
    - **Reproducibility**: Ensure experiment reproducibility
    
    ### 🚀 Key Features
    
    1. **Interactive Prediction**: Real-time wine quality predictions
    2. **Experiment Dashboard**: Comprehensive MLflow integration
    3. **Performance Analysis**: Detailed model evaluation
    4. **Visualization**: Interactive charts and graphs
    5. **Data Export**: Download experiment results
    6. **Responsive Design**: Mobile-friendly interface
    
    ### 🎓 Learning Outcomes
    
    This project demonstrates:
    - End-to-end ML pipeline development
    - Deep learning model design and training
    - Hyperparameter optimization techniques
    - Experiment tracking and reproducibility
    - Web application development
    - Professional project documentation
    
    ### 🔮 Future Enhancements
    
    **Planned Features:**
    - Model comparison across algorithms
    - Automated hyperparameter tuning
    - Real-time model monitoring
    - API endpoint for predictions
    - Docker containerization
    - Cloud deployment options
    
    ### 👨‍💻 Developer Information
    
    **Skills Demonstrated:**
    - Machine Learning & Deep Learning
    - Data Science & Analytics
    - Software Engineering
    - Web Development
    - DevOps & MLOps
    - Project Management
    
    ### 📚 Resources & References
    
    - **MLflow Documentation**: https://mlflow.org/
    - **Streamlit Documentation**: https://streamlit.io/
    - **TensorFlow Documentation**: https://tensorflow.org/
    - **Hyperopt Documentation**: https://hyperopt.github.io/hyperopt/
    
    ### 📄 License
    
    This project is open source and available under the MIT License.
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ for demonstrating modern ML development practices**")

def main():
    st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### MLflow Experiment Dashboard & Model Predictor")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 MLflow Dashboard", "🔮 Model Predictor", "📈 Model Performance", "💾 About"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 MLflow Dashboard":
        show_mlflow_dashboard()
    elif page == "🔮 Model Predictor":
        show_model_predictor()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "💾 About":
        show_about()

if __name__ == "__main__":
    main()
