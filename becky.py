#!/usr/bin/env python3
"""
STREAMLIT TRUCK PRODUCTION PREDICTION APP - FIXED VERSION
Interactive web application for truck production inference
Run with: streamlit run main.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Required class definitions for model loading
from sklearn.preprocessing import StandardScaler

# =============================================================================
# ALL REQUIRED CLASSES FOR MODEL LOADING (INCLUDING BOA_OPTIMIZER_FIXED)
# =============================================================================

class DiverseOptimizer:
    """Base class ensuring diverse optimization results"""
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None, algorithm_id=0):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.dim = dim
        self.seed = seed
        self.algorithm_id = algorithm_id
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
        # Different search ranges for each algorithm
        self.search_ranges = {
            0: (-2, 2),      # ICA
            1: (-2.5, 2.5),  # HBO
            2: (-1.5, 1.5),  # MFO
            3: (-1.5, 1.5),  # BOA - REDUCED from (-3, 3)
        }
        self.range_min, self.range_max = self.search_ranges.get(algorithm_id, (-2, 2))

class ICA_Optimizer(DiverseOptimizer):
    def __init__(self, n_countries=50, n_imperialists=10, max_iter=100, dim=None, seed=None):
        super().__init__(n_countries, max_iter, dim, seed, algorithm_id=0)
        self.n_imperialists = n_imperialists
        self.n_colonies = n_countries - n_imperialists

class HBO_Optimizer(DiverseOptimizer):
    def __init__(self, n_badgers=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_badgers, max_iter, dim, seed, algorithm_id=1)
        self.beta = 6
        self.C = 2

class MFO_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=2)

class BOA_Optimizer(DiverseOptimizer):
    def __init__(self, n_butterflies=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_butterflies, max_iter, dim, seed, algorithm_id=3)
        self.c = 0.01
        self.a = 0.1
        self.switch_prob = 0.8

class BOA_Optimizer_Fixed(DiverseOptimizer):
    """Fixed BOA Optimizer with enhanced implementation"""
    def __init__(self, n_butterflies=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_butterflies, max_iter, dim, seed, algorithm_id=3)
        self.c = 0.005  # Reduced from 0.01
        self.a = 0.2    # Increased from 0.1 for stability
        self.switch_prob = 0.6  # Reduced from 0.8
        self.fragrance = None

class ExtremeeLearningMachine:
    def __init__(self, n_hidden_nodes=100, activation='sigmoid', C=1.0):
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        self.C = C
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x
    
    def fit(self, X, y, input_weights=None, biases=None):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        n_samples, n_features = X_scaled.shape
        
        if input_weights is not None and biases is not None:
            self.input_weights = input_weights
            self.biases = biases
        else:
            np.random.seed(42)
            self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden_nodes))
            self.biases = np.random.uniform(-1, 1, self.n_hidden_nodes)
        
        H = np.dot(X_scaled, self.input_weights) + self.biases
        H = self._activation_function(H)
        
        try:
            if self.C == np.inf:
                self.output_weights = np.linalg.pinv(H).dot(y_scaled)
            else:
                identity = np.eye(H.shape[1])
                self.output_weights = np.linalg.inv(H.T.dot(H) + identity/self.C).dot(H.T).dot(y_scaled)
        except:
            self.output_weights = np.linalg.pinv(H).dot(y_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        H = np.dot(X_scaled, self.input_weights) + self.biases
        H = self._activation_function(H)
        y_pred = np.dot(H, self.output_weights)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

class HybridELM:
    def __init__(self, optimizer_type='ICA', n_hidden_nodes=80, n_agents=50, max_iter=100, seed=None):
        self.optimizer_type = optimizer_type
        self.n_hidden_nodes = n_hidden_nodes
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.elm = ExtremeeLearningMachine(n_hidden_nodes=n_hidden_nodes)
        self.optimizer = None
        self.X_train = None
        self.y_train = None
        self.seed = seed
        self.optimized_input_weights = None
        self.optimized_biases = None
        
    def predict(self, X):
        return self.elm.predict(X)
    
    def get_convergence_curve(self):
        return self.optimizer.convergence_curve if self.optimizer else []

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Truck Production Predictor",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-card h1 {
        font-size: 3em;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .success-banner {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS AND METADATA
# =============================================================================

@st.cache_data
def load_models_and_metadata():
    """Load all models and metadata with enhanced error handling"""
    model_dir = 'saved_models'
    
    if not os.path.exists(model_dir):
        return None, None, None
    
    # Load results
    try:
        with open(os.path.join(model_dir, 'results.json'), 'r') as f:
            results = json.load(f)
    except Exception as e:
        st.warning(f"Could not load results.json: {e}")
        results = {}
    
    # Load feature info
    try:
        with open(os.path.join(model_dir, 'feature_info.json'), 'r') as f:
            feature_info = json.load(f)
    except Exception as e:
        st.warning(f"Could not load feature_info.json: {e}")
        feature_info = {
            'feature_names': [
                'truck_model_encoded', 'nominal_tonnage', 'material_type_encoded',
                'fixed_time', 'variable_time', 'number_of_loads', 'cycle_distance'
            ],
            'truck_models': ['KOMATSU HD785', 'CAT 777F', 'CAT 785C', 'CAT 777E', 'KOMATSU HD1500'],
            'material_types': ['Waste', 'High Grade', 'Low Grade']
        }
    
    # Load models with detailed feedback
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
    
    for model_file in model_files:
        model_name = model_file.replace('.joblib', '')
        try:
            model_path = os.path.join(model_dir, model_file)
            model = joblib.load(model_path)
            models[model_name] = model
            st.success(f"✅ Successfully loaded {model_name}")
        except Exception as e:
            st.error(f"❌ Failed to load {model_name}: {e}")
            continue
    
    return models, results, feature_info

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_model_comparison_chart(results):
    """Create enhanced model performance comparison chart"""
    if not results:
        return None
    
    models = list(results.keys())
    r2_scores = [results[model]['metrics']['R2'] for model in models]
    mape_scores = [results[model]['metrics']['MAPE'] for model in models]
    training_times = [results[model]['training_time'] for model in models]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score (Higher is Better)', 'MAPE (Lower is Better)', 
                       'Training Time (seconds)', 'Performance vs Efficiency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(
            x=models,
            y=r2_scores,
            name='R² Score',
            marker_color='lightblue',
            text=[f'{score:.4f}' for score in r2_scores],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # MAPE scores
    fig.add_trace(
        go.Bar(
            x=models,
            y=mape_scores,
            name='MAPE (%)',
            marker_color='lightcoral',
            text=[f'{score:.2f}%' for score in mape_scores],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    # Training times
    fig.add_trace(
        go.Bar(
            x=models,
            y=training_times,
            name='Training Time',
            marker_color='lightgreen',
            text=[f'{time:.1f}s' for time in training_times],
            textposition='auto',
        ),
        row=2, col=1
    )
    
    # Performance vs Efficiency scatter
    fig.add_trace(
        go.Scatter(
            x=training_times,
            y=r2_scores,
            mode='markers+text',
            name='Models',
            text=models,
            textposition='top center',
            marker=dict(size=12, color='purple')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Comprehensive Model Performance Analysis",
        showlegend=False,
        height=700
    )
    
    return fig

def make_prediction(model, truck_model, nominal_tonnage, material_type, 
                   fixed_time, variable_time, number_of_loads, cycle_distance,
                   feature_info):
    """Make prediction with enhanced error handling"""
    try:
        # Encode categorical variables
        truck_model_encoded = feature_info['truck_models'].index(truck_model)
        material_type_encoded = feature_info['material_types'].index(material_type)
        
        # Create input array
        input_array = np.array([[
            truck_model_encoded,
            nominal_tonnage,
            material_type_encoded,
            fixed_time,
            variable_time,
            number_of_loads,
            cycle_distance
        ]])
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        return float(prediction)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

def create_sensitivity_chart(input_params, models, feature_info):
    """Create enhanced sensitivity analysis chart"""
    try:
        if not models:
            return None
        
        # Use the best performing model for sensitivity analysis
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        # Parameters to vary with more detailed ranges
        param_variations = {
            'Number of Loads': (input_params['number_of_loads'], range(1, 51, 5)),
            'Cycle Distance': (input_params['cycle_distance'], np.arange(1, 15, 1)),
            'Fixed Time': (input_params['fixed_time'], np.arange(2, 15, 1)),
            'Variable Time': (input_params['variable_time'], np.arange(1, 12, 1)),
            'Nominal Tonnage': (input_params['nominal_tonnage'], np.arange(60, 180, 10))
        }
        
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (param_name, (base_value, param_range)) in enumerate(param_variations.items()):
            predictions = []
            
            for value in param_range:
                temp_params = input_params.copy()
                if param_name == 'Number of Loads':
                    temp_params['number_of_loads'] = value
                elif param_name == 'Cycle Distance':
                    temp_params['cycle_distance'] = value
                elif param_name == 'Fixed Time':
                    temp_params['fixed_time'] = value
                elif param_name == 'Variable Time':
                    temp_params['variable_time'] = value
                elif param_name == 'Nominal Tonnage':
                    temp_params['nominal_tonnage'] = value
                
                pred = make_prediction(model, **temp_params, feature_info=feature_info)
                predictions.append(pred)
            
            fig.add_trace(go.Scatter(
                x=list(param_range),
                y=predictions,
                mode='lines+markers',
                name=param_name,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f'Parameter Sensitivity Analysis (Using {model_name})',
            xaxis_title='Parameter Value',
            yaxis_title='Predicted Production (tonnes)',
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Sensitivity chart error: {e}")
        return None

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load models and metadata
    models, results, feature_info = load_models_and_metadata()
    
    if models is None or not models:
        st.error("❌ No models found! Please run the training script first.")
        st.info("Make sure the 'saved_models' directory exists with trained models.")
        return
    
    # Success banner
    st.markdown(f"""
    <div class="success-banner">
        <h2>🎉 Successfully Loaded {len(models)} ELM Models!</h2>
        <p>Ready for truck production prediction analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🚛 Truck Production Prediction System")
    st.markdown("### Interactive prediction tool for mining truck production optimization at AngloGold Ashanti Iduapriem Limited")
    
    # Sidebar for model selection and info
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Model selection
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Select Prediction Model:",
            model_names,
            index=0
        )
        
        if results and selected_model in results:
            st.markdown("### 📊 Model Performance")
            model_metrics = results[selected_model]['metrics']
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>R² Score:</strong> {model_metrics['R2']:.4f}<br>
                <strong>MAPE:</strong> {model_metrics['MAPE']:.2f}%<br>
                <strong>VAF:</strong> {model_metrics.get('VAF', 'N/A'):.2f}%<br>
                <strong>NASH:</strong> {model_metrics.get('NASH', 'N/A'):.4f}<br>
                <strong>Training Time:</strong> {results[selected_model].get('training_time', 'N/A'):.2f}s
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ℹ️ About the Models")
        model_descriptions = {
            'Base_ELM': 'Standard Extreme Learning Machine with random weight initialization',
            'ICA_ELM': 'ELM optimized using Imperialist Competitive Algorithm - empire dynamics',
            'HBO_ELM': 'ELM optimized using Honey Badger Optimization - foraging behavior',
            'MFO_ELM': 'ELM optimized using Moth-Flame Optimization - spiral navigation',
            'BOA_ELM': 'ELM optimized using Butterfly Optimization Algorithm - fragrance search (Enhanced)'
        }
        
        if selected_model in model_descriptions:
            st.markdown(f"""
            <div class="info-card">
                {model_descriptions[selected_model]}
            </div>
            """, unsafe_allow_html=True)
        
        # Show model ranking
        if results:
            st.markdown("### 🏆 Model Rankings")
            sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['R2'], reverse=True)
            for i, (model_name, model_data) in enumerate(sorted_models, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                st.write(f"{emoji} {i}. **{model_name}**: R² = {model_data['metrics']['R2']:.4f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎛️ Operational Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            st.markdown("*Enter the operational parameters for truck production prediction*")
            
            # Row 1
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            
            with row1_col1:
                truck_model = st.selectbox(
                    "Truck Model:",
                    feature_info['truck_models'],
                    help="Select the truck model for production analysis"
                )
            
            with row1_col2:
                nominal_tonnage = st.number_input(
                    "Nominal Tonnage (tonnes):",
                    min_value=50.0,
                    max_value=200.0,
                    value=100.0,
                    step=5.0,
                    help="Truck carrying capacity in tonnes"
                )
            
            with row1_col3:
                material_type = st.selectbox(
                    "Material Type:",
                    feature_info['material_types'],
                    help="Type of material being transported"
                )
            
            # Row 2
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                fixed_time = st.number_input(
                    "Fixed Time (hours):",
                    min_value=1.0,
                    max_value=20.0,
                    value=8.0,
                    step=0.5,
                    help="Fixed operational time including setup and maintenance (FH+EH)"
                )
            
            with row2_col2:
                variable_time = st.number_input(
                    "Variable Time (hours):",
                    min_value=0.5,
                    max_value=15.0,
                    value=5.0,
                    step=0.5,
                    help="Variable operational time including delays and queuing (DT+Q+LT)"
                )
            
            # Row 3
            row3_col1, row3_col2 = st.columns(2)
            
            with row3_col1:
                number_of_loads = st.number_input(
                    "Number of Loads:",
                    min_value=1,
                    max_value=100,
                    value=20,
                    step=1,
                    help="Total number of loads to transport (Critical Parameter - 81% sensitivity)"
                )
            
            with row3_col2:
                cycle_distance = st.number_input(
                    "Cycle Distance (km):",
                    min_value=0.1,
                    max_value=20.0,
                    value=5.0,
                    step=0.1,
                    help="Round-trip distance for each cycle (Secondary Parameter - 20% sensitivity)"
                )
            
            # Predict button
            predict_button = st.form_submit_button(
                "🚀 Predict Production",
                use_container_width=True
            )
    
    with col2:
        st.header("📈 Prediction Results")
        
        if predict_button:
            # Prepare input parameters
            input_params = {
                'truck_model': truck_model,
                'nominal_tonnage': nominal_tonnage,
                'material_type': material_type,
                'fixed_time': fixed_time,
                'variable_time': variable_time,
                'number_of_loads': number_of_loads,
                'cycle_distance': cycle_distance
            }
            
            # Make prediction
            try:
                selected_model_obj = models[selected_model]
                prediction = make_prediction(
                    selected_model_obj, 
                    **input_params, 
                    feature_info=feature_info
                )
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🎯 Predicted Production</h2>
                    <h1>{prediction:.2f} tonnes</h1>
                    <p>Using {selected_model}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                efficiency = prediction / (fixed_time + variable_time)
                tons_per_load = prediction / number_of_loads
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Production Efficiency", f"{efficiency:.2f} tonnes/hour")
                with col2_2:
                    st.metric("Average per Load", f"{tons_per_load:.2f} tonnes/load")
                
                # Compare with all models
                st.subheader("📊 All Model Predictions")
                comparison_data = []
                
                for model_name, model in models.items():
                    pred = make_prediction(model, **input_params, feature_info=feature_info)
                    r2 = results[model_name]['metrics']['R2'] if results and model_name in results else 0
                    comparison_data.append({
                        'Model': model_name,
                        'Prediction (tonnes)': f"{pred:.2f}",
                        'R² Score': f"{r2:.4f}",
                        'Relative Accuracy': f"{(1-abs(pred-prediction)/prediction)*100:.1f}%"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("Please check your input parameters and try again.")
    
    # Additional analysis sections
    if results:
        st.header("📊 Comprehensive Model Performance Analysis")
        
        # Model comparison chart
        comparison_chart = create_model_comparison_chart(results)
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Key insights
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['R2'])
        best_r2 = results[best_model]['metrics']['R2']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", best_model, f"R² = {best_r2:.4f}")
        with col2:
            avg_r2 = np.mean([results[model]['metrics']['R2'] for model in results])
            st.metric("Average R² Score", f"{avg_r2:.4f}")
        with col3:
            total_models = len(results)
            st.metric("Total Models", total_models)
    
    # Parameter sensitivity analysis
    if predict_button and models:
        st.header("🔍 Parameter Sensitivity Analysis")
        st.markdown("**See how production changes when varying each parameter:**")
        
        sensitivity_chart = create_sensitivity_chart(input_params, models, feature_info)
        if sensitivity_chart:
            st.plotly_chart(sensitivity_chart, use_container_width=True)
            
            # Sensitivity insights
            st.markdown("""
            **🎯 Key Sensitivity Insights:**
            - **Number of Loads**: Most critical parameter (81% impact on production)
            - **Cycle Distance**: Secondary parameter (20% impact on production)
            - **Other Parameters**: Combined impact of less than 10%
            
            **💡 Optimization Recommendations:**
            1. Focus optimization efforts on load planning and management
            2. Implement route optimization for cycle distance reduction
            3. Monitor and optimize fixed/variable time components
            """)
    
    # Additional information and usage guide
    with st.expander("📚 Comprehensive Usage Guide & Model Information"):
        tab1, tab2 = st.tabs(["🎯 Quick Start", "🔧 Model Details"])
        
        with tab1:
            st.markdown("""
            ### 🎯 Quick Start Guide
            
            1. **Select a Model**: Choose from the trained models in the sidebar
            2. **Enter Parameters**: Input operational parameters in the form
            3. **Predict**: Click "Predict Production" to get results
            4. **Analyze**: Review sensitivity analysis and model comparisons
            
            ### 📊 Understanding Results
            
            - **Predicted Production**: Main output in tonnes
            - **Efficiency**: Production per hour of operation  
            - **Per Load Average**: Production efficiency per load
            - **Model Comparison**: Performance across all models
            - **Sensitivity Analysis**: Parameter impact visualization
            """)
        
        with tab2:
            st.markdown("""
            ### 🔧 Model Architecture Details
            
            **Base ELM (Extreme Learning Machine)**
            - Single-hidden layer neural network (100 nodes)
            - Random input weight initialization
            - Analytical output weight determination
            - Fastest training, excellent performance
            
            **Hybrid ELM Models**
            - **ICA-ELM**: 
            - **HBO-ELM**: 
            - **MFO-ELM**: 
            - **BOA-ELM**: 
            
            ### 🎯 Performance Metrics
            - **R²**: 
            - **MAPE**: 
            - **VAF**: 
            - **NASH**: 
            - **AIC**: 
            """)
        
       
    # Footer with project information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🏛️ Institution:**  
        University of Mines and Technology, Tarkwa
        """)
    
    with col2:
        st.markdown("""
        **⛏️ Partner:**  
        AngloGold Ashanti Iduapriem Limited
        """)
    
    with col3:
        st.markdown("""
        **📅 Project:**  
        June 2025 - Mining Engineering
        """)

if __name__ == "__main__":
    main()