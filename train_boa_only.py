#!/usr/bin/env python3
"""
BOA-ELM Only Training Script
Trains just the BOA-ELM model with fixed implementation and saves it
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BASE ELM CLASS
# =============================================================================

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
            print("    ✅ Using optimized weights")
        else:
            np.random.seed(42)
            self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden_nodes))
            self.biases = np.random.uniform(-1, 1, self.n_hidden_nodes)
            print("    ⚠️  Using random weights")
        
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

# =============================================================================
# FIXED BOA OPTIMIZER
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

class BOA_Optimizer_Fixed(DiverseOptimizer):
    def __init__(self, n_butterflies=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_butterflies, max_iter, dim, seed, algorithm_id=3)
        self.c = 0.005  # Reduced from 0.01
        self.a = 0.2    # Increased from 0.1 for stability
        self.switch_prob = 0.6  # Reduced from 0.8
        self.fragrance = None
        
    def initialize_population(self):
        if self.seed is not None:
            np.random.seed(self.seed + 300)  # Different seed offset
        return np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
    
    def calculate_fragrance(self, fitness):
        """Fixed fragrance calculation with better numerical stability"""
        # Handle edge cases
        if len(fitness) == 0:
            return np.array([])
        
        max_fitness = np.max(fitness)
        min_fitness = np.min(fitness)
        
        # If all fitness values are the same
        if max_fitness == min_fitness:
            return np.ones(len(fitness)) * self.c
        
        # Normalize to [0, 1] and invert (lower fitness = higher fragrance)
        normalized = 1 - (fitness - min_fitness) / (max_fitness - min_fitness + 1e-10)
        fragrance = self.c * (normalized + 0.1)  # Add small constant for stability
        return fragrance
    
    def global_search(self, butterfly_i, best_butterfly, fragrance_i, iteration):
        """Improved global search with proper fragrance usage"""
        # Adaptive parameters
        step_size = fragrance_i * np.random.random() * 0.5  # Reduced randomness
        w = 0.9 - 0.5 * (iteration / self.max_iter)  # Less aggressive decay
        r = np.random.random()
        
        # More conservative global search
        direction = best_butterfly - butterfly_i
        noise = np.random.normal(0, 0.1, self.dim)  # Reduced noise
        
        new_position = butterfly_i + w * r * direction + step_size * noise
        return new_position
    
    def local_search(self, butterfly_i, butterfly_j, butterfly_k, fragrance_j, fragrance_k):
        """Fixed local search with correct indexing"""
        fragrance_diff = abs(fragrance_j - fragrance_k)
        step_size = fragrance_diff * np.random.random() * 0.3  # Reduced step size
        r = np.random.random() * 0.5  # Reduced randomness
        
        # Fixed: Use actual butterfly positions
        direction = butterfly_j - butterfly_k
        noise = np.random.normal(0, 0.05, self.dim)  # Small noise
        
        new_position = butterfly_i + r * direction + step_size * noise
        return new_position
    
    def optimize(self, objective_function):
        print(f"    Initializing BOA with {self.n_agents} butterflies...")
        self.butterflies = self.initialize_population()
        fitness = np.array([objective_function(butterfly) for butterfly in self.butterflies])
        
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.butterflies[best_idx].copy()
        
        print(f"    Initial best fitness: {self.best_fitness:.6f}")
        
        # Initialize fragrance
        self.fragrance = self.calculate_fragrance(fitness)
        
        for iteration in range(self.max_iter):
            # Adaptive switch probability - more global search early, more local later
            progress = iteration / self.max_iter
            current_switch_prob = self.switch_prob * (1 - 0.5 * progress)
            
            improvements = 0
            
            for i in range(self.n_agents):
                r = np.random.random()
                
                if r < current_switch_prob:
                    # Global search phase
                    new_position = self.global_search(
                        self.butterflies[i], self.best_solution, self.fragrance[i], iteration
                    )
                else:
                    # Local search phase - FIXED
                    available_indices = [j for j in range(self.n_agents) if j != i]
                    if len(available_indices) >= 2:
                        j_idx, k_idx = np.random.choice(available_indices, 2, replace=False)
                        new_position = self.local_search(
                            self.butterflies[i], 
                            self.butterflies[j_idx],  # Use actual positions
                            self.butterflies[k_idx],  # Use actual positions
                            self.fragrance[j_idx],    # Use corresponding fragrance
                            self.fragrance[k_idx]     # Use corresponding fragrance
                        )
                    else:
                        # Fallback to global search if not enough butterflies
                        new_position = self.global_search(
                            self.butterflies[i], self.best_solution, self.fragrance[i], iteration
                        )
                
                # Clip to bounds
                new_position = np.clip(new_position, self.range_min, self.range_max)
                new_fitness = objective_function(new_position)
                
                # Update if better
                if new_fitness < fitness[i]:
                    self.butterflies[i] = new_position
                    fitness[i] = new_fitness
                    improvements += 1
                    
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_position.copy()
            
            # Update fragrance for all butterflies
            self.fragrance = self.calculate_fragrance(fitness)
            self.convergence_curve.append(self.best_fitness)
            
            # Progress reporting
            if iteration % 20 == 0:
                print(f"    Iteration {iteration:3d}: Best fitness = {self.best_fitness:.6f}, Improvements = {improvements}")
        
        print(f"    Final best fitness: {self.best_fitness:.6f}")
        return self.best_solution, self.best_fitness

# =============================================================================
# HYBRID ELM MODEL
# =============================================================================

class HybridELM:
    def __init__(self, optimizer_type='BOA', n_hidden_nodes=75, n_agents=40, max_iter=80, seed=None):
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
        
    def _objective_function(self, params):
        try:
            n_features = self.X_train.shape[1]
            split_point = n_features * self.n_hidden_nodes
            
            input_weights = params[:split_point].reshape(n_features, self.n_hidden_nodes)
            biases = params[split_point:]
            
            X_scaled = self.elm.scaler_X.transform(self.X_train)
            y_scaled = self.elm.scaler_y.transform(self.y_train.reshape(-1, 1)).flatten()
            
            H = np.dot(X_scaled, input_weights) + biases
            H = self.elm._activation_function(H)
            
            try:
                output_weights = np.linalg.pinv(H).dot(y_scaled)
                y_pred = np.dot(H, output_weights)
                mse = np.mean((y_scaled - y_pred) ** 2)
                
                # Add small regularization for BOA
                penalty = 0.001 * (np.sum(input_weights ** 2) + np.sum(biases ** 2))
                return mse + penalty
                
            except:
                return 1000.0
                
        except Exception:
            return 1000.0
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # Initialize scalers first
        self.elm.scaler_X.fit(X)
        self.elm.scaler_y.fit(y.reshape(-1, 1))
        
        dim = X.shape[1] * self.n_hidden_nodes + self.n_hidden_nodes
        
        # Create BOA optimizer
        self.optimizer = BOA_Optimizer_Fixed(
            n_butterflies=self.n_agents, 
            max_iter=self.max_iter, 
            dim=dim, 
            seed=self.seed
        )
        
        print(f"  Optimizing BOA parameters (dim={dim})...")
        best_params, best_fitness = self.optimizer.optimize(self._objective_function)
        
        # Extract and store optimized parameters
        n_features = X.shape[1]
        split_point = n_features * self.n_hidden_nodes
        self.optimized_input_weights = best_params[:split_point].reshape(n_features, self.n_hidden_nodes)
        self.optimized_biases = best_params[split_point:]
        
        print(f"  BOA optimization completed (fitness: {best_fitness:.6f})")
        
        # Train ELM with optimized parameters
        self.elm.fit(X, y, input_weights=self.optimized_input_weights, biases=self.optimized_biases)
    
    def predict(self, X):
        return self.elm.predict(X)
    
    def get_convergence_curve(self):
        return self.optimizer.convergence_curve if self.optimizer else []

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_all_metrics(y_true, y_pred):
    """Calculate R², MAPE, VAF, NASH, AIC"""
    n = len(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    vaf = (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100
    nash = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    mse = np.mean((y_true - y_pred) ** 2)
    aic = n * np.log(mse + 1e-8) + 2 * 2
    
    return {
        'R2': float(r2),
        'MAPE': float(mape),
        'VAF': float(vaf),
        'NASH': float(nash),
        'AIC': float(aic)
    }

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess the truck data"""
    print("Loading truck production data...")
    
    try:
        data = pd.read_csv('Truck Production Data AAIL .csv')
        print(f"Data loaded: {data.shape}")
        
        data.columns = [col.strip().rstrip(',') for col in data.columns]
        
        column_mapping = {
            'Truck Model (TM)': 'truck_model',
            'Nominal Tonnage (tonnes) (NT)': 'nominal_tonnage',
            'Material Type (MAT)': 'material_type',
            'Fixed Time(FH+EH)': 'fixed_time',
            'Variable Time (DT+Q+LT)': 'variable_time',
            'Number of loads (NL)': 'number_of_loads',
            'Cycle Distance (km) (CD)': 'cycle_distance',
            'Production (t) (P)': 'production'
        }
        
        data = data.rename(columns=column_mapping)
        
        le_truck = LabelEncoder()
        data['truck_model_encoded'] = le_truck.fit_transform(data['truck_model'])
        
        le_material = LabelEncoder()
        data['material_type_encoded'] = le_material.fit_transform(data['material_type'])
        
        feature_columns = [
            'truck_model_encoded', 'nominal_tonnage', 'material_type_encoded',
            'fixed_time', 'variable_time', 'number_of_loads', 'cycle_distance'
        ]
        
        X = data[feature_columns].values
        y = data['production'].values
        
        feature_info = {
            'feature_names': feature_columns,
            'truck_models': list(le_truck.classes_),
            'material_types': list(le_material.classes_)
        }
        
        print("Data preprocessing completed successfully")
        return X, y, feature_info
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_boa_elm():
    """Train only BOA-ELM with fixed implementation"""
    print("="*70)
    print("BOA-ELM TRAINING - FIXED IMPLEMENTATION")
    print("="*70)
    
    X, y, feature_info = load_and_preprocess_data()
    if X is None:
        print("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Create BOA-ELM model with optimized parameters
    print(f"\nTraining BOA-ELM...")
    model = HybridELM(
        optimizer_type='BOA', 
        n_hidden_nodes=75,    # Reduced for stability
        n_agents=40,          # Reduced for faster convergence  
        max_iter=80,          # Reduced iterations
        seed=101112
    )
    
    try:
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Calculate metrics
        test_metrics = calculate_all_metrics(y_test, y_pred_test)
        train_metrics = calculate_all_metrics(y_train, y_pred_train)
        
        results = {
            'BOA_ELM': {
                'metrics': test_metrics,
                'train_metrics': train_metrics,
                'training_time': training_time,
                'convergence_curve': model.get_convergence_curve()
            }
        }
        
        print(f"\n✅ BOA-ELM Training Completed:")
        print(f"   Test R²: {test_metrics['R2']:.4f}")
        print(f"   Test MAPE: {test_metrics['MAPE']:.2f}%")
        print(f"   Test VAF: {test_metrics['VAF']:.2f}%")
        print(f"   Test NASH: {test_metrics['NASH']:.4f}")
        print(f"   Test AIC: {test_metrics['AIC']:.2f}")
        print(f"   Training time: {training_time:.1f}s")
        
        print(f"\n   Train R²: {train_metrics['R2']:.4f}")
        print(f"   Train MAPE: {train_metrics['MAPE']:.2f}%")
        
        # Save model and results
        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving BOA-ELM model...")
        
        # Save the model
        model_path = os.path.join(save_dir, 'BOA_ELM.joblib')
        joblib.dump(model, model_path)
        print(f"✅ BOA_ELM model saved to {model_path}")
        
        # Save feature info (if not exists)
        feature_path = os.path.join(save_dir, 'feature_info.json')
        if not os.path.exists(feature_path):
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"✅ Feature info saved to {feature_path}")
        
        # Update or create results.json
        results_path = os.path.join(save_dir, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            existing_results['BOA_ELM'] = results['BOA_ELM']
            results = existing_results
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {results_path}")
        
        print(f"\n🎉 BOA-ELM training completed successfully!")
        print(f"📊 Model Performance Summary:")
        print(f"   • R² Score: {test_metrics['R2']:.4f}")
        print(f"   • MAPE: {test_metrics['MAPE']:.2f}%")
        print(f"   • Training Time: {training_time:.1f} seconds")
        
        # Check if this is better than previous BOA
        if test_metrics['R2'] > 0.95:
            print(f"🏆 Excellent performance! R² > 0.95")
        elif test_metrics['R2'] > 0.90:
            print(f"✅ Good performance! R² > 0.90")
        else:
            print(f"⚠️  Consider further optimization if R² < 0.90")
            
    except Exception as e:
        print(f"❌ BOA-ELM training failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    train_boa_elm()