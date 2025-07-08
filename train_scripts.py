#!/usr/bin/env python3
"""
Complete ELM Training Script for Truck Production Prediction
Includes: Base ELM, ICA-ELM, MFO-ELM, HBO-ELM, and BOA-ELM models
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
# OPTIMIZATION ALGORITHMS
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
            3: (-3.0, 3.0),  # BOA
        }
        self.range_min, self.range_max = self.search_ranges.get(algorithm_id, (-2, 2))

class ICA_Optimizer(DiverseOptimizer):
    def __init__(self, n_countries=50, n_imperialists=10, max_iter=100, dim=None, seed=None):
        super().__init__(n_countries, max_iter, dim, seed, algorithm_id=0)
        self.n_imperialists = n_imperialists
        self.n_colonies = n_countries - n_imperialists
        
    def optimize(self, objective_function):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        countries = np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
        fitness = np.array([objective_function(country) for country in countries])
        
        for iteration in range(self.max_iter):
            sorted_indices = np.argsort(fitness)
            imperialists = countries[sorted_indices[:self.n_imperialists]].copy()
            colonies = countries[sorted_indices[self.n_imperialists:]].copy()
            
            beta = 2.0 * np.random.random() * (1 - iteration / self.max_iter)
            
            for i in range(self.n_imperialists):
                empire_size = len(colonies) // self.n_imperialists
                start_idx = i * empire_size
                end_idx = start_idx + empire_size if i < self.n_imperialists - 1 else len(colonies)
                
                for j in range(start_idx, end_idx):
                    if j < len(colonies):
                        direction = imperialists[i] - colonies[j]
                        colonies[j] += beta * direction + 0.1 * np.random.normal(0, 1, self.dim)
                        colonies[j] = np.clip(colonies[j], self.range_min, self.range_max)
            
            revolution_rate = 0.3 * np.exp(-iteration / 30)
            for i in range(len(colonies)):
                if np.random.random() < revolution_rate:
                    colonies[i] = np.random.uniform(self.range_min, self.range_max, self.dim)
            
            all_solutions = np.vstack([imperialists, colonies])
            all_fitness = np.array([objective_function(sol) for sol in all_solutions])
            
            best_idx = np.argmin(all_fitness)
            if all_fitness[best_idx] < self.best_fitness:
                self.best_fitness = all_fitness[best_idx]
                self.best_solution = all_solutions[best_idx].copy()
            
            self.convergence_curve.append(self.best_fitness)
            countries = all_solutions
            fitness = all_fitness
            
        return self.best_solution, self.best_fitness

class MFO_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=2)
        
    def optimize(self, objective_function):
        if self.seed is not None:
            np.random.seed(self.seed + 200)
            
        moths = np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
        fitness = np.array([objective_function(moth) for moth in moths])
        
        for iteration in range(self.max_iter):
            flame_no = round(self.n_agents - iteration * ((self.n_agents - 1) / self.max_iter))
            
            sorted_indices = np.argsort(fitness)
            flames = moths[sorted_indices[:flame_no]]
            
            for i in range(self.n_agents):
                for j in range(self.dim):
                    if i < flame_no:
                        distance = abs(flames[i, j] - moths[i, j])
                        b = 1.2
                        t = (np.random.random() - 0.5) * 2
                        moths[i, j] = distance * np.exp(b * t) * np.cos(t * 2 * np.pi) + flames[i, j]
                    else:
                        distance = abs(flames[0, j] - moths[i, j])
                        t = (np.random.random() - 0.5) * 4
                        moths[i, j] = distance * np.exp(1.5 * t) * np.cos(t * 2 * np.pi) + flames[0, j]
                
                moths[i] = np.clip(moths[i], self.range_min, self.range_max)
            
            fitness = np.array([objective_function(moth) for moth in moths])
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = moths[best_idx].copy()
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness

class HBO_Optimizer(DiverseOptimizer):
    def __init__(self, n_badgers=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_badgers, max_iter, dim, seed, algorithm_id=1)
        self.beta = 6
        self.C = 2
        
    def initialize_population(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
    
    def calculate_intensity(self, badger_pos, source_pos):
        distance = np.linalg.norm(badger_pos - source_pos)
        return np.random.random() / (4 * np.pi * (distance + 1e-10))
    
    def digging_phase(self, badger_pos, source_pos, iteration):
        decreasing_factor = self.C - iteration * (self.C / self.max_iter)
        I = self.calculate_intensity(badger_pos, source_pos)
        r1, r2, r3, r4 = np.random.random(4)
        
        if r1 < 0.5:
            new_pos = source_pos + 2 * r2 * I * source_pos
        else:
            new_pos = source_pos + 2 * r3 * I * badger_pos
        
        new_pos += decreasing_factor * np.random.normal(0, 1, self.dim)
        return new_pos
    
    def honey_phase(self, badger_pos, source_pos, iteration):
        decreasing_factor = self.C - iteration * (self.C / self.max_iter)
        I = self.calculate_intensity(badger_pos, source_pos)
        density = np.random.random()
        r5 = np.random.random()
        
        if r5 < 0.5:
            new_pos = source_pos + decreasing_factor * np.random.random(self.dim)
        else:
            dance = np.random.random() * np.sin(2 * np.pi * np.random.random(self.dim))
            new_pos = source_pos + decreasing_factor * dance * density
        
        return new_pos
    
    def optimize(self, objective_function):
        badgers = self.initialize_population()
        fitness = np.array([objective_function(badger) for badger in badgers])
        
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = badgers[best_idx].copy()
        
        for iteration in range(self.max_iter):
            for i in range(self.n_agents):
                switch_prob = 0.5 + 0.3 * np.cos(2 * np.pi * iteration / self.max_iter)
                
                if np.random.random() < switch_prob:
                    new_pos = self.digging_phase(badgers[i], self.best_solution, iteration)
                else:
                    new_pos = self.honey_phase(badgers[i], self.best_solution, iteration)
                
                new_pos = np.clip(new_pos, self.range_min, self.range_max)
                new_fitness = objective_function(new_pos)
                
                if new_fitness < fitness[i]:
                    badgers[i] = new_pos
                    fitness[i] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_pos.copy()
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness

class BOA_Optimizer(DiverseOptimizer):
    def __init__(self, n_butterflies=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_butterflies, max_iter, dim, seed, algorithm_id=3)
        self.c = 0.01
        self.a = 0.1
        self.switch_prob = 0.8
        
    def initialize_population(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
    
    def calculate_fragrance(self, fitness):
        max_fitness = np.max(fitness)
        normalized_fitness = max_fitness - fitness + 1e-10
        fragrance = self.c * (normalized_fitness ** self.a)
        return fragrance
    
    def global_search(self, butterfly_i, best_butterfly, iteration):
        step_size = np.random.random() * self.fragrance[np.argmin(self.fragrance)]
        w = 0.9 - 0.8 * (iteration / self.max_iter)
        r = np.random.random()
        
        new_position = (butterfly_i + 
                       w * (r ** 2) * best_butterfly - 
                       butterfly_i + 
                       step_size * (np.random.random(self.dim) - 0.5))
        return new_position
    
    def local_search(self, butterfly_i, butterfly_j, butterfly_k):
        fragrance_diff = abs(self.fragrance[butterfly_j] - self.fragrance[butterfly_k])
        step_size = np.random.random() * fragrance_diff
        r = np.random.random()
        
        new_position = (butterfly_i + 
                       (r ** 2) * self.butterflies[butterfly_j] - 
                       self.butterflies[butterfly_k] + 
                       step_size * (np.random.random(self.dim) - 0.5))
        return new_position
    
    def optimize(self, objective_function):
        self.butterflies = self.initialize_population()
        fitness = np.array([objective_function(butterfly) for butterfly in self.butterflies])
        
        best_idx = np.argmin(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.butterflies[best_idx].copy()
        
        self.fragrance = self.calculate_fragrance(fitness)
        
        for iteration in range(self.max_iter):
            current_switch_prob = self.switch_prob * (1 - iteration / self.max_iter)
            
            for i in range(self.n_agents):
                r = np.random.random()
                
                if r < current_switch_prob:
                    new_position = self.global_search(
                        self.butterflies[i], self.best_solution, iteration
                    )
                else:
                    available_indices = [j for j in range(self.n_agents) if j != i]
                    j, k = np.random.choice(available_indices, 2, replace=False)
                    new_position = self.local_search(self.butterflies[i], j, k)
                
                new_position = np.clip(new_position, self.range_min, self.range_max)
                new_fitness = objective_function(new_position)
                
                if new_fitness < fitness[i]:
                    self.butterflies[i] = new_position
                    fitness[i] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_position.copy()
            
            self.fragrance = self.calculate_fragrance(fitness)
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness

# =============================================================================
# HYBRID ELM MODELS
# =============================================================================

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
                
                # Add small diversity penalty based on algorithm type
                diversity_penalties = {
                    'ICA': 0.001 * np.sum(np.abs(input_weights)),
                    'MFO': 0.001 * np.sum(np.sin(input_weights)),
                    'HBO': 0.002 * np.sum(np.abs(input_weights)) + 0.001 * np.sum(biases ** 2),
                    'BOA': 0.001 * (np.sum(input_weights ** 2) + np.sum(biases ** 2))
                }
                
                penalty = diversity_penalties.get(self.optimizer_type, 0)
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
        
        # Create optimizers with distinct behaviors
        optimizer_map = {
            'ICA': ICA_Optimizer(n_countries=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed),
            'MFO': MFO_Optimizer(n_agents=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed),
            'HBO': HBO_Optimizer(n_badgers=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed),
            'BOA': BOA_Optimizer(n_butterflies=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed)
        }
        
        self.optimizer = optimizer_map[self.optimizer_type]
        
        print(f"  Optimizing {self.optimizer_type} parameters...")
        best_params, best_fitness = self.optimizer.optimize(self._objective_function)
        
        # Extract and store optimized parameters
        n_features = X.shape[1]
        split_point = n_features * self.n_hidden_nodes
        self.optimized_input_weights = best_params[:split_point].reshape(n_features, self.n_hidden_nodes)
        self.optimized_biases = best_params[split_point:]
        
        print(f"  {self.optimizer_type} optimization completed (fitness: {best_fitness:.6f})")
        
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

def train_all_models():
    """Train all 5 models with diverse optimization"""
    print("="*70)
    print("COMPLETE ELM TRAINING - ALL OPTIMIZATION ALGORITHMS")
    print("="*70)
    
    X, y, feature_info = load_and_preprocess_data()
    if X is None:
        print("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Models with different seeds and configurations for diversity
    models = {
        'Base_ELM': ExtremeeLearningMachine(n_hidden_nodes=100),
        'ICA_ELM': HybridELM(optimizer_type='ICA', n_hidden_nodes=75, n_agents=50, max_iter=80, seed=42),
        'MFO_ELM': HybridELM(optimizer_type='MFO', n_hidden_nodes=80, n_agents=55, max_iter=85, seed=456),
        'HBO_ELM': HybridELM(optimizer_type='HBO', n_hidden_nodes=80, n_agents=50, max_iter=100, seed=789),
        'BOA_ELM': HybridELM(optimizer_type='BOA', n_hidden_nodes=80, n_agents=50, max_iter=100, seed=101112)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            y_pred_test = model.predict(X_test)
            metrics = calculate_all_metrics(y_test, y_pred_test)
            
            results[model_name] = {
                'metrics': metrics,
                'training_time': training_time
            }
            
            print(f"✅ {model_name} completed:")
            print(f"   R²: {metrics['R2']:.4f}")
            print(f"   MAPE: {metrics['MAPE']:.2f}%")
            print(f"   VAF: {metrics['VAF']:.2f}%")
            print(f"   NASH: {metrics['NASH']:.4f}")
            print(f"   AIC: {metrics['AIC']:.2f}")
            print(f"   Training time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            continue
    
    # Save everything
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving models to {save_dir}/...")
    
    for model_name, model in models.items():
        if model_name in results:
            model_path = os.path.join(save_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            print(f"✅ {model_name} saved")
    
    with open(os.path.join(save_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final results summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED - ALL OPTIMIZATION ALGORITHMS")
    print("="*80)
    print(f"{'Model':<12} {'R²':<8} {'MAPE':<8} {'VAF':<8} {'NASH':<8} {'AIC':<10} {'Time':<8}")
    print("-"*80)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        time = result['training_time']
        print(f"{model_name:<12} {metrics['R2']:<8.4f} {metrics['MAPE']:<8.2f} {metrics['VAF']:<8.2f} {metrics['NASH']:<8.4f} {metrics['AIC']:<10.2f} {time:<8.1f}")
    
    best_model = max(results.keys(), key=lambda x: results[x]['metrics']['R2'])
    print(f"\n🏆 Best Model: {best_model} (R² = {results[best_model]['metrics']['R2']:.4f})")
    
    # Algorithm comparison
    optimization_models = [name for name in results.keys() if name != 'Base_ELM']
    if optimization_models:
        avg_optimized_r2 = np.mean([results[model]['metrics']['R2'] for model in optimization_models])
        base_r2 = results['Base_ELM']['metrics']['R2']
        improvement = ((avg_optimized_r2 - base_r2) / base_r2) * 100
        
        print(f"\n📊 Optimization Analysis:")
        print(f"   Average Optimized R²: {avg_optimized_r2:.4f}")
        print(f"   Base ELM R²: {base_r2:.4f}")
        print(f"   Average Improvement: {improvement:.1f}%")
    
    print(f"\n✅ All models trained and saved successfully!")
    print("🚀 Ready for inference and comparison!")

# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================

def load_model_for_prediction(model_name="Best"):
    """Load saved model for prediction"""
    try:
        if model_name == "Best":
            # Load results to find best model
            with open('saved_models/results.json', 'r') as f:
                results = json.load(f)
            model_name = max(results.keys(), key=lambda x: results[x]['metrics']['R2'])
            print(f"🏆 Loading best model: {model_name}")
        
        model = joblib.load(f'saved_models/{model_name}.joblib')
        
        with open('saved_models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        print(f"✅ Model '{model_name}' loaded successfully!")
        return model, feature_info
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def predict_production(model, feature_info, truck_model, nominal_tonnage, 
                      material_type, fixed_time, variable_time, 
                      number_of_loads, cycle_distance):
    """Make production prediction"""
    try:
        # Encode inputs
        truck_idx = feature_info['truck_models'].index(truck_model) if truck_model in feature_info['truck_models'] else 0
        material_idx = feature_info['material_types'].index(material_type) if material_type in feature_info['material_types'] else 0
        
        # Create feature vector
        features = np.array([[
            truck_idx, nominal_tonnage, material_idx,
            fixed_time, variable_time, number_of_loads, cycle_distance
        ]])
        
        prediction = model.predict(features)[0]
        return float(prediction)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def compare_all_models(truck_model, nominal_tonnage, material_type, 
                      fixed_time, variable_time, number_of_loads, cycle_distance):
    """Compare predictions from all models"""
    print("🔮 Comparing predictions from all models...")
    
    try:
        with open('saved_models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        with open('saved_models/results.json', 'r') as f:
            results = json.load(f)
        
        predictions = {}
        
        for model_name in results.keys():
            try:
                model = joblib.load(f'saved_models/{model_name}.joblib')
                prediction = predict_production(
                    model, feature_info, truck_model, nominal_tonnage,
                    material_type, fixed_time, variable_time, 
                    number_of_loads, cycle_distance
                )
                predictions[model_name] = prediction
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
        
        print(f"\n📊 Prediction Comparison:")
        print(f"{'Model':<12} {'Prediction':<12} {'R²':<8}")
        print("-"*35)
        
        for model_name, prediction in predictions.items():
            r2 = results[model_name]['metrics']['R2']
            print(f"{model_name:<12} {prediction:<12.2f} {r2:<8.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Train all models
    train_all_models()
    
    # Example inference with best model
    print("\n" + "="*60)
    print("🔮 EXAMPLE INFERENCE WITH BEST MODEL")
    print("="*60)
    
    model, feature_info = load_model_for_prediction("Best")
    
    if model and feature_info:
        # Example prediction
        example_prediction = predict_production(
            model=model,
            feature_info=feature_info,
            truck_model=feature_info['truck_models'][0],
            nominal_tonnage=65.0,
            material_type=feature_info['material_types'][0],
            fixed_time=2.8,
            variable_time=7.5,
            number_of_loads=12,
            cycle_distance=4.5
        )
        
        if example_prediction:
            print(f"🎯 Best Model Prediction: {example_prediction:.2f} tonnes")
        
        # Compare all models
        print("\n" + "="*60)
        print("📊 COMPARING ALL MODELS")
        print("="*60)
        
        compare_all_models(
            truck_model=feature_info['truck_models'][0],
            nominal_tonnage=65.0,
            material_type=feature_info['material_types'][0],
            fixed_time=2.8,
            variable_time=7.5,
            number_of_loads=12,
            cycle_distance=4.5
        )
    
    print("\n✅ Complete ELM training with all optimization algorithms finished!")
    print("🎉 Models available: Base ELM, ICA-ELM, MFO-ELM, HBO-ELM, BOA-ELM")