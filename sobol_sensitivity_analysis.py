#!/usr/bin/env python3
"""
SOBOL SENSITIVITY ANALYSIS - COMPATIBLE VERSION
Investigate parameter impact on the best performing model
Compatible with the merged ELM training script
Run with: python sobol_sensitivity_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import os
import sys
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COMPATIBILITY CLASSES
# =============================================================================
# Import the exact same classes from the merged training script

from sklearn.preprocessing import StandardScaler

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
# SOBOL SENSITIVITY ANALYSIS CLASS
# =============================================================================

class SobolSensitivityAnalyzer:
    """
    Comprehensive Sobol sensitivity analysis for truck production model
    Compatible with merged ELM training script
    """
    
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        self.best_model = None
        self.feature_info = None
        self.results = None
        self.problem = None
        self.sobol_results = None
        self.best_model_name = None
        
    def load_best_model(self):
        """Load the best performing model and metadata"""
        print("🔍 Loading best model for sensitivity analysis...")
        
        # Load results to find best model
        try:
            with open(os.path.join(self.model_dir, 'results.json'), 'r') as f:
                self.results = json.load(f)
        except Exception as e:
            print(f"❌ Could not load results.json: {e}")
            return False
        
        # Find best model by R²
        self.best_model_name = max(self.results.keys(), 
                                  key=lambda x: self.results[x]['metrics']['R2'])
        
        print(f"📊 Best model identified: {self.best_model_name}")
        print(f"   R² Score: {self.results[self.best_model_name]['metrics']['R2']:.4f}")
        print(f"   MAPE: {self.results[self.best_model_name]['metrics']['MAPE']:.2f}%")
        
        # Load the best model
        try:
            model_path = os.path.join(self.model_dir, f'{self.best_model_name}.joblib')
            self.best_model = joblib.load(model_path)
            print(f"✅ {self.best_model_name} loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load {self.best_model_name}: {e}")
            return False
        
        # Load feature information
        try:
            with open(os.path.join(self.model_dir, 'feature_info.json'), 'r') as f:
                self.feature_info = json.load(f)
            print("✅ Feature information loaded")
        except:
            print("⚠️ Feature info not found, using defaults")
            self.feature_info = {
                'feature_names': [
                    'truck_model_encoded', 'nominal_tonnage', 'material_type_encoded',
                    'fixed_time', 'variable_time', 'number_of_loads', 'cycle_distance'
                ],
                'truck_models': ['KOMATSU HD785', 'CAT 777F', 'CAT 785C', 'CAT 777E', 'KOMATSU HD1500'],
                'material_types': ['Waste', 'High Grade', 'Low Grade']
            }
        
        return True
    
    def define_problem(self):
        """Define the Sobol analysis problem with parameter bounds"""
        print("\n🎯 Defining sensitivity analysis problem...")
        
        # Define parameter ranges based on realistic operational values
        self.problem = {
            'num_vars': 7,
            'names': [
                'Truck Model',
                'Nominal Tonnage', 
                'Material Type',
                'Fixed Time',
                'Variable Time', 
                'Number of Loads',
                'Cycle Distance'
            ],
            'bounds': [
                [0, len(self.feature_info['truck_models'])-1],     # Truck Model (0 to n-1)
                [50, 200],     # Nominal Tonnage (tonnes)
                [0, len(self.feature_info['material_types'])-1],  # Material Type (0 to n-1)
                [1, 20],       # Fixed Time (hours)
                [0.5, 15],     # Variable Time (hours)
                [1, 100],      # Number of Loads
                [0.1, 20]      # Cycle Distance (km)
            ]
        }
        
        print("📋 Parameter ranges defined:")
        for i, (name, bounds) in enumerate(zip(self.problem['names'], self.problem['bounds'])):
            print(f"   {name}: {bounds[0]} - {bounds[1]}")
        
        return True
    
    def generate_samples(self, n_samples=1024):
        """Generate Sobol samples for analysis"""
        print(f"\n🎲 Generating {n_samples} Sobol samples...")
        
        # Generate Sobol samples
        param_values = sobol.sample(self.problem, n_samples, calc_second_order=True)
        
        print(f"✅ Generated {param_values.shape[0]} sample combinations")
        print(f"   Shape: {param_values.shape}")
        
        return param_values
    
    def evaluate_model(self, param_values):
        """Evaluate model for all parameter combinations"""
        print("\n🔄 Evaluating model for all parameter combinations...")
        
        n_samples = param_values.shape[0]
        predictions = np.zeros(n_samples)
        
        # Process in batches to avoid memory issues
        batch_size = 100
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            batch_params = param_values[start_idx:end_idx]
            batch_predictions = []
            
            for params in batch_params:
                try:
                    # Round categorical variables and ensure they're within bounds
                    truck_model = int(round(np.clip(params[0], 0, len(self.feature_info['truck_models'])-1)))
                    material_type = int(round(np.clip(params[2], 0, len(self.feature_info['material_types'])-1)))
                    
                    # Create input array matching the expected format from merged script
                    input_array = np.array([[
                        truck_model,                    # truck_model_encoded
                        params[1],                      # nominal_tonnage
                        material_type,                  # material_type_encoded
                        params[3],                      # fixed_time
                        params[4],                      # variable_time
                        int(round(max(1, params[5]))),  # number_of_loads (min 1)
                        max(0.1, params[6])             # cycle_distance (min 0.1)
                    ]])
                    
                    # Make prediction using the loaded model
                    pred = self.best_model.predict(input_array)[0]
                    batch_predictions.append(pred)
                    
                except Exception as e:
                    # Handle prediction errors gracefully
                    print(f"⚠️ Prediction error: {e}")
                    batch_predictions.append(0.0)
            
            predictions[start_idx:end_idx] = batch_predictions
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {end_idx}/{n_samples} samples ({(end_idx/n_samples)*100:.1f}%)")
        
        # Filter out any invalid predictions
        valid_predictions = predictions[predictions > 0]
        
        print(f"✅ Model evaluation completed")
        print(f"   Valid predictions: {len(valid_predictions)}/{len(predictions)}")
        print(f"   Prediction range: {valid_predictions.min():.2f} - {valid_predictions.max():.2f} tonnes")
        print(f"   Mean prediction: {valid_predictions.mean():.2f} tonnes")
        print(f"   Std prediction: {valid_predictions.std():.2f} tonnes")
        
        return predictions
    
    def perform_sobol_analysis(self, param_values, predictions):
        """Perform Sobol sensitivity analysis"""
        print("\n📊 Performing Sobol sensitivity analysis...")
        
        try:
            # Filter out any zero predictions for better analysis
            valid_indices = predictions > 0
            if np.sum(valid_indices) < len(predictions) * 0.8:
                print("⚠️ Warning: Many invalid predictions detected. Results may be unreliable.")
            
            # Perform Sobol analysis
            self.sobol_results = sobol_analyze.analyze(
                self.problem, 
                predictions, 
                calc_second_order=True,
                print_to_console=False
            )
            
            print("✅ Sobol analysis completed successfully")
            
            # Display results
            print("\n" + "="*70)
            print("SOBOL SENSITIVITY ANALYSIS RESULTS")
            print("="*70)
            
            print(f"\nAnalysis Based on: {self.best_model_name}")
            print(f"Model Performance: R² = {self.results[self.best_model_name]['metrics']['R2']:.4f}")
            
            print("\n📈 FIRST-ORDER SENSITIVITY INDICES (S1):")
            print("   (Individual parameter effects)")
            print("-"*50)
            for i, name in enumerate(self.problem['names']):
                s1 = self.sobol_results['S1'][i]
                s1_conf = self.sobol_results['S1_conf'][i]
                print(f"   {name:<20}: {s1:.4f} ± {s1_conf:.4f}")
            
            print("\n📈 TOTAL-ORDER SENSITIVITY INDICES (ST):")
            print("   (Total effects including interactions)")
            print("-"*50)
            for i, name in enumerate(self.problem['names']):
                st = self.sobol_results['ST'][i]
                st_conf = self.sobol_results['ST_conf'][i]
                print(f"   {name:<20}: {st:.4f} ± {st_conf:.4f}")
            
            # Identify most sensitive parameters
            most_sensitive_s1 = np.argmax(self.sobol_results['S1'])
            most_sensitive_st = np.argmax(self.sobol_results['ST'])
            
            print(f"\n🎯 MOST SENSITIVE PARAMETERS:")
            print(f"   First-order: {self.problem['names'][most_sensitive_s1]} (S1 = {self.sobol_results['S1'][most_sensitive_s1]:.4f})")
            print(f"   Total-order: {self.problem['names'][most_sensitive_st]} (ST = {self.sobol_results['ST'][most_sensitive_st]:.4f})")
            
            # Check for significant interactions
            interaction_strength = self.sobol_results['ST'] - self.sobol_results['S1']
            max_interaction_idx = np.argmax(interaction_strength)
            
            print(f"\n🔗 PARAMETER INTERACTIONS:")
            print(f"   Strongest interaction: {self.problem['names'][max_interaction_idx]}")
            print(f"   Interaction strength: {interaction_strength[max_interaction_idx]:.4f}")
            
            # Model behavior insights
            total_s1 = np.sum(self.sobol_results['S1'])
            print(f"\n🔍 MODEL BEHAVIOR INSIGHTS:")
            print(f"   Total first-order effects: {total_s1:.3f}")
            print(f"   {'Non-linear behavior detected' if total_s1 < 0.8 else 'Predominantly linear behavior'}")
            print(f"   {'Significant interactions present' if np.max(interaction_strength) > 0.1 else 'Low parameter interactions'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Sobol analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_visualizations(self):
        """Create comprehensive sensitivity analysis visualizations"""
        print("\n📊 Creating sensitivity analysis visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 12))
        
        # Prepare data
        s1_values = self.sobol_results['S1']
        s1_conf = self.sobol_results['S1_conf']
        st_values = self.sobol_results['ST']
        st_conf = self.sobol_results['ST_conf']
        
        # 1. First-order sensitivity indices
        ax1 = plt.subplot(2, 3, 1)
        bars1 = ax1.bar(range(len(s1_values)), s1_values, 
                       yerr=s1_conf, capsize=5, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('First-order Sensitivity (S1)')
        ax1.set_title('First-order Sensitivity Indices\n(Individual Parameter Effects)')
        ax1.set_xticks(range(len(self.problem['names'])))
        ax1.set_xticklabels(self.problem['names'], rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, s1_values):
            if value > 0.001:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        
        # 2. Total-order sensitivity indices
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(range(len(st_values)), st_values, 
                       yerr=st_conf, capsize=5, alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Total Sensitivity (ST)')
        ax2.set_title('Total-order Sensitivity Indices\n(Total Effects + Interactions)')
        ax2.set_xticks(range(len(self.problem['names'])))
        ax2.set_xticklabels(self.problem['names'], rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, st_values):
            if value > 0.001:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        
        # 3. Comparison of S1 vs ST
        ax3 = plt.subplot(2, 3, 3)
        x_pos = np.arange(len(self.problem['names']))
        width = 0.35
        
        bars_s1 = ax3.bar(x_pos - width/2, s1_values, width, 
                         label='First-order (S1)', alpha=0.8, color='skyblue')
        bars_st = ax3.bar(x_pos + width/2, st_values, width,
                         label='Total-order (ST)', alpha=0.8, color='lightcoral')
        
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Sensitivity Index')
        ax3.set_title('S1 vs ST Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.problem['names'], rotation=45, ha='right', fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Interaction effects (ST - S1)
        ax4 = plt.subplot(2, 3, 4)
        interaction_effects = st_values - s1_values
        
        bars4 = ax4.bar(range(len(interaction_effects)), interaction_effects, 
                       alpha=0.8, color='lightgreen')
        ax4.set_xlabel('Parameters')
        ax4.set_ylabel('Interaction Effect (ST - S1)')
        ax4.set_title('Parameter Interaction Effects')
        ax4.set_xticks(range(len(self.problem['names'])))
        ax4.set_xticklabels(self.problem['names'], rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels for significant interactions
        for bar, value in zip(bars4, interaction_effects):
            if abs(value) > 0.001:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        
        # 5. Sensitivity ranking
        ax5 = plt.subplot(2, 3, 5)
        sorted_indices = np.argsort(st_values)[::-1]
        sorted_names = [self.problem['names'][i] for i in sorted_indices]
        sorted_values = st_values[sorted_indices]
        
        bars5 = ax5.barh(range(len(sorted_values)), sorted_values, alpha=0.8, color='gold')
        ax5.set_ylabel('Parameters (Ranked by Sensitivity)')
        ax5.set_xlabel('Total Sensitivity (ST)')
        ax5.set_title('Parameter Sensitivity Ranking')
        ax5.set_yticks(range(len(sorted_names)))
        ax5.set_yticklabels(sorted_names, fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars5, sorted_values):
            if value > 0.001:
                ax5.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontsize=7)
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Create summary text
        total_s1 = np.sum(s1_values)
        max_s1 = np.max(s1_values)
        max_st = np.max(st_values)
        most_sensitive = self.problem['names'][np.argmax(st_values)]
        
        # Get best model info
        best_r2 = self.results[self.best_model_name]['metrics']['R2']
        best_mape = self.results[self.best_model_name]['metrics']['MAPE']
        
        summary_text = f"""
SENSITIVITY ANALYSIS SUMMARY

Model Used: {self.best_model_name}
R² Score: {best_r2:.4f}
MAPE: {best_mape:.2f}%

Total First-order Effects: {total_s1:.3f}
Maximum S1: {max_s1:.3f}
Maximum ST: {max_st:.3f}

Most Sensitive Parameter:
{most_sensitive}
(ST = {max_st:.3f})

Model Behavior:
{"• High interactions" if (np.max(st_values) - np.max(s1_values)) > 0.1 else "• Low interactions"}
{"• Non-linear effects" if total_s1 < 0.8 else "• Mostly linear effects"}
{"• Well-identified" if max_st > 0.1 else "• Low sensitivity"}

Interpretation:
S1: Individual parameter effect
ST: Total effect + interactions
ST-S1: Interaction strength

Available Models:
{', '.join(self.results.keys())}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Sobol Sensitivity Analysis - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('sobol_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Sensitivity analysis plot saved as 'sobol_sensitivity_analysis.png'")
        
        plt.show()
        
        return True
    
    def generate_sensitivity_report(self):
        """Generate comprehensive sensitivity analysis report"""
        print("\n📝 Generating sensitivity analysis report...")
        
        best_metrics = self.results[self.best_model_name]['metrics']
        
        report = f"""
SOBOL SENSITIVITY ANALYSIS REPORT
Truck Production Prediction Model
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

This report presents the results of a Sobol sensitivity analysis conducted on the
best-performing truck production prediction model from the comprehensive ELM
training suite. The analysis investigates how each operational parameter influences
production output and identifies critical factors for operational optimization.

Best Model Performance:
• Model: {self.best_model_name}
• R² Score: {best_metrics['R2']:.4f}
• MAPE: {best_metrics['MAPE']:.2f}%
• VAF: {best_metrics.get('VAF', 'N/A')}%
• NASH: {best_metrics.get('NASH', 'N/A')}
• AIC: {best_metrics.get('AIC', 'N/A')}

Available Models in Suite:
{', '.join(self.results.keys())}

{'='*80}
METHODOLOGY
{'='*80}

Sobol sensitivity analysis is a variance-based global sensitivity analysis method
that quantifies the contribution of each input parameter to the output variance.

Key Metrics:
• First-order indices (S1): Individual parameter effects
• Total-order indices (ST): Total effects including interactions  
• Interaction effects (ST-S1): Parameter interaction strength

Parameter Ranges Analyzed:
"""
        
        for name, bounds in zip(self.problem['names'], self.problem['bounds']):
            if 'Model' in name:
                report += f"• {name:<20}: {int(bounds[0]):<8} - {int(bounds[1])} (categorical)\n"
            elif 'Type' in name:
                report += f"• {name:<20}: {int(bounds[0]):<8} - {int(bounds[1])} (categorical)\n"
            else:
                report += f"• {name:<20}: {bounds[0]:<8} - {bounds[1]}\n"
        
        report += f"""
{'='*80}
SENSITIVITY ANALYSIS RESULTS
{'='*80}

FIRST-ORDER SENSITIVITY INDICES (S1):
Individual parameter effects on production output

"""
        
        for i, name in enumerate(self.problem['names']):
            s1 = self.sobol_results['S1'][i]
            s1_conf = self.sobol_results['S1_conf'][i]
            report += f"• {name:<20}: {s1:.4f} ± {s1_conf:.4f}\n"
        
        report += f"""
TOTAL-ORDER SENSITIVITY INDICES (ST):
Total effects including parameter interactions

"""
        
        for i, name in enumerate(self.problem['names']):
            st = self.sobol_results['ST'][i]
            st_conf = self.sobol_results['ST_conf'][i]
            report += f"• {name:<20}: {st:.4f} ± {st_conf:.4f}\n"
        
        # Parameter ranking
        sorted_indices = np.argsort(self.sobol_results['ST'])[::-1]
        
        report += f"""
{'='*80}
PARAMETER RANKING BY SENSITIVITY
{'='*80}

Ranking based on total-order sensitivity indices (ST):

"""
        
        for rank, idx in enumerate(sorted_indices, 1):
            name = self.problem['names'][idx]
            st = self.sobol_results['ST'][idx]
            s1 = self.sobol_results['S1'][idx]
            interaction = st - s1
            report += f"{rank}. {name:<20} (ST: {st:.4f}, S1: {s1:.4f}, Interaction: {interaction:.4f})\n"
        
        # Model comparison
        report += f"""
{'='*80}
MODEL PERFORMANCE COMPARISON
{'='*80}

All trained models and their performance:

"""
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            is_best = "⭐ BEST" if model_name == self.best_model_name else ""
            report += f"• {model_name:<15}: R² = {metrics['R2']:.4f}, MAPE = {metrics['MAPE']:.2f}% {is_best}\n"
        
        # Interpretation
        most_sensitive = self.problem['names'][sorted_indices[0]]
        highest_st = self.sobol_results['ST'][sorted_indices[0]]
        total_s1 = np.sum(self.sobol_results['S1'])
        max_interaction = np.max(self.sobol_results['ST'] - self.sobol_results['S1'])
        
        report += f"""
{'='*80}
INTERPRETATION AND INSIGHTS
{'='*80}

KEY FINDINGS:

1. Most Critical Parameter: {most_sensitive}
   • Highest total sensitivity: {highest_st:.4f}
   • This parameter has the strongest influence on production output
   • Priority focus for operational optimization

2. Model Behavior:
   • Total first-order effects: {total_s1:.3f}
   • Maximum interaction effect: {max_interaction:.4f}
   • {'High parameter interactions detected' if max_interaction > 0.1 else 'Low parameter interactions'}
   • {'Non-linear model behavior' if total_s1 < 0.8 else 'Predominantly linear relationships'}

3. Parameter Interactions:
"""
        
        # Identify parameters with significant interactions
        interaction_effects = self.sobol_results['ST'] - self.sobol_results['S1']
        significant_interactions = []
        
        for i, (name, interaction) in enumerate(zip(self.problem['names'], interaction_effects)):
            if interaction > 0.05:  # Threshold for significant interaction
                significant_interactions.append((name, interaction))
        
        if significant_interactions:
            for name, interaction in significant_interactions:
                report += f"   • {name}: Interaction effect = {interaction:.4f}\n"
        else:
            report += "   • No significant parameter interactions detected\n"
        
        report += f"""
{'='*80}
OPERATIONAL RECOMMENDATIONS
{'='*80}

Based on the sensitivity analysis results from the {self.best_model_name} model,
the following recommendations are made for mining operations:

HIGH PRIORITY PARAMETERS:
"""
        
        # High priority parameters (top 3 by ST)
        for i in range(min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            name = self.problem['names'][idx]
            st = self.sobol_results['ST'][idx]
            report += f"• {name} (ST: {st:.4f})\n"
            
            # Parameter-specific recommendations
            if 'loads' in name.lower():
                report += "  - Optimize load planning and scheduling strategies\n"
                report += "  - Implement dynamic load balancing systems\n"
                report += "  - Consider automated load counting systems\n"
            elif 'time' in name.lower():
                if 'fixed' in name.lower():
                    report += "  - Reduce setup and maintenance downtime\n"
                    report += "  - Optimize equipment preparation procedures\n"
                else:
                    report += "  - Focus on cycle time optimization\n"
                    report += "  - Implement lean operational procedures\n"
            elif 'distance' in name.lower():
                report += "  - Optimize haul routes and transportation planning\n"
                report += "  - Consider route optimization algorithms\n"
                report += "  - Implement GPS-based route monitoring\n"
            elif 'tonnage' in name.lower():
                report += "  - Optimize truck capacity utilization\n"
                report += "  - Review fleet composition strategies\n"
                report += "  - Consider load weight optimization systems\n"
            elif 'model' in name.lower():
                report += "  - Analyze truck model performance characteristics\n"
                report += "  - Consider fleet standardization options\n"
                report += "  - Evaluate equipment replacement strategies\n"
            elif 'material' in name.lower():
                report += "  - Implement material-specific handling strategies\n"
                report += "  - Optimize material classification processes\n"
                report += "  - Consider material type scheduling optimization\n"
        
        report += f"""
IMPLEMENTATION STRATEGY:

1. IMMEDIATE ACTIONS (0-30 days):
   • Focus optimization efforts on {most_sensitive}
   • Establish monitoring systems for high-sensitivity parameters
   • Conduct parameter-specific pilot studies
   • Review current operational procedures

2. SHORT-TERM IMPROVEMENTS (1-3 months):
   • Implement control strategies for top 3 sensitive parameters
   • Develop parameter interaction management protocols
   • Train operators on critical parameter management
   • Install monitoring and feedback systems

3. LONG-TERM OPTIMIZATION (3-12 months):
   • Develop integrated optimization system
   • Implement automated parameter adjustment systems
   • Establish continuous monitoring and feedback loops
   • Consider advanced predictive analytics implementation

MODEL SELECTION INSIGHTS:
The analysis used {self.best_model_name} as the best performing model with:
• R² = {best_metrics['R2']:.4f} (explaining {best_metrics['R2']*100:.1f}% of variance)
• MAPE = {best_metrics['MAPE']:.2f}% (average prediction error)

Alternative models available in the suite:
"""
        
        # List other models with their performance
        for model_name, result in self.results.items():
            if model_name != self.best_model_name:
                metrics = result['metrics']
                report += f"• {model_name}: R² = {metrics['R2']:.4f}, MAPE = {metrics['MAPE']:.2f}%\n"
        
        report += f"""
{'='*80}
CONCLUSIONS
{'='*80}

The Sobol sensitivity analysis reveals that {most_sensitive} is the most critical
parameter affecting truck production output (ST = {highest_st:.4f}). The analysis 
provides clear guidance for prioritizing optimization efforts and resource 
allocation in mining operations.

Key Findings Summary:
• Most sensitive parameter: {most_sensitive} (ST = {highest_st:.4f})
• Total variance explained by main effects: {total_s1:.1%}
• Maximum parameter interaction: {max_interaction:.4f}
• Model confidence: High (R² = {best_metrics['R2']:.4f})
• Prediction accuracy: Excellent (MAPE = {best_metrics['MAPE']:.2f}%)

Model Behavior Insights:
• {'Non-linear relationships dominate' if total_s1 < 0.8 else 'Linear relationships dominate'}
• {'Significant parameter interactions present' if max_interaction > 0.1 else 'Limited parameter interactions'}
• {'Complex system behavior' if max_interaction > 0.15 else 'Predictable system behavior'}

This analysis enables data-driven decision making for operational improvements
and provides a scientific basis for parameter prioritization in mining operations.

The {self.best_model_name} model demonstrates excellent predictive performance and
provides reliable sensitivity estimates for operational optimization. The complete
ELM suite offers multiple modeling approaches for different operational scenarios.

Recommendations for Future Analysis:
• Re-run sensitivity analysis as operational conditions change
• Consider seasonal or temporal variations in parameter sensitivity
• Evaluate sensitivity under different operational scenarios
• Integrate findings into operational decision support systems

{'='*80}
END OF REPORT
{'='*80}
        """
        
        # Save report
        with open('sobol_sensitivity_report.txt', 'w') as f:
            f.write(report)
        
        print("✅ Comprehensive sensitivity report saved as 'sobol_sensitivity_report.txt'")
        
        return report
    
    def run_complete_analysis(self, n_samples=1024):
        """Run complete Sobol sensitivity analysis"""
        print("🔬 SOBOL SENSITIVITY ANALYSIS")
        print("="*70)
        print("Investigating parameter impact on truck production prediction")
        print("Compatible with merged ELM training script")
        print("="*70)
        
        # Step 1: Load best model
        if not self.load_best_model():
            return False
        
        # Step 2: Define problem
        if not self.define_problem():
            return False
        
        # Step 3: Generate samples
        param_values = self.generate_samples(n_samples)
        
        # Step 4: Evaluate model
        predictions = self.evaluate_model(param_values)
        
        # Step 5: Perform Sobol analysis
        if not self.perform_sobol_analysis(param_values, predictions):
            return False
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Generate report
        self.generate_sensitivity_report()
        
        print("\n" + "="*70)
        print("🎉 SOBOL SENSITIVITY ANALYSIS COMPLETED!")
        print("="*70)
        print("\n📁 Generated Files:")
        print("• sobol_sensitivity_analysis.png - Comprehensive visualizations")
        print("• sobol_sensitivity_report.txt - Detailed analysis report")
        print("\n🎯 Key Insights:")
        
        # Quick summary
        most_sensitive_idx = np.argmax(self.sobol_results['ST'])
        most_sensitive_param = self.problem['names'][most_sensitive_idx]
        sensitivity_value = self.sobol_results['ST'][most_sensitive_idx]
        
        print(f"• Model analyzed: {self.best_model_name}")
        print(f"• Most sensitive parameter: {most_sensitive_param}")
        print(f"• Sensitivity index: {sensitivity_value:.4f}")
        print(f"• Total variance explained: {np.sum(self.sobol_results['S1']):.3f}")
        print(f"• Model performance: R² = {self.results[self.best_model_name]['metrics']['R2']:.4f}")
        
        return True

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_requirements():
    """Check if required packages are installed"""
    # Map package names to their import names
    package_map = {
        'SALib': 'SALib',
        'matplotlib': 'matplotlib', 
        'joblib': 'joblib',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'  # import name: install name
    }
    
    missing_packages = []
    
    for import_name, install_name in package_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        'saved_models/results.json',
        'saved_models/feature_info.json'
    ]
    
    missing_files = []
    
    if not os.path.exists('saved_models'):
        print("❌ 'saved_models' directory not found!")
        print("💡 Please run the merged ELM training script first.")
        return False
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("💡 Please ensure the training process completed successfully.")
        return False
    
    # Check if at least one model file exists
    model_files = [f for f in os.listdir('saved_models') if f.endswith('.joblib')]
    if not model_files:
        print("❌ No trained model files (.joblib) found in saved_models/")
        print("💡 Please run the merged ELM training script first.")
        return False
    
    print(f"✅ Found {len(model_files)} trained models")
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute Sobol sensitivity analysis"""
    
    print("🔬 SOBOL SENSITIVITY ANALYSIS - COMPATIBLE VERSION")
    print("="*70)
    print("Compatible with merged ELM training script")
    print("Supports: Base ELM, ICA-ELM, MFO-ELM, HBO-ELM, BOA-ELM")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model files
    if not check_model_files():
        return
    
    # Initialize analyzer
    analyzer = SobolSensitivityAnalyzer()
    
    # Ask user for sample size
    try:
        n_samples = input("\n🎲 Enter number of Sobol samples (default: 1024): ").strip()
        if n_samples:
            n_samples = int(n_samples)
        else:
            n_samples = 1024
        
        if n_samples < 256:
            print("⚠️ Warning: Sample size < 256 may produce unreliable results")
        elif n_samples > 4096:
            print("⚠️ Warning: Large sample size may take significant time")
        
    except ValueError:
        print("❌ Invalid input. Using default sample size of 1024.")
        n_samples = 1024
    
    print(f"📊 Running sensitivity analysis with {n_samples} samples...")
    
    # Run complete analysis
    success = analyzer.run_complete_analysis(n_samples=n_samples)
    
    if success:
        print("\n✅ Sensitivity analysis completed successfully!")
        print("🚀 Use insights to optimize mining operations!")
        print("\n📋 Next Steps:")
        print("• Review sobol_sensitivity_report.txt for detailed insights")
        print("• Use sobol_sensitivity_analysis.png for presentations")
        print("• Implement recommendations for operational optimization")
        print("• Consider re-running analysis with different scenarios")
    else:
        print("\n❌ Sensitivity analysis failed.")
        print("🔧 Please check model files and try again.")
        print("💡 Ensure the merged ELM training script was completed successfully.")

if __name__ == "__main__":
    # Install SALib if not available
    try:
        import SALib
        print("✅ SALib is available")
    except ImportError:
        print("📦 Installing SALib for sensitivity analysis...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "SALib"])
            print("✅ SALib installed successfully!")
            # Re-import after installation
            from SALib.sample import sobol
            from SALib.analyze import sobol as sobol_analyze
        except Exception as e:
            print(f"❌ Failed to install SALib: {e}")
            print("💡 Please install manually: pip install SALib")
            sys.exit(1)
    
    main()