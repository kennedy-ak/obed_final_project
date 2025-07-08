# Truck Production Prediction System

This project is an interactive web application for predicting and analyzing mining truck production at AngloGold Ashanti Iduapriem Limited. It leverages advanced machine learning models, including Extreme Learning Machine (ELM) and several hybrid ELM variants optimized with metaheuristic algorithms.

## Features

- **Interactive Streamlit App**: User-friendly interface for inputting operational parameters and viewing predictions.
- **Multiple ELM Models**: Includes Base ELM, ICA-ELM, HBO-ELM, MFO-ELM, and BOA-ELM.
- **Model Comparison**: Visual and tabular comparison of model performance (R², MAPE, VAF, NASH, AIC, training time).
- **Sensitivity Analysis**: Visualizes the impact of each parameter on production output.
- **Comprehensive Visualizations**: Generates publication-ready plots for model performance, convergence, and operational insights.
- **Academic Documentation**: Ready-to-use figures and reports for research and thesis work.

## Project Structure

```
.
├── .gitignore
├── becky.py                  # Main Streamlit app
├── main.py                   # (Alternate/legacy app version)
├── plot.py                   # Visualization suite (matplotlib/seaborn)
├── sobol_sensitivity_analysis.py # Sobol sensitivity analysis
├── requirements.txt
├── saved_models/             # Trained model files (.joblib, .json)
├── env/                      # Python virtual environment
├── Truck Production Data AAIL .csv # Dataset
├── *.png                     # Generated plots
└── ...                       # Other scripts and outputs
```

## Quick Start

1. **Install dependencies**  
   Create a virtual environment and install requirements:
   ```sh
   python -m venv env
   source env/bin/activate  # or .\env\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**  
   ```sh
   streamlit run becky.py
   ```

3. **Interact with the app**  
   - Select a model in the sidebar.
   - Enter operational parameters.
   - Click "Predict Production" to view results, model comparisons, and sensitivity analysis.

4. **Generate Visualizations**  
   To create all publication-ready plots:
   ```sh
   python plot.py
   ```
   Output images will be saved in the project directory.

## Models Included

- **Base_ELM**: Standard Extreme Learning Machine.
- **ICA_ELM**: ELM optimized with Imperialist Competitive Algorithm.
- **HBO_ELM**: ELM optimized with Honey Badger Optimization.
- **MFO_ELM**: ELM optimized with Moth-Flame Optimization.
- **BOA_ELM**: ELM optimized with Bath Optimization Algorithm.

## Key Files

- [`becky.py`](becky.py): Main Streamlit application.
- [`plot.py`](plot.py): Visualization suite ([`ELMVisualizationSuite`](plot.py)).
- [`saved_models/`](saved_models/): Contains trained models and metadata.
- [`sobol_sensitivity_analysis.py`](sobol_sensitivity_analysis.py): Advanced sensitivity analysis and reporting.

## Data

- **Input**: Operational parameters (truck model, tonnage, material type, times, loads, distance).
- **Output**: Predicted production (tonnes), efficiency metrics, and model performance.

## Acknowledgements

- **Institution**: University of Mines and Technology, Tarkwa
- **Industry Partner**: AngloGold Ashanti Iduapriem Limited
- **Project**: June 2025 - Mining Engineering

---

For questions or contributions, please open an