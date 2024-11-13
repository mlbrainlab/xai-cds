# SHAP Analysis
import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def run_shap_analysis(model, X):
    """
    Compute SHAP values for feature importance analysis.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

if __name__ == "__main__":
    # Load model and data
    model = joblib.load('../models/random_forest_model.pkl')
    X = pd.read_csv('data/processed/feature_matrix.csv').drop(columns=['outcome', 'hadm_id'])  # Adjust as needed
    
    explainer, shap_values = run_shap_analysis(model, X)
    
    # Visualize SHAP values
    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values[1], X)
