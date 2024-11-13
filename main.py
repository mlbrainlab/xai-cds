import argparse
import pandas as pd
import numpy as np
import torch # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from data_loading import load_data
from preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model_training import train_xgb_model, evaluate_model

def main():
    # Load and preprocess data
    print("Loading data...")
    data = load_data()  # Assuming you have a function to load data
    processed_data = preprocess_data(data)  # Preprocess the data
    features, target = feature_engineering(processed_data)  # Extract features and target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode target if necessary
    if y_train.dtype == 'object':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # Train the model (using XGBoost as an example)
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(tree_method='gpu_hist', use_label_encoder=False, max_depth=6, n_estimators=50)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, le)

    # Save the model (if needed)
    model.save_model("xgb_model.json")
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()
