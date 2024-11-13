# Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

def train_random_forest(X, y):
    """
    Train a RandomForest model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model, accuracy, report

if __name__ == "__main__":
    # Load feature data
    feature_data = pd.read_csv('../data/processed/feature_matrix.csv')
    X = feature_data.drop(columns=['outcome', 'hadm_id'])  # Replace 'outcome' with actual target variable
    y = feature_data['outcome']  # Replace 'outcome' with actual target variable
    
    model, accuracy, report = train_random_forest(X, y)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    joblib.dump(model, '../models/random_forest_model.pkl')
