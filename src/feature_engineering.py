# Feature Engineering
import pandas as pd

def create_text_features(data):
    """
    Create features based on text length of 'hpi'.
    """
    if 'hpi' in data.columns:
        data['hpi_length'] = data['hpi'].apply(lambda x: len(str(x).split()))
    return data

def encode_categorical_features(data):
    """
    One-hot encode radiology modality and ICD codes.
    """
    if 'modality' in data.columns:
        data = pd.get_dummies(data, columns=['modality'], prefix='modality')
    if 'icd_code' in data.columns:
        data = pd.get_dummies(data, columns=['icd_code'], prefix='icd')
    return data

def feature_engineering(data):
    """
    Apply feature engineering transformations.
    """
    data = create_text_features(data)
    data = encode_categorical_features(data)
    return data

if __name__ == "__main__":
    from preprocessing import preprocess_data
    from data_loading import load_data
    data = load_data()
    processed_data = preprocess_data(data, data['pathology_ids'])
    feature_data = feature_engineering(processed_data)
    feature_data.to_csv('../data/processed/feature_matrix.csv', index=False)
    print("Feature Engineering Complete")
