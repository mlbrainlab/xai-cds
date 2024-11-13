# Preprocessing
from src import data_loading
import data_loading
import pandas as pd

def merge_data(data):
    """
    Merge all clinical data into a single DataFrame based on 'hadm_id'.
    """
    merged_data = data['discharge_diagnosis'].merge(data['hpi'], on='hadm_id', how='inner') \
                                             .merge(data['physical_examination'], on='hadm_id', how='inner') \
                                             .merge(data['lab_tests'], on='hadm_id', how='inner') \
                                             .merge(data['microbiology'], on='hadm_id', how='inner') \
                                             .merge(data['radiology_reports'], on='hadm_id', how='inner')
    return merged_data

def filter_by_pathologies(data, pathology_ids):
    """
    Filter data based on specific pathology IDs.
    """
    pathology_ids_list = [item for sublist in pathology_ids.values() for item in sublist]
    return data[data['hadm_id'].isin(pathology_ids_list)]

def handle_missing_values(data):
    """
    Handle missing data by filling forward.
    """
    return data.fillna(method='ffill')

def preprocess_data(data, pathology_ids):
    """
    Preprocess and combine data steps.
    """
    combined_data = merge_data(data)
    filtered_data = filter_by_pathologies(combined_data, pathology_ids)
    cleaned_data = handle_missing_values(filtered_data)
    return cleaned_data

if __name__ == "__main__":
    from data_loading import load_data
    data = load_data()
    processed_data = preprocess_data(data, data['pathology_ids'])
    print("Data Preprocessing Complete")