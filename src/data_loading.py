# Data Loading
import pandas as pd
import json

def load_data():
    """Load all relevant datasets."""
    discharge_diagnosis = pd.read_csv('../data/raw/discharge_diagnosis.csv')
    discharge_procedures = pd.read_csv('../data/raw/discharge_procedures.csv')
    history_of_present_illness = pd.read_csv('../data/raw/history_of_present_illness.csv')
    icd_diagnosis = pd.read_csv('../data/raw/icd_diagnosis.csv')
    icd_procedures = pd.read_csv('../data/raw/icd_procedures.csv')
    lab_test_mapping = pd.read_csv('../data/raw/lab_test_mapping.csv')
    laboratory_tests = pd.read_csv('../data/raw/laboratory_tests.csv')
    microbiology = pd.read_csv('../data/raw/microbiology.csv')
    physical_examination = pd.read_csv('../data/raw/physical_examination.csv')
    radiology_reports = pd.read_csv('../data/raw/radiology_reports.csv')

    with open('../data/raw/pathology_ids.json') as f:
        pathology_ids = json.load(f)

    return {
        "discharge_diagnosis": discharge_diagnosis,
        "discharge_procedures": discharge_procedures,
        "hpi": history_of_present_illness,
        "icd_diagnosis": icd_diagnosis,
        "icd_procedures": icd_procedures,
        "lab_test_mapping": lab_test_mapping,
        "lab_tests": laboratory_tests,
        "microbiology": microbiology,
        "physical_examination": physical_examination,
        "radiology_reports": radiology_reports,
        "pathology_ids": pathology_ids
    }

if __name__ == "__main__":
    data = load_data()
    print("Data Loaded Successfully")
