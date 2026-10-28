import pandas as pd
import ast
CSV_PATH = (
    "archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
    "ptbxl_database.csv"
)

df = pd.read_csv(CSV_PATH, index_col="ecg_id")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
target_rhythms = ["SBRAD", "SR", "AFIB", "STACH", "AFLT", "SARRH", "SVTAC"]

def extract_rhythm(code_dict):
    for rhythm in target_rhythms:
        if rhythm in code_dict:
            return rhythm
    return None

df["rhythm_label"] = df["scp_codes"].apply(extract_rhythm)


filtered_df = df.dropna(subset=["rhythm_label"]) #filtering
print(f"Total Unique Patients: {df['patient_id'].nunique()}")
print(f"Total ECG Records: {len(df)}")
print(f"Filtered Target Rhythm Records: {len(filtered_df)}")
print("\n--- Class Breakdown (Proves Imbalance Analysis) ---")
print(filtered_df["rhythm_label"].value_counts())
