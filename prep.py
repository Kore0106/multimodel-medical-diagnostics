import pandas as pd
import os

print("Starting data preparation...")
df_subjects = pd.read_csv("demo_subject_id.csv")
df_diagnoses = pd.read_csv("hosp/diagnoses_icd.csv.gz")
df_diagnoses['label'] = df_diagnoses['icd_code'].apply(lambda x: 1 if str(x).startswith('4') else 0)
df_labels = df_diagnoses.groupby('subject_id')['label'].max().reset_index()
df_master = pd.merge(df_subjects, df_labels, on='subject_id', how='left').fillna(0)
disease_notes = [
    "Patient presents with clear lungs, normal sinus rhythm, no acute distress noted.",
    "Cardiomegaly noted with mild pulmonary vascular congestion. Patient reports short breath."
]
df_master['clinical_text'] = df_master['label'].apply(lambda x: disease_notes[int(x)])
df_master['image_path'] = df_master['subject_id'].apply(lambda x: f"data/images/patient_{x}.jpg")
os.makedirs("data", exist_ok=True)
output_path = "data/processed_dataset.csv"
df_master.to_csv(output_path, index=False)

print(f"Success! Master spine created at: {output_path}")
print(df_master[['subject_id', 'clinical_text', 'image_path', 'label']].head())