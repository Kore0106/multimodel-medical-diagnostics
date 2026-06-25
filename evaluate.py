import os  
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.dataset import MultimodalMedicalDataset  
from src.models import MultimodalFusionModel
import pandas as pd

print("⚙️ Initializing Multimodal Evaluation Engine...")

# 1. Setup Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Load the Testing/Validation Dataset Path directly into the Loader
try:
    test_dataset = MultimodalMedicalDataset("data/processed_dataset.csv", text_model_name=TEXT_MODEL_NAME)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"📋 Loaded dataset records for validation tracking.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# 3. Load Trained Model Architecture and Weights with Dynamic Shape Matching
model = MultimodalFusionModel(text_model_name=TEXT_MODEL_NAME)

if os.path.exists("models/best_multimodal_model.pth"):
    model.load_state_dict(torch.load("models/best_multimodal_model.pth", map_location=device))
    print("🎯 Successfully hooked newly trained weights into evaluation stream.")
else:
    print("⚠️ Warning: Best weights not found.")

model.to(device)
model.eval()
# 4. Evaluation Loop
import torch.nn.functional as F

all_predictions = []
all_ground_truths = []

print("🚀 Running validation batch inference loops...")
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tabular = batch['tabular'].to(device)
        labels = batch['label'].to(device)
        
        # 💡 Dynamic Tabular Padding: Match the new 24D grid architecture
        if tabular.shape[1] == 16:
            tabular = F.pad(tabular, (0, 8), "constant", 0)
        
        # Forward pass
        outputs = model(images, input_ids, attention_mask, tabular).squeeze(-1)
        probabilities = torch.sigmoid(outputs)
        
        # Convert continuous probabilities to binary decisions (Threshold = 0.5)
        predictions = (probabilities > 0.5).long()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_ground_truths.extend(labels.cpu().numpy())

# 5. Compute Advanced Performance Metrics
acc = accuracy_score(all_ground_truths, all_predictions) * 100
precision = precision_score(all_ground_truths, all_predictions, zero_division=0) * 100
recall = recall_score(all_ground_truths, all_predictions, zero_division=0) * 100
f1 = f1_score(all_ground_truths, all_predictions, zero_division=0) * 100
cm = confusion_matrix(all_ground_truths, all_predictions)

# 6. Render Metrics Matrix Terminal Report
print(f"""
===================================================================
📊 CDSS MODEL VALIDATION PERFORMANCE REPORT
===================================================================
[METRICS COMPUTED VIA SCIKIT-LEARN MATRIX]

➡️ Overall Accuracy : {acc:.2f}%  (Total correctly categorized)
➡️ Precision        : {precision:.2f}%  (When model says High Risk, how often it's right)
➡️ Recall (Sens.)   : {recall:.2f}%  (Out of all actual sick patients, how many it caught)
➡️ F1-Score         : {f1:.2f}%  (Harmonic mean balance of Precision & Recall)

-------------------------------------------------------------------
📝 CONFUSION MATRIX STRUCTURAL MAP:
    Predicted Low    Predicted High
True Low:    {cm[0][0]}                {cm[0][1]}
True High:   {cm[1][0]}                {cm[1][1]}
===================================================================
""")