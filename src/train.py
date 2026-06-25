import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
from src.dataset import MultimodalMedicalDataset
from src.models import MultimodalFusionModel

print("🏋️ Initializing Multimodal Training Engine with Data Splitting...")

# 1. Hardware & Core Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4

# 2. Load and Split Dataset
if not os.path.exists("data/processed_dataset.csv"):
    print("❌ Error: data/processed_dataset.csv not found.")
    exit()

full_dataset = MultimodalMedicalDataset("data/processed_dataset.csv", text_model_name=TEXT_MODEL_NAME)
total_records = len(full_dataset)

train_size = int(0.8 * total_records)
val_size = total_records - train_size

train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 Dataset Split Complete: {train_size} training samples | {val_size} validation samples.")

# 3. Instantiate Model
model = MultimodalFusionModel(text_model_name=TEXT_MODEL_NAME)
model.to(device)

# 4. Optimization Matrix Setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 5. Training Loop
os.makedirs("models", exist_ok=True)
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train_loss = 0.0
    
    # --- Training Phase ---
    for batch in train_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tabular = batch['tabular'].to(device)
        
        # Ground-truth labels kept as flat 1D float vectors [16]
        labels = batch['label'].to(device).float()
        
        # Dynamic Tabular Padding (16 -> 24)
        if tabular.shape[1] == 16:
            tabular = F.pad(tabular, (0, 8), "constant", 0)
            
        optimizer.zero_grad()
        
        # Explicitly squeeze the model outputs from [16, 1] to a flat 1D vector [16]
        outputs = model(images, input_ids, attention_mask, tabular).squeeze(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * images.size(0)
        
    epoch_train_loss = running_train_loss / train_size
    
    # --- Validation Phase ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular = batch['tabular'].to(device)
            
            labels = batch['label'].to(device).float()
            
            if tabular.shape[1] == 16:
                tabular = F.pad(tabular, (0, 8), "constant", 0)
                
            outputs = model(images, input_ids, attention_mask, tabular).squeeze(-1)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)
            
    epoch_val_loss = running_val_loss / val_size
    
    print(f"Epoch [{epoch}/{EPOCHS}] ➡️ Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "models/best_multimodal_model.pth")
        print("💾 New optimal model weights saved to models/best_multimodal_model.pth")

print("\n🎯 Training fully complete! New shapes locked in.")