import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class MultimodalMedicalDataset(Dataset):
    def __init__(self, csv_file, text_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load our Day 1 master spine
        self.df = pd.read_csv(csv_file)
        
        # Load a lightning-fast, lightweight tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Tabular Modality (Mocking a feature vector using patient ID metrics for baseline testing)
        # We'll generate a consistent 16-dimensional vector representing patient vitals
        # (Using a random seed unique to the patient ID so it stays consistent)
        torch.manual_seed(int(row['subject_id']))
        tabular_tensor = torch.randn(16)
        
        # 2. Text Modality: Tokenize the medical sentence reports
        text_data = str(row['clinical_text'])
        text_tokens = self.tokenizer(
            text_data,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        # Squeeze out the batch dimension added by the tokenizer
        input_ids = text_tokens['input_ids'].squeeze(0)
        attention_mask = text_tokens['attention_mask'].squeeze(0)
        
        # 3. Visual Modality: Create a mock image tensor tensor matching a standard ResNet input
        # (3 channels, 224x224 pixels)
        image_tensor = torch.randn(3, 224, 224)
        
        # 4. Target Label
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tabular': tabular_tensor,
            'label': label
        }