import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

class MultimodalFusionModel(nn.Module):
    def __init__(self, text_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(MultimodalFusionModel, self).__init__()
        
        # 1. Vision Stream (Frozen ResNet50 -> 2048 features)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 2. Text Stream (Frozen MiniLM -> 384 features)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # 3. EXPANDED Merging Stream
        # Vision (2048) + Text (384) + Advanced Tabular Features (Increased to 24)
        total_dense_features = 2048 + 384 + 24
        
        # 4. Deep Predictive Network Layers
        self.classifier = nn.Sequential(
            nn.Linear(total_dense_features, 256), # Expanded hidden layer size for complex features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, image, input_ids, attention_mask, tabular):
        vision_features = self.vision_encoder(image)
        vision_features = torch.flatten(vision_features, 1)
        
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Fuses the text, images, and new 24-dimensional lab metrics vector together
        fused_features = torch.cat((vision_features, text_features, tabular), dim=1)
        
        output = self.classifier(fused_features)
        return output.squeeze(1)