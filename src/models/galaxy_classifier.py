"""
Galaxy Morphology Classification using Vision Transformers

Classifies galaxies into different morphological types:
spiral, elliptical, irregular, lenticular, etc.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from typing import Dict, List
from PIL import Image


class GalaxyClassifier(nn.Module):
    """
    Vision Transformer-based galaxy morphology classifier.
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        """
        Initialize galaxy classifier.
        
        Args:
            num_classes: Number of galaxy types
            pretrained: Use pretrained weights
        """
        super(GalaxyClassifier, self).__init__()
        
        # Simplified ViT-like architecture
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )
        
        self.classes = ['spiral', 'elliptical', 'irregular', 'lenticular', 'peculiar']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)  # (B, 768, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, N, 768)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def predict(self, image: np.ndarray, device: str = 'cpu') -> Dict:
        """Predict galaxy type from image."""
        self.eval()
        self.to(device)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.forward(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'class': self.classes[predicted.item()],
            'confidence': confidence.item(),
            'probabilities': {
                self.classes[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }


if __name__ == "__main__":
    print("Galaxy Classifier initialized")
    model = GalaxyClassifier(num_classes=5)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
