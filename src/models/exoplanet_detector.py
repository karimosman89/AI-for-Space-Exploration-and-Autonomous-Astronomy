"""
Exoplanet Detection from Light Curve Data

Uses 1D CNN + LSTM to detect exoplanets from stellar brightness
time series data (transit method).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class ExoplanetDetector(nn.Module):
    """
    Deep learning model for exoplanet detection from light curves.
    """
    
    def __init__(self, sequence_length: int = 2000, hidden_dim: int = 128):
        """
        Initialize exoplanet detector.
        
        Args:
            sequence_length: Length of light curve sequence
            hidden_dim: Hidden dimension for LSTM
        """
        super(ExoplanetDetector, self).__init__()
        
        # 1D CNN for feature extraction
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary: exoplanet or not
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.conv1d(x)  # (B, 256, L')
        
        # Reshape for LSTM
        x = x.transpose(1, 2)  # (B, L', 256)
        
        # LSTM temporal modeling
        x, _ = self.lstm(x)  # (B, L', hidden*2)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, hidden*2)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def detect(
        self,
        light_curve: np.ndarray,
        device: str = 'cpu'
    ) -> Dict:
        """
        Detect exoplanet from light curve data.
        
        Args:
            light_curve: Time series of stellar brightness
            device: Device to run on
            
        Returns:
            Detection result with confidence
        """
        self.eval()
        self.to(device)
        
        # Normalize light curve
        light_curve = (light_curve - light_curve.mean()) / light_curve.std()
        
        # Convert to tensor
        x = torch.FloatTensor(light_curve).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'exoplanet_detected': bool(predicted.item()),
            'confidence': confidence.item(),
            'probability': probabilities[0][1].item()
        }


if __name__ == "__main__":
    print("Exoplanet Detector initialized")
    model = ExoplanetDetector(sequence_length=2000)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
