"""
Satellite Image Classification Model

This module implements a deep learning model for classifying satellite imagery
into different terrain types (water, vegetation, urban, desert, mountains, etc.)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image


class SatelliteImageClassifier(nn.Module):
    """
    Advanced satellite image classifier using ResNet50 backbone
    with custom classification head for terrain analysis.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the satellite image classifier.
        
        Args:
            num_classes: Number of terrain classes to predict
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
        """
        super(SatelliteImageClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify the first conv layer to accept multi-spectral input if needed
        # For now, keeping RGB (3 channels)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )
        
        # Class labels
        self.classes = [
            'water', 'forest', 'grassland', 'urban', 'desert',
            'mountains', 'ice', 'agricultural', 'wetland', 'barren'
        ]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def predict(
        self,
        image: np.ndarray,
        device: str = 'cpu'
    ) -> Dict[str, any]:
        """
        Predict terrain class for a single image.
        
        Args:
            image: Input image as numpy array or PIL Image
            device: Device to run inference on ('cpu' or 'cuda')
            
        Returns:
            Dictionary containing predicted class, confidence, and all probabilities
        """
        self.eval()
        self.to(device)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = self.forward(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Prepare results
        result = {
            'class': self.classes[predicted.item()],
            'confidence': confidence.item(),
            'class_id': predicted.item(),
            'probabilities': {
                self.classes[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        return result
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        device: str = 'cpu',
        batch_size: int = 32
    ) -> List[Dict[str, any]]:
        """
        Predict terrain classes for multiple images.
        
        Args:
            images: List of input images
            device: Device to run inference on
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        self.eval()
        self.to(device)
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                batch_tensors.append(self.transform(img))
            
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = self.forward(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
            
            # Process results
            for j, pred in enumerate(predicted):
                result = {
                    'class': self.classes[pred.item()],
                    'confidence': confidences[j].item(),
                    'class_id': pred.item(),
                    'probabilities': {
                        self.classes[k]: prob.item()
                        for k, prob in enumerate(probabilities[j])
                    }
                }
                results.append(result)
        
        return results
    
    def get_feature_maps(
        self,
        image: np.ndarray,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            image: Input image
            device: Device to run on
            
        Returns:
            Dictionary of feature maps from different layers
        """
        self.eval()
        self.to(device)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        feature_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        hooks.append(self.backbone.layer1.register_forward_hook(hook_fn('layer1')))
        hooks.append(self.backbone.layer2.register_forward_hook(hook_fn('layer2')))
        hooks.append(self.backbone.layer3.register_forward_hook(hook_fn('layer3')))
        hooks.append(self.backbone.layer4.register_forward_hook(hook_fn('layer4')))
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(img_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return feature_maps


def load_pretrained_satellite_classifier(
    checkpoint_path: str,
    device: str = 'cpu'
) -> SatelliteImageClassifier:
    """
    Load a pretrained satellite classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model instance
    """
    model = SatelliteImageClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Example usage
    print("Satellite Image Classifier initialized")
    model = SatelliteImageClassifier(num_classes=10)
    print(f"Model architecture:\n{model}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
