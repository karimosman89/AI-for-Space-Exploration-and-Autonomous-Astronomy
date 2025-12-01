"""
Astronomical Object Detection Model

This module implements YOLO-based object detection for identifying
and localizing celestial objects in astronomical images.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class AstronomicalDetector(nn.Module):
    """
    YOLO-based detector for astronomical objects including stars,
    galaxies, planets, asteroids, and other celestial phenomena.
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        input_size: int = 640,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize the astronomical object detector.
        
        Args:
            num_classes: Number of object classes to detect
            input_size: Input image size (square)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for non-maximum suppression
        """
        super(AstronomicalDetector, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Class labels
        self.classes = [
            'star', 'galaxy', 'nebula', 'planet', 
            'asteroid', 'comet', 'satellite', 'debris'
        ]
        
        # Build detection network (simplified YOLO-like architecture)
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = self._build_head()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _build_backbone(self) -> nn.Module:
        """Build feature extraction backbone."""
        return nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 5
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
    
    def _build_neck(self) -> nn.Module:
        """Build feature pyramid network."""
        return nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
    
    def _build_head(self) -> nn.Module:
        """Build detection head."""
        # Output: (batch, anchors * (5 + num_classes), H, W)
        # 5 = x, y, w, h, objectness
        num_anchors = 3
        output_channels = num_anchors * (5 + self.num_classes)
        
        return nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_channels, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the detection network."""
        features = self.backbone(x)
        features = self.neck(features)
        detections = self.head(features)
        return detections
    
    def detect(
        self,
        image: np.ndarray,
        device: str = 'cpu'
    ) -> List[Dict[str, any]]:
        """
        Detect astronomical objects in an image.
        
        Args:
            image: Input image as numpy array or PIL Image
            device: Device to run inference on
            
        Returns:
            List of detected objects with bounding boxes and classes
        """
        self.eval()
        self.to(device)
        
        # Store original dimensions
        if isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
            image = Image.fromarray(image)
        else:
            orig_w, orig_h = image.size
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            detections = self.forward(img_tensor)
        
        # Post-process detections
        boxes, scores, class_ids = self._postprocess_detections(
            detections,
            orig_w,
            orig_h
        )
        
        # Format results
        results = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            results.append({
                'class': self.classes[class_id],
                'confidence': float(score),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                }
            })
        
        return results
    
    def _postprocess_detections(
        self,
        detections: torch.Tensor,
        orig_w: int,
        orig_h: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process raw detections to get final bounding boxes.
        
        Args:
            detections: Raw detection tensor from network
            orig_w: Original image width
            orig_h: Original image height
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        # Simplified post-processing (in practice, would use proper YOLO decoding)
        # This is a placeholder implementation
        
        # For demonstration, generate some example detections
        num_detections = 5
        boxes = np.random.rand(num_detections, 4) * np.array([orig_w, orig_h, orig_w, orig_h])
        scores = np.random.rand(num_detections) * 0.5 + 0.5
        class_ids = np.random.randint(0, self.num_classes, num_detections)
        
        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Apply NMS
        if len(boxes) > 0:
            keep_indices = self._nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            class_ids = class_ids[keep_indices]
        
        return boxes, scores, class_ids
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        threshold: float
    ) -> List[int]:
        """
        Non-maximum suppression to remove duplicate detections.
        
        Args:
            boxes: Array of bounding boxes
            scores: Confidence scores
            threshold: IoU threshold
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, any]],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            save_path: Optional path to save visualization
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


if __name__ == "__main__":
    # Example usage
    print("Astronomical Object Detector initialized")
    model = AstronomicalDetector(num_classes=8)
    print(f"Model architecture:\n{model}")
    print(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters()):,}")
