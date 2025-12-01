# 📚 API Reference

Complete API documentation for AI Space Exploration platform.

## 🌐 REST API Endpoints

Base URL: `http://localhost:8000`

### Health & Status

#### `GET /`
Get API information and available endpoints.

**Response:**
```json
{
  "message": "AI Space Exploration API",
  "version": "1.0.0",
  "endpoints": {...}
}
```

#### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["satellite_classifier", "object_detector"]
}
```

---

### Satellite Image Classification

#### `POST /api/v1/satellite/classify`
Classify satellite imagery into terrain types.

**Parameters:**
- `file` (form-data): Image file (JPG, PNG, TIF)

**Response:**
```json
{
  "class": "forest",
  "confidence": 0.942,
  "class_id": 1,
  "probabilities": {
    "water": 0.02,
    "forest": 0.942,
    "grassland": 0.015,
    ...
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/satellite/classify" \
  -F "file=@satellite.jpg"
```

---

### Astronomical Object Detection

#### `POST /api/v1/astronomy/detect`
Detect celestial objects in telescope images.

**Parameters:**
- `file` (form-data): Image file (JPG, PNG, FITS)

**Response:**
```json
{
  "detections": [
    {
      "class": "star",
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 120,
        "y2": 170,
        "center_x": 110,
        "center_y": 160,
        "width": 20,
        "height": 20
      }
    }
  ]
}
```

---

### Trajectory Planning

#### `POST /api/v1/navigation/plan`
Plan optimal spacecraft trajectory.

**Request Body:**
```json
{
  "start_state": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  "goal_state": [10, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "max_steps": 1000
}
```

**Response:**
```json
{
  "success": true,
  "num_actions": 45,
  "total_reward": 95.3,
  "final_state": [9.8, 4.9, 3.1],
  "distance_to_goal": 0.15
}
```

---

### Galaxy Classification

#### `POST /api/v1/galaxy/classify`
Classify galaxy morphology.

**Parameters:**
- `file` (form-data): Galaxy image file

**Response:**
```json
{
  "class": "spiral",
  "confidence": 0.921,
  "probabilities": {
    "spiral": 0.921,
    "elliptical": 0.045,
    "irregular": 0.023,
    ...
  }
}
```

---

### Exoplanet Detection

#### `POST /api/v1/exoplanet/detect`
Detect exoplanets from light curve data.

**Request Body:**
```json
{
  "flux_values": [1.0, 1.01, 0.98, ...]
}
```

**Response:**
```json
{
  "exoplanet_detected": true,
  "confidence": 0.873,
  "probability": 0.873
}
```

---

## 🐍 Python SDK

### Installation

```python
from src.models import (
    SatelliteImageClassifier,
    AstronomicalDetector,
    NavigationAgent,
    GalaxyClassifier,
    ExoplanetDetector
)
```

### SatelliteImageClassifier

```python
classifier = SatelliteImageClassifier(
    num_classes=10,
    pretrained=True,
    dropout_rate=0.5
)

# Predict single image
result = classifier.predict(image, device='cuda')

# Predict batch
results = classifier.predict_batch(images, batch_size=32)

# Get feature maps
features = classifier.get_feature_maps(image)
```

**Methods:**
- `predict(image, device='cpu')` → Dict
- `predict_batch(images, device='cpu', batch_size=32)` → List[Dict]
- `get_feature_maps(image, device='cpu')` → Dict[str, Tensor]

---

### AstronomicalDetector

```python
detector = AstronomicalDetector(
    num_classes=8,
    input_size=640,
    confidence_threshold=0.5,
    nms_threshold=0.4
)

# Detect objects
detections = detector.detect(image, device='cuda')

# Visualize detections
vis_image = detector.visualize_detections(
    image, 
    detections, 
    save_path='output.jpg'
)
```

**Methods:**
- `detect(image, device='cpu')` → List[Dict]
- `visualize_detections(image, detections, save_path=None)` → np.ndarray

---

### NavigationAgent

```python
agent = NavigationAgent(
    state_dim=12,
    action_dim=6,
    learning_rate=0.001,
    gamma=0.99
)

# Select action
action = agent.select_action(state, training=True)

# Store experience
agent.store_transition(state, action, reward, next_state, done)

# Train step
loss = agent.train_step()

# Plan trajectory
actions, states, reward = agent.plan_trajectory(
    start_state,
    goal_state,
    max_steps=1000
)

# Save/load model
agent.save('agent.pth')
agent.load('agent.pth')
```

**Methods:**
- `select_action(state, training=True)` → int
- `store_transition(state, action, reward, next_state, done)` → None
- `train_step()` → Optional[float]
- `plan_trajectory(start, goal, max_steps=1000)` → Tuple[List, List, float]
- `save(filepath)` → None
- `load(filepath)` → None

---

### GalaxyClassifier

```python
classifier = GalaxyClassifier(
    num_classes=5,
    pretrained=True
)

# Predict galaxy type
result = classifier.predict(image, device='cuda')
```

**Methods:**
- `predict(image, device='cpu')` → Dict

---

### ExoplanetDetector

```python
detector = ExoplanetDetector(
    sequence_length=2000,
    hidden_dim=128
)

# Detect exoplanet
result = detector.detect(light_curve, device='cuda')
```

**Methods:**
- `detect(light_curve, device='cpu')` → Dict

---

## 🛠️ Utility Functions

### Image Processing

```python
from src.utils import (
    load_image,
    preprocess_image,
    augment_data,
    normalize_image,
    enhance_contrast,
    remove_noise
)

# Load image
image = load_image('path/to/image.jpg', color_mode='RGB')

# Preprocess
processed = preprocess_image(image, target_size=(224, 224))

# Augment
augmented = augment_data(image, augmentation_type='training')

# Normalize
normalized = normalize_image(image, method='imagenet')
```

### Visualization

```python
from src.utils import (
    visualize_results,
    plot_training_curves,
    plot_detection_results,
    plot_trajectory
)

# Visualize classification results
visualize_results(images, predictions, save_path='results.png')

# Plot training curves
plot_training_curves(train_losses, val_losses, train_accs, val_accs)

# Plot detections
plot_detection_results(image, detections, save_path='detections.png')

# Plot trajectory
plot_trajectory(states, goal, save_path='trajectory.png')
```

---

## 📊 Data Structures

### Classification Result

```python
{
    'class': str,           # Predicted class name
    'confidence': float,    # Confidence score [0, 1]
    'class_id': int,        # Class index
    'probabilities': dict   # All class probabilities
}
```

### Detection Result

```python
{
    'class': str,
    'confidence': float,
    'bbox': {
        'x1': float, 'y1': float,
        'x2': float, 'y2': float,
        'center_x': float, 'center_y': float,
        'width': float, 'height': float
    }
}
```

---

## 🔒 Error Handling

All API endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

Error Response Format:
```json
{
  "detail": "Error message description"
}
```

---

## 📝 Notes

- All image inputs should be in RGB format
- State vectors for navigation have 12 dimensions: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, orientation, fuel, ...]
- Light curves should be normalized before exoplanet detection
- GPU is recommended for inference but not required

---

For more examples, see the [Examples Directory](../examples/) and [Jupyter Notebooks](../notebooks/).
