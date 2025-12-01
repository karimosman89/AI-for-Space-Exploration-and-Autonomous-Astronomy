"""
AI for Space Exploration and Autonomous Astronomy

A comprehensive AI/ML platform for space exploration tasks including:
- Satellite image analysis
- Astronomical object detection
- Autonomous navigation
- Trajectory optimization
"""

__version__ = "1.0.0"
__author__ = "Karim Osman"
__email__ = "karim.osman@example.com"

from src.models import (
    SatelliteImageClassifier,
    AstronomicalDetector,
    NavigationAgent,
    GalaxyClassifier,
    ExoplanetDetector
)

from src.utils import (
    load_image,
    preprocess_image,
    augment_data,
    visualize_results
)

__all__ = [
    "SatelliteImageClassifier",
    "AstronomicalDetector",
    "NavigationAgent",
    "GalaxyClassifier",
    "ExoplanetDetector",
    "load_image",
    "preprocess_image",
    "augment_data",
    "visualize_results",
]
