"""
AI Models for Space Exploration

This module contains implementations of various deep learning models
for space exploration and astronomical analysis tasks.
"""

from src.models.satellite_classifier import SatelliteImageClassifier
from src.models.object_detector import AstronomicalDetector
from src.models.navigation_agent import NavigationAgent
from src.models.galaxy_classifier import GalaxyClassifier
from src.models.exoplanet_detector import ExoplanetDetector

__all__ = [
    "SatelliteImageClassifier",
    "AstronomicalDetector",
    "NavigationAgent",
    "GalaxyClassifier",
    "ExoplanetDetector",
]
