"""
Unit tests for AI models.
"""

import pytest
import torch
import numpy as np
from PIL import Image

from src.models import (
    SatelliteImageClassifier,
    AstronomicalDetector,
    NavigationAgent,
    GalaxyClassifier,
    ExoplanetDetector
)


class TestSatelliteImageClassifier:
    """Test suite for Satellite Image Classifier."""
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = SatelliteImageClassifier(num_classes=10)
        assert model is not None
        assert len(model.classes) == 10
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = SatelliteImageClassifier(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_predict(self):
        """Test prediction on dummy image."""
        model = SatelliteImageClassifier(num_classes=10)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model.predict(image)
        
        assert 'class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert len(result['probabilities']) == 10


class TestAstronomicalDetector:
    """Test suite for Astronomical Object Detector."""
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = AstronomicalDetector(num_classes=8)
        assert model is not None
        assert len(model.classes) == 8
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = AstronomicalDetector(num_classes=8)
        x = torch.randn(2, 3, 640, 640)
        output = model(x)
        assert output is not None
    
    def test_detect(self):
        """Test detection on dummy image."""
        model = AstronomicalDetector(num_classes=8)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        detections = model.detect(image)
        
        assert isinstance(detections, list)
        if len(detections) > 0:
            assert 'class' in detections[0]
            assert 'confidence' in detections[0]
            assert 'bbox' in detections[0]


class TestNavigationAgent:
    """Test suite for Navigation Agent."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = NavigationAgent(state_dim=12, action_dim=6)
        assert agent is not None
        assert agent.state_dim == 12
        assert agent.action_dim == 6
    
    def test_select_action(self):
        """Test action selection."""
        agent = NavigationAgent(state_dim=12, action_dim=6)
        state = np.random.randn(12)
        action = agent.select_action(state, training=False)
        
        assert isinstance(action, int)
        assert 0 <= action < 6
    
    def test_trajectory_planning(self):
        """Test trajectory planning."""
        agent = NavigationAgent(state_dim=12, action_dim=6)
        start = np.zeros(12)
        goal = np.array([5, 5, 5] + [0] * 9)
        
        actions, states, reward = agent.plan_trajectory(start, goal, max_steps=10)
        
        assert len(actions) <= 10
        assert len(states) == len(actions) + 1
        assert isinstance(reward, float)


class TestGalaxyClassifier:
    """Test suite for Galaxy Classifier."""
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = GalaxyClassifier(num_classes=5)
        assert model is not None
        assert len(model.classes) == 5
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = GalaxyClassifier(num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 5)
    
    def test_predict(self):
        """Test prediction on dummy image."""
        model = GalaxyClassifier(num_classes=5)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model.predict(image)
        
        assert 'class' in result
        assert 'confidence' in result


class TestExoplanetDetector:
    """Test suite for Exoplanet Detector."""
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = ExoplanetDetector(sequence_length=2000)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = ExoplanetDetector(sequence_length=2000)
        x = torch.randn(2, 1, 2000)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_detect(self):
        """Test detection on dummy light curve."""
        model = ExoplanetDetector(sequence_length=2000)
        light_curve = np.random.randn(2000)
        result = model.detect(light_curve)
        
        assert 'exoplanet_detected' in result
        assert 'confidence' in result
        assert isinstance(result['exoplanet_detected'], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
