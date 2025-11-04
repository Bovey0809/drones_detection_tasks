"""Tests for configuration module."""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drones_detection import DetectionConfig


class TestDetectionConfig(unittest.TestCase):
    """Test cases for DetectionConfig."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = DetectionConfig()
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.min_size, 50)
        self.assertEqual(config.max_size, 500)
    
    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = DetectionConfig(
            confidence_threshold=0.7,
            min_size=100,
            max_size=600
        )
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.min_size, 100)
        self.assertEqual(config.max_size, 600)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = DetectionConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['confidence_threshold'], 0.5)
        self.assertEqual(config_dict['min_size'], 50)
        self.assertEqual(config_dict['max_size'], 500)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'confidence_threshold': 0.8,
            'min_size': 75,
            'max_size': 400
        }
        config = DetectionConfig.from_dict(config_dict)
        
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.min_size, 75)
        self.assertEqual(config.max_size, 400)
    
    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = DetectionConfig()
        self.assertTrue(config.validate())
    
    def test_validate_invalid_confidence(self):
        """Test validation with invalid confidence threshold."""
        config = DetectionConfig(confidence_threshold=1.5)
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_invalid_min_size(self):
        """Test validation with invalid min_size."""
        config = DetectionConfig(min_size=-10)
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validate_invalid_size_relationship(self):
        """Test validation with max_size < min_size."""
        config = DetectionConfig(min_size=500, max_size=100)
        
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == '__main__':
    unittest.main()
