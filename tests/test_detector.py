"""Tests for DroneDetector class."""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drones_detection import DroneDetector


class TestDroneDetector(unittest.TestCase):
    """Test cases for DroneDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = DroneDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertEqual(self.detector.min_size, 50)
        self.assertEqual(self.detector.max_size, 500)
    
    def test_initialization_with_config(self):
        """Test detector initialization with custom config."""
        config = {
            'confidence_threshold': 0.7,
            'min_size': 100,
            'max_size': 600
        }
        detector = DroneDetector(config)
        self.assertEqual(detector.confidence_threshold, 0.7)
        self.assertEqual(detector.min_size, 100)
        self.assertEqual(detector.max_size, 600)
    
    def test_detect_empty_image(self):
        """Test detection with empty image."""
        empty_image = np.array([])
        detections = self.detector.detect(empty_image)
        self.assertEqual(len(detections), 0)
    
    def test_detect_none_image(self):
        """Test detection with None image."""
        detections = self.detector.detect(None)
        self.assertEqual(len(detections), 0)
    
    def test_detect_simple_image(self):
        """Test detection with a simple test image."""
        # Create a test image with a white square on black background
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        # Draw a square that should be detected as a drone
        image[150:250, 150:250] = 255
        
        detections = self.detector.detect(image)
        
        # Should detect at least some contours
        self.assertIsInstance(detections, list)
        
        # Each detection should have the required keys
        for detection in detections:
            self.assertIn('bbox', detection)
            self.assertIn('confidence', detection)
            self.assertIn('class_name', detection)
            self.assertEqual(detection['class_name'], 'drone')
    
    def test_draw_detections(self):
        """Test drawing detections on image."""
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        detections = [
            {
                'bbox': (100, 100, 50, 50),
                'confidence': 0.85,
                'class_name': 'drone'
            }
        ]
        
        annotated = self.detector.draw_detections(image, detections)
        
        # Check that image was modified
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, image.shape)
        # Image should have been modified (not all zeros anymore)
        self.assertTrue(np.any(annotated != 0))


if __name__ == '__main__':
    unittest.main()
