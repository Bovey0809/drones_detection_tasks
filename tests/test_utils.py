"""Tests for utility functions."""

import unittest
import numpy as np
import sys
import os
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drones_detection.utils import (
    load_image, save_image, resize_image, calculate_iou
)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_resize_image_by_width(self):
        """Test resizing image by width."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(image, width=100)
        
        self.assertEqual(resized.shape[1], 100)
        self.assertEqual(resized.shape[0], 50)  # Maintains aspect ratio
    
    def test_resize_image_by_height(self):
        """Test resizing image by height."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(image, height=50)
        
        self.assertEqual(resized.shape[0], 50)
        self.assertEqual(resized.shape[1], 100)  # Maintains aspect ratio
    
    def test_resize_image_no_change(self):
        """Test resizing with no dimensions specified."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = resize_image(image)
        
        np.testing.assert_array_equal(image, resized)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 10, 10)
        
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_complete_overlap(self):
        """Test IoU calculation with complete overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)
        
        iou = calculate_iou(box1, box2)
        self.assertEqual(iou, 1.0)
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 10, 10)
        
        iou = calculate_iou(box1, box2)
        
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 ≈ 0.143
        self.assertAlmostEqual(iou, 25/175, places=3)
    
    def test_save_and_load_image(self):
        """Test saving and loading an image."""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save image
            result = save_image(test_image, tmp_path)
            self.assertTrue(result)
            
            # Load image
            loaded_image = load_image(tmp_path)
            self.assertIsNotNone(loaded_image)
            self.assertEqual(loaded_image.shape, test_image.shape)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_load_nonexistent_image(self):
        """Test loading a non-existent image."""
        loaded = load_image('/nonexistent/path/image.png')
        self.assertIsNone(loaded)


if __name__ == '__main__':
    unittest.main()
