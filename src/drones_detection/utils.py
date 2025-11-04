"""Utility functions for drone detection."""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None if loading fails
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image from {image_path}")
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
        
    Returns:
        True if successful, False otherwise
    """
    return cv2.imwrite(output_path, image)


def resize_image(image: np.ndarray, width: Optional[int] = None, 
                 height: Optional[int] = None) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / float(h)
        width = int(w * ratio)
    elif height is None:
        ratio = width / float(w)
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box (x, y, w, h)
        box2: Second box (x, y, w, h)
        
    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union
