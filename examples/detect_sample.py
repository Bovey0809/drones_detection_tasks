"""Example script demonstrating drone detection usage."""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drones_detection import DroneDetector, DetectionConfig
from drones_detection.utils import save_image


def create_sample_image():
    """Create a sample image with a drone-like object."""
    # Create a 600x600 image
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Add some background texture
    image[:, :] = [30, 30, 30]
    
    # Draw a "drone" (white rectangle with some features)
    # Main body
    image[200:280, 250:350] = [200, 200, 200]
    
    # Propellers (four circles)
    import cv2
    cv2.circle(image, (260, 210), 15, (150, 150, 150), -1)
    cv2.circle(image, (340, 210), 15, (150, 150, 150), -1)
    cv2.circle(image, (260, 270), 15, (150, 150, 150), -1)
    cv2.circle(image, (340, 270), 15, (150, 150, 150), -1)
    
    return image


def main():
    """Run example detection."""
    print("Drone Detection Example")
    print("-" * 50)
    
    # Create configuration
    config = DetectionConfig(
        confidence_threshold=0.3,
        min_size=30,
        max_size=600
    )
    
    print(f"\nConfiguration:")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Min size: {config.min_size}")
    print(f"  Max size: {config.max_size}")
    
    # Validate configuration
    try:
        config.validate()
        print("  Configuration is valid ✓")
    except ValueError as e:
        print(f"  Configuration error: {e}")
        return
    
    # Initialize detector
    detector = DroneDetector(config.to_dict())
    print("\nDetector initialized ✓")
    
    # Create a sample image
    print("\nCreating sample image...")
    image = create_sample_image()
    print(f"  Image shape: {image.shape}")
    
    # Detect drones
    print("\nRunning detection...")
    detections = detector.detect(image)
    
    print(f"\nDetection Results:")
    print(f"  Found {len(detections)} potential drone(s)")
    
    for i, detection in enumerate(detections, 1):
        bbox = detection['bbox']
        confidence = detection['confidence']
        print(f"\n  Detection {i}:")
        print(f"    Bounding box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        print(f"    Confidence: {confidence:.3f}")
        print(f"    Class: {detection['class_name']}")
    
    # Draw detections
    annotated_image = detector.draw_detections(image, detections)
    
    # Save result
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'detection_result.png')
    
    if save_image(annotated_image, output_path):
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        print("\nFailed to save annotated image")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
