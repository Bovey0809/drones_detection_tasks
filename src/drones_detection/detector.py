"""Core drone detection module."""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2


class DroneDetector:
    """Drone detection class for identifying drones in images and video streams."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the drone detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.min_size = self.config.get('min_size', 50)
        self.max_size = self.config.get('max_size', 500)
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect drones in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with keys:
                - bbox: Bounding box coordinates (x, y, w, h)
                - confidence: Detection confidence score
                - class_name: Detected object class name
        """
        if image is None or image.size == 0:
            return []
            
        detections = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < self.min_size or h < self.min_size:
                continue
            if w > self.max_size or h > self.max_size:
                continue
                
            # Calculate area ratio (drones are typically roughly square/rectangular)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Calculate confidence based on contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            confidence = min(0.95, max(0.1, circularity))
            
            if confidence >= self.confidence_threshold:
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': float(confidence),
                    'class_name': 'drone'
                })
        
        return detections
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> List[List[Dict[str, Any]]]:
        """
        Detect drones in a video file.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            
        Returns:
            List of detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        all_detections = []
        writer = None
        
        try:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect drones in frame
                detections = self.detect(frame)
                all_detections.append(detections)
                
                # Draw detections if saving output
                if output_path and writer:
                    annotated_frame = self.draw_detections(frame, detections)
                    writer.write(annotated_frame)
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        return all_detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Drone: {confidence:.2f}"
            cv2.putText(annotated, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated
