# Drones Detection Tasks

A Python-based drone detection system for identifying drones in images and video streams using computer vision techniques.

## Features

- 🎯 Real-time drone detection in images
- 📹 Video stream processing
- 🔧 Configurable detection parameters
- 📊 Bounding box visualization
- 🧪 Comprehensive test suite
- 🚀 Easy-to-use API

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/Bovey0809/drones_detection_tasks.git
cd drones_detection_tasks

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from drones_detection import DroneDetector
from drones_detection.utils import load_image, save_image

# Initialize detector
detector = DroneDetector()

# Load an image
image = load_image('path/to/image.jpg')

# Detect drones
detections = detector.detect(image)

# Draw detections
annotated_image = detector.draw_detections(image, detections)

# Save result
save_image(annotated_image, 'output.jpg')

# Print results
for detection in detections:
    bbox = detection['bbox']
    confidence = detection['confidence']
    print(f"Drone detected at ({bbox[0]}, {bbox[1]}) "
          f"with confidence {confidence:.2f}")
```

### Custom Configuration

```python
from drones_detection import DroneDetector, DetectionConfig

# Create custom configuration
config = DetectionConfig(
    confidence_threshold=0.7,  # Higher threshold for stricter detection
    min_size=50,               # Minimum object size in pixels
    max_size=500               # Maximum object size in pixels
)

# Validate configuration
config.validate()

# Initialize detector with custom config
detector = DroneDetector(config.to_dict())
```

### Video Processing

```python
from drones_detection import DroneDetector

detector = DroneDetector()

# Process video and save annotated output
all_detections = detector.detect_video(
    video_path='input_video.mp4',
    output_path='output_video.mp4'
)

print(f"Processed {len(all_detections)} frames")
```

## Running Examples

Try the included example script:

```bash
python examples/detect_sample.py
```

This will create a sample image with a drone-like object, detect it, and save the annotated result to the `output` directory.

## Running Tests

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_detector

# Run with verbose output
python -m unittest discover tests -v
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_threshold` | float | 0.5 | Minimum confidence score for detections (0.0-1.0) |
| `min_size` | int | 50 | Minimum bounding box size in pixels |
| `max_size` | int | 500 | Maximum bounding box size in pixels |

## API Reference

### DroneDetector

Main class for drone detection.

#### Methods

- `__init__(config: Optional[Dict[str, Any]] = None)`: Initialize detector
- `detect(image: np.ndarray) -> List[Dict[str, Any]]`: Detect drones in an image
- `detect_video(video_path: str, output_path: Optional[str] = None) -> List[List[Dict[str, Any]]]`: Process video file
- `draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray`: Draw bounding boxes

### DetectionConfig

Configuration dataclass for detection parameters.

#### Methods

- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `from_dict(config_dict: Dict[str, Any]) -> DetectionConfig`: Create from dictionary
- `validate() -> bool`: Validate configuration parameters

### Utility Functions

- `load_image(image_path: str) -> np.ndarray`: Load image from file
- `save_image(image: np.ndarray, output_path: str) -> bool`: Save image to file
- `resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray`: Resize image
- `calculate_iou(box1: Tuple, box2: Tuple) -> float`: Calculate Intersection over Union

## Project Structure

```
drones_detection_tasks/
├── src/
│   └── drones_detection/
│       ├── __init__.py          # Package initialization
│       ├── detector.py          # Main detector class
│       ├── config.py            # Configuration module
│       └── utils.py             # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_detector.py        # Detector tests
│   ├── test_config.py          # Configuration tests
│   └── test_utils.py           # Utility tests
├── examples/
│   └── detect_sample.py        # Example usage script
├── requirements.txt            # Project dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## How It Works

The drone detection system uses computer vision techniques:

1. **Preprocessing**: Images are converted to grayscale and blurred to reduce noise
2. **Edge Detection**: Canny edge detection identifies object boundaries
3. **Contour Detection**: Find contours in the edge-detected image
4. **Filtering**: Filter detections by size and aspect ratio
5. **Confidence Scoring**: Calculate confidence based on contour properties
6. **Post-processing**: Return detections with bounding boxes and confidence scores

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with OpenCV and NumPy
- Uses computer vision techniques for object detection