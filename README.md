
# Object Detection, Tracking, and Counting in ROI

This project detects and tracks objects (e.g., people or vehicles) in a video feed, and counts them within a defined Region of Interest (ROI). Built with YOLO for detection and SORT for tracking.

## Features

-  Object Detection (YOLO)
-  Object Tracking (SORT)
-  Counting objects inside a custom ROI
-  Supports webcam and video files
-  Optional video output saving

## Setup

### 1. Clone the Repo


### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Download a YOLOv5 Model
Get a YOLOv8vn PyTorch model 

## Usage

### Run with Webcam:
```bash
python main.py --model
```

### Run with Video File:
```bash
python main.py --video path/to/video.mp4 --model --output output.mp4
```

## ROI Customization

You can modify the `DEFAULT_COUNTER_ROI` list in `main.py` to set your own polygon region:
```python
DEFAULT_COUNTER_ROI = [(x1, y1), (x2, y2), ..., (xn, yn)]
```

## File Structure

- `main.py` — Entry point
- `detector.py` — YOLO detection
- `tracker.py` — SORT tracking
- `counter.py` — ROI-based counter
- `util.py` — Drawing utilities

