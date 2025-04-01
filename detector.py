from typing import List

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.6) -> np.ndarray:
        # Perform detection using the YOLO model
        results = self.model(frame)

        # Extract bounding boxes and filter based on the confidence score
        boxes = results[0].boxes.data.numpy()  # Use numpy() to convert tensor to numpy array

        # Filter out boxes where confidence is less than the threshold
        boxes = boxes[boxes[:, 4] >= confidence_threshold]  # confidence is at index 4

        return boxes