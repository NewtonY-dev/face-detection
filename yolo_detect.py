from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        # Load the YOLO model (nano model by default for speed)
        self.model = YOLO(model_path)
    
    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detects people in the given frame using YOLO.
        Returns a list of bounding boxes (x, y, w, h) for detected people.
        """
        results = self.model(frame, classes=[0], verbose=False) # class 0 is 'person' in COCO dataset
        
        boxes = []
        for result in results:
            for box in result.boxes:
                # get bounding box coordinates (xyxy)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w = x2 - x1
                h = y2 - y1
                boxes.append((x1, y1, w, h))
        return boxes
