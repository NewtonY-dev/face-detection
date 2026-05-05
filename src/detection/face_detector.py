from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.utils.config import CASCADE_PATH, ensure_project_dirs


# Load the OpenCV Haar Cascade model used to detect faces in images and video frames.
def get_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade face detector.")
    return detector


# Detect face bounding boxes in a grayscale frame using the Haar Cascade classifier.
def detect_faces(
    gray_frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.2,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (80, 80),
) -> List[Tuple[int, int, int, int]]:
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return list(faces)
