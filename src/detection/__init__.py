from __future__ import annotations

from src.detection.face_detector import get_face_detector, detect_faces
from src.detection.person_detector import PersonDetector

__all__ = [
    "get_face_detector",
    "detect_faces",
    "PersonDetector",
]
