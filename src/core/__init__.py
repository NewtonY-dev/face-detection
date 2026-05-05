from __future__ import annotations

from src.core.capture import capture_faces
from src.core.recognize import run_recognition
from src.core.train import train_model

__all__ = [
    "capture_faces",
    "run_recognition",
    "train_model",
]
