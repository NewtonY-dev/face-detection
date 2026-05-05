from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection import get_face_detector, detect_faces
from src.utils import DATASET_DIR, TRAINER_DIR, ensure_project_dirs, parse_label_map, save_label_map


def load_training_data() -> tuple[list[np.ndarray], list[int]]:
    ensure_project_dirs()
    detector = get_face_detector()
    faces: list[np.ndarray] = []
    labels: list[int] = []

    for user_dir in DATASET_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        parts = user_dir.name.split("_", 2)
        if len(parts) < 3 or parts[0] != "user":
            continue

        try:
            person_id = int(parts[1])
        except ValueError:
            continue

        for image_path in user_dir.glob("*.jpg"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            detected_faces = detect_faces(image, detector)
            if detected_faces:
                x, y, w, h = detected_faces[0]
                image = image[y : y + h, x : x + w]

            faces.append(image)
            labels.append(person_id)

    return faces, labels


def train_model() -> Path:
    faces, labels = load_training_data()
    if not faces:
        raise RuntimeError("No training data found. Capture dataset images first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    output_path = TRAINER_DIR / "trainer.yml"
    recognizer.write(str(output_path))
    save_label_map(parse_label_map())
    return output_path


if __name__ == "__main__":
    model_path = train_model()
    print(f"Model trained and saved to: {model_path}")
