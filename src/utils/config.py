from __future__ import annotations

from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
TRAINER_DIR = BASE_DIR / "data" / "trainer"
ATTENDANCE_DIR = BASE_DIR / "data" / "attendance"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"
LABELS_FILE = TRAINER_DIR / "labels.npy"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# Create the required project folders and initialize the attendance CSV header if missing.
def ensure_project_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

    if not ATTENDANCE_FILE.exists():
        import csv
        with ATTENDANCE_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["person_id", "name", "timestamp"])
