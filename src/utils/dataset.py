from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.utils.config import DATASET_DIR, LABELS_FILE, ensure_project_dirs


# Convert a person's name into a normalized folder-friendly format.
def normalize_name(name: str) -> str:
    return "_".join(name.strip().split()).lower()


# Build the dataset folder name pattern for one student using ID and normalized name.
def user_folder_name(person_id: int, name: str) -> str:
    return f"user_{person_id}_{normalize_name(name)}"


# Create and return the dataset folder where one student's face samples will be stored.
def create_user_dataset_dir(person_id: int, name: str) -> Path:
    ensure_project_dirs()
    folder = DATASET_DIR / user_folder_name(person_id, name)
    folder.mkdir(exist_ok=True)
    return folder


# Read dataset folder names and reconstruct a map from student ID to display name.
def parse_label_map() -> Dict[int, str]:
    ensure_project_dirs()
    labels: Dict[int, str] = {}

    for folder in DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.split("_", 2)
        if len(parts) < 3 or parts[0] != "user":
            continue
        try:
            person_id = int(parts[1])
        except ValueError:
            continue
        labels[person_id] = parts[2].replace("_", " ").title()

    return labels


# Save the ID-to-name label map so recognition can translate predictions into names.
def save_label_map(labels: Dict[int, str]) -> None:
    ensure_project_dirs()
    np.save(LABELS_FILE, labels, allow_pickle=True)


# Load the saved label map, or rebuild it from dataset folders if the saved file is missing.
def load_label_map() -> Dict[int, str]:
    ensure_project_dirs()
    if not LABELS_FILE.exists():
        return parse_label_map()

    loaded = np.load(LABELS_FILE, allow_pickle=True)
    if hasattr(loaded, "item"):
        data = loaded.item()
        if isinstance(data, dict):
            return {int(person_id): str(name) for person_id, name in data.items()}

    return parse_label_map()


# Return all registered students together with the number of dataset images for each one.
def list_registered_students() -> List[Tuple[int, str, int]]:
    ensure_project_dirs()
    students: List[Tuple[int, str, int]] = []
    labels = parse_label_map()

    for person_id, name in sorted(labels.items()):
        folder = DATASET_DIR / user_folder_name(person_id, name)
        image_count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        students.append((person_id, name, image_count))

    return students


# Generate the next available student ID based on the currently registered dataset folders.
def get_next_person_id() -> int:
    students = list_registered_students()
    if not students:
        return 1
    return max(person_id for person_id, _, _ in students) + 1
