from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_DIR = BASE_DIR / "trainer"
ATTENDANCE_DIR = BASE_DIR / "attendance"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"
LABELS_FILE = TRAINER_DIR / "labels.npy"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# Create the required project folders and initialize the attendance CSV header if missing.
def ensure_project_dirs() -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    TRAINER_DIR.mkdir(exist_ok=True)
    ATTENDANCE_DIR.mkdir(exist_ok=True)

    if not ATTENDANCE_FILE.exists():
        with ATTENDANCE_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["person_id", "name", "timestamp"])


# Load the OpenCV Haar Cascade model used to detect faces in images and video frames.
def get_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError("Failed to load Haar Cascade face detector.")
    return detector


CAMERA_BACKENDS = {
    "default": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


class MjpegHttpStream:
    def __init__(self, source_url: str, timeout: float = 8.0) -> None:
        self.source_url = source_url
        self.timeout = timeout
        self._response = None
        self._buffer = bytearray()
        self._open()

    def _open(self) -> None:
        self._response = urlopen(self.source_url, timeout=self.timeout)

    def isOpened(self) -> bool:
        return self._response is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._response is None:
            return False, None

        while True:
            start = self._buffer.find(b"\xff\xd8")
            end = self._buffer.find(b"\xff\xd9", start + 2 if start != -1 else 0)
            if start != -1 and end != -1:
                jpeg_bytes = bytes(self._buffer[start : end + 2])
                del self._buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame

            chunk = self._response.read(4096)
            if not chunk:
                return False, None
            self._buffer.extend(chunk)

    def release(self) -> None:
        if self._response is not None:
            self._response.close()
            self._response = None


def _can_read_frame(camera) -> bool:
    for _ in range(30):
        success, frame = camera.read()
        if success and frame is not None and getattr(frame, "size", 0) > 0:
            return True
        time.sleep(0.1)
    return False


def normalize_video_source(source: str | None) -> str | None:
    if not source:
        return None

    stream_url = source.strip()
    if not stream_url:
        return None

    if "://" not in stream_url:
        stream_url = f"http://{stream_url}"

    parsed = urlparse(stream_url)
    if parsed.scheme == "https":
        parsed = parsed._replace(scheme="http")

    if not parsed.path or parsed.path == "/":
        parsed = parsed._replace(path="/video")

    return urlunparse(parsed)


# Apply camera settings that aim for stable, low-latency capture.
def configure_camera(camera: cv2.VideoCapture) -> None:
    camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


# Open a webcam using a selected OpenCV backend, or try multiple backends in auto mode.
def open_camera(camera_index: int = 0, backend_name: str = "default") -> cv2.VideoCapture:
    if backend_name == "auto":
        backends = list(CAMERA_BACKENDS.items())
    else:
        if backend_name not in CAMERA_BACKENDS:
            allowed = ", ".join(CAMERA_BACKENDS.keys()) + ", auto"
            raise RuntimeError(f"Unsupported backend '{backend_name}'. Use one of: {allowed}")
        backends = [(backend_name, CAMERA_BACKENDS[backend_name])]

    for _, backend in backends:
        camera = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
        configure_camera(camera)
        if camera.isOpened():
            return camera
        camera.release()

    raise RuntimeError(
        "Unable to access the webcam. Try --camera 1 if you use an external camera or another app is holding the webcam."
    )


# Open either a webcam or an external video stream such as an IP camera URL.
def open_video_source(
    source: str | None = None,
    camera_index: int = 0,
    backend_name: str = "default",
) -> cv2.VideoCapture | MjpegHttpStream:
    if source:
        stream_url = normalize_video_source(source)
        if not stream_url:
            raise RuntimeError("The video source URL is empty.")

        # IP webcam streams are not local devices, so avoid webcam-only property tuning here.
        for open_args in ((stream_url,), (stream_url, cv2.CAP_FFMPEG)):
            camera = cv2.VideoCapture(*open_args)
            if camera.isOpened() and _can_read_frame(camera):
                return camera
            camera.release()

        if stream_url.lower().startswith("http://"):
            try:
                camera = MjpegHttpStream(stream_url)
                if camera.isOpened() and _can_read_frame(camera):
                    return camera
                camera.release()
            except (OSError, TimeoutError, ValueError, URLError):
                pass

        raise RuntimeError(
            "Unable to read frames from the video source. Make sure the phone app is running and use the direct "
            "HTTP stream URL, such as 'http://<ip>:8080/video'."
        )

    return open_camera(camera_index, backend_name=backend_name)


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


# Read all attendance rows from the CSV file into a list of dictionaries.
def read_attendance_records() -> List[Dict[str, str]]:
    ensure_project_dirs()
    records: List[Dict[str, str]] = []

    if not ATTENDANCE_FILE.exists():
        return records

    with ATTENDANCE_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(
                {
                    "person_id": row.get("person_id", ""),
                    "name": row.get("name", ""),
                    "timestamp": row.get("timestamp", ""),
                }
            )

    return records


# Check whether a specific student already has attendance recorded for a given date.
def has_attendance_for_date(person_id: int, date_text: str) -> bool:
    ensure_project_dirs()
    if not ATTENDANCE_FILE.exists():
        return False

    with ATTENDANCE_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row_person_id = int(row["person_id"])
            except (KeyError, TypeError, ValueError):
                continue

            timestamp = row.get("timestamp", "")
            if row_person_id == person_id and timestamp.startswith(date_text):
                return True

    return False


# Append a new attendance record only if that student has not already been marked today.
def mark_attendance(person_id: int, name: str) -> bool:
    ensure_project_dirs()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if has_attendance_for_date(person_id, today):
        return False

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with ATTENDANCE_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([person_id, name, timestamp])
    return True
