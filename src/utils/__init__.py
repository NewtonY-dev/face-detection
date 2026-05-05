from __future__ import annotations

from src.utils.config import BASE_DIR, DATASET_DIR, TRAINER_DIR, ATTENDANCE_DIR, ATTENDANCE_FILE, LABELS_FILE, CASCADE_PATH
from src.utils.camera import MjpegHttpStream, CAMERA_BACKENDS, normalize_video_source, configure_camera, open_camera, open_video_source, _can_read_frame
from src.utils.dataset import normalize_name, user_folder_name, create_user_dataset_dir, parse_label_map, save_label_map, load_label_map, list_registered_students, get_next_person_id
from src.utils.attendance import read_attendance_records, has_attendance_for_date, mark_attendance

__all__ = [
    # config
    "BASE_DIR", "DATASET_DIR", "TRAINER_DIR", "ATTENDANCE_DIR", "ATTENDANCE_FILE", "LABELS_FILE", "CASCADE_PATH",
    # camera
    "MjpegHttpStream", "CAMERA_BACKENDS", "normalize_video_source", "configure_camera", "open_camera", "open_video_source", "_can_read_frame",
    # dataset
    "normalize_name", "user_folder_name", "create_user_dataset_dir", "parse_label_map", "save_label_map", "load_label_map", "list_registered_students", "get_next_person_id",
    # attendance
    "read_attendance_records", "has_attendance_for_date", "mark_attendance",
]
