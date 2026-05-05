from __future__ import annotations

import csv
from datetime import datetime
from typing import Dict, List

from src.utils.config import ATTENDANCE_FILE, ensure_project_dirs


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
