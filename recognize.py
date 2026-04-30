from __future__ import annotations

import argparse
import time

import cv2

from utils import (
    TRAINER_DIR,
    detect_faces,
    ensure_project_dirs,
    get_face_detector,
    load_label_map,
    mark_attendance,
    open_video_source,
)
from yolo_detect import PersonDetector


def run_recognition(camera_index: int, source: str | None, backend: str) -> None:
    ensure_project_dirs()
    model_path = TRAINER_DIR / "trainer.yml"
    if not model_path.exists():
        raise RuntimeError("Model file not found. Run train.py first.")

    labels = load_label_map()
    detector = get_face_detector()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path))

    person_detector = PersonDetector("yolov8n.pt")

    camera = open_video_source(source=source, camera_index=camera_index, backend_name=backend)

    logged_ids: set[int] = set()
    status_message = "System ready. Show a face to mark attendance."
    status_color = (255, 255, 255)
    status_expires_at = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            people_boxes = person_detector.detect(frame)
            current_time = time.time()

            faces = []
            for px, py, pw, ph in people_boxes:
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                person_gray = gray[py : py + ph, px : px + pw]
                if person_gray.size == 0:
                    continue

                person_faces = detect_faces(person_gray, detector)
                for fx, fy, fw, fh in person_faces:
                    faces.append((px + fx, py + fy, fw, fh))

            for x, y, w, h in faces:
                face = gray[y : y + h, x : x + w]
                person_id, confidence = recognizer.predict(face)

                is_known = confidence < 85 and person_id in labels
                name = labels.get(person_id, "Unknown") if is_known else "Unknown"
                color = (40, 220, 80) if is_known else (0, 0, 255)

                if is_known and person_id not in logged_ids:
                    marked_now = mark_attendance(person_id, name)
                    logged_ids.add(person_id)
                    if marked_now:
                        status_message = f"Attendance marked for {name}"
                        status_color = (40, 220, 80)
                    else:
                        status_message = f"Already marked today: {name}"
                        status_color = (0, 215, 255)
                    status_expires_at = current_time + 3
                elif not is_known:
                    status_message = "Unknown face detected"
                    status_color = (0, 0, 255)
                    status_expires_at = current_time + 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({confidence:.1f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                f"People Count (YOLO): {len(people_boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )
            
            cv2.putText(
                frame,
                "Press 'q' to exit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            if current_time > status_expires_at and not faces:
                status_message = "System ready. Show a face to mark attendance."
                status_color = (255, 255, 255)

            cv2.rectangle(frame, (10, 45), (630, 85), (30, 30, 30), -1)
            cv2.putText(
                frame,
                status_message,
                (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.imshow("Smart Attendance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live face recognition and attendance logging.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to open. Try 1 for an external camera.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional video source URL, such as an IP camera or DroidCam stream.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Camera backend: default, dshow, msmf, or auto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_recognition(args.camera, args.source, args.backend)
