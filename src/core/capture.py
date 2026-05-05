from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.detection import get_face_detector, detect_faces
from src.utils import create_user_dataset_dir, ensure_project_dirs, open_video_source


def capture_faces(
    person_id: int,
    name: str,
    samples: int,
    camera_index: int,
    source: str | None,
    backend: str,
) -> Path:
    ensure_project_dirs()
    detector = get_face_detector()
    output_dir = create_user_dataset_dir(person_id, name)

    camera = open_video_source(source=source, camera_index=camera_index, backend_name=backend)

    saved_images = 0

    try:
        while saved_images < samples:
            success, frame = camera.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = detect_faces(
                gray,
                detector,
                scale_factor=1.1,
                min_neighbors=4,
                min_size=(50, 50),
            )

            for x, y, w, h in faces:
                face = gray[y : y + h, x : x + w]
                saved_images += 1
                image_path = output_dir / f"{saved_images:03d}.jpg"
                cv2.imwrite(str(image_path), face)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 220, 80), 2)
                cv2.putText(
                    frame,
                    f"Saved: {saved_images}/{samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                if saved_images >= samples:
                    break

            cv2.putText(
                frame,
                f"Capturing for {name}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            status_text = f"Faces detected: {len(faces)}"
            status_color = (40, 220, 80) if faces else (0, 0, 255)
            if not faces:
                status_text = "No face detected. Move closer and improve lighting."

            cv2.putText(
                frame,
                status_text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.imshow("Dataset Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture face images for a user dataset.")
    parser.add_argument("--id", type=int, required=True, help="Numeric person ID.")
    parser.add_argument("--name", type=str, required=True, help="Person name.")
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of face images to capture.",
    )
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
    output_dir = capture_faces(args.id, args.name, args.samples, args.camera, args.source, args.backend)
    print(f"Saved dataset images to: {output_dir}")
