from __future__ import annotations

import argparse

import cv2


BACKENDS = [
    ("default", None),
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
]


def test_camera(index: int) -> None:
    print(f"\nTesting camera index {index}")

    for backend_name, backend in BACKENDS:
        camera = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        opened = camera.isOpened()
        read_ok = False
        frame_shape = None

        if opened:
            read_ok, frame = camera.read()
            if read_ok and frame is not None:
                frame_shape = frame.shape

        print(
            f"  backend={backend_name:<7} opened={opened!s:<5} "
            f"read={read_ok!s:<5} frame={frame_shape}"
        )
        camera.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test available camera indexes and OpenCV backends.")
    parser.add_argument("--start", type=int, default=0, help="Start camera index.")
    parser.add_argument("--end", type=int, default=4, help="End camera index.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for index in range(args.start, args.end + 1):
        test_camera(index)
