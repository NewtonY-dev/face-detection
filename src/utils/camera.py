from __future__ import annotations

import time
from typing import Tuple
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

import cv2
import numpy as np

from src.utils.config import ensure_project_dirs


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

    def read(self) -> Tuple[bool, np.ndarray | None]:
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


def configure_camera(camera: cv2.VideoCapture) -> None:
    camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


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


def open_video_source(
    source: str | None = None,
    camera_index: int = 0,
    backend_name: str = "default",
) -> cv2.VideoCapture | MjpegHttpStream:
    if source:
        stream_url = normalize_video_source(source)
        if not stream_url:
            raise RuntimeError("The video source URL is empty.")

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
