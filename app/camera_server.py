from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse


CAMERAS = {
    0: os.getenv("CAMERA_0", "0"),
    1: os.getenv("CAMERA_1", "1"),
}
WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
FPS = int(os.getenv("CAMERA_FPS", "10"))
USE_SUDO = os.getenv("CAMERA_USE_SUDO", "1") == "1"

app = FastAPI(title="Hailo Camera Viewer", version="1.0.4")


def camera_command(camera: str) -> list[str]:
    command = [
        "rpicam-vid",
        "--camera", camera,
        "--codec", "mjpeg",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--timeout", "0",
        "--output", "-",
        "--nopreview",
    ]
    if camera == "0":
        command += ["--rotation", "180"]
    return (["sudo", "-n"] if USE_SUDO else []) + command


class CameraCapture:
    """One rpicam process per physical camera, shared by all HTTP clients."""

    def __init__(self, camera: str) -> None:
        self.camera = camera
        self.condition = threading.Condition()
        self.latest: bytes | None = None
        self.sequence = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        self.process = subprocess.Popen(
            camera_command(self.camera),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        assert self.process.stdout is not None
        buffer = b""
        while True:
            chunk = self.process.stdout.read(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer = buffer[-1:]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    buffer = buffer[start:]
                    break
                frame = buffer[start:end + 2]
                buffer = buffer[end + 2:]
                with self.condition:
                    self.latest = frame
                    self.sequence += 1
                    self.condition.notify_all()

    def frames(self) -> Iterator[bytes]:
        last_sequence = -1
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.latest is not None and self.sequence != last_sequence,
                    timeout=5,
                )
                if self.latest is None:
                    continue
                frame = self.latest
                last_sequence = self.sequence
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame + b"\r\n"
            )


captures: dict[str, CameraCapture] = {}
captures_lock = threading.Lock()


def jpeg_stream(camera: str) -> Iterator[bytes]:
    with captures_lock:
        capture = captures.get(camera)
        if capture is None:
            capture = CameraCapture(camera)
            captures[camera] = capture
    yield from capture.frames()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caméras du Raspberry Pi</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}section{background:#222;padding:1rem;border-radius:8px}img{width:100%;height:auto;background:#000;display:block}</style>
</head><body><h1>Caméras du Raspberry Pi</h1><main>
<section><h2>Caméra 0 — OV5647</h2><img src="/camera/0" alt="Flux caméra 0"></section>
<section><h2>Caméra 1 — IMX708</h2><img src="/camera/1" alt="Flux caméra 1"></section>
</main></body></html>"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cameras": CAMERAS, "resolution": [WIDTH, HEIGHT], "fps": FPS}


@app.get("/camera/{camera_id}")
def camera(camera_id: int) -> StreamingResponse:
    if camera_id not in CAMERAS:
        raise HTTPException(404, "Caméra inconnue")
    return StreamingResponse(
        jpeg_stream(CAMERAS[camera_id]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
