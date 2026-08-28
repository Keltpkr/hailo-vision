from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .people_store import enroll, list_people, rename
from .person_detector import PersonDetector


CAMERAS = {
    0: os.getenv("CAMERA_0", "0"),
    1: os.getenv("CAMERA_1", "1"),
}
WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))
FPS = int(os.getenv("CAMERA_FPS", "5"))
IMX708_WIDTH = int(os.getenv("CAMERA_1_WIDTH", "1280"))
IMX708_HEIGHT = int(os.getenv("CAMERA_1_HEIGHT", "720"))
USE_SUDO = os.getenv("CAMERA_USE_SUDO", "1") == "1"
RESOLUTIONS = ("640x480", "1280x720", "1920x1080")
FPS_OPTIONS = (5, 10, 15, 20, 30)
settings = {
    "0": {"resolution": f"{WIDTH}x{HEIGHT}", "fps": FPS},
    "1": {"resolution": f"{IMX708_WIDTH}x{IMX708_HEIGHT}", "fps": FPS},
}

detector: PersonDetector | None = None
detector_lock = threading.Lock()
detection_results: dict[str, dict[str, Any]] = {}
detection_lock = threading.Lock()

app = FastAPI(title="Hailo Camera Viewer", version="1.2.3")
_cpu_previous: tuple[int, int] | None = None


def cpu_percent() -> float:
    global _cpu_previous
    with open("/proc/stat", encoding="ascii") as proc_stat:
        fields = proc_stat.readline().split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]
    total = sum(values)
    current = (total, idle)
    if _cpu_previous is None:
        _cpu_previous = current
        return 0.0
    total_delta = total - _cpu_previous[0]
    idle_delta = idle - _cpu_previous[1]
    _cpu_previous = current
    return round(max(0.0, min(100.0, 100 * (1 - idle_delta / total_delta))), 1) if total_delta else 0.0


def camera_command(camera: str, width: int, height: int, fps: int) -> list[str]:
    command = [
        "rpicam-vid",
        "--camera", camera,
        "--codec", "mjpeg",
        "--width", str(width),
        "--height", str(height),
        "--framerate", str(fps),
        "--timeout", "0",
        "--output", "-",
        "--nopreview",
    ]
    if camera == "1":
        command += [
            "--autofocus-mode", "continuous",
            "--autofocus-range", "full",
            "--autofocus-speed", "fast",
            "--autofocus-window", "0.20,0.20,0.60,0.60",
        ]
    return (["sudo", "-n"] if USE_SUDO else []) + command


class CameraCapture:
    """One rpicam process per physical camera, shared by all HTTP clients."""

    def __init__(self, camera: str, camera_settings: dict) -> None:
        self.camera = camera
        self.camera_settings = camera_settings
        self.condition = threading.Condition()
        self.latest: bytes | None = None
        self.sequence = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.detection_thread = threading.Thread(target=self._detect, daemon=True)
        self.thread.start()
        self.detection_thread.start()

    def _detect(self) -> None:
        global detector
        last_sequence = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stop_event.is_set()
                    or (self.latest is not None and self.sequence != last_sequence),
                    timeout=5,
                )
                if self.stop_event.is_set():
                    return
                frame = self.latest
                last_sequence = self.sequence
            if frame is None:
                continue
            try:
                if detector is None:
                    with detector_lock:
                        if detector is None:
                            detector = PersonDetector()
                detections = detector.detect(frame)
                with detection_lock:
                    previous = detection_results.get(self.camera, {})
                    detection_results[self.camera] = {
                        "camera": int(self.camera),
                        "person_count": len(detections),
                        "alert": len(detections) > 0 and previous.get("person_count", 0) == 0,
                        "detections": detections,
                    }
            except Exception as exc:
                with detection_lock:
                    detection_results[self.camera] = {
                        "camera": int(self.camera),
                        "person_count": 0,
                        "alert": False,
                        "detections": [],
                        "error": str(exc),
                    }

    def _read(self) -> None:
        self.process = subprocess.Popen(
            camera_command(
                self.camera,
                *map(int, self.camera_settings["resolution"].split("x")),
                self.camera_settings["fps"],
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
        )
        if self.stop_event.is_set():
            self.stop()
            return
        assert self.process.stdout is not None
        buffer = b""
        while not self.stop_event.is_set():
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
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stop_event.is_set()
                    or (self.latest is not None and self.sequence != last_sequence),
                    timeout=5,
                )
                if self.stop_event.is_set():
                    return
                if self.latest is None:
                    continue
                frame = self.latest
                last_sequence = self.sequence
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                + frame + b"\r\n"
            )

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        if self.detection_thread is not threading.current_thread():
            self.detection_thread.join(timeout=3)


captures: dict[str, CameraCapture] = {}
captures_lock = threading.Lock()


def jpeg_stream(camera: str) -> Iterator[bytes]:
    with captures_lock:
        capture = captures.get(camera)
        if capture is None or not capture.thread.is_alive():
            if capture is not None:
                capture.stop()
            capture = CameraCapture(camera, settings[camera].copy())
            captures[camera] = capture
    yield from capture.frames()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caméras du Raspberry Pi</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:2rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}section{background:#222;padding:1rem;border-radius:8px}.camera-view{position:relative;background:#000}.camera-view img{width:100%;height:auto;display:block}.camera-view canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.rotate-180{transform:rotate(180deg)}#cpu{color:#8f8;font-weight:bold}.status{font-size:1.2rem}.alert{color:#ff7070;font-weight:bold}</style>
</head><body><h1>Caméras du Raspberry Pi</h1><p>Occupation CPU : <span id="cpu">--</span> %</p><main>
<section><h2>Caméra 0 — OV5647</h2><label>Résolution <select data-camera="0" class="resolution"></select></label> <label>FPS <select data-camera="0" class="fps"></select></label><p class="status">Personnes : <span id="count-0">0</span> <span id="alert-0"></span></p><div class="camera-view"><img src="/camera/0" alt="Flux caméra 0"><canvas id="canvas-0"></canvas></div><button onclick="enrollPerson(0)">Enregistrer une personne depuis la caméra 0</button></section>
<section><h2>Caméra 1 — IMX708</h2><label>Résolution <select data-camera="1" class="resolution"></select></label> <label>FPS <select data-camera="1" class="fps"></select></label><p class="status">Personnes : <span id="count-1">0</span> <span id="alert-1"></span></p><div class="camera-view"><img class="rotate-180" src="/camera/1" alt="Flux caméra 1"><canvas class="rotate-180" id="canvas-1"></canvas></div><button onclick="enrollPerson(1)">Enregistrer une personne depuis la caméra 1</button></section>
</main><script>
const resolutions = ["640x480", "1280x720", "1920x1080"];
const fpsOptions = [5, 10, 15, 20, 30];
async function init() {
  const current = await fetch('/health').then(r => r.json());
  document.querySelectorAll('.resolution').forEach(select => {
    resolutions.forEach(value => select.add(new Option(value, value)));
    select.value = current.settings[select.dataset.camera].resolution;
  });
  document.querySelectorAll('.fps').forEach(select => {
    fpsOptions.forEach(value => select.add(new Option(value + ' FPS', value)));
    select.value = current.settings[select.dataset.camera].fps;
  });
  document.getElementById('cpu').textContent = current.cpu_percent;
}
async function refreshCpu() {
  const current = await fetch('/health').then(r => r.json());
  document.getElementById('cpu').textContent = current.cpu_percent;
}
let previousCounts = [0, 0];
async function refreshDetections(camera) {
  const result = await fetch('/detections/' + camera).then(r => r.json());
  document.getElementById('count-' + camera).textContent = result.person_count;
  const alert = document.getElementById('alert-' + camera);
  alert.textContent = result.alert ? ' — PERSONNE DÉTECTÉE' : '';
  alert.className = result.alert ? 'alert' : '';
  if (result.person_count > 0 && previousCounts[camera] === 0) {
    try { const audio = new AudioContext(); const oscillator = audio.createOscillator(); oscillator.connect(audio.destination); oscillator.start(); oscillator.stop(audio.currentTime + 0.12); } catch (_) {}
  }
  previousCounts[camera] = result.person_count;
  const img = document.querySelector('img[src^="/camera/' + camera + '"]');
  const canvas = document.getElementById('canvas-' + camera);
  if (!img.naturalWidth) return;
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(3, canvas.width / 320); ctx.font = Math.max(16, canvas.width / 35) + 'px sans-serif';
  result.detections.forEach((d) => { const [x1,y1,x2,y2] = d.box; ctx.strokeStyle='#00ff66'; ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.fillStyle='#00ff66'; ctx.fillText((d.name || 'Personne non identifiée') + ' ' + Math.round(d.confidence * 100) + '%', x1, Math.max(20,y1-6)); });
}
async function changeCamera(event) {
  const camera = event.target.dataset.camera;
  const resolution = document.querySelector('.resolution[data-camera="' + camera + '"]').value;
  const fps = document.querySelector('.fps[data-camera="' + camera + '"]').value;
  await fetch('/settings/' + camera + '?resolution=' + encodeURIComponent(resolution) + '&fps=' + fps, {method: 'POST'});
  document.querySelector('img[src^="/camera/' + camera + '"]').src = '/camera/' + camera + '?t=' + Date.now();
}
async function enrollPerson(camera) {
  const name = prompt('Nom de la personne :', 'Personne 1');
  if (name === null) return;
  const response = await fetch('/people/enroll/' + camera + '?name=' + encodeURIComponent(name), {method: 'POST'});
  const result = await response.json();
  alert(response.ok ? 'Image de référence sauvegardée pour ' + result.name : result.detail);
}
document.querySelectorAll('select').forEach(select => select.addEventListener('change', changeCamera));
init();
setInterval(refreshCpu, 2000);
setInterval(() => { refreshDetections(0); refreshDetections(1); }, 500);
</script></body></html>"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cameras": CAMERAS, "settings": settings, "cpu_percent": cpu_percent()}


@app.get("/detections/{camera_id}")
def detections(camera_id: int) -> dict[str, Any]:
    if camera_id not in CAMERAS:
        raise HTTPException(404, "Caméra inconnue")
    with detection_lock:
        return detection_results.get(CAMERAS[camera_id], {
            "camera": camera_id,
            "person_count": 0,
            "alert": False,
            "detections": [],
            "status": "starting",
        })


@app.get("/people")
def people() -> list[dict[str, Any]]:
    return list_people()


@app.post("/people/enroll/{camera_id}")
def enroll_person(camera_id: int, name: str = "Personne 1") -> dict[str, Any]:
    if camera_id not in CAMERAS:
        raise HTTPException(404, "Caméra inconnue")
    with captures_lock:
        capture = captures.get(CAMERAS[camera_id])
    if capture is None or capture.latest is None:
        raise HTTPException(409, "Le flux caméra doit être actif")
    with detection_lock:
        result = detection_results.get(CAMERAS[camera_id], {})
    if result.get("person_count", 0) < 1:
        raise HTTPException(409, "Enrôlement refusé : aucune personne détectée")
    return enroll(capture.latest, name, camera_id)


@app.patch("/people/{person_id}")
def rename_person(person_id: str, name: str) -> dict[str, Any]:
    person = rename(person_id, name)
    if person is None:
        raise HTTPException(404, "Personne inconnue")
    return person


@app.get("/people/{person_id}/image")
def person_image(person_id: str) -> FileResponse:
    person = next((item for item in list_people() if item["id"] == person_id), None)
    if person is None:
        raise HTTPException(404, "Personne inconnue")
    image = Path(__file__).resolve().parent.parent / "data" / "people" / person["image"]
    if not image.is_file():
        raise HTTPException(404, "Image de référence introuvable")
    return FileResponse(image, media_type="image/jpeg")


@app.post("/settings/{camera_id}")
def update_settings(camera_id: int, resolution: str, fps: int) -> dict:
    key = str(camera_id)
    if key not in settings:
        raise HTTPException(404, "Caméra inconnue")
    if resolution not in RESOLUTIONS:
        raise HTTPException(400, "Résolution non supportée")
    if fps not in FPS_OPTIONS:
        raise HTTPException(400, "FPS non supportés")
    settings[key] = {"resolution": resolution, "fps": fps}
    physical_camera = CAMERAS[camera_id]
    with captures_lock:
        old_capture = captures.pop(physical_camera, None)
    if old_capture:
        old_capture.stop()
    return {"camera": camera_id, "settings": settings[key]}


@app.get("/camera/{camera_id}")
def camera(camera_id: int) -> StreamingResponse:
    if camera_id not in CAMERAS:
        raise HTTPException(404, "Caméra inconnue")
    return StreamingResponse(
        jpeg_stream(CAMERAS[camera_id]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
