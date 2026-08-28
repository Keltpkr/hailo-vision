from __future__ import annotations

import io
import os
import threading
from typing import Any

import numpy as np
from PIL import Image

from .hailo_engine import HailoEngine


DEFAULT_HEF = "/usr/local/hailo/resources/models/hailo8/yolov8m.hef"


class PersonDetector:
    """YOLO person detector. The inference itself is always executed by Hailo."""

    def __init__(self) -> None:
        self.engine = HailoEngine(os.getenv("HEF_PATH", DEFAULT_HEF))
        self.lock = threading.Lock()

    def detect(self, jpeg: bytes) -> list[dict[str, Any]]:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        source_width, source_height = image.size
        height, width, channels = self.engine.input_shape[-3:]
        if channels != 3:
            raise ValueError("Le modèle personne doit accepter 3 canaux")
        tensor = np.array(
            image.resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
            copy=True,
        )
        with self.lock:
            outputs = self.engine.infer(tensor)
        output = next(iter(outputs.values()))
        detections: list[dict[str, Any]] = []
        # Hailo YOLO NMS output: [batch][class][detection][ymin,xmin,ymax,xmax,score].
        for batch in output[:1]:
            if not batch:
                continue
            for class_id, class_detections in enumerate(batch):
                if class_id != 0:  # COCO: person
                    continue
                for detection in class_detections:
                    if len(detection) < 5:
                        continue
                    ymin, xmin, ymax, xmax, score = map(float, detection[:5])
                    if score < float(os.getenv("PERSON_CONFIDENCE", "0.45")):
                        continue
                    detections.append({
                        "label": "person",
                        "confidence": round(score, 3),
                        "box": [
                            round(max(0, min(1, xmin)) * source_width),
                            round(max(0, min(1, ymin)) * source_height),
                            round(max(0, min(1, xmax)) * source_width),
                            round(max(0, min(1, ymax)) * source_height),
                        ],
                    })
        return detections

