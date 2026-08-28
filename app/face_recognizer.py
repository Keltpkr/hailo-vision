from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from .hailo_engine import HailoEngine


SCRFD_HEF = "/usr/local/hailo/resources/models/hailo8/scrfd_10g.hef"
ARCFACE_HEF = "/usr/local/hailo/resources/models/hailo8/arcface_mobilefacenet.hef"


def _nms(boxes: list[tuple[float, float, float, float, float]]) -> list[tuple[float, float, float, float, float]]:
    kept: list[tuple[float, float, float, float, float]] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, _ = box
        for other in kept:
            ox1, oy1, ox2, oy2, _ = other
            overlap = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            union = (x2 - x1) * (y2 - y1) + (ox2 - ox1) * (oy2 - oy1) - overlap
            if union and overlap / union > 0.4:
                break
        else:
            kept.append(box)
    return kept


class FaceRecognizer:
    """Face detection and embedding extraction, with both models run on Hailo."""

    def __init__(self) -> None:
        self.detector = HailoEngine(SCRFD_HEF)
        self.embedder = HailoEngine(ARCFACE_HEF)

    def faces(self, jpeg: bytes) -> list[dict[str, Any]]:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        original_width, original_height = image.size
        network_image = image.resize((640, 640), Image.Resampling.BILINEAR)
        raw = self.detector.infer(np.array(network_image, dtype=np.uint8, copy=True))
        outputs = [np.asarray(value)[0] for value in raw.values()]
        classes = sorted((value for value in outputs if value.shape[-1] == 2), key=lambda value: value.shape[0], reverse=True)
        boxes = sorted((value for value in outputs if value.shape[-1] == 8), key=lambda value: value.shape[0], reverse=True)
        candidates: list[tuple[float, float, float, float, float]] = []
        for stride, scores, deltas, sizes in zip((8, 16, 32), classes, boxes, ((16, 32), (64, 128), (256, 512)), strict=True):
            height, width = scores.shape[:2]
            scores = scores.reshape(height, width, 2)
            deltas = deltas.reshape(height, width, 2, 4)
            for y, x, anchor in np.argwhere(scores > 0.4):
                score = float(scores[y, x, anchor])
                size = sizes[int(anchor)]
                cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
                dx1, dy1, dx2, dy2 = deltas[y, x, anchor]
                candidates.append((
                    max(0.0, cx - float(dx1) * 0.1 * size),
                    max(0.0, cy - float(dy1) * 0.1 * size),
                    min(640.0, cx + float(dx2) * 0.1 * size),
                    min(640.0, cy + float(dy2) * 0.1 * size),
                    score,
                ))
        faces: list[dict[str, Any]] = []
        for x1, y1, x2, y2, score in _nms(candidates):
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop = network_image.crop((x1, y1, x2, y2)).resize((112, 112), Image.Resampling.BILINEAR)
            embedding_raw = self.embedder.infer(np.array(crop, dtype=np.uint8, copy=True))
            embedding = np.asarray(next(iter(embedding_raw.values())), dtype=np.float32).reshape(-1)
            embedding /= max(np.linalg.norm(embedding), 1e-12)
            ox1, oy1 = x1 * original_width / 640, y1 * original_height / 640
            ox2, oy2 = x2 * original_width / 640, y2 * original_height / 640
            reference = image.crop((ox1, oy1, ox2, oy2))
            encoded = io.BytesIO(); reference.save(encoded, format="JPEG", quality=90)
            faces.append({"confidence": round(score, 3), "box": [round(ox1), round(oy1), round(ox2), round(oy2)], "embedding": embedding.tolist(), "image": encoded.getvalue()})
        return faces
