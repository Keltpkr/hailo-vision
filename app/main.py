from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .hailo_engine import HailoEngine
from . import __version__

engine: HailoEngine | None = None
max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", "10485760"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    hef_path = os.getenv("HEF_PATH")
    if not hef_path:
        raise RuntimeError("HEF_PATH doit pointer vers un fichier .hef")
    engine = HailoEngine(hef_path)
    yield
    engine.close()
    engine = None


app = FastAPI(title="Hailo Vision API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    if engine is None:
        return {"status": "starting", "accelerator": "hailo"}
    return {
        "status": "ok",
        "accelerator": "hailo",
        "input_shape": engine.input_shape,
        "outputs": engine.output_names,
    }


@app.post("/v1/infer")
async def infer(image: UploadFile = File(...)) -> dict:
    if engine is None:
        raise HTTPException(503, "Moteur Hailo indisponible")
    content = await image.read(max_image_bytes + 1)
    if len(content) > max_image_bytes:
        raise HTTPException(413, "Image trop volumineuse")
    try:
        decoded = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Image JPEG/PNG invalide") from exc

    height, width, channels = engine.input_shape[-3:]
    if channels != 3:
        raise HTTPException(500, f"Le modèle attend {channels} canaux, API RGB uniquement")
    resized = decoded.resize((width, height), Image.Resampling.BILINEAR)
    # HailoRT écrit/épingle le buffer d'entrée : il doit être modifiable.
    tensor = np.array(resized, dtype=np.uint8, copy=True)
    if tuple(tensor.shape) != engine.input_shape:
        raise HTTPException(500, f"Le modèle attend la forme {engine.input_shape}")
    return {"outputs": engine.infer(tensor)}
