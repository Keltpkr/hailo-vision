from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np


class HailoEngine:
    """Single loaded Hailo model. There is deliberately no CPU inference path."""

    def __init__(self, hef_path: str) -> None:
        path = Path(hef_path)
        if not path.is_file():
            raise FileNotFoundError(f"HEF introuvable: {path}")

        try:
            from hailo_platform import HEF, VDevice
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform est requis. Installez le wheel HailoRT officiel."
            ) from exc

        self._lock = threading.Lock()
        params = VDevice.create_params()
        params.group_id = "SHARED"
        self._vdevice = VDevice(params)
        self._hef = HEF(str(path))
        self._model = self._vdevice.create_infer_model(str(path))
        self._config_context = self._model.configure()
        self._configured_model = self._config_context.__enter__()
        self._input = self._model.inputs[0]
        self._outputs = list(self._model.outputs)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self._input.shape)

    @property
    def output_names(self) -> list[str]:
        return [output.name for output in self._outputs]

    def infer(self, tensor: np.ndarray) -> dict[str, Any]:
        expected = self.input_shape
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Shape attendue {expected}, reçue {tuple(tensor.shape)}")

        # HailoRT n'autorise pas nécessairement plusieurs appels concurrents sur
        # le même InferModel ; la file HTTP reste donc sûre et déterministe.
        with self._lock:
            output_buffers: dict[str, np.ndarray] = {}
            for output in self._outputs:
                info = next(
                    item for item in self._hef.get_output_vstream_infos()
                    if item.name == output.name
                )
                # Hailo NMS streams must use a byte buffer; regular streams use
                # the type declared by the HEF.
                format_order = str(getattr(info.format, "order", ""))
                format_type = str(getattr(info.format, "type", "")).upper()
                dtype = np.uint8 if "BYTE_MASK" in format_order else {
                    "UINT8": np.uint8,
                    "UINT16": np.uint16,
                    "FLOAT32": np.float32,
                }.get(format_type.split(".")[-1], np.float32)
                buffer = np.empty(output.shape, dtype=dtype)
                output_buffers[output.name] = buffer
            bindings = self._configured_model.create_bindings(
                output_buffers=output_buffers
            )
            bindings.input().set_buffer(np.asarray(tensor))
            self._configured_model.run([bindings], timeout=10_000)
            return {name: value.tolist() for name, value in output_buffers.items()}

    def close(self) -> None:
        context = getattr(self, "_config_context", None)
        if context:
            context.__exit__(None, None, None)
        release = getattr(getattr(self, "_vdevice", None), "release", None)
        if release:
            release()
