from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class HailoEngine:
    """HailoRT VStreams engine; inference never falls back to the CPU."""

    def __init__(self, hef_path: str) -> None:
        path = Path(hef_path)
        if not path.is_file():
            raise FileNotFoundError(f"HEF introuvable: {path}")

        try:
            from hailo_platform import (
                ConfigureParams,
                FormatType,
                HailoStreamInterface,
                HEF,
                InferVStreams,
                InputVStreamParams,
                OutputVStreamParams,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform est requis. Installez le wheel HailoRT officiel."
            ) from exc

        self._lock = threading.Lock()
        params = VDevice.create_params()
        params.group_id = "SHARED"
        self._vdevice = VDevice(params)
        self._hef = HEF(str(path))
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(
            self._hef, configure_params
        )[0]
        self._network_group_params = self._network_group.create_params()
        self._input_info = self._hef.get_input_vstream_infos()[0]
        self._output_infos = self._hef.get_output_vstream_infos()
        self._input_name = self._input_info.name
        input_params = InputVStreamParams.make_from_network_group(
            self._network_group, quantized=False, format_type=FormatType.UINT8
        )
        output_params = OutputVStreamParams.make_from_network_group(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self._pipeline = InferVStreams(
            self._network_group, input_params, output_params
        )
        self._pipeline.__enter__()

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self._input_info.shape)

    @property
    def output_names(self) -> list[str]:
        return [info.name for info in self._output_infos]

    def infer(self, tensor: np.ndarray) -> dict[str, Any]:
        if tuple(tensor.shape) != self.input_shape:
            raise ValueError(
                f"Shape attendue {self.input_shape}, reçue {tuple(tensor.shape)}"
            )
        if not tensor.flags.writeable:
            tensor = np.array(tensor, copy=True)

        with self._lock:
            # HailoRT 4.23 exige l'activation explicite du network group.
            with self._network_group.activate(self._network_group_params):
                # InferVStreams attend toujours [batch, height, width, channels].
                result = self._pipeline.infer(
                    {self._input_name: np.expand_dims(tensor, axis=0)}
                )
            return {name: _jsonable(value) for name, value in result.items()}

    def close(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        if pipeline:
            pipeline.__exit__(None, None, None)
        release = getattr(getattr(self, "_vdevice", None), "release", None)
        if release:
            release()
