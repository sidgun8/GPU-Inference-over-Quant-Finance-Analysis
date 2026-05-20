from __future__ import annotations

import statistics
import threading
import time
from typing import Any


class GpuTelemetrySampler:
    def __init__(self, device_index: int = 0, interval_seconds: float = 0.05) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any | None = None
        self._handle: Any | None = None

    def __enter__(self) -> "GpuTelemetrySampler":
        try:
            import pynvml

            self._nvml = pynvml
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as exc:
            raise RuntimeError(f"NVML GPU telemetry is unavailable. No telemetry fallback is allowed. Original error: {exc}") from exc
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            self._nvml.nvmlShutdown()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append(self._read_sample())
            time.sleep(self.interval_seconds)

    def _read_sample(self) -> dict[str, float]:
        if self._nvml is None or self._handle is None:
            raise RuntimeError("NVML sampler was not initialized.")
        memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        sample = {
            "gpu_memory_used_mb": float(memory.used / (1024**2)),
            "gpu_memory_total_mb": float(memory.total / (1024**2)),
            "gpu_utilization_percent": float(utilization.gpu),
            "gpu_memory_utilization_percent": float(utilization.memory),
        }
        try:
            sample["gpu_power_watts"] = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0)
        except Exception:
            sample["gpu_power_watts"] = float("nan")
        try:
            sample["gpu_temperature_c"] = float(self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU))
        except Exception:
            sample["gpu_temperature_c"] = float("nan")
        try:
            sample["gpu_graphics_clock_mhz"] = float(self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_GRAPHICS))
        except Exception:
            sample["gpu_graphics_clock_mhz"] = float("nan")
        try:
            sample["gpu_memory_clock_mhz"] = float(self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_MEM))
        except Exception:
            sample["gpu_memory_clock_mhz"] = float("nan")
        return sample

    def summary(self) -> dict[str, float]:
        if not self._samples:
            return {}
        result: dict[str, float] = {"gpu_telemetry_sample_count": float(len(self._samples))}
        for key in self._samples[0].keys():
            values = [sample[key] for sample in self._samples if sample[key] == sample[key]]
            if not values:
                continue
            result[f"{key}_mean"] = float(statistics.fmean(values))
            result[f"{key}_max"] = float(max(values))
            result[f"{key}_min"] = float(min(values))
        return result
