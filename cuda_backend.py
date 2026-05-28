"""
CUDA OpenCV backend for bolt-hole detection.

Requires OpenCV built with CUDA (standard pip opencv-python is CPU-only).
Uses cv2.cuda for color conversion and signal-mask generation; connected
components still run on CPU (no CUDA implementation in OpenCV mainline).
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

# Set USE_CUDA=0 to force CPU even when CUDA OpenCV is installed
_FORCE_CPU = os.environ.get("USE_CUDA", "1").strip().lower() in (
    "0",
    "false",
    "no",
    "off",
)

_probe_done = False
_cuda_available = False
_device_name = "N/A"
_opencv_build = ""


def _th(gpu_src, thresh: int, inv: bool = False):
    """GPU threshold -> 255 where condition holds."""
    dst = cv2.cuda_GpuMat(gpu_src.size(), cv2.CV_8UC1)
    ttype = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    cv2.cuda.threshold(gpu_src, thresh, 255, ttype, dst)
    return dst


def _and(a, b):
    dst = cv2.cuda_GpuMat(a.size(), cv2.CV_8UC1)
    cv2.cuda.bitwise_and(a, b, dst)
    return dst


def _or(a, b):
    dst = cv2.cuda_GpuMat(a.size(), cv2.CV_8UC1)
    cv2.cuda.bitwise_or(a, b, dst)
    return dst


def _not_mask(m):
    dst = cv2.cuda_GpuMat(m.size(), cv2.CV_8UC1)
    cv2.cuda.bitwise_not(m, dst)
    return dst


def probe_cuda(force: bool = False) -> bool:
    """Detect CUDA-enabled OpenCV once per process."""
    global _probe_done, _cuda_available, _device_name, _opencv_build

    if _probe_done and not force:
        return _cuda_available

    _probe_done = True
    _cuda_available = False
    _device_name = "N/A"
    _opencv_build = getattr(cv2, "__version__", "unknown")

    if _FORCE_CPU:
        print("[CUDA] Disabled via USE_CUDA=0")
        return False

    if not hasattr(cv2, "cuda"):
        print("[CUDA] cv2.cuda module missing (CPU-only OpenCV build)")
        return False

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except cv2.error as exc:
        print(f"[CUDA] getCudaEnabledDeviceCount failed: {exc}")
        return False

    if count < 1:
        print("[CUDA] No CUDA devices reported by OpenCV")
        return False

    try:
        cv2.cuda.setDevice(0)
        dev_info = cv2.cuda.DeviceInfo(0)
        _device_name = dev_info.name()
        _cuda_available = True
        print(f"[CUDA] Active: device 0 = {_device_name} (count={count})")
    except cv2.error as exc:
        print(f"[CUDA] Device init failed: {exc}")
        _cuda_available = False

    return _cuda_available


def is_cuda_available() -> bool:
    return probe_cuda()


def get_backend_info() -> dict[str, Any]:
    probe_cuda()
    return {
        "cuda_available": _cuda_available,
        "device_name": _device_name,
        "opencv_version": _opencv_build,
        "forced_cpu": _FORCE_CPU,
        "note": (
            "GPU accelerates signal-mask generation. "
            "Blob labeling (connected components) runs on CPU."
        ),
    }


def get_signal_mask_cuda(roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    GPU equivalent of colour_rules.get_signal_mask (same logic).
    Returns (signal_bool_mask, roi_rgb).
    """
    if not is_cuda_available():
        raise RuntimeError("CUDA OpenCV not available")

    gpu_bgr = cv2.cuda_GpuMat()
    gpu_bgr.upload(roi_bgr)

    gpu_rgb = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.cuda.split(gpu_rgb)

    tmp = cv2.cuda.add(r, g)
    sum_rgb = cv2.cuda.add(tmp, b)

    brightness = cv2.cuda_GpuMat(r.size(), cv2.CV_8UC1)
    cv2.cuda.convertScaleAbs(sum_rgb, brightness, alpha=1.0 / 3.0, beta=0)

    rg = cv2.cuda.absdiff(r, g)
    gb = cv2.cuda.absdiff(g, b)
    rb = cv2.cuda.absdiff(r, b)
    sat = cv2.cuda.max(cv2.cuda.max(rg, gb), rb)

    dark_or_sat = _or(_th(brightness, 195, inv=True), _th(sat, 20))
    signal = _and(dark_or_sat, _th(brightness, 15))

    green = _and(_and(_th(g, 180), _th(r, 80, inv=True)), _th(b, 80, inv=True))
    blue = _and(_and(_th(b, 130), _th(r, 100, inv=True)), _th(g, 100, inv=True))
    red = _and(_and(_th(r, 150), _th(g, 80, inv=True)), _th(b, 80, inv=True))

    exclude = _or(_or(green, blue), red)
    signal = _and(signal, _not_mask(exclude))

    roi_rgb = gpu_rgb.download()
    signal_u8 = signal.download()
    signal_bool = signal_u8 > 0
    return signal_bool, roi_rgb
