"""Signal pixel detection and blob colour classification."""

import cv2
import numpy as np

from cuda_backend import get_signal_mask_cuda, is_cuda_available


def get_signal_mask(roi_bgr, use_cuda: bool | None = None):
    """
    Find ALL non-background pixels in the ROI.
    Background = light grey (R>205, G>205, B>205 roughly equal).
    Signal pixels are anything that deviates from background.
    Also exclude green overlay and blue line pixels.

    use_cuda: None = auto-detect CUDA OpenCV; True/False to force.
    """
    if use_cuda is None:
        use_cuda = is_cuda_available()
    if use_cuda:
        try:
            return get_signal_mask_cuda(roi_bgr)
        except Exception as exc:
            print(f"[CUDA] mask fallback to CPU: {exc}")

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    R = roi_rgb[:, :, 0].astype(int)
    G = roi_rgb[:, :, 1].astype(int)
    B = roi_rgb[:, :, 2].astype(int)

    brightness = (R + G + B) / 3.0
    sat = np.max([np.abs(R - G), np.abs(G - B), np.abs(R - B)], axis=0)

    signal = (
        ((brightness < 195) | (sat > 20))
        & (brightness > 15)
    )

    green_overlay = (G > 180) & (R < 80) & (B < 80)
    blue_line = (B > 130) & (R < 100) & (G < 100)
    red_line = (R > 150) & (G < 80) & (B < 80)

    signal = signal & ~green_overlay & ~blue_line & ~red_line
    return signal, roi_rgb


def classify_blob_colour(roi_rgb, blob_mask):
    """
    Classify the dominant colour of a blob by its AVERAGE colour.
    Returns: 'purple', 'red', or 'grey'
    """
    ys, xs = np.where(blob_mask)
    if len(ys) == 0:
        return "unknown"

    r_mean = float(np.mean(roi_rgb[ys, xs, 0]))
    g_mean = float(np.mean(roi_rgb[ys, xs, 1]))
    b_mean = float(np.mean(roi_rgb[ys, xs, 2]))

    if (r_mean - g_mean > 10) and (b_mean - g_mean > 10):
        return "purple"

    if (r_mean - g_mean > 20) and (r_mean - b_mean > 20):
        return "red"

    if (
        abs(r_mean - g_mean) < 22
        and abs(g_mean - b_mean) < 22
        and r_mean < 195
    ):
        return "grey"

    if r_mean >= g_mean and r_mean >= b_mean:
        return "red"
    if b_mean >= g_mean:
        return "purple"
    return "grey"
