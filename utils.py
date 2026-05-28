"""Frame processing pipeline and debug helpers."""

import cv2
import numpy as np

from panel_finder import draw_roi_overlay, find_bscan_roi


def _draw_outlined_text(img, text, org, font_scale=0.6, thickness=2):
    """White text with black outline for readability."""
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
        img,
        text,
        org,
        font,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def process_frame(
    frame,
    detector,
    tracker,
    min_blob_area=12,
    max_pair_distance=55,
    show_roi=True,
    frame_num=0,
    roi_dict=None,
    show_blobs=True,
):
    """
    Run ROI extraction, detection, tracking, and annotation for one frame.
    """
    if roi_dict is None:
        roi_dict = find_bscan_roi(frame)

    if not roi_dict.get("found", False):
        return {
            "bolt_hole_count": 0,
            "annotated_frame": frame.copy(),
            "tracked_holes": [],
            "total_unique": tracker.next_id - 1,
            "found": False,
        }

    detect_result = detector.detect(
        roi_dict["roi"],
        min_blob_area=min_blob_area,
        max_pair_distance=max_pair_distance,
    )

    annotated_roi = detect_result["annotated_roi"]
    if not show_blobs:
        annotated_roi = roi_dict["roi"].copy()
        for cx, cy in detect_result["bolt_hole_positions"]:
            cv2.circle(annotated_roi, (cx, cy), 18, (0, 220, 0), 2)

    active = tracker.update(
        detect_result["bolt_hole_positions"], frame_num
    )
    annotated_roi = tracker.draw_labels(annotated_roi, active)

    annotated_frame = frame.copy()
    y_top = roi_dict["y_top"]
    y_bot = roi_dict["y_bot"]
    x_left = roi_dict["x_left"]
    x_right = roi_dict["x_right"]

    rh, rw = annotated_roi.shape[:2]
    paste_h = min(rh, y_bot - y_top)
    paste_w = min(rw, x_right - x_left)
    annotated_frame[y_top : y_top + paste_h, x_left : x_left + paste_w] = (
        annotated_roi[:paste_h, :paste_w]
    )

    banner = (
        f"ACTIVE: {len(active)}  |  "
        f"TOTAL UNIQUE: {tracker.next_id - 1}"
    )
    _draw_outlined_text(annotated_frame, banner, (10, 28))

    if show_roi:
        annotated_frame = draw_roi_overlay(annotated_frame, roi_dict)

    return {
        "bolt_hole_count": detect_result["bolt_hole_count"],
        "annotated_frame": annotated_frame,
        "tracked_holes": active,
        "total_unique": tracker.next_id - 1,
        "found": True,
    }
