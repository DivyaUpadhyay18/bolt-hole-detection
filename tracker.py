"""Persistent bolt hole numbering across frames."""

import cv2
import numpy as np


class BoltHoleTracker:
    """Track bolt holes across frames with stable BH-N labels."""

    def __init__(self, max_distance=35, max_missing_frames=6):
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.next_id = 1
        self.active_holes = {}
        self.hole_history = []

    def update(self, detected_positions, current_frame):
        """
        Match detections to active holes; assign new IDs as needed.
        Returns list of {id, label, cx, cy} for currently active holes.
        """
        detections = list(detected_positions)
        matched_det = set()
        matched_act = set()

        pairs = []
        for det_idx, (dx, dy) in enumerate(detections):
            for hole_id, hole in self.active_holes.items():
                dist = np.sqrt(
                    (dx - hole["cx"]) ** 2 + (dy - hole["cy"]) ** 2
                )
                if dist <= self.max_distance:
                    pairs.append((dist, det_idx, hole_id))

        pairs.sort(key=lambda x: x[0])

        for _dist, det_idx, hole_id in pairs:
            if det_idx in matched_det or hole_id in matched_act:
                continue
            dx, dy = detections[det_idx]
            self.active_holes[hole_id]["cx"] = dx
            self.active_holes[hole_id]["cy"] = dy
            self.active_holes[hole_id]["missing"] = 0
            self.active_holes[hole_id]["last_frame"] = current_frame
            matched_det.add(det_idx)
            matched_act.add(hole_id)

        for det_idx, (dx, dy) in enumerate(detections):
            if det_idx in matched_det:
                continue
            hole_id = self.next_id
            self.next_id += 1
            label = f"BH-{hole_id}"
            self.active_holes[hole_id] = {
                "cx": dx,
                "cy": dy,
                "missing": 0,
                "first_frame": current_frame,
                "last_frame": current_frame,
            }
            self.hole_history.append(
                {
                    "id": hole_id,
                    "label": label,
                    "first_frame": current_frame,
                    "last_frame": current_frame,
                    "cx": dx,
                    "cy": dy,
                }
            )

        to_retire = []
        for hole_id, hole in self.active_holes.items():
            if hole_id in matched_act:
                for hist in self.hole_history:
                    if hist["id"] == hole_id:
                        hist["last_frame"] = hole["last_frame"]
                        hist["cx"] = hole["cx"]
                        hist["cy"] = hole["cy"]
                        break
                continue
            hole["missing"] += 1
            if hole["missing"] > self.max_missing_frames:
                to_retire.append(hole_id)

        for hole_id in to_retire:
            del self.active_holes[hole_id]

        return [
            {
                "id": hole_id,
                "label": f"BH-{hole_id}",
                "cx": hole["cx"],
                "cy": hole["cy"],
            }
            for hole_id, hole in self.active_holes.items()
        ]

    def draw_labels(self, annotated_roi, active_holes):
        """Draw numbered BH-N labels on the annotated ROI."""
        for hole in active_holes:
            cx, cy = int(hole["cx"]), int(hole["cy"])
            label = hole["label"]
            cv2.circle(annotated_roi, (cx, cy), 20, (0, 180, 0), -1)
            cv2.circle(annotated_roi, (cx, cy), 20, (255, 255, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.42
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(
                annotated_roi,
                label,
                (tx, ty),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return annotated_roi
