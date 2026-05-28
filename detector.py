"""Blob-based bolt hole detection."""

import cv2
import numpy as np

from colour_rules import classify_blob_colour, get_signal_mask
from cuda_backend import get_backend_info, is_cuda_available


class BoltHoleDetector:
    """Detect bolt holes by pairing different-colour signal blobs."""

    def __init__(self, use_cuda: bool | None = None):
        self.use_cuda = use_cuda
        self._backend = get_backend_info()

    def detect(self, roi_bgr, min_blob_area=12, max_pair_distance=55):
        empty_result = {
            "bolt_hole_count": 0,
            "bolt_hole_positions": [],
            "annotated_roi": (
                roi_bgr.copy()
                if roi_bgr is not None and roi_bgr.size > 0
                else np.zeros((1, 1, 3), dtype=np.uint8)
            ),
            "valid_blobs": [],
        }

        if roi_bgr is None or roi_bgr.size == 0:
            return empty_result

        use_cuda = self.use_cuda
        if use_cuda is None:
            use_cuda = is_cuda_available()

        signal_mask, roi_rgb = get_signal_mask(roi_bgr, use_cuda=use_cuda)
        signal_uint8 = signal_mask.astype(np.uint8) * 255

        # connectedComponentsWithStats has no CUDA build in standard OpenCV
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            signal_uint8, connectivity=8
        )
        backend = "CUDA+CPU" if use_cuda and self._backend["cuda_available"] else "CPU"
        print(f"[DETECT] backend={backend}")

        valid_blobs = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_blob_area:
                continue
            blob_mask = labels == label_id
            cx = float(centroids[label_id, 0])
            cy = float(centroids[label_id, 1])
            colour = classify_blob_colour(roi_rgb, blob_mask)
            valid_blobs.append(
                {
                    "label_id": label_id,
                    "area": area,
                    "cx": cx,
                    "cy": cy,
                    "colour": colour,
                    "mask": blob_mask,
                }
            )

        print(
            f"[DETECT] signal_px={signal_mask.sum()} "
            f"total_blobs={num_labels - 1} "
            f"valid(>={min_blob_area}px)={len(valid_blobs)}"
        )
        for b in valid_blobs:
            print(
                f"  blob colour={b['colour']} "
                f"area={b['area']} cx={b['cx']:.0f} cy={b['cy']:.0f}"
            )

        bolt_hole_positions = []
        used_ids = set()

        blobs_sorted = sorted(
            valid_blobs, key=lambda b: b["area"], reverse=True
        )

        for i, blob_a in enumerate(blobs_sorted):
            if blob_a["label_id"] in used_ids:
                continue

            best_partner = None
            best_dist = float("inf")

            for j, blob_b in enumerate(blobs_sorted):
                if i == j:
                    continue
                if blob_b["label_id"] in used_ids:
                    continue
                if blob_a["colour"] == blob_b["colour"]:
                    continue

                dist = np.sqrt(
                    (blob_a["cx"] - blob_b["cx"]) ** 2
                    + (blob_a["cy"] - blob_b["cy"]) ** 2
                )

                if dist <= max_pair_distance and dist < best_dist:
                    best_dist = dist
                    best_partner = blob_b

            if best_partner is not None:
                cx = int((blob_a["cx"] + best_partner["cx"]) / 2)
                cy = int((blob_a["cy"] + best_partner["cy"]) / 2)
                bolt_hole_positions.append((cx, cy))
                used_ids.add(blob_a["label_id"])
                used_ids.add(best_partner["label_id"])
                print(
                    f"[DETECT] ACCEPTED: "
                    f"{blob_a['colour']}(area={blob_a['area']}) + "
                    f"{best_partner['colour']}(area={best_partner['area']}) "
                    f"dist={best_dist:.1f}px -> ({cx},{cy})"
                )
            else:
                print(
                    f"[DETECT] NO PAIR: "
                    f"{blob_a['colour']} area={blob_a['area']} "
                    f"cx={blob_a['cx']:.0f} - single colour, rejected"
                )

        annotated = roi_bgr.copy()
        for blob in valid_blobs:
            ys, xs = np.where(blob["mask"])
            if blob["colour"] == "purple":
                annotated[ys, xs] = (180, 0, 180)
            elif blob["colour"] == "red":
                annotated[ys, xs] = (0, 0, 220)
            elif blob["colour"] == "grey":
                annotated[ys, xs] = (180, 180, 0)

        for cx, cy in bolt_hole_positions:
            cv2.circle(annotated, (cx, cy), 18, (0, 220, 0), 2)

        count = len(bolt_hole_positions)
        cv2.putText(
            annotated,
            f"Holes: {count}",
            (5, 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        print(f"[DETECT] FINAL count={count}")

        return {
            "bolt_hole_count": count,
            "bolt_hole_positions": bolt_hole_positions,
            "annotated_roi": annotated,
            "valid_blobs": valid_blobs,
        }
