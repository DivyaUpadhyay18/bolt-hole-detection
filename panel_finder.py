"""Locate B-Scan panel ROI in each video frame."""

import re

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except ImportError:
    pytesseract = None
    TesseractOutput = None


def group_rows(rows, gap=10):
    """Group consecutive row indices within gap pixels; return median y per group."""
    if not rows:
        return []
    sorted_rows = sorted(rows)
    groups = [[sorted_rows[0]]]
    for y in sorted_rows[1:]:
        if y - groups[-1][-1] <= gap:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [int(np.median(g)) for g in groups]


def read_dst_value(frame):
    """
    Read the DST distance value from top-right of frame.
    The value looks like: 0074:0009:0590
    It sits directly below the label "DST:Km:Mt:MM INC"
    in the top-right corner of the SRT BScan software window.
    """
    import re

    H, W = frame.shape[:2]

    search_regions = [
        (0.00, 0.15, 0.60, 1.00),
        (0.00, 0.20, 0.55, 1.00),
        (0.00, 0.25, 0.50, 1.00),
        (0.00, 0.30, 0.40, 1.00),
    ]

    pattern = re.compile(r"\b(\d{3,4}[:\-]\d{3,4}[:\-]\d{3,4})\b")

    for (ys, ye, xs, xe) in search_regions:
        y0 = int(H * ys)
        y1 = int(H * ye)
        x0 = int(W * xs)
        x1 = int(W * xe)
        roi = frame[y0:y1, x0:x1]

        if roi.size == 0:
            continue

        preprocessed = []

        scale = 3
        h_r, w_r = roi.shape[:2]
        big = cv2.resize(
            roi, (w_r * scale, h_r * scale), interpolation=cv2.INTER_CUBIC
        )
        grey_a = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        preprocessed.append(grey_a)

        _, thresh_b = cv2.threshold(grey_a, 127, 255, cv2.THRESH_BINARY)
        preprocessed.append(thresh_b)

        _, thresh_c = cv2.threshold(grey_a, 127, 255, cv2.THRESH_BINARY_INV)
        preprocessed.append(thresh_c)

        _, thresh_d = cv2.threshold(
            grey_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        preprocessed.append(thresh_d)

        for img in preprocessed:
            try:
                import pytesseract

                for psm in [6, 7, 8, 11, 13]:
                    config = (
                        f"--psm {psm} --oem 3 "
                        "-c tessedit_char_whitelist="
                        "0123456789:.-"
                    )
                    text = pytesseract.image_to_string(img, config=config)
                    matches = pattern.findall(text)
                    if matches:
                        dst = matches[0]
                        print(
                            f"[DST] FOUND: {dst} "
                            f"region=({ys:.0%},{xs:.0%}) "
                            f"psm={psm}"
                        )
                        return dst
            except Exception as e:
                print(f"[DST] OCR error: {e}")
                continue

    try:
        import pytesseract

        grey_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(grey_full, config="--psm 11 --oem 3")
        matches = pattern.findall(text)
        if matches:
            for m in matches:
                parts = re.split(r"[:\-]", m)
                if len(parts) == 3 and int(parts[0]) > 10:
                    print(f"[DST] full frame fallback: {m}")
                    return m
    except Exception as e:
        print(f"[DST] full frame fallback error: {e}")

    print("[DST] could not read distance from this frame")
    return None


def _search_region(frame):
    """Bottom 70%, left 60% of frame."""
    h, w = frame.shape[:2]
    y0 = int(h * 0.30)
    x1 = int(w * 0.60)
    return frame[y0:, :x1], y0, 0


def _is_blue_pixel(bgr_row):
    """Blue line pixel: R<100, G<100, B>130."""
    b, g, r = bgr_row[:, 0], bgr_row[:, 1], bgr_row[:, 2]
    return (r < 100) & (g < 100) & (b > 130)


def _find_blue_line_rows(region, min_span_frac=0.08):
    """Rows where blue pixels span at least min_span_frac of region width."""
    h, w = region.shape[:2]
    min_span = max(1, int(w * min_span_frac))
    blue_rows = []
    for y in range(h):
        if _is_blue_pixel(region[y]).sum() >= min_span:
            blue_rows.append(y)
    return group_rows(blue_rows)


def _find_y_top_ocr(region, y_offset):
    """Method A: OCR for Start Km / B Scan header row."""
    if pytesseract is None or TesseractOutput is None:
        return None

    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    try:
        data = pytesseract.image_to_data(
            rgb, output_type=TesseractOutput.DICT, config="--psm 6"
        )
    except Exception:
        return None

    keywords = ("start", "km", "line", "bscan", "b scan", "scan")
    matching_bottoms = []

    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip().lower()
        if not text:
            continue
        normalized = re.sub(r"[^a-z0-9]", "", text)
        if not any(k.replace(" ", "") in normalized or k in text for k in keywords):
            if not any(
                kw in text
                for kw in ("start", "km", "line", "b scan", "bscan", "scan")
            ):
                continue
        bottom = data["top"][i] + data["height"][i]
        matching_bottoms.append(bottom)

    if not matching_bottoms:
        lines = {}
        for i in range(n):
            text = (data["text"][i] or "").strip().lower()
            if not text:
                continue
            line_num = data["line_num"][i]
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(text)

        for line_num, words in lines.items():
            combined = " ".join(words)
            if any(
                kw in combined
                for kw in ("start", "km", "line", "b scan", "bscan", "scan")
            ):
                for i in range(n):
                    if data["line_num"][i] == line_num:
                        bottom = data["top"][i] + data["height"][i]
                        matching_bottoms.append(bottom)

    if matching_bottoms:
        y_top = int(max(matching_bottoms)) + 2 + y_offset
        print(f"[ANCHOR] OCR found text at y_top={y_top}")
        return y_top
    return None


def _find_y_top_method_b(region, y_offset):
    """Method B: pixel scan above topmost blue line."""
    h, w = region.shape[:2]
    blue_groups = _find_blue_line_rows(region)
    if not blue_groups:
        return None

    topmost_blue = min(blue_groups)

    for scan_up in range(2, 26):
        y = topmost_blue - scan_up
        if y < 0:
            break
        row = region[y]
        b, g, r = row[:, 0], row[:, 1], row[:, 2]
        light_grey = (
            (r > 160)
            & (g > 160)
            & (b > 160)
            & (np.abs(r.astype(int) - g.astype(int)) < 25)
            & (np.abs(g.astype(int) - b.astype(int)) < 25)
        )
        grey_frac = light_grey.sum() / w
        dark = ((r < 100) & (g < 100) & (b < 100)).sum()

        if grey_frac > 0.40 and 5 <= dark <= 150:
            y_top = y + 2 + y_offset
            print(f"[ANCHOR] Method B found header at y_top={y_top}")
            return y_top

    y_top = topmost_blue + y_offset
    print(f"[ANCHOR] Method B found header at y_top={y_top} (blue line fallback)")
    return y_top


def _find_y_bot(frame, y_top, x_left, x_right):
    """Red TR baseline below y_top."""
    h, w = frame.shape[:2]
    search = frame[y_top:, x_left:x_right]
    sh, sw = search.shape[:2]

    for min_frac in (0.08, 0.04):
        min_span = max(1, int(sw * min_frac))
        for y in range(sh):
            row = search[y]
            r, g, b = row[:, 2], row[:, 1], row[:, 0]
            red = (r > 150) & (g < 80) & (b < 80)
            if red.sum() >= min_span:
                y_bot = y_top + y - 2
                print(f"[ROI] red line at y_bot={y_bot}")
                return y_bot
    return None


def _find_x_bounds(frame, y_top, y_bot):
    """Median horizontal extent of blue lines between y_top and y_bot."""
    h, w = frame.shape[:2]
    roi_strip = frame[y_top:y_bot, :]
    rh = roi_strip.shape[0]

    lefts = []
    rights = []

    for y in range(rh):
        row = roi_strip[y]
        blue = _is_blue_pixel(row)
        if blue.sum() < max(1, int(w * 0.02)):
            continue
        xs = np.where(blue)[0]
        if len(xs) == 0:
            continue
        lefts.append(int(xs[0]))
        rights.append(int(xs[-1]))

    blue_line_ys = []
    for y in range(rh):
        if _is_blue_pixel(roi_strip[y]).sum() >= max(1, int(w * 0.08)):
            blue_line_ys.append(y)

    for y_med in group_rows(blue_line_ys):
        for dy in range(-3, 4):
            yy = y_med + dy
            if yy < 0 or yy >= rh:
                continue
            row = roi_strip[yy]
            blue = _is_blue_pixel(row)
            xs = np.where(blue)[0]
            if len(xs) > 0:
                lefts.append(int(xs[0]))
                rights.append(int(xs[-1]))

    if lefts and rights:
        x_left = int(np.median(lefts)) - 2
        x_right = int(np.median(rights)) + 2
        print(f"[ROI] x_left={x_left} x_right={x_right}")
        return x_left, x_right

    x_left = 28
    x_right = int(w * 0.43)
    print(f"[ROI] x_left={x_left} x_right={x_right} (fallback)")
    return x_left, x_right


def find_bscan_roi(frame):
    """
    Locate B-Scan detection zone. Never returns None.
    Returns dict with roi, bounds, and found flag.
    """
    h, w = frame.shape[:2]
    region, y_off, x_off = _search_region(frame)

    y_top = _find_y_top_ocr(region, y_off)
    if y_top is None:
        y_top = _find_y_top_method_b(region, y_off)

    if y_top is None:
        return {
            "roi": np.zeros((1, 1, 3), dtype=np.uint8),
            "y_top": 0,
            "y_bot": 1,
            "x_left": 0,
            "x_right": 1,
            "found": False,
        }

    # Initial horizontal search uses frame below header for red-line detection
    x_left_init = 0
    x_right_init = int(w * 0.60)
    y_bot = _find_y_bot(frame, y_top, x_left_init, x_right_init)
    if y_bot is None or y_bot <= y_top:
        return {
            "roi": np.zeros((1, 1, 3), dtype=np.uint8),
            "y_top": 0,
            "y_bot": 1,
            "x_left": 0,
            "x_right": 1,
            "found": False,
        }

    x_left, x_right = _find_x_bounds(frame, y_top, y_bot)

    x_left = max(0, x_left)
    x_right = min(w, x_right)
    y_top = max(0, y_top)
    y_bot = min(h, y_bot)

    roi = frame[y_top:y_bot, x_left:x_right]
    print(
        f"[ROI] FINAL y_top={y_top} y_bot={y_bot} "
        f"x_left={x_left} x_right={x_right} roi_shape={roi.shape}"
    )

    return {
        "roi": roi,
        "y_top": y_top,
        "y_bot": y_bot,
        "x_left": x_left,
        "x_right": x_right,
        "found": True,
    }


def draw_roi_overlay(frame, roi_dict):
    """Draw bright green rectangle around detection zone."""
    annotated = frame.copy()
    if not roi_dict.get("found", False):
        return annotated

    y_top = roi_dict["y_top"]
    y_bot = roi_dict["y_bot"]
    x_left = roi_dict["x_left"]
    x_right = roi_dict["x_right"]

    cv2.rectangle(
        annotated,
        (x_left, y_top),
        (x_right, y_bot),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated,
        "B-SCAN ROI",
        (x_left + 4, max(y_top - 6, 16)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated
