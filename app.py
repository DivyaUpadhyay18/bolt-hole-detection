"""Streamlit dashboard for automated bolt hole detection."""

import os
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cuda_backend import get_backend_info, probe_cuda
from detector import BoltHoleDetector
from panel_finder import find_bscan_roi, read_dst_value
from tracker import BoltHoleTracker
from utils import process_frame

MAX_STORED_FRAMES = 40
REDETECT_EVERY = 60
UI_UPDATE_EVERY = 2  # refresh preview every N processed frames


def _save_upload_to_disk(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name or "video.mp4")[1] or ".mp4"
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.close()
    uploaded_file.seek(0)
    with open(tfile.name, "wb") as out:
        while True:
            chunk = uploaded_file.read(64 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return tfile.name


def _update_cached_roi(cached_roi, frame, frame_num):
    need_detect = (
        cached_roi is None
        or not cached_roi.get("found", False)
        or frame_num % REDETECT_EVERY == 0
    )
    if not need_detect:
        return cached_roi

    new_roi = find_bscan_roi(frame)
    if new_roi["found"]:
        return new_roi
    if cached_roi is not None and cached_roi.get("found", False):
        return cached_roi
    return new_roi


def _store_frame_snapshot(stored_frames, frame_num, annotated_frame):
    stored_frames[frame_num] = annotated_frame.copy()
    if len(stored_frames) > MAX_STORED_FRAMES:
        del stored_frames[min(stored_frames.keys())]


def _reset_processing_state():
    st.session_state.processing = False
    st.session_state.done = False
    st.session_state.stop_requested = False
    st.session_state.read_pos = 0
    st.session_state.frame_results = []
    st.session_state.stored_frames = {}
    st.session_state.cached_roi = None
    st.session_state.error = None


def _init_session_state():
    defaults = {
        "processing": False,
        "done": False,
        "stop_requested": False,
        "video_path": None,
        "video_name": "",
        "read_pos": 0,
        "total_frames": 1,
        "frame_results": [],
        "stored_frames": {},
        "cached_roi": None,
        "detector": None,
        "tracker": None,
        "error": None,
        "frame_skip": 2,
        "min_blob": 12,
        "max_pair_dist": 55,
        "show_roi": True,
        "show_blobs": True,
        "dst_debug_shown": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_video_processing(frame_ph, count_ph, progress_ph, status_ph):
    """Process entire video in one pass (no page reruns = no flicker)."""
    cap = cv2.VideoCapture(st.session_state.video_path)
    total = st.session_state.total_frames
    processed_count = 0
    last_result = None
    cached_distance = None
    last_dst_read_frame = -10

    try:
        while cap.isOpened():
            if st.session_state.stop_requested:
                break

            ret, frame = cap.read()
            if not ret:
                break

            st.session_state.read_pos += 1
            frame_num = st.session_state.read_pos

            if frame_num % st.session_state.frame_skip != 0:
                if frame_num % 50 == 0:
                    progress_ph.progress(min(frame_num / total, 1.0))
                continue

            if (
                not st.session_state.dst_debug_shown
                and frame_num == st.session_state.frame_skip
            ):
                H_f, W_f = frame.shape[:2]
                dst_region = frame[0 : int(H_f * 0.20), int(W_f * 0.60) : W_f]
                dst_region_rgb = cv2.cvtColor(dst_region, cv2.COLOR_BGR2RGB)
                st.sidebar.image(
                    dst_region_rgb,
                    caption="DST region being read by OCR",
                    use_container_width=True,
                )
                st.sidebar.write(
                    "If distance shows N/A, check that the DST value is visible in this image."
                )
                st.session_state.dst_debug_shown = True

            st.session_state.cached_roi = _update_cached_roi(
                st.session_state.cached_roi, frame, frame_num
            )
            cached_roi = st.session_state.cached_roi

            if not cached_roi.get("found", False):
                continue

            roi_dict = cached_roi.copy()
            roi_dict["roi"] = frame[
                cached_roi["y_top"] : cached_roi["y_bot"],
                cached_roi["x_left"] : cached_roi["x_right"],
            ]

            # Only re-read DST every 10 processed frames (or if missing)
            if (len(st.session_state.frame_results) % 10 == 0) or (
                cached_distance is None
            ):
                new_dst = read_dst_value(frame)
                last_dst_read_frame = frame_num
                if new_dst is not None:
                    cached_distance = new_dst

            result = process_frame(
                frame,
                st.session_state.detector,
                st.session_state.tracker,
                min_blob_area=st.session_state.min_blob,
                max_pair_distance=st.session_state.max_pair_dist,
                show_roi=st.session_state.show_roi,
                frame_num=frame_num,
                roi_dict=roi_dict,
                show_blobs=st.session_state.show_blobs,
                current_distance=cached_distance,
            )

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            st.session_state.frame_results.append(
                {
                    "frame": frame_num,
                    "time_sec": round(ts, 2),
                    "active_holes": len(result["tracked_holes"]),
                    "total_unique": result["total_unique"],
                    "distance": result.get("current_distance"),
                }
            )
            processed_count += 1

            if processed_count % 5 == 0:
                _store_frame_snapshot(
                    st.session_state.stored_frames,
                    frame_num,
                    result["annotated_frame"],
                )

            last_result = result

            if processed_count % UI_UPDATE_EVERY == 0:
                rgb = cv2.cvtColor(
                    result["annotated_frame"], cv2.COLOR_BGR2RGB
                )
                frame_ph.image(
                    rgb,
                    use_container_width=True,
                    caption=f"Frame {frame_num}",
                )
                dist_text = result.get("current_distance", None) or "Reading..."
                count_ph.markdown(
                    f"## 🟢 {len(result['tracked_holes'])}\n"
                    f"### Active holes\n\n"
                    f"**Total unique:** {result['total_unique']}\n\n"
                    f"**Distance:** `{dist_text}`"
                )
                progress_ph.progress(min(frame_num / total, 1.0))
                status_ph.caption(
                    f"Frame **{frame_num}** / **{total}** | "
                    f"Processed **{processed_count}**"
                )

    finally:
        cap.release()

    if last_result is not None:
        rgb = cv2.cvtColor(last_result["annotated_frame"], cv2.COLOR_BGR2RGB)
        frame_ph.image(
            rgb,
            use_container_width=True,
            caption=f"Frame {st.session_state.frame_results[-1]['frame']}",
        )
        dist_text = last_result.get("current_distance", None) or "Reading..."
        count_ph.markdown(
            f"## 🟢 {len(last_result['tracked_holes'])}\n"
            f"### Active holes\n\n"
            f"**Total unique:** {last_result['total_unique']}\n\n"
            f"**Distance:** `{dist_text}`"
        )

    progress_ph.progress(1.0)


def _show_results():
    frame_results = st.session_state.frame_results
    tracker = st.session_state.tracker
    stored_frames = st.session_state.stored_frames

    if st.session_state.stop_requested:
        st.warning("Processing stopped by user.")
    elif st.session_state.error:
        st.error(st.session_state.error)
    elif frame_results:
        st.success(f"Done. {len(frame_results)} frames processed.")
    else:
        st.warning(
            "No frames were processed. Check that the B-Scan panel "
            "was detected in the video."
        )

    if len(frame_results) >= 2:
        df_chart = pd.DataFrame(frame_results)
        st.line_chart(df_chart.set_index("frame")["active_holes"])

    if frame_results:
        df = pd.DataFrame(frame_results)
        st.dataframe(df)
        st.download_button(
            "Download Results CSV",
            df.to_csv(index=False),
            "results.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.header("Bolt Hole Distance Table")

        if tracker and tracker.hole_history:
            simple_table = pd.DataFrame(
                [
                    {
                        "Hole": h["label"],
                        "Distance": h.get("first_distance") or "N/A",
                    }
                    for h in tracker.hole_history
                ]
            )

            st.subheader(f"Total unique holes: {len(tracker.hole_history)}")

            st.dataframe(
                simple_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Hole": st.column_config.TextColumn("Hole", width="small"),
                    "Distance": st.column_config.TextColumn(
                        "Distance (Km:Mt:MM)", width="medium"
                    ),
                },
            )

            st.download_button(
                "Download Distance Table CSV",
                simple_table.to_csv(index=False),
                "bolt_hole_distances.csv",
            )


_init_session_state()

st.set_page_config(
    page_title="Bolt Hole Detection",
    layout="wide",
    page_icon="🔍",
)
st.title("Automated Bolt Hole Detection System")

probe_cuda()
_gpu = get_backend_info()
if _gpu["cuda_available"]:
    st.sidebar.success(f"GPU: {_gpu['device_name']}")
    st.sidebar.caption("CUDA OpenCV active for signal masks")
else:
    st.sidebar.info("GPU: not active (CPU OpenCV)")
    st.sidebar.caption(
        "Install CUDA OpenCV for RTX 4050 - see INSTALL_CUDA_OPENCV.md"
    )

busy = st.session_state.processing
uploaded_file = None

# --- Sidebar ---
st.sidebar.markdown("---")

if busy:
    st.sidebar.markdown(f"**Video:** {st.session_state.video_name}")
    st.sidebar.caption("Processing - do not refresh the page.")
    stop_processing = st.sidebar.button("Stop", type="primary")
    if stop_processing:
        st.session_state.stop_requested = True
else:
    frame_skip = st.sidebar.slider("Process every N frames", 1, 10, 2)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Settings")
    min_blob = st.sidebar.slider(
        "Min blob size (pixels)",
        min_value=5,
        max_value=50,
        value=12,
        step=1,
        help="Noise=1-4px. Real holes=12-200px.",
    )
    max_pair_dist = st.sidebar.slider(
        "Max distance between colour blobs (pixels)",
        min_value=20,
        max_value=100,
        value=55,
        step=5,
    )
    show_roi = st.sidebar.checkbox(
        "Show detection zone outline", value=True
    )
    show_blobs = st.sidebar.checkbox(
        "Show colour blob highlights", value=True
    )

    uploaded_file = st.file_uploader(
        "Upload B-Scan Video",
        type=["mp4", "avi", "mov"],
        help="Maximum file size: 5 GB",
    )

    start_processing = st.sidebar.button(
        "Start Processing",
        type="primary",
        disabled=uploaded_file is None,
    )

    if uploaded_file is not None and start_processing:
        if st.session_state.video_path and os.path.exists(
            st.session_state.video_path
        ):
            try:
                os.unlink(st.session_state.video_path)
            except OSError:
                pass

        with st.spinner("Saving video to disk..."):
            st.session_state.video_path = _save_upload_to_disk(
                uploaded_file
            )
            st.session_state.video_name = uploaded_file.name or "video"

        cap_probe = cv2.VideoCapture(st.session_state.video_path)
        st.session_state.total_frames = (
            int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        )
        cap_probe.release()

        st.session_state.processing = True
        st.session_state.done = False
        st.session_state.stop_requested = False
        st.session_state.read_pos = 0
        st.session_state.frame_results = []
        st.session_state.stored_frames = {}
        st.session_state.cached_roi = None
        st.session_state.detector = BoltHoleDetector()
        st.session_state.tracker = BoltHoleTracker(
            max_distance=35, max_missing_frames=6
        )
        st.session_state.frame_skip = frame_skip
        st.session_state.min_blob = min_blob
        st.session_state.max_pair_dist = max_pair_dist
        st.session_state.show_roi = show_roi
        st.session_state.show_blobs = show_blobs
        st.session_state.error = None
        busy = True

# --- Main area (never show upload info while processing) ---
if st.session_state.processing and not st.session_state.done:
    st.subheader(f"Processing: {st.session_state.video_name}")
    col1, col2 = st.columns([3, 1])
    frame_ph = col1.empty()
    count_ph = col2.empty()
    progress_ph = st.progress(0)
    status_ph = st.empty()

    try:
        _run_video_processing(frame_ph, count_ph, progress_ph, status_ph)
    except Exception as exc:
        st.session_state.error = str(exc)
        st.error(f"Processing stopped: {exc}")
    finally:
        st.session_state.processing = False
        st.session_state.done = True

    _show_results()

    if st.button("Process another video"):
        if st.session_state.video_path and os.path.exists(
            st.session_state.video_path
        ):
            try:
                os.unlink(st.session_state.video_path)
            except OSError:
                pass
        st.session_state.video_path = None
        st.session_state.video_name = ""
        _reset_processing_state()
        st.rerun()

elif st.session_state.done and not st.session_state.processing:
    _show_results()
    if st.button("Process another video"):
        if st.session_state.video_path and os.path.exists(
            st.session_state.video_path
        ):
            try:
                os.unlink(st.session_state.video_path)
            except OSError:
                pass
        st.session_state.video_path = None
        st.session_state.video_name = ""
        _reset_processing_state()
        st.rerun()

else:
    st.markdown(
        """
        Upload a screen recording of **SRT BScan USFD** software, then click
        **Start Processing** in the sidebar.

        - Locates the B-Scan panel automatically
        - Detects bolt holes from paired colour blobs
        - Numbers each hole (BH-1, BH-2, ...) across frames
        """
    )
    if uploaded_file is not None:
        st.caption(
            f"Selected: **{uploaded_file.name}** - click Start Processing."
        )
