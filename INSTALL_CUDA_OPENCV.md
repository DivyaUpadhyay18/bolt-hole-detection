# CUDA OpenCV for NVIDIA RTX 4050 (Windows)

The default package `opencv-python` from pip is **CPU-only**. Your RTX 4050 will not be used until you install an OpenCV build compiled with CUDA.

## 1. Install NVIDIA driver + CUDA Toolkit

1. Update the GPU driver from [NVIDIA](https://www.nvidia.com/Download/index.aspx).
2. Install **CUDA Toolkit 12.x** from [CUDA downloads](https://developer.nvidia.com/cuda-downloads) (match the wheel you install below).

## 2. Install CUDA-enabled OpenCV (Python)

### Option A – Prebuilt CUDA wheels (recommended on Windows)

Community builds ship `cv2.cuda` without compiling OpenCV yourself:

- [cudawarped/opencv-python-cuda-wheels](https://github.com/cudawarped/opencv-python-cuda-wheels/releases)

Steps:

```powershell
cd bolt_hole_detection
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
# Download the .whl that matches your Python version + CUDA 12.x from Releases
pip install path\to\opencv_contrib_python-*+cuda*.whl
python scripts\check_cuda_opencv.py
```

You should see `OK: GpuMat upload + cuda.cvtColor succeeded.`

### Option B – Build OpenCV from source

Only if prebuilt wheels are unavailable. Requires Visual Studio, CMake, CUDA Toolkit, and several hours. See [OpenCV CUDA docs](https://docs.opencv.org/4.x/d2/dbc/cuda_intro.html).

## 3. Run the app

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The sidebar shows **GPU: Active (RTX …)** when CUDA OpenCV is detected.

## What runs on the GPU?

| Step | Device |
|------|--------|
| BGR→RGB, signal mask (colour thresholds) | **GPU** (`cv2.cuda`) |
| Connected components (blob labeling) | **CPU** (no CUDA in OpenCV) |
| Blob pairing, tracking, UI | **CPU** |

So the GPU speeds up the heaviest per-pixel colour work; blob counting still uses the CPU.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cv2.cuda is missing` | Wrong OpenCV wheel; reinstall CUDA wheel |
| `CUDA devices: 0` | CUDA Toolkit not installed or driver mismatch |
| App shows **CPU only** | Run `python scripts\check_cuda_opencv.py` |
| Force CPU | `$env:USE_CUDA="0"` before `streamlit run app.py` |

## Verify from this project

```powershell
python scripts\check_cuda_opencv.py
```
