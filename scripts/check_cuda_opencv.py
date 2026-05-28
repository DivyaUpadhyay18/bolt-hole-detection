"""Verify CUDA OpenCV installation for RTX GPUs."""

import sys

import cv2


def main() -> int:
    print("OpenCV version:", cv2.__version__)
    print("Build info (CUDA lines):")
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "CUDA" in line or "NVIDIA" in line or "cuDNN" in line:
            print(" ", line)

    if not hasattr(cv2, "cuda"):
        print("\nFAIL: cv2.cuda is missing.")
        print("You have the standard CPU-only pip wheel.")
        print("See INSTALL_CUDA_OPENCV.md for RTX 4050 setup.")
        return 1

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except cv2.error as exc:
        print(f"\nFAIL: getCudaEnabledDeviceCount: {exc}")
        return 1

    print(f"\nCUDA devices reported by OpenCV: {count}")
    if count < 1:
        print("FAIL: No CUDA devices. Install CUDA Toolkit + CUDA OpenCV wheel.")
        return 1

    cv2.cuda.setDevice(0)
    dev = cv2.cuda.DeviceInfo(0)
    print("Device 0:", dev.name())
    print("Compute capability:", dev.majorVersion(), dev.minorVersion())

    # Quick smoke test
    import numpy as np

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    g = cv2.cuda_GpuMat()
    g.upload(img)
    out = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _ = out.download()
    print("\nOK: GpuMat upload + cuda.cvtColor succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
