"""Diagnose which OpenCV camera backend works on this machine.

Tries each backend, times how long the first frame takes, and reports
frame size and brightness (an all-black frame means Windows is blocking
camera access rather than the camera being missing).
"""

import time

import cv2
import numpy as np

BACKENDS = [
    ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("CAP_MSMF  (Media Foundation)", cv2.CAP_MSMF),
    ("CAP_ANY   (auto)", cv2.CAP_ANY),
]


def try_backend(name: str, flag: int, index: int = 0) -> bool:
    print(f"\n--- {name} on camera index {index} ---", flush=True)
    print("  opening...", end="", flush=True)
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(index, flag)
    print(f" {time.perf_counter() - t0:.1f}s  isOpened={cap.isOpened()}", flush=True)

    if not cap.isOpened():
        cap.release()
        return False

    print("  reading first frame (this is where it may hang)...", end="", flush=True)
    t0 = time.perf_counter()
    ok, frame = cap.read()
    print(f" {time.perf_counter() - t0:.1f}s  ok={ok}", flush=True)

    if ok and frame is not None:
        brightness = float(np.mean(frame))
        print(f"  size={frame.shape[1]}x{frame.shape[0]}  brightness={brightness:.1f}", flush=True)
        if brightness < 1.0:
            print("  !! all-black frame - Windows camera permission is likely blocking it", flush=True)
        else:
            print("  >> THIS BACKEND WORKS", flush=True)
        cap.release()
        return brightness >= 1.0

    cap.release()
    return False


if __name__ == "__main__":
    print(f"OpenCV {cv2.__version__}")
    winners = [n for n, f in BACKENDS if try_backend(n, f)]
    print("\n==============================")
    print("WORKING BACKENDS:", winners or "NONE")
