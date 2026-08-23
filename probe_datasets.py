"""Measure MediaPipe's face detection rate on candidate datasets.

Blendshape extraction is worthless if the landmarker cannot find a face
in the source image, so this decides which dataset we can actually use
before committing to a multi-gigabyte download.

Small images are also tested upscaled: MediaPipe has a minimum useful
face size, and it costs nothing to check whether naive upsampling gets
a 48x48 thumbnail over that line.
"""

import io
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

warnings.filterwarnings("ignore")

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
SAMPLE_SIZE = 120

CANDIDATES = [
    # (hf id, image column, label column, upscale factors to try)
    ("Jeneral/fer-2013", "img_bytes", "labels", (1, 2, 4)),
    ("deanngkl/raf-db-7emotions", "image", "label", (1, 2)),
    ("Piro17/affectnethq", "image", "label", (1,)),
]


def make_detector() -> FaceLandmarker:
    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
        )
    )


def to_rgb(value) -> np.ndarray:
    """Rows arrive either as a PIL image or as raw encoded bytes."""
    if isinstance(value, Image.Image):
        img = value
    else:
        raw = bytes(value) if isinstance(value, (list, tuple)) else value
        img = Image.open(io.BytesIO(raw))
    return np.array(img.convert("RGB"))


def probe(dataset_id: str, image_col: str, label_col: str, scales) -> None:
    from datasets import load_dataset

    print(f"\n=== {dataset_id} ===", flush=True)
    try:
        stream = load_dataset(dataset_id, split="train", streaming=True)
    except Exception as exc:
        print(f"  could not stream: {exc}")
        return

    detector = make_detector()
    images, labels = [], []
    for row in stream:
        try:
            images.append(to_rgb(row[image_col]))
            labels.append(row[label_col])
        except Exception:
            continue
        if len(images) >= SAMPLE_SIZE:
            break

    if not images:
        print("  no images pulled")
        return

    h, w = images[0].shape[:2]
    print(f"  pulled {len(images)} images at {w}x{h}", flush=True)

    for scale in scales:
        found = 0
        for img in images:
            if scale != 1:
                img = cv2.resize(
                    img, (img.shape[1] * scale, img.shape[0] * scale),
                    interpolation=cv2.INTER_CUBIC,
                )
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
            if detector.detect(mp_img).face_blendshapes:
                found += 1
        rate = found / len(images)
        size = f"{w * scale}x{h * scale}"
        verdict = "USABLE" if rate >= 0.85 else ("marginal" if rate >= 0.5 else "TOO LOW")
        print(f"  {size:>12}  detected {found:>3}/{len(images)}  = {rate:5.1%}  {verdict}", flush=True)

    detector.close()


if __name__ == "__main__":
    for dataset_id, image_col, label_col, scales in CANDIDATES:
        probe(dataset_id, image_col, label_col, scales)
    print("\nNeed >=85% detection for the blendshape pipeline to be viable.")
