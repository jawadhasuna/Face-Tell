"""Stream a face dataset, extract MediaPipe blendshapes, optionally keep images.

Downloading is the slow part - roughly 3 images per second from an
unauthenticated HuggingFace stream - so a single pass produces both
training inputs:

  the CSV    52 muscle activation scores per face, for the baseline model
  the images resized JPEGs on disk, for fine-tuning a CNN

Saving images resized rather than at source resolution turns 7.9GB of
1547px originals into a few hundred MB, which is all a 224px model needs.

Rows are flushed as they are produced. A dropped connection twenty
minutes in should cost you the remainder, not the whole run.

Usage:
    python extract.py affectnet --limit 12000 --save-images
    python extract.py rafdb --limit 500
"""

import argparse
import csv
import io
import sys
import time
import warnings
from pathlib import Path

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

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
FLUSH_EVERY = 100

SOURCES = {
    "affectnet": {
        "hf_id": "Piro17/affectnethq",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "out": "affectnet_blendshapes.csv",
    },
    "rafdb": {
        "hf_id": "deanngkl/raf-db-7emotions",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "out": "rafdb_blendshapes.csv",
    },
}

# Both datasets use their own class ordering; normalise to one vocabulary
# so a model trained on one can be tested on the other.
CANONICAL = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgusted", "disgusted": "disgusted",
    "fear": "fear", "fearful": "fear",
    "happy": "happy", "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised",
    "contempt": "contempt",
}


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
    if isinstance(value, Image.Image):
        img = value
    else:
        raw = bytes(value) if isinstance(value, (list, tuple)) else value
        img = Image.open(io.BytesIO(raw))
    return np.array(img.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="read over the network instead of downloading first. Slower "
        "(a single throttled request rather than parallel transfer) and the "
        "shuffle buffer holds decoded images in RAM, which is ruinous for "
        "large source images. Only worth it when disk space is short.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=200,
        help="streaming only: rows held in memory for shuffling. These "
        "datasets are stored sorted by class, so without it a partial run "
        "yields only the first few emotions.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="also write resized JPEGs to data/images/<source>/<label>/, "
        "so the same download can train a CNN as well as the baseline",
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="edge length for saved JPEGs"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    cfg = SOURCES[args.source]
    out_path = ROOT / "data" / cfg["out"]
    out_path.parent.mkdir(exist_ok=True)

    image_root = ROOT / "data" / "images" / args.source
    if args.save_images:
        image_root.mkdir(parents=True, exist_ok=True)
        print(f"Also saving {args.image_size}px JPEGs to {image_root}", flush=True)

    print(f"Reading {cfg['hf_id']} -> {out_path.name}", flush=True)

    if args.stream:
        print("Streaming over the network...", flush=True)
        stream = load_dataset(cfg["hf_id"], split=cfg["split"], streaming=True)
        label_feature = stream.features[cfg["label_col"]]
        if args.shuffle_buffer:
            stream = stream.shuffle(seed=0, buffer_size=args.shuffle_buffer)
            print(
                f"Filling a {args.shuffle_buffer}-image shuffle buffer first "
                "(silent for a while, this is normal).",
                flush=True,
            )
    else:
        # Downloading is several times faster than streaming: parallel
        # transfer instead of one throttled request. The cached copy is
        # memory-mapped, so shuffling is an index permutation rather than
        # a buffer full of decoded bitmaps.
        print("Downloading (cached after the first run, progress bar below)...", flush=True)
        dataset = load_dataset(cfg["hf_id"], split=cfg["split"])
        label_feature = dataset.features[cfg["label_col"]]
        dataset = dataset.shuffle(seed=0)
        if args.limit:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Downloaded. Processing {len(dataset)} images.", flush=True)
        stream = dataset

    class_names = getattr(label_feature, "names", None)
    print(f"Source classes: {class_names}", flush=True)

    detector = make_detector()
    names: list[str] = []
    seen = missed = written = 0
    start = time.perf_counter()
    buffer: list[list] = []

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        for row in stream:
            if args.limit and seen >= args.limit:
                break
            seen += 1

            try:
                rgb = to_rgb(row[cfg["image_col"]])
            except Exception:
                missed += 1
                continue

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = detector.detect(mp_img)
            if not result.face_blendshapes:
                missed += 1
                continue

            shapes = result.face_blendshapes[0]
            if not names:
                names = [c.category_name for c in shapes]
                writer.writerow(["label"] + names)

            raw_label = row[cfg["label_col"]]
            if class_names is not None and isinstance(raw_label, int):
                raw_label = class_names[raw_label]
            label = CANONICAL.get(str(raw_label).lower(), str(raw_label).lower())

            if args.save_images:
                # Only faces that MediaPipe accepted are saved, so the image
                # folder and the CSV describe exactly the same population and
                # the two models can be compared fairly.
                class_dir = image_root / label
                class_dir.mkdir(exist_ok=True)
                Image.fromarray(rgb).resize(
                    (args.image_size, args.image_size), Image.LANCZOS
                ).save(class_dir / f"{written:06d}.jpg", quality=92)

            buffer.append([label] + [round(c.score, 6) for c in shapes])
            written += 1

            if len(buffer) >= FLUSH_EVERY:
                writer.writerows(buffer)
                fh.flush()
                buffer.clear()
                rate = seen / (time.perf_counter() - start)
                print(
                    f"  seen {seen:>6}  kept {written:>6}  missed {missed:>5}"
                    f"  ({rate:.1f} img/s)",
                    flush=True,
                )

        if buffer:
            writer.writerows(buffer)

    detector.close()
    elapsed = time.perf_counter() - start
    keep_rate = written / seen if seen else 0
    print(
        f"\nDone in {elapsed/60:.1f} min. {written} rows kept from {seen} images "
        f"({keep_rate:.1%} detected, {missed} skipped).",
        flush=True,
    )
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
