"""Record blendshape samples for one expression.

Usage:  python collect.py happy
        python collect.py neutral --frames 400

Shows the webcam with a countdown, then records the 52 blendshape scores
once per frame into data/samples.csv. Frames where no face is found are
skipped, so the count on screen is real samples, not elapsed time.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
CSV_PATH = ROOT / "data" / "samples.csv"
COUNTDOWN_SECONDS = 3


def build_landmarker() -> FaceLandmarker:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(options)


def banner(frame, text, colour=(0, 255, 255), scale=1.0, y=40) -> None:
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="expression name, e.g. happy")
    parser.add_argument("--frames", type=int, default=300, help="samples to record")
    args = parser.parse_args()

    CSV_PATH.parent.mkdir(exist_ok=True)
    is_new_file = not CSV_PATH.exists()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    landmarker = build_landmarker()
    window = f"FaceTell - recording '{args.label}'"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window, 960, 720)

    rows: list[list] = []
    names: list[str] = []
    started_at = None

    print(f"Get ready to look '{args.label}'. Recording {args.frames} samples.", flush=True)

    while len(rows) < args.frames:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(time.perf_counter() * 1000))

        has_face = bool(result.face_blendshapes)

        if started_at is None:
            started_at = time.perf_counter()
        remaining = COUNTDOWN_SECONDS - (time.perf_counter() - started_at)

        if remaining > 0:
            banner(frame, f"'{args.label}' in {remaining:.0f}", (0, 200, 255), 1.4, 60)
            banner(frame, "hold the expression", (200, 200, 200), 0.7, 100)
        elif has_face:
            shapes = result.face_blendshapes[0]
            if not names:
                names = [c.category_name for c in shapes]
            rows.append([args.label] + [round(c.score, 6) for c in shapes])

        pct = len(rows) / args.frames
        banner(frame, f"{args.label}   {len(rows)}/{args.frames}", (0, 255, 120), 1.0, 35)
        if not has_face:
            banner(frame, "NO FACE - paused", (0, 0, 255), 0.8, 130)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (15, h - 40), (w - 15, h - 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, h - 40), (15 + int((w - 30) * pct), h - 20), (0, 255, 120), -1)

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            print("Aborted - nothing saved.", flush=True)
            rows = []
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if not rows:
        return

    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new_file:
            writer.writerow(["label"] + names)
        writer.writerows(rows)

    print(f"Saved {len(rows)} '{args.label}' samples to {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
