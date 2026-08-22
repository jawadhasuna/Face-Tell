"""Live expression recognition from the webcam.

Loads the trained classifier and names your expression in real time.
Predictions are averaged over a short window so the label does not
flicker between frames. Press q or Esc to quit.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
CLF_PATH = ROOT / "models" / "expression_clf.joblib"

SMOOTHING_FRAMES = 8      # ~0.3s at 26 FPS
CONFIDENCE_FLOOR = 0.45   # below this we say "unsure" rather than guess

COLOURS = {
    "happy": (80, 220, 100),
    "sad": (220, 140, 60),
    "angry": (60, 60, 235),
    "surprised": (40, 200, 240),
    "disgusted": (150, 90, 200),
    "neutral": (190, 190, 190),
}


def build_landmarker() -> FaceLandmarker:
    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
    )


def text(frame, s, org, scale, colour, weight=2) -> None:
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight, cv2.LINE_AA)


def draw_bars(frame, labels, probs) -> None:
    h = frame.shape[0]
    top = h - 20 - len(labels) * 26
    for i, (label, p) in enumerate(zip(labels, probs)):
        y = top + i * 26
        colour = COLOURS.get(label, (200, 200, 200))
        cv2.rectangle(frame, (15, y), (15 + int(260 * p), y + 18), colour, -1)
        cv2.rectangle(frame, (15, y), (275, y + 18), (90, 90, 90), 1)
        text(frame, f"{label} {p:.0%}", (285, y + 15), 0.5, colour, 1)


def main() -> None:
    bundle = joblib.load(CLF_PATH)
    clf, labels = bundle["model"], bundle["labels"]
    print(f"Loaded classifier for: {', '.join(labels)}", flush=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    landmarker = build_landmarker()
    window = "FaceTell - live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window, 960, 720)

    history: deque = deque(maxlen=SMOOTHING_FRAMES)
    fps, last = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(time.perf_counter() * 1000))

        if result.face_blendshapes:
            scores = np.array([[c.score for c in result.face_blendshapes[0]]])
            history.append(clf.predict_proba(scores)[0])

            probs = np.mean(history, axis=0)
            best = int(np.argmax(probs))
            name, confidence = labels[best], probs[best]

            if confidence >= CONFIDENCE_FLOOR:
                text(frame, name.upper(), (15, 75), 1.8, COLOURS.get(name, (255, 255, 255)), 3)
                text(frame, f"{confidence:.0%} confident", (15, 110), 0.7, (220, 220, 220), 2)
            else:
                text(frame, "UNSURE", (15, 75), 1.8, (140, 140, 140), 3)

            draw_bars(frame, labels, probs)
        else:
            history.clear()
            text(frame, "no face", (15, 75), 1.2, (0, 0, 255), 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        text(frame, f"FPS {fps:4.0f}", (frame.shape[1] - 150, 35), 0.7, (0, 255, 255), 2)

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
