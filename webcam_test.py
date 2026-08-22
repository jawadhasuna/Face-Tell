"""Stage 0: prove the camera + face landmark pipeline works.

Opens the webcam, runs MediaPipe FaceLandmarker on every frame, and draws
the 478 landmarks, live FPS, and the strongest facial-muscle signals
(blendshapes) on screen. Press q or Esc to quit.
"""

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

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def build_landmarker() -> FaceLandmarker:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    return FaceLandmarker.create_from_options(options)


def draw_landmarks(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)


def draw_blendshapes(frame, blendshapes, top_n: int = 6) -> None:
    ranked = sorted(blendshapes, key=lambda c: c.score, reverse=True)[:top_n]
    for i, cat in enumerate(ranked):
        y = 60 + i * 22
        cv2.putText(
            frame,
            f"{cat.category_name:<22} {cat.score:.2f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}")

    # CAP_DSHOW avoids the multi-second camera warmup on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0). Is another app using it?")

    landmarker = build_landmarker()

    # Force the window in front - it otherwise opens behind the editor.
    window = "FaceTell - Stage 0"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window, 960, 720)
    print("Window is open (press q or Esc in the window to quit)", flush=True)

    fps, last = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Dropped frame, retrying...")
            continue

        frame = cv2.flip(frame, 1)  # mirror, so moving left looks like left
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.perf_counter() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            draw_landmarks(frame, result.face_landmarks[0])
        if result.face_blendshapes:
            draw_blendshapes(frame, result.face_blendshapes[0])

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        cv2.putText(
            frame,
            f"FPS {fps:5.1f}   faces {len(result.face_landmarks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
