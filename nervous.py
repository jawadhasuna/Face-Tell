"""Nervousness estimation from temporal facial behaviour.

Nervousness is not an expression - there is no single face you can pull
that means "nervous". It is a pattern over time. So this is not a model:
it is arithmetic on four signals, each measured against your own baseline.

  blink rate          nervous people blink faster than their resting rate
  gaze instability    eyes dart around instead of holding still
  expression churn    the face flickers between states
  tension expressions sustained anger / sadness / surprise

Run it, sit normally through the calibration window, then act nervous.
Press q or Esc to quit, c to recalibrate.
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

CALIBRATION_SECONDS = 30.0  # 20s caught only ~3 blinks; too noisy a baseline
WINDOW_SECONDS = 15.0      # rolling window for the live metrics
BLINK_CLOSE = 0.45         # eye counts as shut above this
BLINK_OPEN = 0.25          # and must fall below this before the next blink
TENSION_LABELS = ("angry", "sad", "surprised")

# How much each signal contributes to the final score.
WEIGHTS = {"blink": 0.40, "gaze": 0.25, "churn": 0.20, "tension": 0.15}

# Floors stop a near-zero baseline from making its signal hair-trigger.
# A ratio against 0.5 blinks/min would read as maximum panic forever.
# The blink floor sits at the low end of a normal resting rate (15-20/min),
# so a quiet 20 seconds during calibration cannot make normal blinking
# register as anxiety.
BASELINE_FLOORS = {"blink": 10.0, "gaze": 0.05, "churn": 4.0}

# A signal reads as maximum at this multiple above baseline. At 2.0 the
# score saturated on ordinary behaviour; anxious blinking runs roughly
# 2-3x resting, so full scale belongs at 3x.
FULL_SCALE_MULTIPLE = 3.0

# Tension is already a probability, so it is scored on its own absolute
# scale rather than as a ratio: this much above baseline reads as maximum.
TENSION_FULL_SCALE = 0.5

GAZE_FIELDS = (
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
)


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


class BlinkCounter:
    """Edge-triggered blink detector with hysteresis.

    A single threshold would count one slow blink as several, because the
    signal jitters back and forth across the line. Requiring the eye to
    reopen properly before arming again fixes that.
    """

    def __init__(self) -> None:
        self.closed = False
        self.timestamps: deque = deque()

    def update(self, blink_score: float, now: float) -> None:
        if not self.closed and blink_score > BLINK_CLOSE:
            self.closed = True
            self.timestamps.append(now)
        elif self.closed and blink_score < BLINK_OPEN:
            self.closed = False

    def rate_per_minute(self, now: float, window: float) -> float:
        while self.timestamps and now - self.timestamps[0] > window:
            self.timestamps.popleft()
        return len(self.timestamps) * 60.0 / window


class Signals:
    """Rolling buffers for the three frame-wise signals."""

    def __init__(self) -> None:
        self.gaze: deque = deque()
        self.labels: deque = deque()
        self.tension: deque = deque()

    def add(self, now, gaze_vec, label, tension_prob) -> None:
        self.gaze.append((now, gaze_vec))
        self.labels.append((now, label))
        self.tension.append((now, tension_prob))

    def prune(self, now, window) -> None:
        for buf in (self.gaze, self.labels, self.tension):
            while buf and now - buf[0][0] > window:
                buf.popleft()

    def gaze_instability(self) -> float:
        if len(self.gaze) < 5:
            return 0.0
        return float(np.mean(np.std(np.array([g for _, g in self.gaze]), axis=0)))

    def churn_per_minute(self, window) -> float:
        labels = [l for _, l in self.labels]
        flips = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        return flips * 60.0 / window

    def mean_tension(self) -> float:
        return float(np.mean([t for _, t in self.tension])) if self.tension else 0.0


def ratio(current: float, baseline: float, floor: float) -> float:
    """Deviation above baseline, 0.0 at or below normal, 1.0 at full scale.

    The floor matters. Calibrating for 20 seconds can easily yield a
    baseline of 3 blinks/min, and a ratio against that would peg the
    signal at maximum on the first extra blink.
    """
    reference = max(baseline, floor)
    excess = (current / reference) - 1.0
    return float(np.clip(excess / (FULL_SCALE_MULTIPLE - 1.0), 0.0, 1.0))


def absolute_rise(current: float, baseline: float, full_scale: float) -> float:
    """Rise above baseline on a fixed 0-1 scale, for signals that are
    already probabilities and can legitimately sit at zero when calm."""
    return float(np.clip((current - baseline) / full_scale, 0.0, 1.0))


def main() -> None:
    bundle = joblib.load(CLF_PATH)
    clf, labels = bundle["model"], bundle["labels"]
    feature_names = bundle["features"]
    gaze_idx = [feature_names.index(f) for f in GAZE_FIELDS]
    blink_idx = [feature_names.index(f) for f in ("eyeBlinkLeft", "eyeBlinkRight")]
    tension_idx = [labels.index(l) for l in TENSION_LABELS if l in labels]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    landmarker = build_landmarker()
    window_name = "FaceTell - nervousness"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, 1000, 750)

    blinks = BlinkCounter()
    signals = Signals()
    probs_history: deque = deque(maxlen=8)

    baseline = None
    calib_start = time.perf_counter()
    score_smoothed = 0.0

    print(f"Calibrating for {CALIBRATION_SECONDS:.0f}s - sit normally and relax.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        now = time.perf_counter()
        result = landmarker.detect_for_video(mp_image, int(now * 1000))

        if not result.face_blendshapes:
            text(frame, "no face", (15, 60), 1.2, (0, 0, 255), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue

        scores = np.array([c.score for c in result.face_blendshapes[0]])
        probs_history.append(clf.predict_proba(scores.reshape(1, -1))[0])
        probs = np.mean(probs_history, axis=0)
        label = labels[int(np.argmax(probs))]

        blinks.update(float(np.mean(scores[blink_idx])), now)
        signals.add(now, scores[gaze_idx], label, float(np.sum(probs[tension_idx])))

        elapsed = now - calib_start

        if baseline is None:
            signals.prune(now, CALIBRATION_SECONDS)
            remaining = CALIBRATION_SECONDS - elapsed
            if remaining <= 0:
                baseline = {
                    "blink": blinks.rate_per_minute(now, CALIBRATION_SECONDS),
                    "gaze": signals.gaze_instability(),
                    "churn": signals.churn_per_minute(CALIBRATION_SECONDS),
                    "tension": signals.mean_tension(),
                }
                summary = "  ".join(f"{k}={v:.2f}" for k, v in baseline.items())
                print("Baseline: " + summary, flush=True)
            else:
                text(frame, "CALIBRATING", (15, 60), 1.3, (0, 200, 255), 3)
                text(frame, f"relax, {remaining:.0f}s left", (15, 100), 0.8, (220, 220, 220), 2)
                text(frame, f"blinks so far: {len(blinks.timestamps)}", (15, 130), 0.6, (180, 180, 180), 1)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

        signals.prune(now, WINDOW_SECONDS)
        current = {
            "blink": blinks.rate_per_minute(now, WINDOW_SECONDS),
            "gaze": signals.gaze_instability(),
            "churn": signals.churn_per_minute(WINDOW_SECONDS),
            "tension": signals.mean_tension(),
        }
        parts = {
            k: ratio(current[k], baseline[k], BASELINE_FLOORS[k])
            for k in ("blink", "gaze", "churn")
        }
        parts["tension"] = absolute_rise(
            current["tension"], baseline["tension"], TENSION_FULL_SCALE
        )
        raw_score = 100.0 * sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
        score_smoothed = 0.92 * score_smoothed + 0.08 * raw_score

        if score_smoothed < 25:
            verdict, colour = "CALM", (90, 220, 120)
        elif score_smoothed < 55:
            verdict, colour = "SLIGHTLY UNEASY", (60, 200, 240)
        else:
            verdict, colour = "NERVOUS", (60, 60, 240)

        text(frame, verdict, (15, 65), 1.5, colour, 3)
        text(frame, f"score {score_smoothed:3.0f}/100", (15, 105), 0.8, (230, 230, 230), 2)
        text(frame, f"expression: {label}", (15, 135), 0.7, (200, 200, 200), 2)

        w = frame.shape[1]
        cv2.rectangle(frame, (15, 150), (w - 15, 172), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, 150), (15 + int((w - 30) * score_smoothed / 100), 172), colour, -1)

        base_y = frame.shape[0] - 20 - len(WEIGHTS) * 26
        for i, key in enumerate(WEIGHTS):
            y = base_y + i * 26
            cv2.rectangle(frame, (15, y), (15 + int(200 * parts[key]), y + 18), colour, -1)
            cv2.rectangle(frame, (15, y), (215, y + 18), (90, 90, 90), 1)
            caption = f"{key:8} {current[key]:6.1f}  (base {baseline[key]:.1f})"
            text(frame, caption, (225, y + 15), 0.5, (210, 210, 210), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            baseline = None
            calib_start = time.perf_counter()
            blinks = BlinkCounter()
            signals = Signals()
            print("Recalibrating...", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
