"""Live webcam expression recognition with the fine-tuned CNN.

Runs both trained models on the same frame so their disagreements are
visible: the blendshape classifier (52 muscle scores, 73.8% on RAF-DB)
and the fine-tuned EfficientNet (pixels, 90.6% on RAF-DB).

MediaPipe still does the face finding. Its landmarks give the crop that
gets fed to the CNN, and its blendshapes feed the baseline model, so one
pass drives both.

Keys: q/Esc quit, b toggle baseline, t toggle flip averaging,
      +/- widen or tighten the crop fed to the CNN.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as TF

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from finetune import BACKBONES, build_model

ROOT = Path(__file__).parent
LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

SMOOTHING_FRAMES = 8

# Measured against the training set rather than guessed: across 199 RAF-DB
# images the frame is 0.97x the MediaPipe landmark box, with the landmark
# centre sitting 0.034 x side below the image centre. The 478 landmarks
# already reach the hairline, so the box is the whole head. Feeding a wider
# crop than this shifts the input away from what the model was trained on.
CROP_MARGIN = 0.0
CROP_Y_SHIFT = -0.034

COLOURS = {
    "happy": (80, 220, 100),
    "sad": (220, 140, 60),
    "angry": (60, 60, 235),
    "surprised": (40, 200, 240),
    "disgusted": (150, 90, 200),
    "fear": (200, 120, 255),
    "neutral": (190, 190, 190),
}


def text(frame, s, org, scale, colour, weight=2) -> None:
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight, cv2.LINE_AA)


def load_cnn(checkpoint: Path, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    name = ckpt["args"]["model"]
    # Rebuild with the dropout it was trained with: a non-zero value adds a
    # layer to the head, which shifts the parameter names in the state dict.
    # Dropout is inactive under eval() regardless.
    dropout = ckpt["args"].get("dropout", 0.0)
    model = build_model(name, len(ckpt["classes"]), dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt["classes"], name, ckpt.get("val_acc")


def face_crop(frame, landmarks, margin: float = CROP_MARGIN):
    """Square crop around the landmarks, clamped to the frame."""
    h, w = frame.shape[:2]
    xs = np.array([lm.x for lm in landmarks]) * w
    ys = np.array([lm.y for lm in landmarks]) * h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    side = max(x1 - x0, y1 - y0) * (1 + margin)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + CROP_Y_SHIFT * side
    half = side / 2

    left = int(max(cx - half, 0))
    top = int(max(cy - half, 0))
    right = int(min(cx + half, w))
    bottom = int(min(cy + half, h))
    if right - left < 32 or bottom - top < 32:
        return None, None
    return frame[top:bottom, left:right], (left, top, right, bottom)


def preprocess(crop_bgr, device):
    """Match the evaluation transform used during training."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    off = (256 - 224) // 2
    cropped = resized[off:off + 224, off:off + 224]

    tensor = torch.from_numpy(cropped).permute(2, 0, 1).float().div_(255.0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0).to(device)


def draw_bars(frame, labels, probs, x, y, title, width=210) -> None:
    text(frame, title, (x, y - 8), 0.5, (200, 200, 200), 1)
    for i, (label, p) in enumerate(zip(labels, probs)):
        row = y + i * 22
        colour = COLOURS.get(label, (200, 200, 200))
        cv2.rectangle(frame, (x, row), (x + int(width * p), row + 15), colour, -1)
        cv2.rectangle(frame, (x, row), (x + width, row + 15), (90, 90, 90), 1)
        text(frame, f"{label} {p:.0%}", (x + width + 8, row + 13), 0.42, colour, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/finetuned_rafdb_best.pt")
    ap.add_argument("--baseline", default="models/rafdb_blendshape.joblib")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--hide", nargs="*", default=[],
                    help="classes to suppress: their logit is set to -inf before "
                         "softmax, so probability redistributes across the rest. "
                         "The website hides 'disgusted'; this tool shows everything "
                         "unless asked, so the full model stays inspectable.")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, classes, backbone, val_acc = load_cnn(ROOT / args.checkpoint, device)
    hidden = [i for i, c in enumerate(classes) if c in args.hide]
    shown = [c for c in classes if c not in args.hide]
    print(f"CNN: {backbone} on {device}, val {val_acc:.1%}", flush=True)
    print(f"showing {shown}" + (f", hiding {args.hide}" if hidden else ""), flush=True)

    baseline = None
    baseline_path = ROOT / args.baseline
    if baseline_path.exists():
        bundle = joblib.load(baseline_path)
        if bundle["labels"] == classes:
            baseline = bundle
            print(f"Baseline: blendshape model, {len(bundle['features'])} features", flush=True)
        else:
            print("Baseline skipped - its classes differ from the CNN's", flush=True)

    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=baseline is not None,
        )
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    window = "FaceTell - fine-tuned CNN"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window, 1100, 780)

    cnn_history: deque = deque(maxlen=SMOOTHING_FRAMES)
    base_history: deque = deque(maxlen=SMOOTHING_FRAMES)
    show_baseline = True
    margin = CROP_MARGIN
    flip_tta = True
    fps, last, infer_ms = 0.0, time.perf_counter(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(time.perf_counter() * 1000))

        if result.face_landmarks:
            crop, box = face_crop(frame, result.face_landmarks[0], margin)

            if crop is not None:
                t0 = time.perf_counter()
                batch = preprocess(crop, device)
                if flip_tta:
                    # Averaging a prediction with its mirror costs one extra
                    # forward pass and removes left/right asymmetry noise.
                    batch = torch.cat([batch, torch.flip(batch, dims=[3])])
                with torch.no_grad():
                    logits = model(batch)
                    if hidden:
                        logits[:, hidden] = float("-inf")
                    probs = TF.softmax(logits, dim=1).mean(0).float().cpu().numpy()
                infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t0) * 1000
                cnn_history.append(probs)

                left, top, right, bottom = box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 220, 255), 2)

                # Show exactly what the model sees - the fastest way to spot a
                # framing mismatch against the training data.
                thumb = cv2.resize(crop, (140, 140), interpolation=cv2.INTER_AREA)
                fh, fw = frame.shape[:2]
                frame[fh - 150:fh - 10, fw - 150:fw - 10] = thumb
                cv2.rectangle(frame, (fw - 150, fh - 150), (fw - 10, fh - 10), (0, 220, 255), 1)
                text(frame, f"model input  margin {margin:+.2f}",
                     (fw - 300, fh - 158), 0.45, (200, 200, 200), 1)

            if baseline is not None and result.face_blendshapes:
                scores = np.array([[c.score for c in result.face_blendshapes[0]]])
                base_history.append(baseline["model"].predict_proba(scores)[0])

            if cnn_history:
                cnn_probs = np.mean(cnn_history, axis=0)
                best = int(np.argmax(cnn_probs))
                name = classes[best]
                text(frame, name.upper(), (15, 70), 1.7, COLOURS.get(name, (255, 255, 255)), 3)
                text(frame, f"{cnn_probs[best]:.0%}  ({backbone}, {infer_ms:.0f} ms)",
                     (15, 104), 0.6, (220, 220, 220), 2)
                visible = [(c, cnn_probs[classes.index(c)]) for c in shown]
                draw_bars(frame, [c for c, _ in visible], [v for _, v in visible],
                          15, 140, "FINE-TUNED CNN  90.6%")

                if show_baseline and base_history:
                    base_probs = np.mean(base_history, axis=0)
                    bx = frame.shape[1] - 320
                    vis_base = [(c, base_probs[classes.index(c)]) for c in shown]
                    draw_bars(frame, [c for c, _ in vis_base], [v for _, v in vis_base],
                              bx, 140, "BLENDSHAPE BASELINE  73.8%")
                    other = max(shown, key=lambda c: base_probs[classes.index(c)])
                    if other != name:
                        text(frame, f"baseline says: {other}", (bx, 120), 0.5, (60, 200, 255), 1)
        else:
            cnn_history.clear()
            base_history.clear()
            text(frame, "no face", (15, 70), 1.2, (0, 0, 255), 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        text(frame, f"FPS {fps:4.0f}", (frame.shape[1] - 140, 35), 0.7, (0, 255, 255), 2)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("b"):
            show_baseline = not show_baseline
        elif key == ord("t"):
            flip_tta = not flip_tta
            print(f"flip averaging {'on' if flip_tta else 'off'}", flush=True)
        elif key in (ord("+"), ord("=")):
            margin = round(margin + 0.05, 2)
            cnn_history.clear()
        elif key in (ord("-"), ord("_")):
            margin = round(margin - 0.05, 2)
            cnn_history.clear()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
