"""Test a trained model on a dataset it was never trained on.

Held-out accuracy tells you how well a model does on new photos from the
same collection - same photographers, same crops, same labelling team.
That flatters it. The honest question is whether it works on faces from
somewhere else entirely, and only a second dataset can answer that.

Handles both model types so the comparison stays fair:
  a fine-tuned CNN checkpoint (.pt)  scored on images
  a blendshape classifier (.joblib)  scored on the extracted CSV

Usage:
    python cross_eval.py --checkpoint models/finetuned_affectnet_best.pt --on rafdb
    python cross_eval.py --baseline models/affectnet_blendshape.joblib \
                         --on-csv data/rafdb_blendshapes.csv
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from finetune import IMAGENET_MEAN, IMAGENET_STD, build_model

ROOT = Path(__file__).parent


def eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def print_report(y_true, y_pred, classes, title: str) -> float:
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = cm.diagonal().sum() / cm.sum()
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    macro = recall.mean()

    print(f"\n{title}")
    print(f"  accuracy      {acc:.1%}")
    print(f"  macro recall  {macro:.1%}   ({cm.sum()} images)\n")

    print("  per-class recall:")
    for name, r, support in zip(classes, recall, cm.sum(1)):
        print(f"    {name:<12}{r:>7.1%}   (n={support})")

    width = max(len(c) for c in classes) + 4
    print("\n  confusion - rows are truth, columns are predictions:")
    print(" " * width + "".join(f"{c[:6]:>8}" for c in classes))
    for name, row in zip(classes, cm):
        print(f"  {name:<{width-2}}" + "".join(f"{v:>8}" for v in row))
    return macro


def eval_cnn(checkpoint: Path, dataset: str, batch_size: int, workers: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["args"].get("img_size", 224)

    model = build_model(ckpt["args"]["model"], len(classes),
                        ckpt["args"].get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    root = ROOT / "data" / "images" / dataset
    folder = datasets.ImageFolder(root, transform=eval_transform(img_size))

    if folder.classes != classes:
        raise SystemExit(
            f"Class mismatch.\n  model : {classes}\n  {dataset:<6}: {folder.classes}"
        )

    loader = DataLoader(folder, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
            y_pred += logits.argmax(1).cpu().tolist()
            y_true += labels.tolist()

    trained_on = ckpt["args"].get("dataset", "?")
    return print_report(
        y_true, y_pred, classes,
        f"CNN trained on {trained_on.upper()}, tested on {dataset.upper()}",
    )


def eval_baseline(model_path: Path, csv_path: Path):
    import joblib
    import pandas as pd

    bundle = joblib.load(model_path)
    classes = bundle["labels"]
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(classes)]

    index = {c: i for i, c in enumerate(classes)}
    y_true = [index[l] for l in df["label"]]
    y_pred = [index[p] for p in bundle["model"].predict(df[bundle["features"]].to_numpy())]

    return print_report(
        y_true, y_pred, classes,
        f"Blendshape model {model_path.name}, tested on {csv_path.name}",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="fine-tuned CNN .pt to evaluate")
    p.add_argument("--on", default="rafdb", help="image dataset to test the CNN on")
    p.add_argument("--baseline", help="blendshape .joblib to evaluate")
    p.add_argument("--on-csv", help="blendshape CSV to test the baseline on")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if not args.checkpoint and not args.baseline:
        raise SystemExit("Give --checkpoint or --baseline (or both).")

    if args.baseline:
        if not args.on_csv:
            raise SystemExit("--baseline needs --on-csv")
        eval_baseline(ROOT / args.baseline, ROOT / args.on_csv)

    if args.checkpoint:
        eval_cnn(ROOT / args.checkpoint, args.on, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
