"""Train a classifier on extracted blendshape features.

This is the baseline arm of the experiment: 52 muscle activation scores
straight into a small classical model, no pixels involved. It is what
the fine-tuned CNN has to beat.

Splitting matters more than the model choice here:

  random    correct for independent photographs
  temporal  correct for webcam captures, where neighbouring rows are
            consecutive frames of the same face and a random split
            would put near-duplicates on both sides

Pass --test-csv to evaluate on a different dataset entirely. That is the
only number that honestly answers "does this work on other people".

Usage:
    python train.py                                   # webcam data
    python train.py --csv data/affectnet_blendshapes.csv --split random
    python train.py --csv data/affectnet_blendshapes.csv --split random \
                    --test-csv data/rafdb_blendshapes.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"


def build_models() -> dict:
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1
        ),
        "hist_boost": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.1, random_state=0
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000,
                          early_stopping=True, random_state=0),
        ),
    }


def looks_like_video(df: pd.DataFrame) -> bool:
    """Rows recorded from video sit in long same-label runs.

    Independent photos alternate labels constantly. If transitions are
    rare, a random split will leak near-identical neighbouring frames.
    """
    labels = df["label"].to_numpy()
    transitions = (labels[1:] != labels[:-1]).sum()
    return transitions < len(labels) / 20


def temporal_split(df: pd.DataFrame, test_frac: float):
    train_parts, test_parts = [], []
    for _, group in df.groupby("label", sort=False):
        cut = int(len(group) * (1 - test_frac))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def xy(df: pd.DataFrame, features: list[str]):
    return df[features].to_numpy(), df["label"].to_numpy()


def print_confusion(y_true, y_pred, labels) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(l) for l in labels) + 2
    print(" " * width + "".join(f"{l[:6]:>8}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:<{width}}" + "".join(f"{v:>8}" for v in row))
    return cm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/samples.csv")
    p.add_argument("--test-csv", default="", help="evaluate on a second dataset")
    p.add_argument("--split", choices=("auto", "random", "temporal"), default="auto")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--out", default="models/expression_clf.joblib")
    p.add_argument("--tag", default="blendshape")
    args = p.parse_args()

    df = pd.read_csv(ROOT / args.csv)
    features = [c for c in df.columns if c != "label"]
    labels = sorted(df["label"].unique())

    split = args.split
    if split == "auto":
        split = "temporal" if looks_like_video(df) else "random"
        print(f"Split mode auto-detected as '{split}'.")
    elif split == "random" and looks_like_video(df):
        print("WARNING: rows sit in long same-label runs, which looks like video.\n"
              "         A random split will leak near-duplicate frames and\n"
              "         report an accuracy you will not see live.")

    print(f"{len(df)} samples, {len(features)} features, {len(labels)} classes")
    print("class balance: " + "  ".join(
        f"{k}={v}" for k, v in df['label'].value_counts().sort_index().items()) + "\n")

    if split == "temporal":
        df_tr, df_te = temporal_split(df, args.test_frac)
    else:
        df_tr, df_te = train_test_split(
            df, test_size=args.test_frac, stratify=df["label"], random_state=0
        )

    Xtr, ytr = xy(df_tr, features)
    Xte, yte = xy(df_te, features)

    external = None
    if args.test_csv:
        ext = pd.read_csv(ROOT / args.test_csv)
        shared = [c for c in features if c in ext.columns]
        if len(shared) != len(features):
            raise SystemExit("Feature columns differ between the two CSVs.")
        ext = ext[ext["label"].isin(labels)]
        external = (ext[features].to_numpy(), ext["label"].to_numpy())
        print(f"Cross-dataset test set: {len(ext)} rows from {Path(args.test_csv).name}\n")

    header = f"{'model':<16}{'held-out':>11}"
    if external is not None:
        header += f"{'cross-dataset':>16}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        line = f"{name:<16}{acc:>10.1%}"
        ext_acc = None
        if external is not None:
            ext_acc = accuracy_score(external[1], model.predict(external[0]))
            line += f"{ext_acc:>15.1%}"
        print(line)
        results[name] = {"model": model, "acc": acc, "ext_acc": ext_acc}

    # Rank by cross-dataset accuracy when we have it; that is the number
    # that reflects performance on people the model has never seen.
    key = "ext_acc" if external is not None else "acc"
    best_name = max(results, key=lambda k: results[k][key])
    best = results[best_name]
    print(f"\nBest: {best_name}  (held-out {best['acc']:.1%}"
          + (f", cross-dataset {best['ext_acc']:.1%}" if external is not None else "") + ")\n")

    preds = best["model"].predict(Xte)
    print(classification_report(yte, preds, zero_division=0))
    print("Confusion matrix - rows are truth, columns are predictions:")
    cm = print_confusion(yte, preds, labels)

    if external is not None:
        print("\nCross-dataset confusion matrix:")
        ext_labels = sorted(set(external[1]))
        print_confusion(external[1], best["model"].predict(external[0]), ext_labels)

    final = build_models()[best_name]
    final.fit(*xy(df, features))
    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": features, "labels": labels}, out_path)

    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / f"{args.tag}.json").write_text(json.dumps({
        "csv": args.csv, "test_csv": args.test_csv, "split": split,
        "n_samples": len(df), "classes": labels,
        "scores": {k: {"acc": v["acc"], "ext_acc": v["ext_acc"]} for k, v in results.items()},
        "best": best_name, "confusion": cm.tolist(),
    }, indent=2))

    print(f"\nSaved {best_name} (retrained on all {len(df)} rows) to {out_path}")
    print(f"Run log: {RUNS_DIR / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
