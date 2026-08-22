"""Train an expression classifier on collected blendshape samples.

Reports accuracy two ways:
  random split   - optimistic, leaks near-identical neighbouring frames
  temporal split - honest, holds out the tail of each recording

The temporal number is the one to trust. The model saved to disk is the
one that scores best on it.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "samples.csv"
MODEL_OUT = ROOT / "models" / "expression_clf.joblib"
TEST_TAIL = 0.25  # fraction of each recording held out, taken from the end


def build_models() -> dict:
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=0),
        ),
    }


def temporal_split(df: pd.DataFrame):
    """Hold out the final TEST_TAIL of each label's rows, preserving order."""
    train_parts, test_parts = [], []
    for label, group in df.groupby("label", sort=False):
        cut = int(len(group) * (1 - TEST_TAIL))
        train_parts.append(group.iloc[:cut])
        test_parts.append(group.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def xy(df: pd.DataFrame):
    return df.drop(columns=["label"]).to_numpy(), df["label"].to_numpy()


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    features = [c for c in df.columns if c != "label"]
    print(f"{len(df)} samples, {len(features)} features, {df['label'].nunique()} classes\n")

    # --- optimistic: random split ---
    X, y = xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_TAIL, stratify=y, random_state=0)

    # --- honest: temporal split ---
    df_tr, df_te = temporal_split(df)
    Xtr_t, ytr_t = xy(df_tr)
    Xte_t, yte_t = xy(df_te)

    print(f"{'model':<15}{'random split':>15}{'temporal split':>17}")
    print("-" * 47)

    results = {}
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        random_acc = accuracy_score(yte, model.predict(Xte))

        honest = build_models()[name]
        honest.fit(Xtr_t, ytr_t)
        temporal_acc = accuracy_score(yte_t, honest.predict(Xte_t))

        results[name] = (temporal_acc, honest)
        print(f"{name:<15}{random_acc:>14.1%}{temporal_acc:>17.1%}")

    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_model = results[best_name]
    print(f"\nBest on the honest split: {best_name} at {best_acc:.1%}\n")

    preds = best_model.predict(Xte_t)
    labels = sorted(df["label"].unique())

    print("Per-class detail (temporal split):")
    print(classification_report(yte_t, preds, zero_division=0))

    print("Confusion matrix - rows are truth, columns are predictions:")
    cm = confusion_matrix(yte_t, preds, labels=labels)
    width = max(len(l) for l in labels) + 2
    print(" " * width + "".join(f"{l[:6]:>8}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:<{width}}" + "".join(f"{v:>8}" for v in row))

    # Retrain the winner on everything before shipping it.
    final = build_models()[best_name]
    final.fit(X, y)
    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": features, "labels": labels}, MODEL_OUT)
    print(f"\nSaved {best_name} (retrained on all {len(df)} samples) to {MODEL_OUT}")


if __name__ == "__main__":
    main()
