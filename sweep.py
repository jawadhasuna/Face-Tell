"""Run a hyperparameter sweep and tabulate what actually mattered.

One factor at a time, from a fixed baseline. Grid search over every
combination would be thousands of runs; changing one knob at a time
against a common reference tells you which knobs move the number, which
is the question worth answering first.

Every run writes runs/<tag>.json, so results survive a crash and the
table can be rebuilt later with --report.

Usage:
    python sweep.py --dataset rafdb --group lr
    python sweep.py --dataset rafdb --group all
    python sweep.py --report
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"

# The reference configuration every experiment is measured against.
BASELINE = {
    "model": "efficientnet_b0",
    "epochs": "12",
    "lr": "3e-4",
    "head-lr": "1e-3",
    "freeze-epochs": "2",
    "batch-size": "48",
    "weight-decay": "1e-4",
    "dropout": "0.2",
    "label-smoothing": "0.1",
    "aug": "medium",
}

# group -> list of (label, overrides). None means "the baseline itself".
GROUPS = {
    "baseline": [("baseline", {})],
    "lr": [
        ("lr_1e4", {"lr": "1e-4"}),
        ("lr_3e4", {"lr": "3e-4"}),
        ("lr_1e3", {"lr": "1e-3"}),
    ],
    "aug": [
        ("aug_light", {"aug": "light"}),
        ("aug_medium", {"aug": "medium"}),
        ("aug_strong", {"aug": "strong"}),
    ],
    # Kept separate: the no-augmentation control is the clearest
    # demonstration of overfitting, but it is not needed to pick a winner.
    "aug_control": [("aug_none", {"aug": "none"})],
    "freeze": [
        ("freeze_0", {"freeze-epochs": "0"}),
        ("freeze_2", {"freeze-epochs": "2"}),
        ("freeze_5", {"freeze-epochs": "5"}),
    ],
    "backbone": [
        ("bb_b0", {"model": "efficientnet_b0"}),
        ("bb_b2", {"model": "efficientnet_b2"}),
        ("bb_mobilenet", {"model": "mobilenet_v3_large"}),
        ("bb_resnet50", {"model": "resnet50", "batch-size": "32"}),
        ("bb_convnext", {"model": "convnext_tiny", "batch-size": "32"}),
    ],
    "regularisation": [
        ("reg_none", {"weight-decay": "0", "dropout": "0", "label-smoothing": "0"}),
        ("reg_default", {}),
        ("reg_heavy", {"weight-decay": "1e-3", "dropout": "0.4", "label-smoothing": "0.2"}),
    ],
    "longer": [
        ("epochs_25", {"epochs": "25", "patience": "6"}),
    ],
}


def run_one(dataset: str, tag: str, overrides: dict, class_weights: bool, extra: list[str]) -> bool:
    cfg = dict(BASELINE)
    cfg.update(overrides)

    cmd = [sys.executable, str(ROOT / "finetune.py"), "--dataset", dataset, "--tag", tag]
    for key, value in cfg.items():
        cmd += [f"--{key}", value]
    if class_weights:
        cmd.append("--class-weights")
    cmd += extra

    changed = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "reference config"
    print(f"\n{'=' * 70}\n{tag}   ({changed})\n{'=' * 70}", flush=True)

    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    mins = (time.perf_counter() - started) / 60

    if result.returncode != 0:
        print(f"  {tag} FAILED (exit {result.returncode})", flush=True)
        return False
    print(f"  {tag} finished in {mins:.1f} min", flush=True)
    return True


def report() -> None:
    rows = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if "test_acc" not in data:
            continue  # blendshape run log, different shape

        confusion = data.get("confusion")
        macro = None
        if confusion:
            per_class = [
                row[i] / sum(row) if sum(row) else 0.0
                for i, row in enumerate(confusion)
            ]
            macro = sum(per_class) / len(per_class)

        args = data.get("args", {})
        rows.append({
            "tag": path.stem,
            "model": args.get("model", "?"),
            "lr": args.get("lr", "?"),
            "aug": args.get("aug", "?"),
            "freeze": args.get("freeze_epochs", "?"),
            "val": data.get("best_val_acc", 0.0),
            "test": data["test_acc"],
            "macro": macro,
            "epochs_run": len(data.get("history", [])),
        })

    if not rows:
        print("No fine-tuning runs found in runs/")
        return

    rows.sort(key=lambda r: r["macro"] if r["macro"] is not None else r["test"], reverse=True)

    print(f"\n{'tag':<22}{'backbone':<20}{'lr':>7}{'aug':>8}{'frz':>5}"
          f"{'val':>8}{'test':>8}{'macro':>8}{'ep':>4}")
    print("-" * 90)
    for r in rows:
        macro = f"{r['macro']:.1%}" if r["macro"] is not None else "  -  "
        print(f"{r['tag']:<22}{r['model']:<20}{str(r['lr']):>7}{r['aug']:>8}"
              f"{str(r['freeze']):>5}{r['val']:>8.1%}{r['test']:>8.1%}{macro:>8}{r['epochs_run']:>4}")

    best = rows[0]
    print(f"\nBest by macro recall: {best['tag']}  "
          f"(test {best['test']:.1%}, macro {best['macro']:.1%})")
    print("\nMacro recall treats every emotion equally; plain accuracy is "
          "dominated by whichever classes happen to be common.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="rafdb")
    p.add_argument("--group", default="", choices=[""] + sorted(GROUPS) + ["all"])
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument(
        "--redo",
        action="store_true",
        help="rerun configurations that already have a log in runs/. "
        "By default they are skipped, so a sweep can be resumed.",
    )
    p.add_argument("extra", nargs="*", help="extra flags passed to finetune.py")
    args = p.parse_args()

    if args.report or not args.group:
        report()
        return

    groups = sorted(GROUPS) if args.group == "all" else [args.group]
    plan = []
    seen = set()
    for group in groups:
        if group == "baseline" and args.group == "all":
            continue
        for label, overrides in GROUPS[group]:
            if label in seen:
                continue
            seen.add(label)
            plan.append((f"{args.dataset}_{label}", overrides))

    if not args.redo:
        already = [t for t, _ in plan if (RUNS_DIR / f"{t}.json").exists()]
        if already:
            print(f"Skipping {len(already)} already-run configs "
                  f"(pass --redo to force): {', '.join(already)}\n")
        plan = [(t, o) for t, o in plan if t not in already]

    if not plan:
        print("Nothing left to run.\n")
        report()
        return

    print(f"Sweep on {args.dataset}: {len(plan)} runs")
    for tag, overrides in plan:
        print(f"  {tag:<28}{overrides or 'reference config'}")

    ok = 0
    started = time.perf_counter()
    for tag, overrides in plan:
        ok += run_one(args.dataset, tag, overrides, not args.no_class_weights, args.extra)

    print(f"\n{ok}/{len(plan)} runs completed in "
          f"{(time.perf_counter() - started)/60:.0f} min\n")
    report()


if __name__ == "__main__":
    main()
