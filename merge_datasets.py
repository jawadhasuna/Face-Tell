"""Merge extracted datasets into one training set, dropping duplicates.

Training on a single collection teaches its house style as much as it
teaches emotion: AffectNet's "happy" is a mild adult smile, RAF-DB's is a
wide toothy laugh. A model trained on one calls the other's happy faces
angry. Merging forces it to learn what the expressions have in common.

The two sources also genuinely overlap - 5,130 byte-identical neutral
images appear in both - so files are hashed and duplicates kept once.
Without that, those images would be over-weighted and could land in both
the training and test halves of a split.

Filenames record their origin (affectnet_000123.jpg), so a per-source
breakdown is still possible afterwards.

Usage:
    python merge_datasets.py
    python merge_datasets.py --sources rafdb affectnet --out combined
"""

import argparse
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "data" / "images"


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["affectnet", "rafdb"])
    p.add_argument("--out", default="combined")
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of hard-linking (uses real disk space)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="class names to leave out of the merged set")
    args = p.parse_args()

    out_root = IMAGE_DIR / args.out
    if out_root.exists():
        print(f"Removing existing {out_root}")
        shutil.rmtree(out_root)

    seen: dict[str, str] = {}          # hash -> "source/class" it was first kept from
    kept = Counter()                   # class -> files written
    duplicates = Counter()             # class -> duplicates skipped
    per_source = defaultdict(Counter)  # source -> class -> kept
    conflicts = []                     # same image, different label

    for source in args.sources:
        src_root = IMAGE_DIR / source
        if not src_root.exists():
            raise SystemExit(f"Missing {src_root} - run extract.py for '{source}' first")

        for class_dir in sorted(d for d in src_root.iterdir() if d.is_dir()):
            label = class_dir.name
            if label in args.exclude:
                continue
            dest_dir = out_root / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            for image in sorted(class_dir.glob("*.jpg")):
                digest = file_hash(image)
                if digest in seen:
                    duplicates[label] += 1
                    previous = seen[digest]
                    if previous.split("/")[1] != label:
                        conflicts.append((previous, f"{source}/{label}", image.name))
                    continue

                seen[digest] = f"{source}/{label}"
                dest = dest_dir / f"{source}_{image.name}"
                if args.copy:
                    shutil.copy2(image, dest)
                else:
                    try:
                        dest.hardlink_to(image)
                    except OSError:
                        shutil.copy2(image, dest)

                kept[label] += 1
                per_source[source][label] += 1

        print(f"  {source}: {sum(per_source[source].values())} kept", flush=True)

    labels = sorted(kept)
    total = sum(kept.values())

    print(f"\nMerged into {out_root}")
    print(f"{'class':<12}" + "".join(f"{s:>12}" for s in args.sources) + f"{'total':>9}{'dupes':>8}")
    print("-" * (12 + 12 * len(args.sources) + 17))
    for label in labels:
        row = "".join(f"{per_source[s][label]:>12}" for s in args.sources)
        print(f"{label:<12}{row}{kept[label]:>9}{duplicates[label]:>8}")
    print("-" * (12 + 12 * len(args.sources) + 17))
    print(f"{'TOTAL':<12}" + "".join(f"{sum(per_source[s].values()):>12}" for s in args.sources)
          + f"{total:>9}{sum(duplicates.values()):>8}")

    imbalance = max(kept.values()) / max(min(kept.values()), 1)
    print(f"\nimbalance {imbalance:.1f}:1   duplicates dropped {sum(duplicates.values())}")

    if conflicts:
        print(f"\nWARNING: {len(conflicts)} identical images carry different labels "
              f"in different sources. First few:")
        for previous, current, name in conflicts[:5]:
            print(f"  {name}: {previous} vs {current}")
        print("These were kept once, under the label of whichever source was read first.")


if __name__ == "__main__":
    main()
