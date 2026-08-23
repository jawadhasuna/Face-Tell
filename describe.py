"""Turn measured blendshapes into written descriptions of a face.

To fine-tune a model that *describes* an expression rather than labelling
it, you need image-to-sentence pairs. No such dataset exists for facial
expressions, so this builds one from measurements already taken: every
clause traces back to a number, nothing is invented.

    browDownLeft 0.71, browDownRight 0.68, mouthPressLeft 0.44
    -> "brows pulled down hard, lips pressed together. Reads as anger."

Intensity bands come from the value distribution across both datasets
(69% of all readings fall below 0.05, p90 = 0.32, p95 = 0.53, p99 = 0.84),
so "faint" and "strong" mean something measured rather than guessed.

Usage:
    python describe.py --preview 12
    python describe.py --format text   --out data/describe_text.jsonl
    python describe.py --format vision --out data/describe_vision.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# Below p75 a signal is indistinguishable from a resting face.
FLOOR = 0.08
BANDS = [(0.84, "very strongly"), (0.53, "strongly"), (0.32, "clearly"), (FLOOR, "faintly")]

# Where the eyes point says where attention is, not what is felt, so gaze is
# only mentioned when little else is happening. Without this the descriptions
# fill up with "the gaze turned inward" and crowd out the real signals.
DEMOTED = {"eyeLookIn", "eyeLookOut", "eyeLookUp", "eyeLookDown",
           "jawLeft", "jawRight", "mouthLeft", "mouthRight"}

# A face cannot be broadly smiling and afraid at once. When the measurement
# flatly contradicts the dataset label, one of the two is wrong and the pair
# is not worth training on.
CONTRADICTIONS = {
    "happy":     [("mouthSmile", "min", 0.20)],
    "neutral":   [("mouthSmile", "max", 0.35), ("jawOpen", "max", 0.40),
                  ("browDown", "max", 0.45), ("browInnerUp", "max", 0.45)],
    "sad":       [("mouthSmile", "max", 0.40)],
    "angry":     [("mouthSmile", "max", 0.40)],
    "fear":      [("mouthSmile", "max", 0.40)],
    "disgusted": [("mouthSmile", "max", 0.45)],
    "surprised": [("mouthSmile", "max", 0.50)],
}


def contradicts(signals: dict, label: str) -> str:
    """Return the reason this row disagrees with its label, or an empty string."""
    for name, kind, limit in CONTRADICTIONS.get(label, []):
        value = signals.get(name, 0.0)
        if kind == "max" and value > limit:
            return f"{name}={value:.2f} too high for {label}"
        if kind == "min" and value < limit:
            return f"{name}={value:.2f} too low for {label}"
    return ""


# Blendshape base name -> how a person would describe it.
PHRASES = {
    "mouthSmile": "the mouth corners pulled up",
    "mouthFrown": "the mouth corners pulled down",
    "mouthPucker": "the lips pushed forward",
    "mouthPress": "the lips pressed together",
    "mouthStretch": "the mouth stretched wide",
    "mouthShrugLower": "the lower lip pushed up",
    "mouthShrugUpper": "the upper lip pushed up",
    "mouthUpperUp": "the upper lip lifted",
    "mouthLowerDown": "the lower lip pulled down",
    "mouthDimple": "dimples set at the mouth corners",
    "mouthRollLower": "the lower lip rolled inward",
    "mouthRollUpper": "the upper lip rolled inward",
    "mouthFunnel": "the lips funnelled open",
    "mouthClose": "the lips held shut",
    "mouthLeft": "the mouth pulled to one side",
    "mouthRight": "the mouth pulled to one side",
    "jawOpen": "the jaw dropped open",
    "jawForward": "the jaw pushed forward",
    "jawLeft": "the jaw shifted sideways",
    "jawRight": "the jaw shifted sideways",
    "browDown": "the brows pulled down",
    "browInnerUp": "the inner brows lifted",
    "browOuterUp": "the outer brows raised",
    "eyeSquint": "the eyes narrowed",
    "eyeWide": "the eyes opened wide",
    "eyeBlink": "the eyelids lowered",
    "eyeLookDown": "the gaze dropped downward",
    "eyeLookUp": "the gaze lifted upward",
    "eyeLookIn": "the gaze turned inward",
    "eyeLookOut": "the gaze turned to the side",
    "cheekSquint": "the cheeks bunched up",
    "cheekPuff": "the cheeks puffed out",
    "noseSneer": "the nose wrinkled",
}

# How each label reads once the muscles are accounted for.
READS = {
    "happy": ["reads as happiness", "this is a smile", "the overall read is happy"],
    "sad": ["reads as sadness", "the overall read is sad", "this reads as low mood"],
    "angry": ["reads as anger", "the overall read is angry", "this reads as irritation"],
    "fear": ["reads as fear", "the overall read is fearful", "this reads as alarm"],
    "surprised": ["reads as surprise", "the overall read is surprised", "this reads as startled"],
    "disgusted": ["reads as disgust", "the overall read is disgusted", "this reads as revulsion"],
    "neutral": ["reads as neutral", "the face is at rest", "no strong expression is present"],
}

OPENERS = ["", "Looking at the face, ", "In this face, ", "Here, "]
JOINERS = [". ", ", and ", ", with "]

QUESTIONS = [
    "What expression is this person making?",
    "Describe this facial expression.",
    "What is this face showing?",
    "Read this expression and explain what you see.",
    "What emotion does this face convey, and why?",
]


def band(value: float) -> str | None:
    for threshold, word in BANDS:
        if value >= threshold:
            return word
    return None


def group_signals(row, features) -> dict:
    """Average Left/Right pairs so the text says 'brows' not 'browDownLeft'."""
    grouped: dict[str, list[float]] = {}
    for name in features:
        base = name.removesuffix("Left").removesuffix("Right")
        if base in PHRASES:
            grouped.setdefault(base, []).append(float(row[name]))
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def describe(row, features, label: str, rng: random.Random) -> str:
    signals = group_signals(row, features)
    ranked = sorted(((k, v) for k, v in signals.items() if v >= FLOOR),
                    key=lambda kv: kv[1], reverse=True)
    strong = [kv for kv in ranked if kv[0] not in DEMOTED]
    active = (strong or ranked)[:3]

    if not active:
        return rng.choice([
            "Nothing is moving much; the muscles all sit near rest. "
            + rng.choice(READS[label]).capitalize() + ".",
            "No muscle group is meaningfully engaged. "
            + rng.choice(READS[label]).capitalize() + ".",
        ])

    clauses = [f"{PHRASES[name]} {band(value)}" for name, value in active]
    if len(clauses) == 1:
        body = clauses[0]
    elif len(clauses) == 2:
        body = clauses[0] + rng.choice(JOINERS[1:]) + clauses[1]
    else:
        body = clauses[0] + ", " + clauses[1] + ", " + clauses[2]

    opener = rng.choice(OPENERS)
    sentence = opener + body[0].lower() + body[1:] if opener else body[0].upper() + body[1:]
    return f"{sentence}. {rng.choice(READS[label]).capitalize()}."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+",
                   default=["data/affectnet_blendshapes.csv", "data/rafdb_blendshapes.csv"])
    p.add_argument("--format", choices=("text", "vision"), default="text")
    p.add_argument("--out", default="")
    p.add_argument("--preview", type=int, default=0, help="print N examples and stop")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-contradictions", action="store_true",
                   help="do not drop rows whose measurements disagree with the label")
    args = p.parse_args()

    frames = []
    for path in args.csv:
        df = pd.read_csv(ROOT / path)
        df["source"] = Path(path).stem.replace("_blendshapes", "")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    features = [c for c in data.columns if c not in ("label", "source")]

    rng = random.Random(args.seed)

    if args.preview:
        # pandas 3 drops the grouping column inside groupby.apply, so sample
        # each label separately instead.
        per_label = max(1, args.preview // data.label.nunique())
        for label in sorted(data.label.unique()):
            subset = data[data.label == label].sample(frac=1, random_state=args.seed)
            kept = 0
            for _, row in subset.iterrows():
                if not args.keep_contradictions and contradicts(
                        group_signals(row, features), label):
                    continue
                print(f"[{label}]  {describe(row, features, label, rng)}\n")
                kept += 1
                if kept >= per_label:
                    break
        return

    if args.limit:
        data = data.sample(min(args.limit, len(data)), random_state=args.seed)

    out_path = ROOT / (args.out or f"data/describe_{args.format}.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    written = 0
    dropped = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in data.iterrows():
            label = row["label"]

            if not args.keep_contradictions:
                why = contradicts(group_signals(row, features), label)
                if why:
                    dropped[label] = dropped.get(label, 0) + 1
                    continue

            answer = describe(row, features, label, rng)

            if args.format == "text":
                readings = ", ".join(
                    f"{k} {v:.2f}" for k, v in
                    sorted(group_signals(row, features).items(), key=lambda kv: -kv[1])[:8])
                record = {"messages": [
                    {"role": "system",
                     "content": "You read facial muscle measurements and describe the expression in plain language."},
                    {"role": "user",
                     "content": f"Facial muscle activations (0-1): {readings}\n\n{rng.choice(QUESTIONS)}"},
                    {"role": "assistant", "content": answer},
                ]}
            else:
                record = {
                    "image": f"data/images/{row['source']}/{label}/",
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": rng.choice(QUESTIONS)}]},
                        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                    ],
                }

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {written} examples to {out_path} ({size_mb:.1f} MB)")
    print(f"Label balance:\n{data.label.value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
