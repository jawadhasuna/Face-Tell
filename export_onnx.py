"""Export a fine-tuned checkpoint to ONNX, quantise it, and verify both.

PyTorch checkpoints only run in PyTorch. ONNX describes the network as a
portable graph, which is what lets the same model run in a browser via
onnxruntime-web.

Quantising to INT8 makes the file roughly four times smaller - the
difference between a page that loads instantly and one that does not.
It also costs accuracy, so this script measures the cost on the real test
split rather than assuming the usual "about 1%".

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint models/finetuned_combined_best.pt
    python export_onnx.py --skip-quant
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cross_eval import eval_transform
from finetune import build_model, stratified_split

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["args"]["model"], len(ckpt["classes"]),
                        ckpt["args"].get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def test_images(dataset: str, img_size: int, limit: int | None = None):
    """The same held-out split the training run reported on."""
    from torchvision import datasets as tvd

    root = ROOT / "data" / "images" / dataset
    base = tvd.ImageFolder(root)
    targets = [s[1] for s in base.samples]
    _, _, test_idx = stratified_split(targets, 0.15, 0.15, 0)
    if limit:
        rng = np.random.default_rng(0)
        test_idx = rng.permutation(test_idx)[:limit].tolist()

    transform = eval_transform(img_size)
    for i in test_idx:
        path, label = base.samples[i]
        tensor = transform(Image.open(path).convert("RGB"))
        yield tensor.numpy()[None], label


def run_onnx(model_path: Path, dataset: str, img_size: int, limit=None):
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    correct = total = 0
    elapsed = 0.0
    for array, label in test_images(dataset, img_size, limit):
        t0 = time.perf_counter()
        logits = session.run(None, {input_name: array})[0]
        elapsed += time.perf_counter() - t0
        correct += int(logits.argmax()) == label
        total += 1

    return correct / total, elapsed / total * 1000, total


class Calibrator:
    """Feeds real images to the quantiser so it can pick sensible scales."""

    def __init__(self, dataset: str, img_size: int, input_name: str, count: int = 200):
        self.input_name = input_name
        self.samples = [a for a, _ in test_images(dataset, img_size, count)]
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        array = self.samples[self.index]
        self.index += 1
        return {self.input_name: array}

    def rewind(self):
        self.index = 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/finetuned_rafdb_best.pt")
    p.add_argument("--dataset", default="", help="defaults to whatever the model trained on")
    p.add_argument("--out", default="", help="output basename, defaults to the model name")
    p.add_argument("--eval-limit", type=int, default=1000,
                   help="images used to score each variant (0 = the whole test split)")
    p.add_argument("--skip-quant", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt_path = ROOT / args.checkpoint
    model, ckpt = load_checkpoint(ckpt_path)
    classes = ckpt["classes"]
    img_size = ckpt["args"].get("img_size", 224)
    dataset = args.dataset or ckpt["args"]["dataset"]
    stem = args.out or ckpt_path.stem

    WEB_DIR.mkdir(exist_ok=True)
    fp32_path = WEB_DIR / f"{stem}.onnx"
    int8_path = WEB_DIR / f"{stem}.int8.onnx"

    print(f"{ckpt['args']['model']}  {len(classes)} classes  {img_size}px  "
          f"trained on {dataset}\n")

    # --- export -----------------------------------------------------------
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(fp32_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"exported {fp32_path.name}  ({fp32_path.stat().st_size/1e6:.1f} MB)")

    # --- does the exported graph match PyTorch? ---------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    drift = float(np.abs(torch_out - onnx_out).max())
    print(f"max difference against PyTorch: {drift:.2e}  "
          f"({'identical enough' if drift < 1e-4 else 'TOO LARGE - investigate'})\n")

    limit = args.eval_limit or None
    results = []

    acc, ms, n = run_onnx(fp32_path, dataset, img_size, limit)
    results.append(("float32", fp32_path, acc, ms))
    print(f"float32  accuracy {acc:.1%} on {n} images,  {ms:.1f} ms/image (CPU)")

    # --- quantise ---------------------------------------------------------
    if not args.skip_quant:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process

        prepared = WEB_DIR / f"{stem}.prepared.onnx"
        quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=False)

        quantize_static(
            str(prepared), str(int8_path),
            Calibrator(dataset, img_size, "input"),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        prepared.unlink(missing_ok=True)

        acc8, ms8, _ = run_onnx(int8_path, dataset, img_size, limit)
        results.append(("int8", int8_path, acc8, ms8))
        print(f"int8     accuracy {acc8:.1%} on {n} images,  {ms8:.1f} ms/image (CPU)")

    # --- summary ----------------------------------------------------------
    print(f"\n{'variant':<10}{'size':>10}{'accuracy':>11}{'ms/image':>11}")
    print("-" * 42)
    for name, path, accuracy, ms in results:
        print(f"{name:<10}{path.stat().st_size/1e6:>9.1f}M{accuracy:>11.1%}{ms:>11.1f}")

    if len(results) == 2:
        size_drop = 1 - results[1][1].stat().st_size / results[0][1].stat().st_size
        acc_drop = results[0][2] - results[1][2]
        print(f"\nint8 is {size_drop:.0%} smaller and costs {acc_drop*100:+.1f} "
              f"accuracy points.")
        pick = "int8" if acc_drop <= 0.02 else "float32"
        print(f"Recommended for the web app: {pick}"
              + ("" if pick == "int8" else "  (quantisation cost too much here)"))

    meta = {
        "classes": classes,
        "img_size": img_size,
        "trained_on": dataset,
        "backbone": ckpt["args"]["model"],
        "normalise": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "variants": {name: path.name for name, path, _, _ in results},
        "accuracy": {name: round(a, 4) for name, _, a, _ in results},
    }
    (WEB_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {WEB_DIR / 'model_meta.json'}")


if __name__ == "__main__":
    main()
