"""Fine-tune a pretrained image model on facial expressions.

This is transfer learning, not training from scratch. A backbone that
already learned edges, textures and face structure from ImageNet keeps
those weights; only the classifier head is new.

Two stages, which matters:

  warmup   backbone frozen, train the new head alone. A randomly
           initialised head produces large gradients that would otherwise
           tear up good pretrained features on the first few batches.
  finetune everything unfrozen at a much lower learning rate, cosine
           decayed. This is where the backbone specialises to faces.

Usage:
    python finetune.py --model efficientnet_b0 --epochs 12
    python finetune.py --model resnet50 --lr 3e-4 --freeze-epochs 3
    python finetune.py --list-models
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision import models as tvm

ROOT = Path(__file__).parent
IMAGE_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Backbone name -> (constructor, default weights, attribute holding the head)
BACKBONES = {
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "efficientnet_b2": (tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights.IMAGENET1K_V1, "classifier"),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "classifier"),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier"),
}


def build_model(name: str, num_classes: int, dropout: float) -> nn.Module:
    """Load pretrained weights and swap the 1000-class head for ours."""
    ctor, weights, head_attr = BACKBONES[name]
    model = ctor(weights=weights)

    head = getattr(model, head_attr)
    if isinstance(head, nn.Sequential):
        # Find the final Linear and replace it, keeping the rest of the head.
        idx = max(i for i, m in enumerate(head) if isinstance(m, nn.Linear))
        in_features = head[idx].in_features
        layers = list(head)
        layers[idx] = nn.Linear(in_features, num_classes)
        if dropout > 0:
            layers.insert(idx, nn.Dropout(dropout))
        setattr(model, head_attr, nn.Sequential(*layers))
    else:
        in_features = head.in_features
        new_head = nn.Linear(in_features, num_classes)
        if dropout > 0:
            new_head = nn.Sequential(nn.Dropout(dropout), new_head)
        setattr(model, head_attr, new_head)

    return model


def head_parameters(model: nn.Module, name: str):
    return getattr(model, BACKBONES[name][2]).parameters()


def set_backbone_frozen(model: nn.Module, name: str, frozen: bool) -> None:
    head_attr = BACKBONES[name][2]
    for attr, module in model.named_children():
        if attr == head_attr:
            continue
        for p in module.parameters():
            p.requires_grad = not frozen


def build_transforms(img_size: int, aug_strength: str):
    train_steps = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0))]

    if aug_strength != "none":
        train_steps.append(transforms.RandomHorizontalFlip())
    if aug_strength in ("medium", "strong"):
        train_steps.append(transforms.ColorJitter(0.3, 0.3, 0.3, 0.05))
    if aug_strength == "strong":
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=7))

    train_steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug_strength == "strong":
        train_steps.append(transforms.RandomErasing(p=0.25))

    eval_steps = [
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def stratified_split(targets, val_frac: float, test_frac: float, seed: int):
    """Equal class proportions in every split, so rare emotions are present."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for idx, t in enumerate(targets):
        by_class.setdefault(int(t), []).append(idx)

    train, val, test = [], [], []
    for _, idxs in sorted(by_class.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_test = int(len(idxs) * test_frac)
        n_val = int(len(idxs) * val_frac)
        test += idxs[:n_test].tolist()
        val += idxs[n_test:n_test + n_val].tolist()
        train += idxs[n_test + n_val:].tolist()
    return train, val, test


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int):
    model.eval()
    loss_sum = correct = total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        preds = logits.argmax(1)
        loss_sum += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[t, p] += 1

    return loss_sum / total, correct / total, confusion


def run_epoch(model, loader, device, criterion, optimizer, scheduler):
    model.train()
    loss_sum = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


def print_confusion(confusion, classes) -> None:
    width = max(len(c) for c in classes) + 2
    print(" " * width + "".join(f"{c[:6]:>8}" for c in classes))
    for name, row in zip(classes, confusion.tolist()):
        print(f"{name:<{width}}" + "".join(f"{v:>8}" for v in row))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--dataset", default="rafdb", help="folder under data/images/")
    p.add_argument("--model", default="efficientnet_b0", choices=sorted(BACKBONES))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--freeze-epochs", type=int, default=2, help="head-only warmup epochs")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4, help="peak LR for the fine-tune stage")
    p.add_argument("--head-lr", type=float, default=1e-3, help="LR during the frozen warmup")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--aug", default="medium", choices=("none", "light", "medium", "strong"))
    p.add_argument("--class-weights", action="store_true", help="reweight the loss for imbalance")
    p.add_argument("--patience", type=int, default=4, help="early stop after N epochs without gain")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="name for this run's output files")
    args = p.parse_args()

    if args.list_models:
        for name, (_, weights, _) in BACKBONES.items():
            print(f"  {name}")
        return

    image_root = IMAGE_DIR / args.dataset
    if not image_root.exists():
        available = sorted(p.name for p in IMAGE_DIR.glob("*") if p.is_dir())
        raise SystemExit(
            f"No images at {image_root}\n"
            f"Available: {available or 'none'}\n"
            f"Run: python extract.py {args.dataset} --save-images"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = build_transforms(args.img_size, args.aug)
    base = datasets.ImageFolder(image_root)
    classes = base.classes
    targets = [s[1] for s in base.samples]

    train_idx, val_idx, test_idx = stratified_split(targets, 0.15, 0.15, args.seed)

    train_ds = Subset(datasets.ImageFolder(image_root, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(image_root, transform=eval_tf), val_idx)
    test_ds = Subset(datasets.ImageFolder(image_root, transform=eval_tf), test_idx)

    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **common)

    counts = Counter(targets)
    print(f"device {device}  |  {len(classes)} classes  |  "
          f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    print("class balance: " + "  ".join(f"{c}={counts[i]}" for i, c in enumerate(classes)))
    print(f"model {args.model}  batch {args.batch_size}  aug {args.aug}  "
          f"lr {args.lr}  head_lr {args.head_lr}\n")

    model = build_model(args.model, len(classes), args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last)

    weight = None
    if args.class_weights:
        freq = torch.tensor([counts[i] for i in range(len(classes))], dtype=torch.float)
        weight = (freq.sum() / (len(classes) * freq)).to(device)
        print("loss weights: " + "  ".join(f"{c}={w:.2f}" for c, w in zip(classes, weight.tolist())) + "\n")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    RUNS_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or f"{args.dataset}_{args.model}_{time.strftime('%H%M%S')}"
    ckpt_path = OUT_DIR / f"finetuned_{tag}.pt"

    history = []
    best_acc, best_epoch, epochs_since_best = 0.0, -1, 0

    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        set_backbone_frozen(model, args.model, frozen)

        if epoch == 0 or epoch == args.freeze_epochs:
            params = [p for p in model.parameters() if p.requires_grad]
            lr = args.head_lr if frozen else args.lr
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
            remaining = (args.freeze_epochs if frozen else args.epochs - args.freeze_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining, 1) * len(train_loader)
            )
            stage = "warmup (head only)" if frozen else "finetune (all layers)"
            trainable = sum(p.numel() for p in params) / 1e6
            print(f"--- epoch {epoch + 1}: {stage}, {trainable:.1f}M trainable params, lr {lr:g} ---")

        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer, scheduler)
        va_loss, va_acc, _ = evaluate(model, val_loader, device, criterion, len(classes))
        secs = time.perf_counter() - t0

        marker = ""
        if va_acc > best_acc:
            best_acc, best_epoch, epochs_since_best = va_acc, epoch, 0
            torch.save({"model": model.state_dict(), "classes": classes,
                        "args": vars(args), "val_acc": va_acc}, ckpt_path)
            marker = "  <- best, saved"
        else:
            epochs_since_best += 1

        print(f"epoch {epoch + 1:>2}/{args.epochs}  "
              f"train {tr_loss:.3f}/{tr_acc:.1%}  val {va_loss:.3f}/{va_acc:.1%}  "
              f"{secs:.0f}s{marker}")
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})

        if epochs_since_best >= args.patience:
            print(f"\nNo improvement for {args.patience} epochs, stopping early.")
            break

    print(f"\nBest val accuracy {best_acc:.1%} at epoch {best_epoch + 1}. Testing that checkpoint.\n")
    model.load_state_dict(torch.load(ckpt_path)["model"])
    te_loss, te_acc, confusion = evaluate(model, test_loader, device, criterion, len(classes))

    print(f"TEST ACCURACY {te_acc:.1%}  (loss {te_loss:.3f})\n")
    print("Confusion matrix - rows are truth, columns are predictions:")
    print_confusion(confusion, classes)

    per_class = confusion.diag().float() / confusion.sum(1).clamp(min=1).float()
    print("\nPer-class recall:")
    for name, acc in zip(classes, per_class.tolist()):
        print(f"  {name:<12}{acc:6.1%}")

    (RUNS_DIR / f"{tag}.json").write_text(json.dumps({
        "args": vars(args), "history": history, "best_val_acc": best_acc,
        "test_acc": te_acc, "classes": classes,
        "confusion": confusion.tolist(),
    }, indent=2))
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Run log:    {RUNS_DIR / (tag + '.json')}")


if __name__ == "__main__":
    main()
