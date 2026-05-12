"""
train_resnet.py — Local port of the ResNet-50 transfer-learning notebook.

Two-phase fine-tuning:
    Phase 1 — freeze backbone, train new classifier head        (5 epochs)
    Phase 2 — unfreeze, fine-tune whole network at lower LR    (10 epochs)

Dataset layout (explicit Train/Val/Test split):
    <data_dir>/
        Training/    Smoking/  Not_Smoking/
        Validation/  Smoking/  Not_Smoking/
        Testing/     Smoking/  Not_Smoking/

Usage:
    python train_resnet.py --data_dir ./dataset --model_out ./models/resnet50.pth
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--model_out", default="./models/resnet50.pth")
    parser.add_argument("--plot_dir", default="./eval_plots")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--phase1_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Set 0 on Windows unless you wrap in __main__")
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    IMG_SIZE = 224
    LR = 1e-4
    LR_BACKBONE = 1e-5
    CLASS_NAMES = ["Not_Smoking", "Smoking"]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "Training"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "Validation"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "Testing"), transform=eval_tf)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError(
            f"class_to_idx mismatch across splits: "
            f"train={train_ds.class_to_idx} val={val_ds.class_to_idx} test={test_ds.class_to_idx}"
        )
    print(f"Classes: {train_ds.class_to_idx}", flush=True)

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}", flush=True)

    model = build_model(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total_p:,}", flush=True)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")

    # Phase 1 — head only
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    print("\n=== Phase 1: classifier head ===", flush=True)
    for ep in range(1, args.phase1_epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate_model(model, val_loader, criterion, device)
        sched.step(vl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), args.model_out)
        print(f"Ep {ep:02d}/{args.phase1_epochs} "
              f"train L={tl:.4f} A={ta:.4f} | val L={vl:.4f} A={va:.4f}", flush=True)

    # Phase 2 — unfreeze
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": LR_BACKBONE},
        {"params": model.fc.parameters(), "lr": LR},
    ])
    phase2_epochs = args.epochs - args.phase1_epochs
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    print("\n=== Phase 2: fine-tune backbone ===", flush=True)
    for ep in range(1, phase2_epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate_model(model, val_loader, criterion, device)
        sched.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), args.model_out)
        print(f"Ep {ep+args.phase1_epochs:02d}/{args.epochs} "
              f"train L={tl:.4f} A={ta:.4f} | val L={vl:.4f} A={va:.4f}", flush=True)

    print(f"\nBest checkpoint saved -> {args.model_out}", flush=True)

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    model.eval()

    @torch.no_grad()
    def collect_preds(loader):
        all_y, all_p, all_pr = [], [], []
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pr = torch.softmax(out, 1).cpu().numpy()
            all_y.extend(y.numpy())
            all_p.extend(out.argmax(1).cpu().numpy())
            all_pr.extend(pr)
        return np.array(all_y), np.array(all_p), np.array(all_pr)

    y_true, y_pred, y_probs = collect_preds(test_loader)
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    ap = average_precision_score(y_true, y_probs[:, 1])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n" + "=" * 50)
    print(" RESNET-50 — TEST SET RESULTS")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"AUC-ROC    : {roc_auc:.4f}")
    print(f"AP         : {ap:.4f}")
    print(f"MCC        : {mcc:.4f}")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}", flush=True)

    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    ep_range = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ep_range, history["train_loss"], "o-", label="train")
    axes[0].plot(ep_range, history["val_loss"], "o-", label="val")
    axes[0].axvline(args.phase1_epochs + 0.5, color="grey", linestyle="--", alpha=0.5)
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep_range, history["train_acc"], "o-", label="train")
    axes[1].plot(ep_range, history["val_acc"], "o-", label="val")
    axes[1].axvline(args.phase1_epochs + 0.5, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("ResNet-50 Training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "resnet_training.png"), dpi=140)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("ResNet-50 — Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "resnet_confusion.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ResNet-50 — ROC", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "resnet_roc.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, color="darkorange", label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="darkorange")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("ResNet-50 — PR Curve", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "resnet_pr.png"), dpi=140)
    plt.close(fig)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "test_accuracy": acc,
        "auc_roc": roc_auc,
        "f1_score": f1,
    }, args.model_out)

    with open(Path(args.model_out).with_suffix(".txt"), "w") as f:
        f.write("RESNET-50 SMOKING DETECTOR — REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        f.write(f"\nAccuracy   : {acc:.4f}\n")
        f.write(f"Precision  : {prec:.4f}\n")
        f.write(f"Recall     : {rec:.4f}\n")
        f.write(f"Specificity: {spec:.4f}\n")
        f.write(f"F1         : {f1:.4f}\n")
        f.write(f"AUC-ROC    : {roc_auc:.4f}\n")
        f.write(f"AP         : {ap:.4f}\n")
        f.write(f"MCC        : {mcc:.4f}\n")
        f.write(f"TP={tp} TN={tn} FP={fp} FN={fn}\n")

    print(f"\nPlots -> {args.plot_dir}/  (resnet_*.png)")
    print(f"Report -> {Path(args.model_out).with_suffix('.txt')}")


if __name__ == "__main__":
    main()
