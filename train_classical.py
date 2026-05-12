"""
train_classical.py — Classical (non-deep) pipeline with explicit Train/Val/Test split.

    HOG + LBP + HSV + Face-ROI  ->  Scaler  ->  PCA(95%)  ->  RBF-SVM

Dataset layout:
    <data_dir>/
        Training/    Smoking/  Not_Smoking/    (augmented)
        Validation/  Smoking/  Not_Smoking/
        Testing/     Smoking/  Not_Smoking/

GridSearchCV uses PredefinedSplit so hyperparameters are picked using the
real Validation set (one fold). Final report is on Testing.

Usage:
    python train_classical.py --data_dir ./dataset --model_out ./models/classical.pkl
    python train_classical.py --data_dir ./dataset --model_out ./models/classical.pkl --fast
"""

import argparse
import os
import pickle
import time
import warnings
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from features import extract_features

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./dataset")
parser.add_argument("--model_out", default="./models/classical.pkl")
parser.add_argument("--fast", action="store_true",
                    help="Skip GridSearchCV (uses C=10, gamma=scale)")
args = parser.parse_args()

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def collect(folder, label):
    paths = []
    for ext in EXTS:
        paths += glob(os.path.join(folder, "**", ext), recursive=True)
        paths += glob(os.path.join(folder, "**", ext.upper()), recursive=True)
    return [(p, label) for p in sorted(set(paths))]


def extract_split(name):
    smoking = collect(os.path.join(args.data_dir, name, "Smoking"), 1)
    not_smoking = collect(os.path.join(args.data_dir, name, "Not_Smoking"), 0)
    samples = smoking + not_smoking
    if not samples:
        raise FileNotFoundError(
            f"No images under {args.data_dir}/{name}. "
            "Expected Smoking/ and Not_Smoking/ subfolders."
        )
    paths, labels = zip(*samples)
    print(f"\n[{name}] {int(sum(labels))} Smoking | {int(len(labels)-sum(labels))} Not_Smoking")
    X, y, bad = [], [], []
    for p, lbl in tqdm(zip(paths, labels), total=len(paths), ncols=80, desc=name):
        feat = extract_features(p)
        if feat is None:
            bad.append(p)
            continue
        X.append(feat)
        y.append(lbl)
    if bad:
        print(f"  skipped {len(bad)} unreadable images")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


X_train, y_train = extract_split("Training")
X_val, y_val = extract_split("Validation")
X_test, y_test = extract_split("Testing")

print(f"\nFeature shapes  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
    ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
])

if args.fast:
    print("\nFast mode -> C=10, gamma=scale")
    pipeline.set_params(svm__C=10, svm__gamma="scale")
    pipeline.fit(X_train, y_train)
    best = pipeline
    best_params, best_cv = {"svm__C": 10, "svm__gamma": "scale"}, None
else:
    # PredefinedSplit: train on Training (-1) and validate on Validation (0)
    X_grid = np.concatenate([X_train, X_val])
    y_grid = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=np.int8),
        np.zeros(len(X_val), dtype=np.int8),
    ])
    ps = PredefinedSplit(test_fold)

    grid = GridSearchCV(
        pipeline,
        {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", "auto", 0.001, 0.01]},
        cv=ps,
        scoring="balanced_accuracy",
        n_jobs=1,
        verbose=3,
        refit=False,
    )
    print("\nGridSearchCV (PredefinedSplit: real Validation fold) ...")
    t0 = time.time()
    grid.fit(X_grid, y_grid)
    print(f"Done in {time.time()-t0:.1f}s")
    best_params = grid.best_params_
    best_cv = grid.best_score_
    print(f"Best params  : {best_params}")
    print(f"Val balanced : {best_cv:.4f}")

    # Refit on Training only with the chosen params (so the model never saw Validation)
    best = pipeline.set_params(**best_params)
    best.fit(X_train, y_train)

# ── Validation report (sanity check) ──────────────────────────────────────────
y_val_pred = best.predict(X_val)
y_val_proba = best.predict_proba(X_val)[:, 1]
val_report = classification_report(y_val, y_val_pred, target_names=["Not_Smoking", "Smoking"])
val_cm = confusion_matrix(y_val, y_val_pred)
val_roc = roc_auc_score(y_val, y_val_proba)
val_bal = balanced_accuracy_score(y_val, y_val_pred)

# ── Test report (final) ───────────────────────────────────────────────────────
y_test_pred = best.predict(X_test)
y_test_proba = best.predict_proba(X_test)[:, 1]
test_report = classification_report(y_test, y_test_pred, target_names=["Not_Smoking", "Smoking"])
test_cm = confusion_matrix(y_test, y_test_pred)
test_roc = roc_auc_score(y_test, y_test_proba)
test_bal = balanced_accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 60)
print(" CLASSICAL — VALIDATION SET")
print("=" * 60)
print(val_report)
print(f"Confusion matrix:\n{val_cm}")
print(f"ROC-AUC           : {val_roc:.4f}")
print(f"Balanced accuracy : {val_bal:.4f}")

print("\n" + "=" * 60)
print(" CLASSICAL — TEST SET (final)")
print("=" * 60)
print(test_report)
print(f"Confusion matrix:\n{test_cm}")
print(f"ROC-AUC           : {test_roc:.4f}")
print(f"Balanced accuracy : {test_bal:.4f}")

pca = best.named_steps["pca"]
print(f"\nPCA: {X_train.shape[1]} -> {pca.n_components_} components")

Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
with open(args.model_out, "wb") as f:
    pickle.dump(best, f)
print(f"\nSaved -> {args.model_out}")

txt = Path(args.model_out).with_suffix(".txt")
with open(txt, "w") as f:
    f.write("CLASSICAL SMOKING DETECTOR — REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")
    if best_cv is not None:
        f.write(f"\nBest params (chosen on Validation): {best_params}\n")
        f.write(f"Validation balanced accuracy      : {best_cv:.4f}\n")
    f.write("\n--- VALIDATION ---\n")
    f.write(val_report)
    f.write(f"Confusion matrix:\n{val_cm}\n")
    f.write(f"ROC-AUC           : {val_roc:.4f}\n")
    f.write(f"Balanced accuracy : {val_bal:.4f}\n")
    f.write("\n--- TEST (final) ---\n")
    f.write(test_report)
    f.write(f"Confusion matrix:\n{test_cm}\n")
    f.write(f"ROC-AUC           : {test_roc:.4f}\n")
    f.write(f"Balanced accuracy : {test_bal:.4f}\n")
print(f"Report -> {txt}")
