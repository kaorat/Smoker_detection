"""
predict.py — CLI inference using either the classical or ResNet-50 model.

Usage:
    python predict.py --model classical --input photo.jpg
    python predict.py --model resnet    --input ./test_images/
"""

import argparse
import os
import pickle
from glob import glob
from pathlib import Path

import numpy as np

CLASS_NAMES = ["Not_Smoking", "Smoking"]
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def collect(target):
    if os.path.isdir(target):
        paths = []
        for ext in EXTS:
            paths += glob(os.path.join(target, "**", ext), recursive=True)
        return sorted(set(paths))
    return [target]


def predict_classical(model_path, paths):
    from features import extract_features
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    out = []
    for p in paths:
        feat = extract_features(p)
        if feat is None:
            out.append((p, None, None))
            continue
        prob = model.predict_proba(feat.reshape(1, -1))[0]
        pred = int(prob.argmax())
        out.append((p, pred, prob))
    return out


def predict_resnet(model_path, paths):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    img_size = ckpt.get("img_size", 224)

    backbone = models.resnet50(weights=None)
    in_f = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
    )
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    backbone.load_state_dict(state)
    backbone = backbone.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                out.append((p, None, None))
                continue
            x = tf(img).unsqueeze(0).to(device)
            prob = torch.softmax(backbone(x), 1)[0].cpu().numpy()
            out.append((p, int(prob.argmax()), prob))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classical", "resnet"], required=True)
    parser.add_argument("--model_path", default=None,
                        help="Overrides default (./models/classical.pkl or ./models/resnet50.pth)")
    parser.add_argument("--input", required=True, help="Image file or folder")
    args = parser.parse_args()

    default = {"classical": "./models/classical.pkl", "resnet": "./models/resnet50.pth"}
    path = args.model_path or default[args.model]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    paths = collect(args.input)
    if not paths:
        print("No images found.")
        return

    runner = predict_classical if args.model == "classical" else predict_resnet
    results = runner(path, paths)

    print(f"\n{'File':<45}  {'P(Smoking)':>11}  {'Prediction':>14}")
    print("-" * 76)
    for p, pred, prob in results:
        name = Path(p).name[:44]
        if pred is None:
            print(f"{name:<45}  {'ERROR':>11}  {'-':>14}")
            continue
        ps = prob[1]
        label = "SMOKING" if pred == 1 else "NOT_SMOKING"
        print(f"{name:<45}  {ps:>11.3f}  {label:>14}")


if __name__ == "__main__":
    main()
