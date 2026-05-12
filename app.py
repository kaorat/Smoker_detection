"""
app.py — Flask server exposing BOTH the classical pipeline and the ResNet-50.

Endpoints:
    GET  /                       -> static/index.html
    GET  /health                 -> which models are loaded
    POST /predict                -> form/json image  + ?model=classical|resnet|both
                                    -> { label, confidence, probabilities, model, inference_ms }

Run:
    python app.py
        --classical_model ./models/classical.pkl
        --resnet_model    ./models/resnet50.pth
        --port 5000
"""

import argparse
import base64
import io
import os
import pickle
import sys
import tempfile
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

parser = argparse.ArgumentParser()
parser.add_argument("--classical_model", default="./models/classical.pkl")
parser.add_argument("--resnet_model", default="./models/resnet50.pth")
parser.add_argument("--enable_dino", action="store_true",
                    help="Load Grounding DINO + SAM at startup (slow first run, large download)")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ["Not_Smoking", "Smoking"]
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_BYTES = 10 * 1024 * 1024

# ── Load classical pipeline ───────────────────────────────────────────────────
CLASSICAL = None
CLASSICAL_META = {}
if os.path.exists(args.classical_model):
    with open(args.classical_model, "rb") as f:
        CLASSICAL = pickle.load(f)
    pca = CLASSICAL.named_steps.get("pca", None)
    CLASSICAL_META = {
        "path": args.classical_model,
        "pca_components": int(pca.n_components_) if pca is not None else None,
    }
    print(f"[classical] loaded ({CLASSICAL_META})")
else:
    print(f"[classical] not found: {args.classical_model} (skipping)")

# ── Load ResNet-50 ────────────────────────────────────────────────────────────
RESNET = None
RESNET_DEVICE = None
RESNET_TF = None
RESNET_META = {}
if os.path.exists(args.resnet_model):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    RESNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.resnet_model, map_location=RESNET_DEVICE, weights_only=False)
    img_size = ckpt.get("img_size", 224) if isinstance(ckpt, dict) else 224

    backbone = models.resnet50(weights=None)
    in_f = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
    )
    state = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    backbone.load_state_dict(state)
    RESNET = backbone.to(RESNET_DEVICE).eval()

    RESNET_TF = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    RESNET_META = {
        "path": args.resnet_model,
        "device": str(RESNET_DEVICE),
        "img_size": img_size,
        "test_accuracy": ckpt.get("test_accuracy") if isinstance(ckpt, dict) else None,
        "auc_roc": ckpt.get("auc_roc") if isinstance(ckpt, dict) else None,
    }
    print(f"[resnet] loaded ({RESNET_META})")
else:
    print(f"[resnet] not found: {args.resnet_model} (skipping)")

# ── Load DINO + SAM (optional, large) ─────────────────────────────────────────
DINO_SAM = None
DINO_SAM_META = {}
if args.enable_dino:
    try:
        from dino_sam_predict import DinoSamPredictor, draw_overlay
        DINO_SAM = DinoSamPredictor()
        DINO_SAM_META = {
            "dino_id": DINO_SAM.cfg.dino_id,
            "sam_id": DINO_SAM.cfg.sam_id,
            "device": str(DINO_SAM.device),
            "box_threshold": DINO_SAM.cfg.box_threshold,
        }
        print(f"[dino_sam] enabled ({DINO_SAM_META})")
    except Exception as e:
        print(f"[dino_sam] failed to load: {e}")
        DINO_SAM = None
else:
    draw_overlay = None
    print("[dino_sam] skipped (pass --enable_dino to load)")

if CLASSICAL is None and RESNET is None and DINO_SAM is None:
    print("\nNo models loaded. Train at least one before starting the server.")
    sys.exit(1)

# Classical feature extractor (only import if needed)
if CLASSICAL is not None:
    from features import extract_features
else:
    extract_features = None

app = Flask(__name__, static_folder="static")
CORS(app)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _classical_predict(tmp_path):
    feat = extract_features(tmp_path)
    if feat is None:
        return None
    feat = feat.reshape(1, -1)
    proba = CLASSICAL.predict_proba(feat)[0]
    pred = int(proba.argmax())
    return {
        "label": "smoking" if pred == 1 else "not_smoking",
        "confidence": round(float(proba[pred]), 4),
        "probabilities": {
            "smoking": round(float(proba[1]), 4),
            "not_smoking": round(float(proba[0]), 4),
        },
    }


def _resnet_predict(img_bytes):
    import torch
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = RESNET_TF(img).unsqueeze(0).to(RESNET_DEVICE)
    with torch.no_grad():
        proba = torch.softmax(RESNET(x), 1)[0].cpu().numpy()
    pred = int(proba.argmax())
    return {
        "label": "smoking" if pred == 1 else "not_smoking",
        "confidence": round(float(proba[pred]), 4),
        "probabilities": {
            "smoking": round(float(proba[1]), 4),
            "not_smoking": round(float(proba[0]), 4),
        },
    }


def _dino_predict(img_bytes):
    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _Image

    img_pil = _Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result = DINO_SAM.predict(img_pil)

    cig = result["cigarette"]
    smk = result["smoke"]
    max_cig = max(cig["scores"]) if cig["scores"] else 0.0
    label = "smoking" if result["is_smoking"] else "not_smoking"

    # Annotated overlay (BGR -> PNG -> base64)
    img_bgr = _cv2.cvtColor(_np.array(img_pil), _cv2.COLOR_RGB2BGR)
    overlay_bgr = draw_overlay(img_bgr, result)
    ok, buf = _cv2.imencode(".png", overlay_bgr)
    overlay_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    return {
        "label": label,
        "confidence": round(float(max_cig), 4) if result["is_smoking"] else 1.0,
        "probabilities": {
            "smoking": round(float(max_cig), 4),
            "not_smoking": round(float(1.0 - max_cig), 4),
        },
        "detections": {
            "cigarettes": [
                {"box": [round(v, 1) for v in b], "score": round(float(s), 4)}
                for b, s in zip(cig["boxes"], cig["scores"])
            ],
            "smoke": [
                {"box": [round(v, 1) for v in b], "score": round(float(s), 4)}
                for b, s in zip(smk["boxes"], smk["scores"])
            ],
        },
        "overlay_b64": overlay_b64,
    }


def _read_image_payload():
    """Returns raw image bytes from either multipart upload or JSON base64."""
    if request.content_type and "multipart" in request.content_type:
        if "image" not in request.files:
            return None, ("No image file in request", 400)
        f = request.files["image"]
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext and ext not in ALLOWED_EXTS:
            return None, (f"Unsupported file type: {ext}", 400)
        return f.read(), None
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return None, ("Expected JSON with 'image' key (base64)", 400)
    try:
        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64), None
    except Exception as e:
        return None, (f"Invalid base64: {e}", 400)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "classical": CLASSICAL_META if CLASSICAL is not None else None,
            "resnet": RESNET_META if RESNET is not None else None,
            "dino": DINO_SAM_META if DINO_SAM is not None else None,
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    which = request.args.get("model", "both").lower()
    valid = {"classical", "resnet", "dino", "both", "all"}
    if which not in valid:
        return jsonify({"error": f"Unknown model '{which}'"}), 400

    img_bytes, err = _read_image_payload()
    if err:
        return jsonify({"error": err[0]}), err[1]
    if len(img_bytes) > MAX_BYTES:
        return jsonify({"error": "Image too large (max 10 MB)"}), 413

    results = {}
    timings = {}

    if which in {"classical", "both", "all"}:
        if CLASSICAL is None:
            results["classical"] = {"error": "Classical model not loaded"}
        else:
            t0 = time.time()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            try:
                r = _classical_predict(tmp_path)
            finally:
                os.unlink(tmp_path)
            timings["classical"] = round((time.time() - t0) * 1000, 1)
            if r is None:
                results["classical"] = {"error": "Could not process image"}
            else:
                r["inference_ms"] = timings["classical"]
                results["classical"] = r

    if which in {"resnet", "both", "all"}:
        if RESNET is None:
            results["resnet"] = {"error": "ResNet model not loaded"}
        else:
            t0 = time.time()
            try:
                r = _resnet_predict(img_bytes)
            except Exception as e:
                r = None
                results["resnet"] = {"error": f"Inference error: {e}"}
            timings["resnet"] = round((time.time() - t0) * 1000, 1)
            if r is not None:
                r["inference_ms"] = timings["resnet"]
                results["resnet"] = r

    if which in {"dino", "all"}:
        if DINO_SAM is None:
            results["dino"] = {"error": "DINO+SAM not loaded (start app with --enable_dino)"}
        else:
            t0 = time.time()
            try:
                r = _dino_predict(img_bytes)
            except Exception as e:
                r = None
                results["dino"] = {"error": f"Inference error: {e}"}
            timings["dino"] = round((time.time() - t0) * 1000, 1)
            if r is not None:
                r["inference_ms"] = timings["dino"]
                results["dino"] = r

    # Backwards-compatible flat payload if only one model was requested
    if which not in {"both", "all"}:
        single = results.get(which, {})
        single["model"] = which
        return jsonify(single)

    return jsonify({"models": results})


if __name__ == "__main__":
    print(f"\nServer  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
