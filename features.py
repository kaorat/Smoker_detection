"""
features.py — HOG + LBP + HSV color + Face-ROI feature extractor.

Used by the classical (non-deep) pipeline:
    Image -> resize 128x128 -> [HOG | LBP | HSV-hist | Face-ROI features] -> vector
"""

import warnings

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = (128, 128)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS

COLOR_BINS = 32

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
_PROFILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


def _load_image(path):
    img = cv2.imread(path)
    if img is None:
        warnings.warn(f"Could not load image: {path}")
    return img


def _resize(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _detect_face_roi(gray):
    for cascade in (_FACE_CASCADE, _PROFILE_CASCADE):
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces):
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return faces[0]
    return None


def extract_hog(img_gray):
    feats = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)


def extract_lbp(img_gray):
    lbp = local_binary_pattern(
        img_gray, LBP_N_POINTS, LBP_RADIUS, method="uniform"
    )
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_color_histogram(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [COLOR_BINS], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)

    smoke_mask = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
    smoke_ratio = np.array([smoke_mask.sum() / smoke_mask.size], dtype=np.float32)

    ember_mask = cv2.inRange(hsv, (5, 150, 150), (25, 255, 255))
    ember_ratio = np.array([ember_mask.sum() / ember_mask.size], dtype=np.float32)

    feats.append(smoke_ratio)
    feats.append(ember_ratio)
    return np.concatenate(feats).astype(np.float32)


def extract_face_roi_features(img_bgr, img_gray):
    roi = _detect_face_roi(img_gray)
    if roi is not None:
        x, y, w, h = roi
        y_mouth = y + int(h * 0.55)
        mouth_bgr = img_bgr[y_mouth: y + h, x: x + w]
        mouth_gray = img_gray[y_mouth: y + h, x: x + w]

        if mouth_bgr.size > 0 and mouth_gray.size > 0:
            mouth_bgr_r = cv2.resize(mouth_bgr, (64, 32), interpolation=cv2.INTER_AREA)
            mouth_gray_r = cv2.resize(mouth_gray, (64, 32), interpolation=cv2.INTER_AREA)
            lbp_f = extract_lbp(mouth_gray_r)
            col_f = extract_color_histogram(mouth_bgr_r)
            return np.concatenate([lbp_f, col_f]).astype(np.float32)

    dummy_gray = np.zeros((32, 64), dtype=np.uint8)
    dummy_bgr = np.zeros((32, 64, 3), dtype=np.uint8)
    return np.concatenate(
        [extract_lbp(dummy_gray), extract_color_histogram(dummy_bgr)]
    ).astype(np.float32)


def extract_features(path):
    """Returns a 1-D float32 feature vector or None on read failure."""
    img_bgr = _load_image(path)
    if img_bgr is None:
        return None

    img_bgr = _resize(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hog_f = extract_hog(img_gray)
    lbp_f = extract_lbp(img_gray)
    col_f = extract_color_histogram(img_bgr)
    face_f = extract_face_roi_features(img_bgr, img_gray)

    return np.concatenate([hog_f, lbp_f, col_f, face_f]).astype(np.float32)
