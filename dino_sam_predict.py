"""
dino_sam_predict.py — Grounding DINO + SAM smoking detector.

Uses HuggingFace ports (no custom CUDA build needed on Windows):
  - IDEA-Research/grounding-dino-tiny  (text-prompted object detection)
  - facebook/sam-vit-base              (segment anything from boxes)

Pipeline:
  Image -> DINO("cigarette") -> boxes -> SAM(boxes) -> masks
       -> DINO("smoke")      -> boxes -> SAM(boxes) -> masks
       -> is_smoking = (any cigarette detected)
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


@dataclass
class DinoSamConfig:
    dino_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_id: str = "facebook/sam-vit-base"
    cigarette_prompt: str = "a cigarette."
    smoke_prompt: str = "smoke."
    detect_smoke: bool = True
    box_threshold: float = 0.30
    text_threshold: float = 0.25


CIG_COLOR = (0, 200, 255)   # BGR yellow-orange
SMK_COLOR = (180, 180, 180)


class DinoSamPredictor:
    def __init__(self, cfg=None, device=None):
        self.cfg = cfg or DinoSamConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[dino_sam] loading DINO {self.cfg.dino_id} ...", flush=True)
        self.dino_processor = AutoProcessor.from_pretrained(self.cfg.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.cfg.dino_id
        ).to(self.device).eval()

        print(f"[dino_sam] loading SAM {self.cfg.sam_id} ...", flush=True)
        self.sam_processor = SamProcessor.from_pretrained(self.cfg.sam_id)
        self.sam_model = SamModel.from_pretrained(self.cfg.sam_id).to(self.device).eval()
        print("[dino_sam] ready.", flush=True)

    # ── DINO detection ────────────────────────────────────────────────────
    @torch.no_grad()
    def _detect(self, image_pil: Image.Image, prompt: str):
        inputs = self.dino_processor(
            images=image_pil, text=prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.dino_model(**inputs)
        # HuggingFace API changed names between versions; try both.
        post = self.dino_processor.post_process_grounded_object_detection
        try:
            results = post(
                outputs,
                inputs.input_ids,
                threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[image_pil.size[::-1]],
            )[0]
        except TypeError:
            results = post(
                outputs,
                inputs.input_ids,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[image_pil.size[::-1]],
            )[0]
        boxes = results["boxes"].detach().cpu().numpy().tolist()
        scores = results["scores"].detach().cpu().numpy().tolist()
        return boxes, scores

    # ── SAM segmentation ──────────────────────────────────────────────────
    @torch.no_grad()
    def _segment(self, image_pil: Image.Image, boxes_xyxy):
        if not boxes_xyxy:
            return []
        inputs = self.sam_processor(
            image_pil,
            input_boxes=[boxes_xyxy],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.sam_model(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]   # tensor (num_boxes, 1, H, W)
        return [m[0].numpy().astype(bool) for m in masks]

    # ── Public predict ────────────────────────────────────────────────────
    def predict(self, image_pil: Image.Image):
        cig_boxes, cig_scores = self._detect(image_pil, self.cfg.cigarette_prompt)
        cig_masks = self._segment(image_pil, cig_boxes) if cig_boxes else []

        smoke_boxes, smoke_scores, smoke_masks = [], [], []
        if self.cfg.detect_smoke:
            smoke_boxes, smoke_scores = self._detect(image_pil, self.cfg.smoke_prompt)
            smoke_masks = self._segment(image_pil, smoke_boxes) if smoke_boxes else []

        return {
            "is_smoking": bool(cig_boxes),
            "cigarette": {"boxes": cig_boxes, "scores": cig_scores},
            "smoke": {"boxes": smoke_boxes, "scores": smoke_scores},
            "cig_masks": cig_masks,
            "smoke_masks": smoke_masks,
        }


def draw_overlay(image_bgr: np.ndarray, result: dict) -> np.ndarray:
    overlay = image_bgr.copy()

    def _draw(masks, boxes, color, label):
        for i, mask in enumerate(masks):
            if mask.shape[:2] == overlay.shape[:2]:
                colored = np.zeros_like(overlay)
                colored[mask] = color
                cv2.addWeighted(colored, 0.45, overlay, 1.0, 0, overlay)
            if i < len(boxes):
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _draw(result["cig_masks"], result["cigarette"]["boxes"], CIG_COLOR, "cigarette")
    _draw(result["smoke_masks"], result["smoke"]["boxes"], SMK_COLOR, "smoke")

    status = "SMOKING" if result["is_smoking"] else "NOT SMOKING"
    banner = (0, 0, 220) if result["is_smoking"] else (0, 180, 0)
    cv2.rectangle(overlay, (0, 0), (220, 36), banner, -1)
    cv2.putText(overlay, status, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return overlay


if __name__ == "__main__":
    import sys
    pred = DinoSamPredictor()
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path:
        print("Usage: python dino_sam_predict.py <image_path>")
        sys.exit(1)
    img = Image.open(img_path).convert("RGB")
    r = pred.predict(img)
    print(f"is_smoking: {r['is_smoking']}")
    print(f"cigarette boxes: {len(r['cigarette']['boxes'])}  scores: {r['cigarette']['scores']}")
    print(f"smoke boxes    : {len(r['smoke']['boxes'])}    scores: {r['smoke']['scores']}")
