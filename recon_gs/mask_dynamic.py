"""
Step 2: Mask dynamic objects using HuggingFace GroundingDINO + SAM2.

Uses:
  - transformers.AutoModelForZeroShotObjectDetection  (GroundingDINO, no CUDA compilation)
  - sam2.SAM2ImagePredictor                           (SAM2)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from recon_gs.config import (
    GROUNDING_DINO_BOX_THRESHOLD,
    GROUNDING_DINO_TEXT_THRESHOLD,
    MASK_PROMPTS,
)

_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_SAM2_MODEL_ID = "facebook/sam2-hiera-large"


def _load_models(device: torch.device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino_processor = AutoProcessor.from_pretrained(_GDINO_MODEL_ID)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_MODEL_ID).to(device)
    gdino_model.eval()

    sam2_model = build_sam2_hf(_SAM2_MODEL_ID, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return gdino_processor, gdino_model, sam2_predictor


def _detect_boxes(
    processor,
    model,
    image_pil: Image.Image,
    prompt: str,
    device: torch.device,
) -> np.ndarray:
    """
    Run GroundingDINO and return detected boxes as [x1, y1, x2, y2] pixel coords.
    """
    inputs = processor(
        images=image_pil,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=GROUNDING_DINO_BOX_THRESHOLD,
        text_threshold=GROUNDING_DINO_TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],  # (H, W)
    )[0]

    boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy pixel coords
    return boxes


def mask_frames(frames_dir: Path, masks_dir: Path) -> None:
    """
    For each frame in frames_dir, generate a binary mask (white = dynamic object)
    and save it to masks_dir as a PNG.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gdino_processor, gdino_model, sam2_predictor = _load_models(device)

    # GroundingDINO expects a period-separated prompt string
    prompt = " . ".join(MASK_PROMPTS) + " ."

    frame_paths = sorted(frames_dir.glob("*.png"))
    for frame_path in frame_paths:
        mask_path = masks_dir / frame_path.name

        image_pil = Image.open(frame_path).convert("RGB")
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]

        boxes = _detect_boxes(gdino_processor, gdino_model, image_pil, prompt, device)

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if len(boxes) > 0:
            sam2_predictor.set_image(image_np)
            sam_masks, _, _ = sam2_predictor.predict(
                box=boxes,
                multimask_output=False,
            )
            # sam_masks: (N, 1, H, W) or (N, H, W)
            if sam_masks.ndim == 4:
                sam_masks = sam_masks[:, 0]
            for m in sam_masks:
                combined_mask = np.logical_or(combined_mask, m).astype(np.uint8)

        cv2.imwrite(str(mask_path), combined_mask * 255)
