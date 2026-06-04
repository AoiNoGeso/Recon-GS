"""Step 2: Mask dynamic objects using Grounded-SAM2."""

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

# GroundingDINO model weights (downloaded on first use via supervision)
_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_SAM2_MODEL_ID = "facebook/sam2-hiera-large"


def _load_models(device: torch.device):
    from groundingdino.util.inference import load_model_from_hub
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino = load_model_from_hub(_GDINO_MODEL_ID)
    gdino.to(device).eval()

    sam2 = build_sam2(_SAM2_MODEL_ID, device=device)
    predictor = SAM2ImagePredictor(sam2)

    return gdino, predictor


def _detect_boxes(gdino, image_path: Path, prompt: str, device: torch.device):
    from groundingdino.util.inference import load_image, predict

    _, image_tensor = load_image(str(image_path))
    boxes, logits, _ = predict(
        model=gdino,
        image=image_tensor.to(device),
        caption=prompt,
        box_threshold=GROUNDING_DINO_BOX_THRESHOLD,
        text_threshold=GROUNDING_DINO_TEXT_THRESHOLD,
    )
    return boxes, logits


def _boxes_to_pixel(boxes, w: int, h: int) -> np.ndarray:
    """Convert normalised [cx, cy, bw, bh] to [x1, y1, x2, y2] pixel coords."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def mask_frames(frames_dir: Path, masks_dir: Path) -> None:
    """
    For each frame in frames_dir, generate a binary mask (white = dynamic object)
    and save it to masks_dir as a PNG.
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gdino, sam2_predictor = _load_models(device)

    # Join prompts into single GroundingDINO caption
    caption = " . ".join(MASK_PROMPTS) + " ."

    frame_paths = sorted(frames_dir.glob("*.png"))
    for frame_path in frame_paths:
        mask_path = masks_dir / frame_path.name

        image_np = np.array(Image.open(frame_path).convert("RGB"))
        h, w = image_np.shape[:2]

        boxes, _ = _detect_boxes(gdino, frame_path, caption, device)
        pixel_boxes = _boxes_to_pixel(boxes.cpu().numpy(), w, h)

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if len(pixel_boxes) > 0:
            sam2_predictor.set_image(image_np)
            sam_masks, _, _ = sam2_predictor.predict(
                box=pixel_boxes,
                multimask_output=False,
            )
            # sam_masks shape: (N, 1, H, W) or (N, H, W)
            if sam_masks.ndim == 4:
                sam_masks = sam_masks[:, 0]
            for m in sam_masks:
                combined_mask = np.logical_or(combined_mask, m).astype(np.uint8)

        cv2.imwrite(str(mask_path), combined_mask * 255)
