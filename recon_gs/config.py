"""
Pipeline configuration constants.
Edit MASK_PROMPTS to control which dynamic objects are masked out before SfM/3DGS.
"""

# Dynamic object categories to mask (Grounded-SAM2 / GroundingDINO format).
# Sourced from Vid2Sim's DEVA segmentation configuration.
MASK_PROMPTS: list[str] = [
    "person",
    "pedestrian",
    "cyclist",
    "child",
    "adult",
    "bag",
    "backpack",
    "handbag",
    "suitcase",
    "hat",
    "shoes",
    "cloth",
    "wheelchair",
]

# GroundingDINO detection thresholds
GROUNDING_DINO_BOX_THRESHOLD: float = 0.3
GROUNDING_DINO_TEXT_THRESHOLD: float = 0.25

# Frame extraction
EXTRACT_FPS: int = 15

# 3DGS training
TRAIN_ITERATIONS: int = 30_000
