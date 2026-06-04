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

# Adaptive Density Control (ADC)
DENSIFY_FROM_ITER: int = 500
DENSIFY_UNTIL_ITER: int = 15_000
DENSIFICATION_INTERVAL: int = 100
OPACITY_RESET_INTERVAL: int = 3_000
DENSIFY_GRAD_THRESHOLD: float = 0.0002
PERCENT_DENSE: float = 0.01      # scale ratio threshold: clone vs split
MIN_OPACITY: float = 0.005       # prune gaussians below this opacity
MAX_GAUSSIANS: int = 300_000     # safety cap
