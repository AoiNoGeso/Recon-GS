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
DENSIFY_GRAD_THRESHOLD: float = (
    0.00001  # gsplat returns NDC-normalized grads (smaller than pixel-space)
)
PERCENT_DENSE: float = 0.01  # scale ratio threshold: clone vs split
MIN_OPACITY: float = 0.005  # prune gaussians below this opacity
MAX_GAUSSIANS: int = 300_000  # safety cap

# --------------------------------------------------------------------------- #
# Mesh export (TSDF Fusion)
# --------------------------------------------------------------------------- #

# TSDF volume parameters
MESH_VOXEL_SIZE: float = 0.05          # voxel size [m]; smaller = finer mesh
MESH_MAX_DEPTH: float = 8.0            # depth cutoff [m]; set to scene diagonal
MESH_MIN_CLUSTER_TRIANGLES: int = 500  # remove isolated clusters below this size

# Horizontal surface masking (normal-based)
# Surfaces whose world-space normal is within MESH_SURFACE_ANGLE_DEG of the
# up/down axis are masked before TSDF integration.
MESH_SURFACE_ANGLE_DEG: float = 30.0  # half-cone angle for horizontal surface detection

MESH_MASK_FLOOR: bool = True     # mask upward-facing horizontal surfaces (indoor floor)
MESH_MASK_CEILING: bool = True   # mask downward-facing horizontal surfaces (indoor ceiling)
MESH_MASK_GROUND: bool = False   # same as floor, intended for outdoor ground
MESH_MASK_SKY: bool = False      # mask sky regions (alpha-based + color heuristic)

# World-space up vector (COLMAP convention: Y points down → world-up = [0,-1,0])
# Change to [0, 1, 0] if your reconstruction has Y pointing up.
MESH_WORLD_UP: list[float] = [0.0, -1.0, 0.0]

# Sky mask heuristics (only used when MESH_MASK_SKY=True)
MESH_SKY_ALPHA_THRESHOLD: float = 0.1  # pixels with alpha below this → no-hit → sky candidate
MESH_SKY_TOP_FRACTION: float = 0.5     # only apply sky mask to the top N fraction of the image
