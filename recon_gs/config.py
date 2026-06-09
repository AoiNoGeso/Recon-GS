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

# --------------------------------------------------------------------------- #
# Surface masking for mesh export (Grounded-SAM2 prompt-based)
# --------------------------------------------------------------------------- #
# Set each flag to True to exclude that surface from TSDF integration.
# Prompts are passed to GroundingDINO; SAM2 refines the detected regions.

MESH_MASK_FLOOR: bool = True     # indoor floor
MESH_MASK_CEILING: bool = True   # indoor ceiling
MESH_MASK_GROUND: bool = False   # outdoor ground / road
MESH_MASK_SKY: bool = False      # outdoor sky

MESH_FLOOR_PROMPTS: list[str] = ["floor", "carpet", "tile floor", "wooden floor", "rug"]
MESH_CEILING_PROMPTS: list[str] = ["ceiling"]
MESH_GROUND_PROMPTS: list[str] = ["ground", "road", "pavement", "sidewalk", "grass"]
MESH_SKY_PROMPTS: list[str] = ["sky"]

# GroundingDINO thresholds for surface masking (can differ from dynamic-object masking)
MESH_GDINO_BOX_THRESHOLD: float = 0.25
MESH_GDINO_TEXT_THRESHOLD: float = 0.20

# --------------------------------------------------------------------------- #
# RANSAC plane filling
# Replace excluded surface regions with fitted flat planes.
# --------------------------------------------------------------------------- #
MESH_FILL_PLANES: bool = True              # enable/disable plane filling
MESH_PLANE_RANSAC_DISTANCE: float = 0.05  # inlier distance threshold [m]
MESH_PLANE_RANSAC_ITERATIONS: int = 1000  # RANSAC iterations
MESH_PLANE_MIN_POINTS: int = 500          # skip plane if fewer masked points remain
MESH_PLANE_MAX_PLANES: int = 4            # max planes to extract (iterative RANSAC)
MESH_PLANE_PIXEL_STRIDE: int = 8          # collect every N-th masked pixel per frame
MESH_PLANE_VOXEL_SIZE: float = 0.05       # voxel size for downsampling before RANSAC [m]
