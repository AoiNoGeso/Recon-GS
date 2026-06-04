"""
Step 3: Camera pose estimation using pycolmap Python API (no COLMAP CLI required).

pycolmap >= 3.8 bundles the full COLMAP pipeline as Python bindings:
  - extract_features   : SIFT keypoints
  - match_exhaustive   : exhaustive feature matching
  - incremental_mapping: incremental SfM reconstruction
"""

import sys
from pathlib import Path

import pycolmap


def run_sfm(frames_dir: Path, masks_dir: Path, colmap_dir: Path) -> pycolmap.Reconstruction:
    """
    Run COLMAP SfM pipeline via pycolmap Python API and return the best reconstruction.

    Args:
        frames_dir: Directory containing extracted PNG frames.
        masks_dir:  Directory containing binary masks (white = dynamic, ignored).
        colmap_dir: Output directory for COLMAP database and sparse model.

    Returns:
        pycolmap.Reconstruction of the largest registered model.
    """
    colmap_dir.mkdir(parents=True, exist_ok=True)

    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Feature extraction (SIFT)
    # ------------------------------------------------------------------ #
    print("  [sfm] Feature extraction...")
    pycolmap.extract_features(
        database_path=db_path,
        image_path=frames_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        sift_options=pycolmap.SiftExtractionOptions(
            max_num_features=8192,
        ),
        camera_options=pycolmap.CameraOptions(
            camera_model="SIMPLE_RADIAL",
        ),
    )

    # ------------------------------------------------------------------ #
    # 2. Exhaustive matching
    # ------------------------------------------------------------------ #
    print("  [sfm] Exhaustive matching...")
    pycolmap.match_exhaustive(
        database_path=db_path,
    )

    # ------------------------------------------------------------------ #
    # 3. Incremental reconstruction
    # ------------------------------------------------------------------ #
    print("  [sfm] Incremental reconstruction...")
    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=frames_dir,
        output_path=sparse_dir,
        options=pycolmap.IncrementalPipelineOptions(
            min_num_matches=15,
        ),
    )

    if not maps:
        print("ERROR: pycolmap produced no reconstruction.", file=sys.stderr)
        sys.exit(1)

    # Pick the model with the most registered images
    best_model = max(maps.values(), key=lambda m: m.num_reg_images())
    print(f"  [sfm] Registered {best_model.num_reg_images()} images, "
          f"{best_model.num_points3D()} 3D points")

    # Save the best model
    best_model.write(str(sparse_dir / "0"))

    return best_model
