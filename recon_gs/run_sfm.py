"""
Step 3: Camera pose estimation using pycolmap Python API (no COLMAP CLI required).
"""

import sys
from pathlib import Path

import pycolmap


def run_sfm(frames_dir: Path, masks_dir: Path, colmap_dir: Path) -> pycolmap.Reconstruction:
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
        extraction_options=pycolmap.FeatureExtractionOptions(),
        device=pycolmap.Device.cpu,
    )

    # ------------------------------------------------------------------ #
    # 2. Exhaustive matching
    # ------------------------------------------------------------------ #
    print("  [sfm] Exhaustive matching...")
    pycolmap.match_exhaustive(
        database_path=db_path,
        device=pycolmap.Device.cpu,
    )

    # ------------------------------------------------------------------ #
    # 3. Incremental reconstruction
    # ------------------------------------------------------------------ #
    print("  [sfm] Incremental reconstruction...")
    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=frames_dir,
        output_path=sparse_dir,
        options=pycolmap.IncrementalPipelineOptions(min_num_matches=15),
    )

    if not maps:
        print("ERROR: pycolmap produced no reconstruction.", file=sys.stderr)
        sys.exit(1)

    best_model = max(maps.values(), key=lambda m: m.num_reg_images())
    print(f"  [sfm] Registered {best_model.num_reg_images()} images, "
          f"{best_model.num_points3D()} 3D points")

    best_model.write(str(sparse_dir / "0"))
    return best_model
