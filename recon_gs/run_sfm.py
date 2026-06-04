"""
Step 3: Camera pose estimation using COLMAP CLI + pycolmap.

Requires COLMAP to be installed on the system:
    sudo apt install colmap        # Ubuntu
    brew install colmap            # macOS

Pipeline:
    feature_extractor  → SIFT keypoints per frame
    exhaustive_matcher → match all pairs (suitable for short clips)
    mapper             → incremental SfM reconstruction
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pycolmap


def _check_colmap() -> str:
    """Return the colmap binary path or raise if not found."""
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        print(
            "ERROR: 'colmap' binary not found.\n"
            "  Ubuntu: sudo apt install colmap\n"
            "  macOS:  brew install colmap",
            file=sys.stderr,
        )
        sys.exit(1)
    return colmap_bin


def _run(cmd: list[str]) -> None:
    """Run a shell command, streaming output to stdout."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_sfm(frames_dir: Path, masks_dir: Path, colmap_dir: Path) -> pycolmap.Reconstruction:
    """
    Run COLMAP SfM pipeline and return the best reconstruction.

    Args:
        frames_dir: Directory containing extracted PNG frames.
        masks_dir:  Directory containing binary masks (white = dynamic, ignored).
        colmap_dir: Output directory for COLMAP database and sparse model.

    Returns:
        pycolmap.Reconstruction of the largest registered model.
    """
    colmap_bin = _check_colmap()
    colmap_dir.mkdir(parents=True, exist_ok=True)

    db_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Feature extraction (SIFT, GPU-accelerated if available)
    #    Masks are passed so keypoints inside dynamic regions are skipped.
    # ------------------------------------------------------------------ #
    _run([
        colmap_bin, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.mask_path", str(masks_dir),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.max_num_features", "8192",
    ])

    # ------------------------------------------------------------------ #
    # 2. Exhaustive matching
    #    Works well for short videos (< ~300 frames).
    #    Switch to sequential_matcher for longer sequences.
    # ------------------------------------------------------------------ #
    _run([
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ])

    # ------------------------------------------------------------------ #
    # 3. Incremental reconstruction (mapper)
    # ------------------------------------------------------------------ #
    _run([
        colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.num_threads", "8",
        "--Mapper.init_min_tri_angle", "4",
    ])

    # ------------------------------------------------------------------ #
    # 4. Load and return the largest model
    # ------------------------------------------------------------------ #
    model_dirs = sorted(sparse_dir.iterdir())
    if not model_dirs:
        print("ERROR: COLMAP mapper produced no reconstruction.", file=sys.stderr)
        sys.exit(1)

    # Pick the model with the most registered images
    best_dir = max(
        model_dirs,
        key=lambda d: len(list(d.glob("images.bin"))) and
                      pycolmap.Reconstruction(str(d)).num_reg_images()
    )
    model = pycolmap.Reconstruction(str(best_dir))
    return model
