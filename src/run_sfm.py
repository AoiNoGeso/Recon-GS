"""Step 3: Camera pose estimation with hloc (SuperPoint + SuperGlue) + COLMAP."""

from pathlib import Path

import pycolmap

# hloc imports
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
)


# hloc feature/matcher configs tuned for indoor scenes
_FEATURE_CONF = extract_features.confs["superpoint_inloc"]
_MATCHER_CONF = match_features.confs["superglue-indoor"]


def run_sfm(frames_dir: Path, masks_dir: Path, colmap_dir: Path) -> pycolmap.Reconstruction:
    """
    Run hloc feature extraction + matching + COLMAP reconstruction.
    Masked regions (white pixels) are ignored during feature extraction.
    Returns the pycolmap Reconstruction object.
    """
    colmap_dir.mkdir(parents=True, exist_ok=True)

    db_path = colmap_dir / "database.db"
    sfm_pairs = colmap_dir / "pairs.txt"
    features_path = colmap_dir / "features.h5"
    matches_path = colmap_dir / "matches.h5"
    sfm_dir = colmap_dir / "sparse"

    # Build image list (frames only, masks supplied separately)
    image_list = [p.name for p in sorted(frames_dir.glob("*.png"))]

    # Exhaustive pairing (suitable for short videos; swap to sequential for long ones)
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list)

    # Feature extraction — pass masks so keypoints inside masked areas are dropped
    extract_features.main(
        conf=_FEATURE_CONF,
        image_dir=frames_dir,
        image_list=image_list,
        feature_path=features_path,
        mask_dir=masks_dir,
    )

    # Feature matching
    match_features.main(
        conf=_MATCHER_CONF,
        pairs=sfm_pairs,
        features=features_path,
        matches=matches_path,
    )

    # COLMAP incremental reconstruction
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=frames_dir,
        pairs=sfm_pairs,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.AUTO,
    )

    return model
