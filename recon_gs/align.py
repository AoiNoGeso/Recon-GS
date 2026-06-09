"""Gravity alignment for COLMAP reconstructions.

COLMAP reconstructions are not gravity-aligned by default.  This module
estimates the world "up" direction from the average camera orientation and
computes a rotation that maps it to Y-up (the convention used by most 3DGS
viewers and standard mesh tools).

The computed rotation is cached to ``sparse_dir/gravity_rotation.json`` so
that training and mesh-export use an identical transform.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycolmap


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_or_compute_gravity_rotation(sparse_dir: Path) -> np.ndarray:
    """Return (3, 3) rotation R such that ``R @ x_world = x_yup``.

    The rotation is computed once from the camera poses and cached in
    ``sparse_dir/gravity_rotation.json``.  Subsequent calls reuse the cache.
    """
    cache_path = sparse_dir / "gravity_rotation.json"
    if cache_path.exists():
        return np.array(json.loads(cache_path.read_text()), dtype=np.float32)

    R = _compute_from_cameras(sparse_dir)
    cache_path.write_text(json.dumps(R.tolist()))
    deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"  [align] gravity rotation computed (angle ≈ {deg:.1f}°) → {cache_path}")
    return R


def apply_to_c2w(c2w: "torch.Tensor", R_align: "torch.Tensor") -> "torch.Tensor":
    """Apply world-space rotation R_align to a c2w matrix (4×4 torch tensor).

    After rotation every point x_world is mapped to R_align @ x_world, so:
      - rotation part : R_wc_new = R_align @ R_wc
      - translation   : t_new    = R_align @ t   (camera centre in world)
    """
    c2w_new = c2w.clone()
    c2w_new[:3, :3] = R_align @ c2w[:3, :3]
    c2w_new[:3, 3] = R_align @ c2w[:3, 3]
    return c2w_new


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _compute_from_cameras(sparse_dir: Path) -> np.ndarray:
    """Estimate gravity direction from camera up-vectors, return alignment R."""
    model = pycolmap.Reconstruction(str(sparse_dir))

    up_vectors: list[np.ndarray] = []
    for img in model.images.values():
        cfw = img.cam_from_world()
        # c2w rotation: R_wc = R_cw^T
        R_wc = np.array(cfw.rotation.matrix(), dtype=np.float64).T
        # Camera -Y axis in world ≈ "up" (COLMAP: camera Y points down)
        cam_up_world = -R_wc[:, 1]
        up_vectors.append(cam_up_world)

    avg_up = np.mean(up_vectors, axis=0)
    avg_up /= np.linalg.norm(avg_up)

    # COLMAPはOpenCV規約でY軸が下向き → カメラのup方向は世界座標で [0,-1,0] になるのが正常
    # avg_up を [0,-1,0] に揃えることで、傾き(roll)のみを補正し上下反転を起こさない
    target = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return _rotation_between(avg_up, target).astype(np.float32)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation matrix R such that R @ a = b (unit vectors)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(a, b)

    if cross_norm < 1e-8:
        if dot > 0:
            return np.eye(3, dtype=np.float64)
        # 180° rotation: find an orthogonal axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)          # numerically stable

    # Rodrigues formula
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
