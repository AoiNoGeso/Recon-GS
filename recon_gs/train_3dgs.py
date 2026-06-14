"""Step 4: Train 3D Gaussian Splatting with gsplat and export output.ply."""

from pathlib import Path

import torch
import numpy as np
from torch import Tensor

import pycolmap
from gsplat import rasterization

from recon_gs.config import TRAIN_ITERATIONS
from recon_gs.align import load_or_compute_gravity_rotation, apply_to_c2w


# --------------------------------------------------------------------------- #
# COLMAP → gsplat data loading
# --------------------------------------------------------------------------- #

def _load_colmap_cameras(
    sparse_dir: Path,
    frames_dir: Path,
    masks_dir: Path,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor], list[np.ndarray], list[np.ndarray], int, int]:
    """Parse COLMAP sparse model and return per-image data needed by gsplat.

    Images and masks are stored as uint8 numpy arrays on CPU to avoid VRAM
    exhaustion on long videos.  They are converted to float32 GPU tensors
    one frame at a time in the training loop.

    Returns (c2w_list, K_list, image_list, mask_list, width, height).
      image_list : list of (H, W, 3) uint8 numpy arrays  [0, 255]
      mask_list  : list of (H, W)    uint8 numpy arrays  [0, 255]
    """
    from PIL import Image

    model = pycolmap.Reconstruction(str(sparse_dir))

    R_align = torch.tensor(
        load_or_compute_gravity_rotation(sparse_dir), dtype=torch.float32, device=device
    )

    c2w_list, K_list, image_list, mask_list = [], [], [], []

    for image_id in sorted(model.images):
        img = model.images[image_id]
        cam = model.cameras[img.camera_id]

        cam_from_world = img.cam_from_world()
        R = torch.tensor(cam_from_world.rotation.matrix(), dtype=torch.float32, device=device)
        t = torch.tensor(cam_from_world.translation, dtype=torch.float32, device=device)
        w2c = torch.eye(4, device=device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = apply_to_c2w(torch.linalg.inv(w2c), R_align)
        c2w_list.append(c2w)

        fx = cam.focal_length_x if hasattr(cam, "focal_length_x") else cam.focal_length
        fy = cam.focal_length_y if hasattr(cam, "focal_length_y") else fx
        cx, cy = cam.principal_point_x, cam.principal_point_y
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
        ).to(device)
        K_list.append(K)

        # Store as uint8 CPU numpy to minimise memory (3× smaller than float32)
        img_path = frames_dir / img.name
        image_list.append(np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8))

        mask_path = masks_dir / img.name
        if mask_path.exists():
            mask_list.append(np.array(Image.open(mask_path).convert("L"), dtype=np.uint8))
        else:
            h, w = image_list[-1].shape[:2]
            mask_list.append(np.zeros((h, w), dtype=np.uint8))

    h, w = image_list[0].shape[:2]
    print(f"  Loaded {len(c2w_list)} views into CPU RAM (images as uint8)")
    return c2w_list, K_list, image_list, mask_list, w, h


def _init_gaussians_from_sparse(sparse_dir: Path, device: torch.device):
    """Initialise Gaussian means from COLMAP sparse point cloud."""
    model = pycolmap.Reconstruction(str(sparse_dir))
    points = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)

    # Apply same gravity alignment as cameras
    R_align = torch.tensor(
        load_or_compute_gravity_rotation(sparse_dir), dtype=torch.float32, device=device
    )
    means = (R_align @ torch.tensor(points, device=device).T).T
    n = len(means)

    # Covariance (log-scale + quaternion)
    scales = torch.full((n, 3), -4.0, device=device)  # exp(-4) ≈ 0.018 m
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0  # identity quaternion [w, x, y, z]

    # Opacity (logit 0.1 ≈ sigmoid → 0.52)
    opacities = torch.full((n,), 0.1, device=device)

    # SH degree-0 colour from COLMAP point colours
    # gsplat renders as: color = C0 * sh_dc + 0.5
    # → 初期レンダリングが点群の色に一致するよう逆変換して初期化
    C0 = 0.28209479177387814
    colors_raw = np.array(
        [p.color[:3] for p in model.points3D.values()], dtype=np.float32
    ) / 255.0
    sh_dc_init = (colors_raw - 0.5) / C0  # 逆変換: rendered = C0 * sh_dc + 0.5
    sh_coeffs = torch.tensor(sh_dc_init, device=device).unsqueeze(1)  # (N, 1, 3)

    return (
        means.requires_grad_(True),
        scales.requires_grad_(True),
        quats.requires_grad_(True),
        opacities.requires_grad_(True),
        sh_coeffs.requires_grad_(True),
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_3dgs(colmap_dir: Path, frames_dir: Path, masks_dir: Path, output_ply: Path) -> None:
    """Train 3DGS and write final Gaussians to output_ply."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sparse_dir = colmap_dir / "sparse" / "0"
    c2w_list, K_list, gt_images, gt_masks, W, H = _load_colmap_cameras(
        sparse_dir, frames_dir, masks_dir, device
    )
    n_views = len(c2w_list)

    means, scales, quats, opacities, sh_coeffs = _init_gaussians_from_sparse(
        sparse_dir, device
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4},
            {"params": [scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [opacities], "lr": 5e-2},
            {"params": [sh_coeffs], "lr": 2.5e-3},
        ]
    )

    for step in range(1, TRAIN_ITERATIONS + 1):
        idx = step % n_views
        c2w = c2w_list[idx]
        K = K_list[idx]

        # Convert uint8 CPU numpy → float32 GPU tensor on-the-fly
        gt   = torch.tensor(gt_images[idx], dtype=torch.float32, device=device) / 255.0
        mask = torch.tensor(gt_masks[idx],  dtype=torch.float32, device=device) / 255.0

        viewmat = torch.linalg.inv(c2w)[None]  # (1, 4, 4)
        quats_norm = quats / quats.norm(dim=-1, keepdim=True)

        renders, alphas, _ = rasterization(
            means=means,
            quats=quats_norm,
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=0,
        )

        rendered = renders[0]  # (H, W, 3)

        valid = (1.0 - mask).unsqueeze(-1)
        loss = torch.abs(rendered * valid - gt * valid).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"  [{step}/{TRAIN_ITERATIONS}] loss={loss.item():.4f}  gaussians={len(means)}")

    _export_ply(
        output_ply,
        means.detach().cpu().numpy(),
        torch.exp(scales).detach().cpu().numpy(),
        (quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy(),
        torch.sigmoid(opacities).detach().cpu().numpy(),
        sh_coeffs[:, 0, :].detach().cpu().numpy(),
    )


# --------------------------------------------------------------------------- #
# PLY export
# --------------------------------------------------------------------------- #

def _export_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Write Gaussians to a .ply file compatible with standard GS viewers."""
    from plyfile import PlyData, PlyElement

    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(means)

    # colors are already SH DC coefficients (f_dc).
    # Viewers compute: rendered = C0 * f_dc + 0.5, so store as-is.
    sh_dc = colors

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
    # Store raw logit opacity to match 3DGS viewer convention
    arr["opacity"] = np.log(opacities / (1.0 - opacities + 1e-8))
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = (
        np.log(scales[:, 0]), np.log(scales[:, 1]), np.log(scales[:, 2])
    )
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    )

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"Saved {n} Gaussians → {path}")
