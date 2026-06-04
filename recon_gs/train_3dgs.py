"""Step 4: Train 3D Gaussian Splatting with gsplat and export output.ply."""

from pathlib import Path

import torch
import numpy as np
from torch import Tensor

import pycolmap
from gsplat import rasterization

from recon_gs.config import (
    TRAIN_ITERATIONS,
    DENSIFY_FROM_ITER,
    DENSIFY_UNTIL_ITER,
    DENSIFICATION_INTERVAL,
    OPACITY_RESET_INTERVAL,
    DENSIFY_GRAD_THRESHOLD,
    PERCENT_DENSE,
    MIN_OPACITY,
    MAX_GAUSSIANS,
)


# --------------------------------------------------------------------------- #
# COLMAP → gsplat data loading
# --------------------------------------------------------------------------- #

def _load_colmap_cameras(
    sparse_dir: Path,
    frames_dir: Path,
    masks_dir: Path,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], int, int]:
    """
    Parse COLMAP sparse model and return per-image tensors needed by gsplat.
    Returns (c2w_list, K_list, image_list, mask_list, width, height).
    """
    from PIL import Image

    model = pycolmap.Reconstruction(str(sparse_dir))

    c2w_list, K_list, image_list, mask_list = [], [], [], []

    for image_id in sorted(model.images):
        img = model.images[image_id]
        cam = model.cameras[img.camera_id]

        cam_from_world = img.cam_from_world()
        R = torch.tensor(cam_from_world.rotation.matrix(), dtype=torch.float32)
        t = torch.tensor(cam_from_world.translation, dtype=torch.float32)
        w2c = torch.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = torch.linalg.inv(w2c)
        c2w_list.append(c2w.to(device))

        fx = cam.focal_length_x if hasattr(cam, "focal_length_x") else cam.focal_length
        fy = cam.focal_length_y if hasattr(cam, "focal_length_y") else fx
        cx, cy = cam.principal_point_x, cam.principal_point_y
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
        ).to(device)
        K_list.append(K)

        img_path = frames_dir / img.name
        rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        image_list.append(torch.tensor(rgb).to(device))

        mask_path = masks_dir / img.name
        if mask_path.exists():
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
            mask = torch.tensor(mask_np).to(device)
        else:
            h, w = rgb.shape[:2]
            mask = torch.zeros(h, w, device=device)
        mask_list.append(mask)

    h, w = image_list[0].shape[:2]
    return c2w_list, K_list, image_list, mask_list, w, h


def _init_gaussians_from_sparse(sparse_dir: Path, device: torch.device):
    """Initialise Gaussian means from COLMAP sparse point cloud."""
    model = pycolmap.Reconstruction(str(sparse_dir))
    points = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)
    means = torch.tensor(points, device=device)
    n = len(means)

    scales = torch.full((n, 3), -4.0, device=device)
    quats = torch.zeros((n, 4), device=device)
    quats[:, 0] = 1.0

    opacities = torch.full((n,), 0.1, device=device)

    C0 = 0.28209479177387814
    colors_raw = np.array(
        [p.color[:3] for p in model.points3D.values()], dtype=np.float32
    ) / 255.0
    sh_dc_init = (colors_raw - 0.5) / C0
    sh_coeffs = torch.tensor(sh_dc_init, device=device).unsqueeze(1)  # (N, 1, 3)

    return (
        means.requires_grad_(True),
        scales.requires_grad_(True),
        quats.requires_grad_(True),
        opacities.requires_grad_(True),
        sh_coeffs.requires_grad_(True),
    )


def _scene_extent(sparse_dir: Path) -> float:
    """Estimate scene extent as max camera-to-centroid distance."""
    model = pycolmap.Reconstruction(str(sparse_dir))
    cam_positions = []
    for img in model.images.values():
        cfw = img.cam_from_world()
        R = np.array(cfw.rotation.matrix())
        t = np.array(cfw.translation)
        # camera position in world = -R^T @ t
        cam_positions.append(-R.T @ t)
    cam_positions = np.array(cam_positions)
    centroid = cam_positions.mean(axis=0)
    dists = np.linalg.norm(cam_positions - centroid, axis=1)
    return float(dists.max())


# --------------------------------------------------------------------------- #
# Adaptive Density Control helpers
# --------------------------------------------------------------------------- #

def _extend_optimizer_state(optimizer: torch.optim.Adam, group_idx: int, n_new: int) -> None:
    """Extend Adam state tensors for a parameter group by n_new entries (appended)."""
    param = optimizer.param_groups[group_idx]["params"][0]
    state = optimizer.state.get(param)
    if state is None or len(state) == 0:
        return
    pad_shape = list(state["exp_avg"].shape)
    pad_shape[0] = n_new
    zeros = torch.zeros(pad_shape, dtype=param.dtype, device=param.device)
    state["exp_avg"] = torch.cat([state["exp_avg"], zeros], dim=0)
    state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], zeros], dim=0)


def _filter_optimizer_state(optimizer: torch.optim.Adam, group_idx: int, keep_mask: Tensor) -> None:
    """Keep only the entries selected by keep_mask in Adam state for a parameter group."""
    param = optimizer.param_groups[group_idx]["params"][0]
    state = optimizer.state.get(param)
    if state is None or len(state) == 0:
        return
    state["exp_avg"] = state["exp_avg"][keep_mask]
    state["exp_avg_sq"] = state["exp_avg_sq"][keep_mask]


def _replace_param(optimizer: torch.optim.Adam, group_idx: int, new_tensor: Tensor) -> Tensor:
    """Replace a parameter in-place and update optimizer reference. Returns new leaf."""
    leaf = new_tensor.detach().requires_grad_(True)
    optimizer.param_groups[group_idx]["params"][0] = leaf
    # Move existing state to new leaf key
    old_param = list(optimizer.state.keys())[0] if optimizer.state else None
    # Rebuild state mapping: find the old param for this group
    # We do a clean rebuild by iterating param_groups
    return leaf


def _rebuild_params_and_optimizer(
    params: list[Tensor],
    optimizer: torch.optim.Adam,
) -> tuple[list[Tensor], torch.optim.Adam]:
    """
    Detach all params, create new leaf tensors, rebuild optimizer preserving state.
    params order: [means, scales, quats, opacities, sh_coeffs]
    """
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    old_states = []
    for pg in optimizer.param_groups:
        p = pg["params"][0]
        old_states.append(optimizer.state.get(p, {}))

    new_params = [p.detach().requires_grad_(True) for p in params]
    new_optimizer = torch.optim.Adam(
        [{"params": [p], "lr": lr} for p, lr in zip(new_params, lrs)]
    )
    for i, (pg, st) in enumerate(zip(new_optimizer.param_groups, old_states)):
        if st:
            new_optimizer.state[pg["params"][0]] = {
                "step": st.get("step", torch.tensor(0)),
                "exp_avg": st["exp_avg"].clone(),
                "exp_avg_sq": st["exp_avg_sq"].clone(),
            }

    return new_params, new_optimizer


def _densify_and_prune(
    params: list[Tensor],
    optimizer: torch.optim.Adam,
    grad_accum: Tensor,
    grad_count: Tensor,
    scene_extent: float,
) -> tuple[list[Tensor], torch.optim.Adam, Tensor, Tensor]:
    """Adaptive density control: clone small, split large, prune transparent."""
    means, scales, quats, opacities, sh_coeffs = params

    avg_grad = grad_accum / (grad_count.clamp(min=1))

    # Per-gaussian max scale
    max_scale = torch.exp(scales).max(dim=-1).values

    # Which gaussians have large gradients
    grad_mask = avg_grad > DENSIFY_GRAD_THRESHOLD

    # Clone: small gaussians with large gradient
    clone_mask = grad_mask & (max_scale <= PERCENT_DENSE * scene_extent)
    # Split: large gaussians with large gradient
    split_mask = grad_mask & (max_scale > PERCENT_DENSE * scene_extent)

    n_clone = clone_mask.sum().item()
    n_split = split_mask.sum().item()

    new_means_list = [means]
    new_scales_list = [scales]
    new_quats_list = [quats]
    new_opacities_list = [opacities]
    new_sh_list = [sh_coeffs]

    # ---- Clone ----
    if n_clone > 0:
        new_means_list.append(means[clone_mask].detach())
        new_scales_list.append(scales[clone_mask].detach())
        new_quats_list.append(quats[clone_mask].detach())
        new_opacities_list.append(opacities[clone_mask].detach())
        new_sh_list.append(sh_coeffs[clone_mask].detach())

    # ---- Split ----
    if n_split > 0:
        # Sample two new positions from the gaussian distribution
        s = torch.exp(scales[split_mask])  # (M, 3)
        q = quats[split_mask] / quats[split_mask].norm(dim=-1, keepdim=True)
        # Convert quaternion to rotation matrix (w, x, y, z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x*x + z*z),  2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),       1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)  # (M, 3, 3)

        # Sample offsets along principal axes (2 samples per gaussian)
        noise = torch.randn(n_split, 3, device=means.device)
        offset = (R @ (s * noise).unsqueeze(-1)).squeeze(-1)  # (M, 3)

        for sign in [+1.0, -1.0]:
            new_means_list.append((means[split_mask] + sign * offset).detach())
            new_scales_list.append((scales[split_mask] - np.log(1.6)).detach())
            new_quats_list.append(quats[split_mask].detach())
            new_opacities_list.append(opacities[split_mask].detach())
            new_sh_list.append(sh_coeffs[split_mask].detach())

    # Concatenate new gaussians
    all_means = torch.cat(new_means_list, dim=0)
    all_scales = torch.cat(new_scales_list, dim=0)
    all_quats = torch.cat(new_quats_list, dim=0)
    all_opacities = torch.cat(new_opacities_list, dim=0)
    all_sh = torch.cat(new_sh_list, dim=0)

    # ---- Prune ----
    # Prune: low opacity, or split originals (replaced by two children)
    keep = torch.sigmoid(all_opacities) >= MIN_OPACITY

    # Also remove the original split gaussians (first n positions up to n_split)
    n_orig = means.shape[0]
    if n_split > 0:
        split_orig_indices = split_mask.nonzero(as_tuple=True)[0]
        keep[split_orig_indices] = False

    # Cap total gaussians
    if keep.sum().item() > MAX_GAUSSIANS:
        # Keep the MAX_GAUSSIANS with highest opacity
        opac_vals = torch.sigmoid(all_opacities)
        opac_vals[~keep] = -1.0
        _, top_idx = opac_vals.topk(MAX_GAUSSIANS)
        keep = torch.zeros_like(keep)
        keep[top_idx] = True

    # Build new param list
    new_params_raw = [
        all_means[keep],
        all_scales[keep],
        all_quats[keep],
        all_opacities[keep],
        all_sh[keep],
    ]

    # Extend optimizer state for newly added gaussians, then filter by keep
    # Simpler: rebuild optimizer from scratch (state for new gaussians = zero)
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    new_params = [p.detach().requires_grad_(True) for p in new_params_raw]
    new_optimizer = torch.optim.Adam(
        [{"params": [p], "lr": lr} for p, lr in zip(new_params, lrs)]
    )

    # Reset gradient accumulators
    n_new = new_params[0].shape[0]
    new_grad_accum = torch.zeros(n_new, device=means.device)
    new_grad_count = torch.zeros(n_new, device=means.device)

    print(f"  [ADC] clone={n_clone} split={n_split} "
          f"pruned={keep.shape[0]-keep.sum().item()} total={n_new}")

    return new_params, new_optimizer, new_grad_accum, new_grad_count


def _reset_opacity(
    params: list[Tensor],
    optimizer: torch.optim.Adam,
) -> tuple[list[Tensor], torch.optim.Adam]:
    """Reset all opacities to a low value (sigmoid → 0.01) to prune dead gaussians."""
    means, scales, quats, opacities, sh_coeffs = params
    reset_val = np.log(0.01 / (1.0 - 0.01))  # logit(0.01) ≈ -4.6
    new_opacities = torch.full_like(opacities, reset_val)
    new_params = [means, scales, quats, new_opacities, sh_coeffs]
    return _rebuild_params_and_optimizer(new_params, optimizer)


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
    params = [means, scales, quats, opacities, sh_coeffs]

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4},
            {"params": [scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [opacities], "lr": 5e-2},
            {"params": [sh_coeffs], "lr": 2.5e-3},
        ]
    )

    ext = _scene_extent(sparse_dir)

    n = params[0].shape[0]
    grad_accum = torch.zeros(n, device=device)
    grad_count = torch.zeros(n, device=device)

    for step in range(1, TRAIN_ITERATIONS + 1):
        means, scales, quats, opacities, sh_coeffs = params
        idx = step % n_views
        c2w = c2w_list[idx]
        K = K_list[idx]
        gt = gt_images[idx]
        mask = gt_masks[idx]

        viewmat = torch.linalg.inv(c2w)[None]
        quats_norm = quats / quats.norm(dim=-1, keepdim=True)

        do_densify = DENSIFY_FROM_ITER <= step <= DENSIFY_UNTIL_ITER

        renders, alphas, info = rasterization(
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
            packed=False,   # means2d shape: (1, N, 2) — needed for per-gaussian grad
            absgrad=True,   # sets means2d.absgrad after backward
        )

        # Register retain_grad as fallback in case absgrad isn't populated
        if do_densify:
            info["means2d"].retain_grad()

        rendered = renders[0]

        valid = (1.0 - mask).unsqueeze(-1)
        loss = torch.abs(rendered * valid - gt * valid).mean()

        optimizer.zero_grad()
        loss.backward()

        # Accumulate 2D positional gradients for ADC
        # means2d shape: (1, N, 2); use absgrad if available, else .grad
        if do_densify:
            m2d = info["means2d"]
            absgrad = getattr(m2d, "absgrad", None)
            if absgrad is not None:
                grad2d = absgrad[0].norm(dim=-1)          # (N,)
            elif m2d.grad is not None:
                grad2d = m2d.grad[0].abs().norm(dim=-1)   # (N,)
            else:
                grad2d = None

            if grad2d is not None:
                grad_accum += grad2d.detach()
                grad_count += 1

            # Debug: first densification step直前に実際の勾配統計を表示
            if step == DENSIFY_FROM_ITER + DENSIFICATION_INTERVAL - 1:
                avg = (grad_accum / grad_count.clamp(min=1))
                print(f"  [ADC debug] grad stats: max={avg.max():.6f} "
                      f"mean={avg.mean():.6f} threshold={DENSIFY_GRAD_THRESHOLD}"
                      f" grad_source={'absgrad' if absgrad is not None else 'grad' if m2d.grad is not None else 'none'}")

        optimizer.step()

        # Densification
        if (DENSIFY_FROM_ITER <= step <= DENSIFY_UNTIL_ITER
                and step % DENSIFICATION_INTERVAL == 0):
            params, optimizer, grad_accum, grad_count = _densify_and_prune(
                params, optimizer, grad_accum, grad_count, ext
            )

        # Opacity reset
        if step % OPACITY_RESET_INTERVAL == 0 and step < DENSIFY_UNTIL_ITER:
            params, optimizer = _reset_opacity(params, optimizer)
            # Reset accumulators to match new param size
            n_new = params[0].shape[0]
            grad_accum = torch.zeros(n_new, device=device)
            grad_count = torch.zeros(n_new, device=device)

        if step % 1000 == 0:
            means, scales, quats, opacities, sh_coeffs = params
            print(f"  [{step}/{TRAIN_ITERATIONS}] loss={loss.item():.4f}  gaussians={len(means)}")

    means, scales, quats, opacities, sh_coeffs = params
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
    arr["opacity"] = np.log(opacities / (1.0 - opacities + 1e-8))
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = (
        np.log(scales[:, 0]), np.log(scales[:, 1]), np.log(scales[:, 2])
    )
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    )

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))
    print(f"Saved {n} Gaussians → {path}")
