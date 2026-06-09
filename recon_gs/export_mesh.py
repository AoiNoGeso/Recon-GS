"""Step 5: Export mesh from trained 3DGS via TSDF Fusion.

Pipeline
--------
1. Load trained Gaussians from .ply
2. For each training camera:
   a. Render RGB + depth (z in camera space)
   b. Render world-space normals
   c. Build per-pixel mask (floor / ceiling / ground / sky)
3. Integrate unmasked (depth, RGB) frames into a TSDF volume
4. Extract triangle mesh, clean, and save
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import pycolmap
from plyfile import PlyData
from gsplat import rasterization
import open3d as o3d

from recon_gs.align import load_or_compute_gravity_rotation, apply_to_c2w
from recon_gs.config import (
    MESH_VOXEL_SIZE,
    MESH_MAX_DEPTH,
    MESH_MIN_CLUSTER_TRIANGLES,
    MESH_SURFACE_ANGLE_DEG,
    MESH_MASK_FLOOR,
    MESH_MASK_CEILING,
    MESH_MASK_GROUND,
    MESH_MASK_SKY,
    MESH_WORLD_UP,
    MESH_SKY_ALPHA_THRESHOLD,
    MESH_SKY_TOP_FRACTION,
)


# --------------------------------------------------------------------------- #
# PLY loading
# --------------------------------------------------------------------------- #

def _load_ply_gaussians(ply_path: Path, device: torch.device):
    """Load Gaussians from a .ply file exported by _export_ply().

    Returns (means, scales, quats, opacities, sh_coeffs) — all on `device`.
    scales and quats are activation-applied (raw values, not log/logit).
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32, device=device
    )

    # scales stored as log(scale) in PLY
    scales = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
        dtype=torch.float32,
        device=device,
    ).exp()

    # quaternion stored as-is [w, x, y, z]
    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
        dtype=torch.float32,
        device=device,
    )
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # opacity stored as logit(opacity)
    opacities = torch.tensor(v["opacity"], dtype=torch.float32, device=device).sigmoid()

    # SH DC coefficients stored as-is (f_dc = SH coeff, NOT raw rgb)
    sh_dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
        dtype=torch.float32,
        device=device,
    )
    sh_coeffs = sh_dc.unsqueeze(1)  # (N, 1, 3)

    return means, scales, quats, opacities, sh_coeffs


# --------------------------------------------------------------------------- #
# Gaussian normal computation
# --------------------------------------------------------------------------- #

def _quaternion_to_rotmat(quats: Tensor) -> Tensor:
    """(N, 4) [w,x,y,z] → (N, 3, 3) rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _gaussian_world_normals(scales: Tensor, quats: Tensor) -> Tensor:
    """Compute per-Gaussian world-space normals as the shortest-axis direction.

    The normal of a Gaussian splat is conventionally taken as the direction
    of its smallest scale (thinnest axis).

    Returns: (N, 3) unit normals in world space.
    """
    R = _quaternion_to_rotmat(quats)              # (N, 3, 3)
    min_axis = scales.argmin(dim=-1)              # (N,)
    # Gather the column of R corresponding to the shortest axis
    idx = min_axis.view(-1, 1, 1).expand(-1, 3, 1)
    normals = R.gather(2, idx).squeeze(-1)        # (N, 3)
    return normals  # already unit-length (R is orthogonal)


# --------------------------------------------------------------------------- #
# Per-view rendering
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _render_view(
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
    sh_coeffs: Tensor,
    normals_world: Tensor,
    c2w: Tensor,
    K: Tensor,
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render one camera view.

    Returns:
        rgb   : (H, W, 3) float32 [0,1]
        depth : (H, W)    float32 [m], 0 = no hit
        normal: (H, W, 3) float32, world-space unit normals
        alpha : (H, W)    float32 [0,1]
    """
    viewmat = torch.linalg.inv(c2w)[None]  # (1, 4, 4)

    # ---- RGB rendering ----
    rgb_render, alpha_render, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmat,
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=0,
        packed=False,
    )
    rgb   = rgb_render[0].clamp(0, 1).cpu().numpy()   # (H, W, 3)
    alpha = alpha_render[0, ..., 0].cpu().numpy()      # (H, W)

    # ---- Depth rendering (z in camera space as a 1-channel color) ----
    w2c = torch.linalg.inv(c2w)
    means_cam = (w2c[:3, :3] @ means.T + w2c[:3, 3:]).T   # (N, 3)
    z_vals = means_cam[:, 2].clamp(min=0.0)               # (N,)
    depth_colors = z_vals.view(-1, 1)                      # (N, 1)

    depth_render, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=depth_colors,
        viewmats=viewmat,
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
    )
    depth = depth_render[0, ..., 0].cpu().numpy()          # (H, W)

    # ---- Normal rendering (world-space 3-channel color) ----
    # Normals are in [-1,1]; rasterization treats them as colors so we map to [0,1]
    # We undo this mapping after rendering.
    normal_colors = (normals_world + 1.0) * 0.5               # (N, 3)

    normal_render, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=normal_colors,
        viewmats=viewmat,
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
    )
    normal = normal_render[0].cpu().numpy() * 2.0 - 1.0   # (H, W, 3), back to [-1,1]
    # Re-normalise per-pixel
    norm_mag = np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
    normal = normal / norm_mag

    return rgb, depth, normal, alpha


# --------------------------------------------------------------------------- #
# Mask building
# --------------------------------------------------------------------------- #

def _build_mask(
    normal_world: np.ndarray,
    alpha: np.ndarray,
    H: int,
) -> np.ndarray:
    """Return boolean mask (H, W); True = pixel should be excluded from TSDF.

    Masking rules (all controlled by config flags):
      floor   : normal ≈ world-up   (upward-facing horizontal surface)
      ceiling : normal ≈ -world-up  (downward-facing horizontal surface)
      ground  : same as floor (outdoor alias)
      sky     : low-alpha region in upper portion of image
    """
    world_up = np.array(MESH_WORLD_UP, dtype=np.float32)
    world_up /= np.linalg.norm(world_up)

    cos_thresh = np.cos(np.deg2rad(MESH_SURFACE_ANGLE_DEG))  # cos(30°) ≈ 0.866

    mask = np.zeros(normal_world.shape[:2], dtype=bool)

    if MESH_MASK_FLOOR or MESH_MASK_GROUND:
        # dot(normal, world_up) > cos_thresh → nearly upward-facing → floor/ground
        dot = (normal_world * world_up).sum(axis=-1)
        mask |= dot > cos_thresh

    if MESH_MASK_CEILING:
        # dot(normal, -world_up) > cos_thresh → nearly downward-facing → ceiling
        dot = (normal_world * (-world_up)).sum(axis=-1)
        mask |= dot > cos_thresh

    if MESH_MASK_SKY:
        # Pixels with very low alpha in the upper fraction of the image
        sky_rows = int(H * MESH_SKY_TOP_FRACTION)
        sky_region = np.zeros_like(mask)
        sky_region[:sky_rows, :] = True
        mask |= sky_region & (alpha < MESH_SKY_ALPHA_THRESHOLD)

    return mask


# --------------------------------------------------------------------------- #
# Mesh post-processing
# --------------------------------------------------------------------------- #

def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove small disconnected clusters, unreferenced vertices, and degenerate triangles."""
    import copy
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)

    remove_mask = cluster_n_tris[tri_clusters] < MESH_MIN_CLUSTER_TRIANGLES
    cleaned = copy.deepcopy(mesh)
    cleaned.remove_triangles_by_mask(remove_mask)
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_degenerate_triangles()
    return cleaned


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def export_mesh(
    colmap_dir: Path,
    frames_dir: Path,
    ply_path: Path,
    output_dir: Path,
) -> None:
    """Export mesh from trained Gaussians via TSDF Fusion.

    Args:
        colmap_dir : directory containing sparse/0 COLMAP reconstruction
        frames_dir : directory with input RGB frames
        ply_path   : trained Gaussian .ply file
        output_dir : where to write tsdf_fusion.ply and tsdf_fusion_post.ply
    """
    from PIL import Image as PILImage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load Gaussians ----
    print("Loading Gaussians …")
    means, scales, quats, opacities, sh_coeffs = _load_ply_gaussians(ply_path, device)
    normals_world = _gaussian_world_normals(scales, quats)  # (N, 3)

    print(f"  {len(means):,} Gaussians loaded")
    active = (
        f"floor={MESH_MASK_FLOOR}  ceiling={MESH_MASK_CEILING}  "
        f"ground={MESH_MASK_GROUND}  sky={MESH_MASK_SKY}"
    )
    print(f"  Active masks: {active}")

    # ---- Load COLMAP cameras ----
    model = pycolmap.Reconstruction(str(sparse_dir))

    # Gravity alignment (reuse cached rotation computed during training)
    import torch as _torch
    R_align = _torch.tensor(
        load_or_compute_gravity_rotation(sparse_dir), dtype=_torch.float32, device=device
    )

    # ---- TSDF Volume ----
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=MESH_VOXEL_SIZE,
        sdf_trunc=4.0 * MESH_VOXEL_SIZE,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_images = len(model.images)
    print(f"Rendering & integrating {n_images} views …")

    for i, image_id in enumerate(sorted(model.images)):
        img = model.images[image_id]
        cam = model.cameras[img.camera_id]

        # Camera intrinsics
        fx = cam.focal_length_x if hasattr(cam, "focal_length_x") else cam.focal_length
        fy = cam.focal_length_y if hasattr(cam, "focal_length_y") else fx
        cx, cy = cam.principal_point_x, cam.principal_point_y

        # Image size from actual frame
        img_path = frames_dir / img.name
        pil_img = PILImage.open(img_path).convert("RGB")
        W, H = pil_img.size

        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

        # Camera pose (c2w)
        cfw = img.cam_from_world()
        R_np = np.array(cfw.rotation.matrix(), dtype=np.float32)
        t_np = np.array(cfw.translation, dtype=np.float32)
        w2c_np = np.eye(4, dtype=np.float32)
        w2c_np[:3, :3] = R_np
        w2c_np[:3, 3] = t_np
        c2w = apply_to_c2w(
            torch.tensor(np.linalg.inv(w2c_np), dtype=torch.float32, device=device),
            R_align,
        )

        # Render this view
        rgb, depth, normal_w, alpha = _render_view(
            means, scales, quats, opacities, sh_coeffs,
            normals_world, c2w, K, W, H,
        )

        # Build mask
        pixel_mask = _build_mask(normal_w, alpha, H)

        # Zero-out masked depth before TSDF integration
        depth_masked = depth.copy()
        depth_masked[pixel_mask] = 0.0
        depth_masked[depth_masked > MESH_MAX_DEPTH] = 0.0

        # Integrate into TSDF
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        depth_uint16 = (depth_masked * 1000).clip(0, 65535).astype(np.uint16)  # mm

        o3d_color = o3d.geometry.Image(rgb_uint8)
        o3d_depth = o3d.geometry.Image(depth_uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1000.0,
            depth_trunc=MESH_MAX_DEPTH,
            convert_rgb_to_intensity=False,
        )

        # w2c pose for Open3D (extrinsic: world→camera)
        pose_o3d = np.eye(4, dtype=np.float64)
        pose_o3d[:3, :3] = R_np.astype(np.float64)
        pose_o3d[:3, 3] = t_np.astype(np.float64)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        volume.integrate(rgbd, intrinsic, pose_o3d)

        if (i + 1) % 50 == 0 or (i + 1) == n_images:
            print(f"  [{i + 1}/{n_images}]")

    # ---- Extract & save ----
    print("Extracting triangle mesh …")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    raw_path = output_dir / "tsdf_fusion.ply"
    o3d.io.write_triangle_mesh(
        str(raw_path), mesh,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"  Raw mesh → {raw_path}  ({len(mesh.triangles):,} triangles)")

    print("Cleaning mesh …")
    mesh_clean = _clean_mesh(mesh)
    post_path = output_dir / "tsdf_fusion_post.ply"
    o3d.io.write_triangle_mesh(
        str(post_path), mesh_clean,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"  Cleaned mesh → {post_path}  ({len(mesh_clean.triangles):,} triangles)")
