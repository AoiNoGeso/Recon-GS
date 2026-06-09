"""Step 5: Export mesh from trained 3DGS via TSDF Fusion.

Pipeline
--------
1. Load trained Gaussians from .ply
2. Build surface-exclusion prompts from config flags (floor/ceiling/ground/sky)
3. For each training camera:
   a. Render RGB + depth (z in camera space)
   b. Run Grounded-SAM2 on the RGB frame to detect surface regions
   c. Zero-out masked pixels before TSDF integration
4. Integrate unmasked (depth, RGB) frames into a TSDF volume
5. Extract triangle mesh, clean, and save
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
    MESH_MASK_FLOOR,
    MESH_MASK_CEILING,
    MESH_MASK_GROUND,
    MESH_MASK_SKY,
    MESH_FLOOR_PROMPTS,
    MESH_CEILING_PROMPTS,
    MESH_GROUND_PROMPTS,
    MESH_SKY_PROMPTS,
    MESH_GDINO_BOX_THRESHOLD,
    MESH_GDINO_TEXT_THRESHOLD,
    MESH_FILL_PLANES,
    MESH_PLANE_RANSAC_DISTANCE,
    MESH_PLANE_RANSAC_ITERATIONS,
    MESH_PLANE_MIN_POINTS,
    MESH_PLANE_MAX_PLANES,
    MESH_PLANE_PIXEL_STRIDE,
    MESH_PLANE_VOXEL_SIZE,
    MESH_PLANE_MAX_INPUT_POINTS,
)

_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_SAM2_MODEL_ID = "facebook/sam2-hiera-large"


# --------------------------------------------------------------------------- #
# PLY loading
# --------------------------------------------------------------------------- #

def _load_ply_gaussians(ply_path: Path, device: torch.device):
    """Load Gaussians from a .ply file exported by _export_ply().

    Returns (means, scales, quats, opacities, sh_coeffs) — all on `device`.
    scales are activation-applied (not log).
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=1), dtype=torch.float32, device=device
    )

    scales = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1),
        dtype=torch.float32,
        device=device,
    ).exp()

    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1),
        dtype=torch.float32,
        device=device,
    )
    quats = quats / quats.norm(dim=-1, keepdim=True)

    opacities = torch.tensor(v["opacity"], dtype=torch.float32, device=device).sigmoid()

    sh_dc = torch.tensor(
        np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1),
        dtype=torch.float32,
        device=device,
    )
    sh_coeffs = sh_dc.unsqueeze(1)  # (N, 1, 3)

    return means, scales, quats, opacities, sh_coeffs


# --------------------------------------------------------------------------- #
# Grounded-SAM2 surface masking
# --------------------------------------------------------------------------- #

def _build_surface_prompts() -> list[str]:
    """Collect active surface prompts from config flags."""
    prompts: list[str] = []
    if MESH_MASK_FLOOR:
        prompts.extend(MESH_FLOOR_PROMPTS)
    if MESH_MASK_CEILING:
        prompts.extend(MESH_CEILING_PROMPTS)
    if MESH_MASK_GROUND:
        prompts.extend(MESH_GROUND_PROMPTS)
    if MESH_MASK_SKY:
        prompts.extend(MESH_SKY_PROMPTS)
    return prompts


def _load_gsam2_models(device: torch.device):
    """Load GroundingDINO and SAM2 models."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino_processor = AutoProcessor.from_pretrained(_GDINO_MODEL_ID)
    gdino_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_MODEL_ID)
        .to(device)
        .eval()
    )

    sam2_model = build_sam2_hf(_SAM2_MODEL_ID, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return gdino_processor, gdino_model, sam2_predictor


def _surface_mask_gsam2(
    image_np: np.ndarray,
    gdino_processor,
    gdino_model,
    sam2_predictor,
    prompt: str,
    device: torch.device,
) -> np.ndarray:
    """Return boolean mask (H, W); True = surface pixel to exclude from TSDF.

    Args:
        image_np : (H, W, 3) uint8 RGB image
        prompt   : period-separated GroundingDINO text prompt
    """
    from PIL import Image as PILImage

    image_pil = PILImage.fromarray(image_np)
    h, w = image_np.shape[:2]

    inputs = gdino_processor(
        images=image_pil,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=MESH_GDINO_BOX_THRESHOLD,
        text_threshold=MESH_GDINO_TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy

    mask = np.zeros((h, w), dtype=bool)
    if len(boxes) == 0:
        return mask

    sam2_predictor.set_image(image_np)
    sam_masks, _, _ = sam2_predictor.predict(
        box=boxes,
        multimask_output=False,
    )
    # sam_masks: (N, 1, H, W) or (N, H, W)
    if sam_masks.ndim == 4:
        sam_masks = sam_masks[:, 0]
    for m in sam_masks:
        mask |= m.astype(bool)

    return mask


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
    c2w: Tensor,
    K: Tensor,
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB and depth for one camera view.

    Returns:
        rgb   : (H, W, 3) float32 [0, 1]
        depth : (H, W)    float32 [m], 0 = no hit
    """
    viewmat = torch.linalg.inv(c2w)[None]  # (1, 4, 4)

    # ---- RGB ----
    rgb_render, _, _ = rasterization(
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
    rgb = rgb_render[0].clamp(0, 1).cpu().numpy()  # (H, W, 3)

    # ---- Depth (z in camera space) ----
    w2c = torch.linalg.inv(c2w)
    means_cam = (w2c[:3, :3] @ means.T + w2c[:3, 3:]).T  # (N, 3)
    z_vals = means_cam[:, 2].clamp(min=0.0)               # (N,)

    depth_render, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=z_vals.view(-1, 1),
        viewmats=viewmat,
        Ks=K[None],
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
    )
    depth = depth_render[0, ..., 0].cpu().numpy()  # (H, W)

    return rgb, depth


# --------------------------------------------------------------------------- #
# RANSAC plane fitting
# --------------------------------------------------------------------------- #

def _unproject_to_world(
    depth: np.ndarray,
    mask: np.ndarray,
    rgb: np.ndarray,
    c2w_np: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject masked depth pixels to world-space 3D points.

    Pixels are subsampled by MESH_PLANE_PIXEL_STRIDE to limit point count.

    Returns:
        points : (M, 3) float32 world-space XYZ
        colors : (M, 3) float32 RGB [0, 1]
    """
    # Subsampled pixel grid to reduce point count
    H, W = mask.shape
    stride = MESH_PLANE_PIXEL_STRIDE
    gy, gx = np.meshgrid(
        np.arange(0, H, stride, dtype=np.int32),
        np.arange(0, W, stride, dtype=np.int32),
        indexing="ij",
    )
    gy, gx = gy.ravel(), gx.ravel()
    valid = mask[gy, gx] & (depth[gy, gx] > 0) & (depth[gy, gx] < MESH_MAX_DEPTH)
    ys, xs = gy[valid], gx[valid]

    if len(ys) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    z = depth[ys, xs]
    x_c = (xs - cx) / fx * z
    y_c = (ys - cy) / fy * z
    pts_cam = np.stack([x_c, y_c, z], axis=1)  # (M, 3)

    R_cw = c2w_np[:3, :3]
    t_cw = c2w_np[:3, 3]
    pts_world = (R_cw @ pts_cam.T + t_cw[:, None]).T  # (M, 3)

    colors = rgb[ys, xs]  # (M, 3)
    return pts_world.astype(np.float32), colors.astype(np.float32)


def _make_plane_mesh(
    inlier_pts: np.ndarray,
    plane_model: tuple[float, float, float, float],
    avg_color: np.ndarray,
) -> o3d.geometry.TriangleMesh | None:
    """Build a convex-hull triangle mesh from inlier points projected onto a plane.

    Args:
        inlier_pts  : (M, 3) 3D points already classified as inliers
        plane_model : (a, b, c, d) with ax+by+cz+d=0, [a,b,c] unit normal
        avg_color   : (3,) RGB [0, 1] to paint all vertices

    Returns:
        TriangleMesh or None if convex hull fails.
    """
    from scipy.spatial import ConvexHull

    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Project inlier points onto the plane
    dists = inlier_pts @ normal + d
    projected = inlier_pts - np.outer(dists, normal)  # (M, 3)

    # Build orthonormal basis {u, v} in the plane
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(normal, ref)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    # Project to 2D plane coordinates
    pts_2d = np.stack([projected @ u_axis, projected @ v_axis], axis=1)

    try:
        hull = ConvexHull(pts_2d)
    except Exception:
        return None

    hull_2d = pts_2d[hull.vertices]  # (K, 2)

    # Reconstruct hull vertices in 3D on the plane
    # For any point p on the plane: p = s*u + t*v + (-d)*normal
    hull_3d = (
        hull_2d[:, 0:1] * u_axis
        + hull_2d[:, 1:2] * v_axis
        + (-d) * normal
    ).astype(np.float32)

    # Fan triangulation from centroid
    center = hull_3d.mean(axis=0, keepdims=True)  # (1, 3)
    vertices = np.vstack([center, hull_3d])        # (K+1, 3)
    n_hull = len(hull_3d)
    triangles = [
        [0, j + 1, j % n_hull + 2] for j in range(1, n_hull)
    ]
    triangles.append([0, n_hull, 1])  # close the fan

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Paint with average color
    color_arr = np.tile(avg_color.astype(np.float64), (len(vertices), 1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(color_arr)
    mesh.compute_vertex_normals()
    return mesh


def _ransac_plane_numpy(
    points: np.ndarray,
    distance_threshold: float,
    n_iterations: int,
    seed: int = 0,
) -> tuple[tuple[float, float, float, float] | None, np.ndarray]:
    """Pure numpy RANSAC plane fitting. Avoids Open3D segment_plane heap bugs.

    Returns:
        plane_model : (a, b, c, d) unit-normal plane equation, or None if failed
        inlier_mask : (N,) bool array
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_model: tuple[float, float, float, float] | None = None

    for _ in range(n_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]

        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = float(-normal @ p0)

        dists = np.abs(points @ normal + d)
        mask = dists < distance_threshold
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask
            best_model = (float(normal[0]), float(normal[1]), float(normal[2]), d)

    return best_model, best_mask


def _fit_plane_meshes(
    points: np.ndarray,
    colors: np.ndarray,
) -> list[o3d.geometry.TriangleMesh]:
    """Iteratively extract planes from a masked point cloud via RANSAC.

    Uses pure numpy RANSAC to avoid Open3D segment_plane heap corruption bugs.
    """
    # Voxel downsample via Open3D (stable; only segment_plane triggers the bug)
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points)
    pcd_all.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd_down = pcd_all.voxel_down_sample(MESH_PLANE_VOXEL_SIZE)
    remaining_pts = np.asarray(pcd_down.points, dtype=np.float32)
    remaining_col = np.asarray(pcd_down.colors, dtype=np.float32)
    del pcd_all, pcd_down  # release Open3D objects immediately
    print(f"  After voxel downsampling: {len(remaining_pts):,} points "
          f"(was {len(points):,}, voxel={MESH_PLANE_VOXEL_SIZE}m)")

    meshes: list[o3d.geometry.TriangleMesh] = []

    for plane_idx in range(MESH_PLANE_MAX_PLANES):
        if len(remaining_pts) < MESH_PLANE_MIN_POINTS:
            break

        # Subsample for RANSAC speed; inliers are re-evaluated on full cloud after
        if len(remaining_pts) > MESH_PLANE_MAX_INPUT_POINTS:
            rng = np.random.default_rng(seed=plane_idx)
            idx = rng.choice(len(remaining_pts), MESH_PLANE_MAX_INPUT_POINTS, replace=False)
            sample_pts = remaining_pts[idx]
        else:
            sample_pts = remaining_pts

        plane_model, _ = _ransac_plane_numpy(
            sample_pts,
            distance_threshold=MESH_PLANE_RANSAC_DISTANCE,
            n_iterations=MESH_PLANE_RANSAC_ITERATIONS,
            seed=plane_idx,
        )

        if plane_model is None:
            break

        # Re-evaluate inliers on the full remaining cloud
        a, b, c, d = plane_model
        dists = np.abs(remaining_pts @ np.array([a, b, c], dtype=np.float32) + d)
        inlier_mask = dists < MESH_PLANE_RANSAC_DISTANCE
        n_inliers = int(inlier_mask.sum())

        print(f"  [plane {plane_idx + 1}] {n_inliers:,} inliers  "
              f"model=({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})")

        if n_inliers < MESH_PLANE_MIN_POINTS:
            break

        inlier_pts = remaining_pts[inlier_mask]
        avg_color = remaining_col[inlier_mask].mean(axis=0)

        mesh = _make_plane_mesh(inlier_pts, plane_model, avg_color)
        if mesh is not None:
            meshes.append(mesh)

        remaining_pts = remaining_pts[~inlier_mask]
        remaining_col = remaining_col[~inlier_mask]

    return meshes


# --------------------------------------------------------------------------- #
# Mesh post-processing
# --------------------------------------------------------------------------- #

def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Remove small disconnected clusters and rebuild as an independent mesh.

    Avoids copy.deepcopy which shares Open3D internal buffers and causes
    double-free crashes on destruction.
    """
    cc_result = mesh.cluster_connected_triangles()
    # .copy() ensures numpy owns the data, not Open3D's internal vector
    tri_clusters = np.array(cc_result[0])
    cluster_n_tris = np.array(cc_result[1])
    del cc_result

    keep_mask = cluster_n_tris[tri_clusters] >= MESH_MIN_CLUSTER_TRIANGLES

    all_tris = np.asarray(mesh.triangles)[keep_mask].copy()     # (K, 3)
    all_verts = np.asarray(mesh.vertices).copy()                # (V, 3)
    has_colors = mesh.has_vertex_colors()
    all_colors = np.asarray(mesh.vertex_colors).copy() if has_colors else None

    # Re-index vertices to remove unreferenced ones
    used = np.unique(all_tris)
    remap = np.full(len(all_verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_tris = remap[all_tris]
    new_verts = all_verts[used]

    # Remove degenerate triangles (any vertex index appears twice in same tri)
    is_degenerate = (
        (new_tris[:, 0] == new_tris[:, 1])
        | (new_tris[:, 1] == new_tris[:, 2])
        | (new_tris[:, 0] == new_tris[:, 2])
    )
    new_tris = new_tris[~is_degenerate]

    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices = o3d.utility.Vector3dVector(new_verts)
    cleaned.triangles = o3d.utility.Vector3iVector(new_tris)
    if has_colors and all_colors is not None:
        cleaned.vertex_colors = o3d.utility.Vector3dVector(all_colors[used])
    cleaned.compute_vertex_normals()
    return cleaned


def _merge_meshes(
    base: o3d.geometry.TriangleMesh,
    extras: list[o3d.geometry.TriangleMesh],
) -> o3d.geometry.TriangleMesh:
    """Concatenate base mesh with extra meshes via numpy; avoids += double-free."""
    verts_list = [np.asarray(base.vertices).copy()]
    tris_list = [np.asarray(base.triangles).copy()]
    colors_list = [
        np.asarray(base.vertex_colors).copy()
        if base.has_vertex_colors()
        else np.ones((len(verts_list[0]), 3), dtype=np.float64)
    ]

    offset = len(verts_list[0])
    for pm in extras:
        pm_verts = np.asarray(pm.vertices).copy()
        pm_tris = np.asarray(pm.triangles).copy() + offset
        pm_colors = (
            np.asarray(pm.vertex_colors).copy()
            if pm.has_vertex_colors()
            else np.ones((len(pm_verts), 3), dtype=np.float64)
        )
        verts_list.append(pm_verts)
        tris_list.append(pm_tris)
        colors_list.append(pm_colors)
        offset += len(pm_verts)

    merged = o3d.geometry.TriangleMesh()
    merged.vertices = o3d.utility.Vector3dVector(np.vstack(verts_list))
    merged.triangles = o3d.utility.Vector3iVector(np.vstack(tris_list))
    merged.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors_list))
    merged.compute_vertex_normals()
    return merged


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
    print(f"  {len(means):,} Gaussians loaded")

    # ---- Gravity alignment ----
    R_align = torch.tensor(
        load_or_compute_gravity_rotation(sparse_dir), dtype=torch.float32, device=device
    )

    # ---- Grounded-SAM2 surface masking ----
    surface_prompts = _build_surface_prompts()
    use_surface_mask = len(surface_prompts) > 0
    gdino_processor = gdino_model = sam2_predictor = None

    if use_surface_mask:
        prompt_str = " . ".join(surface_prompts) + " ."
        active = (
            f"floor={MESH_MASK_FLOOR}  ceiling={MESH_MASK_CEILING}  "
            f"ground={MESH_MASK_GROUND}  sky={MESH_MASK_SKY}"
        )
        print(f"Loading Grounded-SAM2 for surface masking ({active}) …")
        gdino_processor, gdino_model, sam2_predictor = _load_gsam2_models(device)
    else:
        print("  Surface masking disabled (all flags are False)")

    # ---- Load COLMAP cameras ----
    model = pycolmap.Reconstruction(str(sparse_dir))

    # ---- TSDF Volume ----
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=MESH_VOXEL_SIZE,
        sdf_trunc=4.0 * MESH_VOXEL_SIZE,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # Accumulate masked surface points for RANSAC plane fitting
    surface_pts_list: list[np.ndarray] = []
    surface_col_list: list[np.ndarray] = []

    n_images = len(model.images)
    print(f"Rendering & integrating {n_images} views …")

    for i, image_id in enumerate(sorted(model.images)):
        img = model.images[image_id]
        cam = model.cameras[img.camera_id]

        fx = cam.focal_length_x if hasattr(cam, "focal_length_x") else cam.focal_length
        fy = cam.focal_length_y if hasattr(cam, "focal_length_y") else fx
        cx, cy = cam.principal_point_x, cam.principal_point_y

        img_path = frames_dir / img.name
        pil_img = PILImage.open(img_path).convert("RGB")
        W, H = pil_img.size

        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device
        )

        # Camera pose (c2w with gravity alignment)
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

        # Render RGB + depth from Gaussians
        rgb, depth = _render_view(means, scales, quats, opacities, sh_coeffs, c2w, K, W, H)

        # Surface mask via Grounded-SAM2
        depth_masked = depth.copy()
        if use_surface_mask:
            image_np = np.array(pil_img)
            surface_mask = _surface_mask_gsam2(
                image_np, gdino_processor, gdino_model, sam2_predictor, prompt_str, device
            )
            depth_masked[surface_mask] = 0.0

            # Collect masked points for RANSAC plane fitting
            if MESH_FILL_PLANES:
                c2w_np = np.linalg.inv(w2c_np)
                pts, cols = _unproject_to_world(
                    depth, surface_mask, rgb, c2w_np, fx, fy, cx, cy
                )
                if len(pts) > 0:
                    surface_pts_list.append(pts)
                    surface_col_list.append(cols)

        depth_masked[depth_masked > MESH_MAX_DEPTH] = 0.0

        # Integrate into TSDF
        rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        depth_uint16 = (depth_masked * 1000).clip(0, 65535).astype(np.uint16)

        o3d_color = o3d.geometry.Image(rgb_uint8)
        o3d_depth = o3d.geometry.Image(depth_uint16)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1000.0,
            depth_trunc=MESH_MAX_DEPTH,
            convert_rgb_to_intensity=False,
        )

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
    del mesh  # release raw mesh now to avoid shared-buffer double-free

    # ---- RANSAC plane filling ----
    if MESH_FILL_PLANES and len(surface_pts_list) > 0:
        all_pts = np.concatenate(surface_pts_list, axis=0)
        all_cols = np.concatenate(surface_col_list, axis=0)
        print(f"Fitting planes to {len(all_pts):,} masked surface points …")
        plane_meshes = _fit_plane_meshes(all_pts, all_cols)
        print(f"  {len(plane_meshes)} plane(s) found")
        if plane_meshes:
            mesh_clean = _merge_meshes(mesh_clean, plane_meshes)
            del plane_meshes

    post_path = output_dir / "tsdf_fusion_post.ply"
    o3d.io.write_triangle_mesh(
        str(post_path), mesh_clean,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    n_tris = len(mesh_clean.triangles)
    del mesh_clean
    print(f"  Final mesh → {post_path}  ({n_tris:,} triangles)")
