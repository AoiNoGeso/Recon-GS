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
    post_path = output_dir / "tsdf_fusion_post.ply"
    o3d.io.write_triangle_mesh(
        str(post_path), mesh_clean,
        write_triangle_uvs=True,
        write_vertex_colors=True,
        write_vertex_normals=True,
    )
    print(f"  Cleaned mesh → {post_path}  ({len(mesh_clean.triangles):,} triangles)")
