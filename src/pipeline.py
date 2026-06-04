"""
CLI entry point.

Usage:
  recon-gs pipeline --video INPUT.mp4 --output OUTPUT_DIR [--resume] [--from-step STEP]
"""

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()

_STEPS = ["extract", "mask", "sfm", "train"]

_STATE_FILE = "pipeline.json"


class Step(str, Enum):
    extract = "extract"
    mask = "mask"
    sfm = "sfm"
    train = "train"


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #

def _load_state(output_dir: Path) -> dict:
    path = output_dir / _STATE_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {s: False for s in _STEPS}


def _save_state(output_dir: Path, state: dict) -> None:
    (output_dir / _STATE_FILE).write_text(json.dumps(state, indent=2))


def _mark_done(output_dir: Path, step: str) -> None:
    state = _load_state(output_dir)
    state[step] = True
    _save_state(output_dir, state)


def _is_done(output_dir: Path, step: str) -> bool:
    return _load_state(output_dir).get(step, False)


def _reset_from(output_dir: Path, from_step: str) -> None:
    state = _load_state(output_dir)
    reset = False
    for s in _STEPS:
        if s == from_step:
            reset = True
        if reset:
            state[s] = False
    _save_state(output_dir, state)


# --------------------------------------------------------------------------- #
# Main command
# --------------------------------------------------------------------------- #

@app.command()
def pipeline(
    video: Path = typer.Option(..., help="Input video file"),
    output: Path = typer.Option(..., help="Output directory"),
    resume: bool = typer.Option(False, "--resume", help="Skip already-completed steps"),
    from_step: Optional[Step] = typer.Option(
        None, "--from-step", help="Re-run from this step onward (extract|mask|sfm|train)"
    ),
) -> None:
    if not video.exists():
        console.print(f"[red]Video not found: {video}[/red]")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)

    frames_dir = output / "frames"
    masks_dir = output / "masks"
    colmap_dir = output / "colmap"
    ply_path = output / "gaussian.ply"

    # Reset state for re-run from a specific step
    if from_step is not None:
        _reset_from(output, from_step.value)
        resume = True

    # ---------------------------------------------------------------------- #
    # Step 1: Extract frames
    # ---------------------------------------------------------------------- #
    if resume and _is_done(output, "extract"):
        console.print("[dim]  [skip] extract frames[/dim]")
    else:
        console.print(Panel("Step 1 / 4 — Extract frames", style="bold blue"))
        from recon_gs.extract_frames import extract_frames
        n = extract_frames(video, frames_dir)
        console.print(f"  Extracted {n} frames → {frames_dir}")
        _mark_done(output, "extract")

    # ---------------------------------------------------------------------- #
    # Step 2: Mask dynamic objects
    # ---------------------------------------------------------------------- #
    if resume and _is_done(output, "mask"):
        console.print("[dim]  [skip] dynamic masking[/dim]")
    else:
        console.print(Panel("Step 2 / 4 — Mask dynamic objects", style="bold blue"))
        from recon_gs.mask_dynamic import mask_frames
        mask_frames(frames_dir, masks_dir)
        console.print(f"  Masks saved → {masks_dir}")
        _mark_done(output, "mask")

    # ---------------------------------------------------------------------- #
    # Step 3: SfM (hloc + COLMAP)
    # ---------------------------------------------------------------------- #
    if resume and _is_done(output, "sfm"):
        console.print("[dim]  [skip] SfM[/dim]")
    else:
        console.print(Panel("Step 3 / 4 — Camera pose estimation (hloc + COLMAP)", style="bold blue"))
        from recon_gs.run_sfm import run_sfm
        model = run_sfm(frames_dir, masks_dir, colmap_dir)
        console.print(f"  Registered {len(model.images)} images, {len(model.points3D)} 3D points")
        _mark_done(output, "sfm")

    # ---------------------------------------------------------------------- #
    # Step 4: 3DGS training
    # ---------------------------------------------------------------------- #
    if resume and _is_done(output, "train"):
        console.print("[dim]  [skip] 3DGS training[/dim]")
    else:
        console.print(Panel("Step 4 / 4 — 3DGS training (gsplat)", style="bold blue"))
        from recon_gs.train_3dgs import train_3dgs
        train_3dgs(colmap_dir, frames_dir, masks_dir, ply_path)
        _mark_done(output, "train")

    console.print(Panel(f"[green]Done![/green]  Output: {ply_path}", style="bold green"))
