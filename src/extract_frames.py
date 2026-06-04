"""Step 1: Extract frames from input video at target FPS."""

from pathlib import Path

import ffmpeg

from recon_gs.config import EXTRACT_FPS


def extract_frames(video_path: Path, output_dir: Path, fps: int = EXTRACT_FPS) -> int:
    """
    Extract frames from video at the given FPS and save as PNG files.
    Returns the number of extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "%06d.png")

    (
        ffmpeg
        .input(str(video_path))
        .filter("fps", fps=fps)
        .output(pattern, start_number=0)
        .overwrite_output()
        .run(quiet=True)
    )

    frames = sorted(output_dir.glob("*.png"))
    return len(frames)
