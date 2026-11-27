"""Object-centric renders of activated 3D Gaussians.

An orbit camera looks at a y-up object from `yaw_deg` (0 = the object's front,
+z). Used to save the views of each candidate and, without a nuScenes scene, as
the 2D detector's input in the CPU quick start. Rasterization is the exact
splatting in `render/gs_rasterizer.py`.
"""

from typing import Optional, Tuple

import numpy as np

from render.gs_rasterizer import GaussianSet, look_at, render
from utils.geometry import pinhole_intrinsics


def camera_pose(yaw_deg: float, distance: float, height: float, look_height: float) -> np.ndarray:
    """World-to-camera matrix of the orbit camera. At yaw 0 the camera sits on +z."""
    a = np.deg2rad(yaw_deg)
    cam = np.array([distance * np.sin(a), height, distance * np.cos(a)], dtype=np.float64)
    return look_at(cam, [0.0, look_height, 0.0], [0.0, 1.0, 0.0])


def render_gaussians(
    gaussians,
    yaw_deg: float = 0.0,
    resolution: Tuple[int, int] = (256, 256),
    cam_distance: float = 6.0,
    cam_height: float = 1.2,
    look_height: float = 0.85,
    fovy_deg: float = 40.0,
    background: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """RGB uint8 image of the Gaussians from `yaw_deg`, composited over `background`."""
    height, width = int(resolution[0]), int(resolution[1])
    bg = _background(height, width) if background is None else np.asarray(background, dtype=np.float32)
    if bg.max() > 1.0:
        bg = bg / 255.0
    gs = gaussians if isinstance(gaussians, GaussianSet) else GaussianSet.from_packed(gaussians, device=device)
    if len(gs) == 0:
        return (bg * 255.0).clip(0, 255).astype(np.uint8)
    w2c = camera_pose(yaw_deg, cam_distance, cam_height, look_height)
    out = render(gs, w2c, pinhole_intrinsics(width, height, fovy_deg), height, width)
    image = out["rgb"] + (1.0 - out["alpha"][..., None]) * bg[..., :3]
    return (image * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def _background(height: int, width: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.float32)
    horizon = int(height * 0.58)
    image[:horizon] = (0.62, 0.72, 0.82)
    image[horizon:] = (0.38, 0.38, 0.36)
    return image
