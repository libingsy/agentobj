"""Multi-view renders of one generated object at the evaluation yaws."""

from typing import Dict, Optional, Tuple

import numpy as np

from render.gaussian_renderer import render_gaussians
from render.gs_rasterizer import GaussianSet
from utils.device import resolve_device


class SceneRenderer:
    """Orbit-camera renders of the object at each yaw (0 = front)."""

    def __init__(
        self,
        device: str = "auto",
        viewpoints=None,
        resolution: Tuple[int, int] = (256, 256),
        fovy_deg: float = 40.0,
        cam_distance: float = 6.0,
    ):
        self.device = resolve_device(device)
        self.viewpoints = list(viewpoints or [0, 90, 180, 270])
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.fovy_deg = fovy_deg
        self.cam_distance = cam_distance

    def render_camera(self, gaussians, yaw: float, resolution: Optional[Tuple[int, int]] = None) -> np.ndarray:
        return render_gaussians(
            gaussians if gaussians is not None else np.zeros((0, 14), dtype=np.float32),
            yaw_deg=yaw,
            resolution=resolution or self.resolution,
            cam_distance=self.cam_distance,
            fovy_deg=self.fovy_deg,
            device=self.device,
        )

    def render_all_views(self, obj: Dict) -> Dict[int, Dict]:
        gaussians = obj.get("gaussians")
        gs = GaussianSet.from_packed(gaussians, device=self.device) if gaussians is not None else None
        return {
            int(yaw): {"image": self.render_camera(gs, yaw), "yaw": int(yaw)}
            for yaw in self.viewpoints
        }
