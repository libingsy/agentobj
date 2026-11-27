"""LiDAR sensor model and ray-casting against the object Gaussians.

`LidarSweep` carries one sweep in the keyframe frame; `BeamModel` measures the
per-ring elevation and the azimuth firing grid; `cast_rays` returns the range
along each ray to the Gaussians' median surface. Pure numpy/torch, Python 3.8
compatible (the MambaFusion worker imports it).
"""

from dataclasses import dataclass, field

import numpy as np

from render.gs_rasterizer import GaussianSet, look_at, trace

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class LidarSweep:
    """One sweep in the keyframe LiDAR frame.

    `points` is (N, 4): x, y, z, intensity. `ring` is the laser index per point.
    `origin` and `rotation` map the sweep's sensor frame into the keyframe frame.
    """

    points: np.ndarray
    ring: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    time_lag: float = 0.0

    def sensor_frame(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float64) - self.origin) @ self.rotation


@dataclass
class BeamModel:
    """Elevation of each ring and the azimuth step between firings (radians)."""

    elevation: np.ndarray
    azimuth_step: float

    @classmethod
    def measure(cls, sensor_xyz: np.ndarray, ring: np.ndarray, n_rings: int = 32) -> "BeamModel":
        xyz = np.asarray(sensor_xyz, dtype=np.float64)
        rng = np.linalg.norm(xyz[:, :2], axis=1)
        keep = rng > 2.5
        xyz, ring, rng = xyz[keep], np.asarray(ring)[keep].astype(int), rng[keep]
        elev = np.arctan2(xyz[:, 2], rng)
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        table = np.full(n_rings, np.nan)
        steps = []
        for r in range(n_rings):
            sel = ring == r
            if sel.sum() < 20:
                continue
            table[r] = float(np.median(elev[sel]))
            d = np.diff(np.sort(az[sel]))
            d = d[(d > 1e-5) & (d < 0.05)]
            if len(d):
                steps.append(np.median(d))
        step = float(np.median(steps)) if steps else np.deg2rad(1.0 / 3.0)
        return cls(elevation=table, azimuth_step=step)

    def phases(self, sensor_xyz: np.ndarray, ring: np.ndarray) -> np.ndarray:
        """Azimuth offset of each ring's firing grid in this sweep (circular mean)."""
        xyz = np.asarray(sensor_xyz, dtype=np.float64)
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        ring = np.asarray(ring).astype(int)
        out = np.zeros(len(self.elevation))
        theta = 2.0 * np.pi * np.mod(az, self.azimuth_step) / self.azimuth_step
        for r in range(len(self.elevation)):
            sel = ring == r
            if sel.sum() < 5:
                continue
            ang = np.arctan2(np.sin(theta[sel]).mean(), np.cos(theta[sel]).mean())
            out[r] = np.mod(ang, 2.0 * np.pi) / (2.0 * np.pi) * self.azimuth_step
        return out


def _ray_frame(gs: GaussianSet, origin: np.ndarray, focal: float = 2000.0):
    """A pinhole camera at `origin` looking at the object, used to trace rays."""
    weights = gs.opacity.detach().cpu().numpy()
    means = gs.means.detach().cpu().numpy()
    center = (means * weights[:, None]).sum(0) / max(float(weights.sum()), 1e-9)
    w2c = look_at(origin, center, UP)
    k = np.array([[focal, 0.0, 0.0], [0.0, focal, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return w2c, k


def cast_rays(gs: GaussianSet, origin: np.ndarray, dirs: np.ndarray, frame=None) -> np.ndarray:
    """Range along each unit ray from `origin` to the Gaussians' median surface (inf on a miss)."""
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    if len(dirs) == 0:
        return np.zeros(0)
    w2c, k = frame if frame is not None else _ray_frame(gs, origin)
    d_cam = dirs @ w2c[:3, :3].astype(np.float64).T
    valid = d_cam[:, 2] > 0.05
    uv = np.zeros((len(dirs), 2))
    uv[valid] = k[0, 0] * d_cam[valid, :2] / d_cam[valid, 2:3]
    out = trace(gs, w2c, k, uv.astype(np.float32))
    depth = out["depth"].detach().cpu().numpy().astype(np.float64)
    rng = np.full(len(dirs), np.inf)
    hit = valid & np.isfinite(depth)
    rng[hit] = depth[hit] / d_cam[hit, 2]
    return rng

