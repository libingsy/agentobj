"""Place a generated object and composite it into the camera and the LiDAR.

Object preprocessing (trim the LGM background, squeeze to a standing body, move
into the LiDAR frame), the LiDAR insertion (ray-cast the body into each sweep,
optionally smeared along a walk to match how a moving pedestrian's returns
accumulate across sweeps), and the camera insertion (paste the render over the
projected body). Pure numpy/torch, unit tested without nuScenes; Python 3.8
compatible.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from render.gs_rasterizer import GaussianSet, composite, footprint, render
from utils.geometry import (
    fit_standing_body,
    project_lidar_to_image,
    subsample_pedestrian,
    visible_lidar_returns,
)

from data.lidar_sensor import UP, BeamModel, LidarSweep, _ray_frame, cast_rays


# --------------------------------------------------------------------------- pose

def object_rotation(translation: Sequence[float], yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation from the y-up object frame to the LiDAR frame (z up).

    At yaw 0 the object's front (+z) faces the sensor at the origin. A positive
    yaw turns the object counter-clockwise seen from above.
    """
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    toward = -t[:2]
    norm = float(np.linalg.norm(toward))
    toward = toward / norm if norm > 1e-9 else np.array([-1.0, 0.0])
    a = np.deg2rad(float(yaw_deg))
    c, s = np.cos(a), np.sin(a)
    front = np.array([c * toward[0] - s * toward[1], s * toward[0] + c * toward[1], 0.0])
    side = np.cross(UP, front)
    return np.stack([side, UP, front], axis=1)


def trim_background(packed, radius: float = 0.78, chroma: float = 0.05, min_keep: int = 500):
    """Drop the gray cube LGM leaves around the person.

    Those Gaussians sit far from the body axis, are almost gray, and are opaque.
    Rasterized, they become a square in the camera image; ray-cast, they become a
    slab around the body. Removing them leaves the person. Colored Gaussians are
    kept even when they are far from the axis.
    """
    g = np.asarray(packed, dtype=np.float32)
    if g.ndim != 2 or g.shape[1] < 14 or len(g) < min_keep:
        return g
    radial = np.linalg.norm(g[:, [0, 2]], axis=1)
    rgb = g[:, 11:14]
    gray = (rgb.max(axis=1) - rgb.min(axis=1)) < chroma
    keep = ~((radial > radius) & gray)
    if int(keep.sum()) < min_keep:
        return g
    kept = np.ascontiguousarray(g[keep])
    if not getattr(trim_background, "_announced", False):
        print(f"[insert] trimmed background gaussians {len(g)} -> {len(kept)}", flush=True)
        trim_background._announced = True
    return kept


def squeeze_standing(packed, max_width: float = 0.6):
    """Same horizontal squeeze as `fit_standing_body`, applied to packed Gaussians.

    Means and the x/z scales shrink together, so the ray-cast silhouette is a
    standing person and not the 1.7 m LGM cube.
    """
    g = np.array(packed, dtype=np.float32, copy=True)
    if g.ndim != 2 or g.shape[1] < 14 or len(g) == 0:
        return g
    xyz = g[:, :3]
    xyz[:, 0] -= 0.5 * (float(xyz[:, 0].min()) + float(xyz[:, 0].max()))
    xyz[:, 2] -= 0.5 * (float(xyz[:, 2].min()) + float(xyz[:, 2].max()))
    extent = max(float(xyz[:, 0].max() - xyz[:, 0].min()), float(xyz[:, 2].max() - xyz[:, 2].min()), 1e-3)
    if extent > max_width:
        scale = max_width / extent
        xyz[:, 0] *= scale
        xyz[:, 2] *= scale
        g[:, 4] *= scale
        g[:, 6] *= scale
    g[:, :3] = xyz
    return g


def place_gaussians(gaussians, translation: Sequence[float], yaw_deg: float, device: str = "cpu") -> GaussianSet:
    """Activated y-up object Gaussians moved into the LiDAR frame."""
    if not isinstance(gaussians, GaussianSet):
        gaussians = trim_background(gaussians)
    gs = gaussians if isinstance(gaussians, GaussianSet) else GaussianSet.from_packed(gaussians, device=device)
    return gs.transformed(object_rotation(translation, yaw_deg), np.asarray(translation, dtype=np.float64))

def insert_into_sweep(
    gs: GaussianSet,
    sweep: LidarSweep,
    beams: BeamModel,
    intensity: float,
    margin_deg: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """LiDAR returns of the object for one sweep, and a keep-mask over `sweep.points`.

    Returns object points (M, 4) in the keyframe frame and a boolean mask that
    drops background returns hidden behind the object.
    """
    keep = np.ones(len(sweep.points), dtype=bool)
    means = gs.means.detach().cpu().numpy()
    solid = gs.opacity.detach().cpu().numpy() > 0.1
    if not solid.any():
        return np.zeros((0, 4), dtype=np.float32), keep
    local = sweep.sensor_frame(means[solid])
    az = np.arctan2(local[:, 1], local[:, 0])
    az_c = float(np.arctan2(np.sin(az).mean(), np.cos(az).mean()))
    rel = np.angle(np.exp(1j * (az - az_c)))
    el = np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1))
    m = np.deg2rad(margin_deg)
    rel_lo, rel_hi = rel.min() - m, rel.max() + m
    el_lo, el_hi = el.min() - m, el.max() + m

    frame = _ray_frame(gs, sweep.origin)

    # Beam rays through the object's angular window.
    bg_local = sweep.sensor_frame(sweep.points[:, :3])
    phases = beams.phases(bg_local, sweep.ring)
    step = beams.azimuth_step
    dirs = []
    for r, e in enumerate(beams.elevation):
        if not np.isfinite(e) or e < el_lo or e > el_hi:
            continue
        lo = az_c + rel_lo - phases[r]
        hi = az_c + rel_hi - phases[r]
        ks = np.arange(np.ceil(lo / step), np.floor(hi / step) + 1)
        a = phases[r] + ks * step
        dirs.append(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.full_like(a, np.sin(e))], axis=1))
    obj = np.zeros((0, 4), dtype=np.float32)
    if dirs:
        d_sensor = np.concatenate(dirs, axis=0)
        d_kf = d_sensor @ sweep.rotation.T
        rng = cast_rays(gs, sweep.origin, d_kf, frame)
        hit = np.isfinite(rng)
        if hit.any():
            xyz = sweep.origin[None, :] + rng[hit, None] * d_kf[hit]
            obj = np.concatenate([xyz, np.full((len(xyz), 1), float(intensity))], axis=1).astype(np.float32)

    # Background returns on rays that now hit the object first.
    bg_az = np.arctan2(bg_local[:, 1], bg_local[:, 0])
    bg_rel = np.angle(np.exp(1j * (bg_az - az_c)))
    bg_rng = np.linalg.norm(bg_local, axis=1)
    bg_el = np.arctan2(bg_local[:, 2], np.maximum(np.linalg.norm(bg_local[:, :2], axis=1), 1e-9))
    window = (bg_rel >= rel_lo) & (bg_rel <= rel_hi) & (bg_el >= el_lo) & (bg_el <= el_hi) & (bg_rng > 0.5)
    idx = np.nonzero(window)[0]
    if len(idx):
        d_kf = (bg_local[idx] / bg_rng[idx, None]) @ sweep.rotation.T
        obj_rng = cast_rays(gs, sweep.origin, d_kf, frame)
        hidden = np.isfinite(obj_rng) & (obj_rng < bg_rng[idx] - 0.05)
        keep[idx[hidden]] = False
    return obj, keep


def insert_lidar(
    gs: GaussianSet,
    sweeps: Sequence[LidarSweep],
    beams: BeamModel,
    intensity: float,
    velocity: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Stack all sweeps with the object inserted: (N, 5) x, y, z, intensity, time lag.

    With `velocity` (m/s, keyframe LiDAR frame) the object is placed at its
    position at each sweep's time: a sweep with lag t sees it at
    `translation - velocity * t`. The ten-sweep accumulation is then smeared
    along the walk, matching how a real moving pedestrian's returns spread across
    sweeps rather than piling into one position. The keyframe (lag 0) is unmoved,
    so the camera image and object box stay at `translation`.
    """
    vel = None if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(3)
    moving = vel is not None and float(np.linalg.norm(vel)) > 1e-6
    eye = np.eye(3)
    stacked = []
    stats = {"object_points": [], "removed": []}
    for sweep in sweeps:
        gs_sweep = gs
        if moving and abs(float(sweep.time_lag)) > 1e-6:
            gs_sweep = gs.transformed(eye, -vel * float(sweep.time_lag))
        obj, keep = insert_into_sweep(gs_sweep, sweep, beams, intensity)
        pts = np.concatenate([sweep.points[keep], obj], axis=0) if len(obj) else sweep.points[keep]
        lag = np.full((len(pts), 1), float(sweep.time_lag), dtype=np.float32)
        stacked.append(np.concatenate([pts.astype(np.float32), lag], axis=1))
        stats["object_points"].append(int(len(obj)))
        stats["removed"].append(int((~keep).sum()))
    return np.concatenate(stacked, axis=0), stats


def walk_velocity(
    translation: Sequence[float],
    yaw_deg: float,
    speed: float = 1.4,
    mode: str = "facing",
) -> np.ndarray:
    """Horizontal pedestrian velocity in the LiDAR frame (m/s).

    `facing`: along the direction the body faces at this yaw (its +z front),
    so the walk and the pose agree. `lateral`: perpendicular to the
    sensor->object ray, a crossing pedestrian. Returns a zero vector when
    `speed` is non-positive, which falls back to a static insertion.
    """
    if speed <= 0:
        return np.zeros(3)
    if mode == "lateral":
        t = np.asarray(translation, dtype=np.float64).reshape(-1)
        d = np.cross(UP, np.array([t[0], t[1], 0.0]))
    else:
        d = object_rotation(translation, yaw_deg)[:, 2].astype(np.float64).copy()
        d[2] = 0.0
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.zeros(3)
    return (d / n) * float(speed)


def stack_sweeps(sweeps: Sequence[LidarSweep]) -> np.ndarray:
    """The unmodified cloud in the same layout as `insert_lidar`."""
    return np.concatenate(
        [
            np.concatenate([s.points.astype(np.float32), np.full((len(s.points), 1), s.time_lag, np.float32)], axis=1)
            for s in sweeps
        ],
        axis=0,
    )


# --------------------------------------------------------------------------- camera

def place_body(points_yup, translation: Sequence[float], yaw_deg: float) -> np.ndarray:
    """Fitted standing body in the keyframe LiDAR frame, feet at `translation`."""
    body = fit_standing_body(points_yup)
    if len(body) == 0:
        return body
    rotation = object_rotation(translation, yaw_deg)
    return (body.astype(np.float64) @ rotation.T).astype(np.float32) + np.asarray(translation, dtype=np.float32)


def pedestrian_shell(points_lidar, n: int = 40) -> np.ndarray:
    """A sparse subset of the body's returns: the visible returns, then `n` of them."""
    return subsample_pedestrian(visible_lidar_returns(points_lidar), n)


def _ellipse_alpha(height: int, width: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, height)[:, None]
    xs = np.linspace(-1.0, 1.0, width)[None, :]
    mask = (xs ** 2 + ys ** 2) <= 1.0
    alpha = np.clip(1.0 - (xs ** 2 + ys ** 2), 0.0, 1.0)
    return (alpha * mask).astype(np.float32)


def _paste_alpha(patch: np.ndarray) -> np.ndarray:
    """Ellipse mask. A flat render backdrop (LGM gray) stays transparent."""
    ellipse = _ellipse_alpha(patch.shape[0], patch.shape[1])
    rgb = patch[..., :3].astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    if float(border.std()) > 12.0:
        return ellipse[..., None]
    bg = np.median(border, axis=0)
    foreground = np.linalg.norm(rgb - bg, axis=2) > 18.0
    if float(foreground.mean()) < 0.02:
        return ellipse[..., None]
    return (ellipse * foreground.astype(np.float32))[..., None]


def paste_render(image, points_lidar, render, lidar_to_cam, intrinsics):
    """Paste `render` over the projected body. The gray backdrop stays transparent.

    The box is the visible person, from pixels whose paste alpha is above 0.5.
    """
    from PIL import Image

    image = np.asarray(image)
    canvas = image.copy()
    pts = np.asarray(points_lidar, dtype=np.float32)
    uv, depth = project_lidar_to_image(pts, lidar_to_cam, intrinsics)
    visible = depth > 0.1
    if not visible.any():
        return canvas, None
    pix = uv[visible]
    h, w = canvas.shape[:2]
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
    if not inside.any():
        return canvas, None
    pix = pix[inside]
    patch = np.asarray(render)
    if patch.dtype != np.uint8:
        patch = np.clip(patch, 0, 255).astype(np.uint8)
    if patch.ndim == 2:
        patch = np.repeat(patch[..., None], 3, axis=2)
    bh = max(int(round(float(pix[:, 1].max() - pix[:, 1].min()))), 8)
    bw = max(int(round(bh * (patch.shape[1] / max(patch.shape[0], 1)))), 8)
    x0 = int(round(float(np.median(pix[:, 0])) - bw / 2))
    y0 = int(round(float(np.median(pix[:, 1])) - bh / 2))
    x1, y1 = x0 + bw, y0 + bh
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return canvas, None
    resized = np.array(Image.fromarray(patch[..., :3]).resize((bw, bh), Image.BILINEAR))
    crop = resized[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    alpha = _paste_alpha(resized)[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    region = canvas[sy0:sy1, sx0:sx1, :3].astype(np.float32)
    canvas[sy0:sy1, sx0:sx1, :3] = (region * (1.0 - alpha) + crop.astype(np.float32) * alpha).astype(canvas.dtype)
    ys, xs = np.nonzero(alpha[..., 0] > 0.5)
    if len(ys) == 0:
        return canvas, None
    box = [float(sx0 + xs.min()), float(sy0 + ys.min()), float(sx0 + xs.max() + 1), float(sy0 + ys.max() + 1)]
    return canvas, box


def insert_camera(image: np.ndarray, gs: GaussianSet, lidar_to_cam, intrinsics, pad: int = 4):
    """Composite the object into a camera image.

    Returns the new image and the object's pixel box (x0, y0, x1, y1) from the
    rendered alpha (> 0.5), or None when the object is not in view.
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    box = footprint(gs, lidar_to_cam, intrinsics)
    if box is None:
        return image.copy(), None
    x0 = int(np.floor(max(box[0] - pad, 0)))
    y0 = int(np.floor(max(box[1] - pad, 0)))
    x1 = int(np.ceil(min(box[2] + pad, w - 1)))
    y1 = int(np.ceil(min(box[3] + pad, h - 1)))
    if x1 <= x0 or y1 <= y0:
        return image.copy(), None
    out = render(gs, lidar_to_cam, intrinsics, y1 - y0 + 1, x1 - x0 + 1, x0=x0, y0=y0)
    result = composite(image, out, x0, y0)
    ys, xs = np.nonzero(out["alpha"] > 0.5)
    if len(ys) == 0:
        return result, None
    return result, [float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)]


