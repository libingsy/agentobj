"""Geometry helpers shared by the generator, renderer, and scene insertion.

Object frame is y-up, metric, feet at y=0, front facing +z (the yaw-0 view).
LiDAR frames are z-up (nuScenes LIDAR_TOP: x right, y forward).
"""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


LGM_AZIMUTHS = (0, 90, 180, 270)


def as_numpy_points(points) -> np.ndarray:
    if points is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(points, torch.Tensor):
        points = points.detach().float().cpu().numpy()
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected points (N, 3+), got {arr.shape}")
    return arr[:, :3]


def box_stats_from_points(points) -> Dict[str, float]:
    """Axis-aligned size in the y-up object frame: width (x), height (y), length (z)."""
    pts = as_numpy_points(points)
    if len(pts) == 0:
        return {"w": 0.0, "h": 0.0, "l": 0.0}
    ext = pts.max(axis=0) - pts.min(axis=0)
    return {"w": float(ext[0]), "h": float(ext[1]), "l": float(ext[2])}


def fit_standing_body(points, max_width: float = 0.6) -> np.ndarray:
    """Keep the height and squeeze the horizontal size to a standing person.

    Gaussian means of a person holding a box fill a cube about 1.7 m on a side.
    A fusion detector sees that volume as a solid object, not a pedestrian.
    """
    pts = as_numpy_points(points).copy()
    if len(pts) == 0:
        return pts
    pts[:, 0] -= 0.5 * (float(pts[:, 0].min()) + float(pts[:, 0].max()))
    pts[:, 2] -= 0.5 * (float(pts[:, 2].min()) + float(pts[:, 2].max()))
    extent = max(float(pts[:, 0].max() - pts[:, 0].min()), float(pts[:, 2].max() - pts[:, 2].min()), 1e-3)
    if extent > max_width:
        scale = max_width / extent
        pts[:, 0] *= scale
        pts[:, 2] *= scale
    return pts.astype(np.float32)


def visible_lidar_returns(
    points_xyz,
    n_azimuth: int = 1080,
    n_beams: int = 32,
    elev_deg: Sequence[float] = (-30.67, 10.67),
    thickness: float = 0.45,
    max_per_bin: int = 3,
) -> np.ndarray:
    """Lidar returns from the origin, with the depth of a body rather than a sheet.

    One angular bin keeps up to `max_per_bin` points within `thickness` meters
    of the first hit. A pedestrian in this frame is about half a meter deep;
    a single facing layer is a flat panel and gets called a barrier.
    """
    pts = as_numpy_points(points_xyz).astype(np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    depth = np.linalg.norm(pts, axis=1)
    valid = depth > 0.5
    pts, depth = pts[valid], depth[valid]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    azimuth = np.arctan2(pts[:, 1], pts[:, 0])
    elevation = np.arcsin(np.clip(pts[:, 2] / np.maximum(depth, 1e-6), -1.0, 1.0))
    el0, el1 = np.deg2rad(float(elev_deg[0])), np.deg2rad(float(elev_deg[1]))
    keep = (elevation >= el0) & (elevation <= el1)
    pts, depth, azimuth, elevation = pts[keep], depth[keep], azimuth[keep], elevation[keep]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    az_id = np.clip(((azimuth + np.pi) / (2 * np.pi) * n_azimuth).astype(np.int32), 0, n_azimuth - 1)
    el_id = np.clip(((elevation - el0) / max(el1 - el0, 1e-6) * n_beams).astype(np.int32), 0, n_beams - 1)
    bins = el_id.astype(np.int64) * int(n_azimuth) + az_id
    order = np.lexsort((depth, bins))
    pts, depth, bins = pts[order], depth[order], bins[order]
    chosen = []
    start = 0
    for end in range(1, len(bins) + 1):
        if end < len(bins) and bins[end] == bins[start]:
            continue
        first = depth[start]
        slab = thickness / max(max_per_bin, 1)
        for piece in range(max_per_bin):
            lo = first + piece * slab
            hi = first + thickness if piece == max_per_bin - 1 else lo + slab
            for k in range(start, end):
                if lo <= depth[k] < hi or (piece == max_per_bin - 1 and depth[k] == first + thickness):
                    chosen.append(k)
                    break
        start = end
    if not chosen:
        return np.zeros((0, 3), dtype=np.float32)
    return pts[np.array(chosen, dtype=np.int64)].astype(np.float32)


def subsample_pedestrian(points_xyz, n: int = 40) -> np.ndarray:
    """Keep `n` returns spread over the height.

    A few hundred points fill every voxel of the body and MambaFusion ignores
    the cluster. About 40 returns on this sweep score as a pedestrian; 80
    already falls off.
    """
    pts = as_numpy_points(points_xyz)
    if len(pts) <= n:
        return pts
    order = np.argsort(pts[:, 2])
    pick = order[np.linspace(0, len(order) - 1, int(n)).astype(int)]
    return pts[pick]


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    """Rotate a y-up point cloud around the up axis."""
    a = np.deg2rad(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def look_at_basis(
    cam_pos: Sequence[float],
    look_at: Sequence[float],
    world_up: Sequence[float] = (0.0, 1.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV camera basis (x right, y down, z forward) expressed in the y-up world.

    Returns cam_pos, right, down, forward, each shape (3,).
    """
    cam = np.asarray(cam_pos, dtype=np.float32)
    target = np.asarray(look_at, dtype=np.float32)
    up = np.asarray(world_up, dtype=np.float32)
    forward = target - cam
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-8)
    down = np.cross(forward, right)
    down = down / max(np.linalg.norm(down), 1e-8)
    return cam, right, down, forward


def world_to_camera(points: np.ndarray, cam, right, down, forward) -> np.ndarray:
    rel = points - cam
    return np.stack(
        [rel @ right, rel @ down, rel @ forward],
        axis=1,
    ).astype(np.float32)


def project_lidar_to_image(
    points_xyz: np.ndarray,
    lidar_to_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points with a 4x4 extrinsic and a 3x3 intrinsic.

    Returns pixel coordinates (N, 2) and camera-frame depth (N,).
    """
    pts = np.asarray(points_xyz, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    homo = np.concatenate([pts[:, :3], np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    cam = (np.asarray(lidar_to_cam, dtype=np.float32) @ homo.T).T
    depth = cam[:, 2]
    pix = (np.asarray(intrinsics, dtype=np.float32) @ cam[:, :3].T).T
    uv = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    return uv.astype(np.float32), depth.astype(np.float32)


def select_views_by_azimuth(images: Sequence, azimuths: Iterable[float], needed: Sequence[int] = LGM_AZIMUTHS):
    """Pick the views LGM was trained on, in azimuth order 0, 90, 180, 270."""
    mapping = {int(round(float(a))): img for a, img in zip(azimuths, images)}
    missing = [a for a in needed if a not in mapping]
    if missing:
        raise KeyError(
            f"Missing azimuths {missing}. Generated {sorted(mapping)} but LGM needs {list(needed)}."
        )
    return [mapping[a] for a in needed]


def pinhole_intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
    return np.array([[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
