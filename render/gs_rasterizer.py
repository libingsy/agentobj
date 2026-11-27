"""3D Gaussian splatting forward pass in plain PyTorch.

This follows the forward rasterizer of `diff-gaussian-rasterization` (Kerbl et
al. 2023): EWA projection of each anisotropic Gaussian, a 0.3 px low-pass
dilation, depth sorting, and front-to-back alpha compositing with alpha capped
at 0.99 and dropped below 1/255. It is written in PyTorch so that it runs in
both environments of this repository (torch 2.5 for Agent-OBJ, torch 2.1 for
the MambaFusion worker) and on CPU, without a CUDA extension. No gradients are
needed: Agent-OBJ is a black-box attack.

Packed Gaussians follow LGM: (N, 14) = xyz, opacity, scale(3), quaternion
(w, x, y, z), rgb, all already activated.

Cameras use the OpenCV convention: x right, y down, z forward. A camera is a
4x4 world-to-camera matrix and a 3x3 intrinsic matrix. Pixel (u, v) is sampled
at integer coordinates, the same convention as the nuScenes intrinsics.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
LOW_PASS = 0.3


def _tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions in (w, x, y, z) order to (N, 3, 3) rotations."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    rot = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return rot.view(-1, 3, 3)


@dataclass
class GaussianSet:
    """Activated Gaussians: means (N, 3), covariances (N, 3, 3), opacity (N,), rgb (N, 3)."""

    means: torch.Tensor
    covs: torch.Tensor
    opacity: torch.Tensor
    rgb: torch.Tensor

    @classmethod
    def from_packed(cls, packed, device: str = "cpu") -> "GaussianSet":
        g = _tensor(packed, device)
        if g.ndim != 2 or g.shape[1] < 14:
            raise ValueError(f"Expected packed gaussians (N, 14), got {tuple(g.shape)}")
        rot = quat_to_rotmat(g[:, 7:11])
        m = rot * g[:, 4:7].abs()[:, None, :]
        return cls(
            means=g[:, 0:3].contiguous(),
            covs=m @ m.transpose(1, 2),
            opacity=g[:, 3].clamp(0.0, 1.0).contiguous(),
            rgb=g[:, 11:14].clamp(0.0, 1.0).contiguous(),
        )

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def device(self):
        return self.means.device

    def transformed(self, rotation, translation) -> "GaussianSet":
        """Rigidly move the set: x -> R x + t."""
        rot = _tensor(rotation, self.device)
        trans = _tensor(translation, self.device).reshape(1, 3)
        return GaussianSet(
            means=self.means @ rot.T + trans,
            covs=rot @ self.covs @ rot.T,
            opacity=self.opacity,
            rgb=self.rgb,
        )


@dataclass
class Projected:
    order: torch.Tensor     # indices into the GaussianSet, near to far
    depth: torch.Tensor     # camera z, sorted
    center: torch.Tensor    # (M, 2) pixel coordinates
    conic: torch.Tensor     # (M, 3) inverse 2D covariance a, b, c
    radius: torch.Tensor    # (M,) 3-sigma radius in pixels
    opacity: torch.Tensor
    rgb: torch.Tensor


def project(gs: GaussianSet, world_to_cam, intrinsics, near: float = 0.05) -> Projected:
    """EWA-project every Gaussian in front of the camera and sort by depth."""
    device = gs.device
    w2c = _tensor(world_to_cam, device)
    k = _tensor(intrinsics, device)
    rot, trans = w2c[:3, :3], w2c[:3, 3]
    cam = gs.means @ rot.T + trans
    z = cam[:, 2]
    keep = (z > near) & (gs.opacity >= ALPHA_MIN)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        empty = torch.zeros(0, device=device)
        return Projected(
            idx, empty, torch.zeros(0, 2, device=device), torch.zeros(0, 3, device=device),
            empty, empty, torch.zeros(0, 3, device=device),
        )
    cam, z = cam[idx], z[idx]
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    x, y = cam[:, 0], cam[:, 1]
    jac = torch.zeros(len(idx), 2, 3, device=device)
    jac[:, 0, 0] = fx / z
    jac[:, 0, 2] = -fx * x / (z * z)
    jac[:, 1, 1] = fy / z
    jac[:, 1, 2] = -fy * y / (z * z)
    cov_cam = rot @ gs.covs[idx] @ rot.T
    cov2d = jac @ cov_cam @ jac.transpose(1, 2)
    a = cov2d[:, 0, 0] + LOW_PASS
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + LOW_PASS
    det = a * c - b * b
    ok = det > 1e-12
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))
    center = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=1)
    conic = torch.stack([c, -b, a], dim=1) / det.clamp_min(1e-12)[:, None]
    sel = torch.nonzero(ok, as_tuple=False).squeeze(1)
    z, center, conic, radius, idx = z[sel], center[sel], conic[sel], radius[sel], idx[sel]
    order = torch.argsort(z)
    gidx = idx[order]
    return Projected(
        order=gidx,
        depth=z[order],
        center=center[order],
        conic=conic[order],
        radius=radius[order],
        opacity=gs.opacity[gidx],
        rgb=gs.rgb[gidx],
    )


def _composite(proj: Projected, sel: torch.Tensor, samples: torch.Tensor, depth_threshold: float,
               budget: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Front-to-back blending of the Gaussians `sel` (already depth sorted) at `samples` (P, 2)."""
    device = samples.device
    n_samples = samples.shape[0]
    color = torch.zeros(n_samples, 3, device=device)
    trans = torch.ones(n_samples, device=device)
    depth = torch.full((n_samples,), float("inf"), device=device)
    if sel.numel() == 0 or n_samples == 0:
        return color, 1.0 - trans, depth
    chunk = max(16, int(budget // max(n_samples, 1)))
    for start in range(0, sel.numel(), chunk):
        ids = sel[start:start + chunk]
        d = samples[None, :, :] - proj.center[ids][:, None, :]
        con = proj.conic[ids]
        power = -0.5 * (con[:, 0:1] * d[..., 0] ** 2 + con[:, 2:3] * d[..., 1] ** 2) - con[:, 1:2] * d[..., 0] * d[..., 1]
        alpha = torch.clamp(proj.opacity[ids][:, None] * torch.exp(torch.clamp(power, max=0.0)), max=ALPHA_MAX)
        alpha = torch.where((power <= 0.0) & (alpha >= ALPHA_MIN), alpha, torch.zeros_like(alpha))
        log_keep = torch.log1p(-alpha)
        incl = torch.cumsum(log_keep, dim=0)
        before = trans[None, :] * torch.exp(incl - log_keep)
        after = trans[None, :] * torch.exp(incl)
        weight = before * alpha
        color = color + weight.transpose(0, 1) @ proj.rgb[ids]
        cross = (before >= depth_threshold) & (after < depth_threshold)
        hit = cross.any(dim=0) & torch.isinf(depth)
        if bool(hit.any()):
            first = cross.float().argmax(dim=0)
            depth = torch.where(hit, proj.depth[ids][first], depth)
        trans = after[-1]
        if float(trans.max()) < 1e-4:
            break
    return color, 1.0 - trans, depth


def trace(
    gs: GaussianSet,
    world_to_cam,
    intrinsics,
    samples,
    depth_threshold: float = 0.5,
    near: float = 0.05,
    budget: int = 1 << 24,
    proj: Optional[Projected] = None,
) -> Dict[str, torch.Tensor]:
    """Blend the Gaussians at arbitrary pixel positions `samples` (P, 2).

    Returns premultiplied `color` (P, 3), accumulated `alpha` (P,), and
    `depth` (P,): the camera z where transmittance first drops below
    `depth_threshold` (the median surface), `inf` where it never does.
    """
    device = gs.device
    pts = _tensor(samples, device).reshape(-1, 2)
    proj = proj if proj is not None else project(gs, world_to_cam, intrinsics, near=near)
    if pts.numel() == 0 or proj.depth.numel() == 0:
        n = pts.shape[0]
        return {
            "color": torch.zeros(n, 3, device=device),
            "alpha": torch.zeros(n, device=device),
            "depth": torch.full((n,), float("inf"), device=device),
        }
    lo = pts.min(dim=0).values
    hi = pts.max(dim=0).values
    r = proj.radius
    overlap = (
        (proj.center[:, 0] + r >= lo[0]) & (proj.center[:, 0] - r <= hi[0])
        & (proj.center[:, 1] + r >= lo[1]) & (proj.center[:, 1] - r <= hi[1])
    )
    sel = torch.nonzero(overlap, as_tuple=False).squeeze(1)
    color, alpha, depth = _composite(proj, sel, pts, depth_threshold, budget)
    return {"color": color, "alpha": alpha, "depth": depth}


def render(
    gs: GaussianSet,
    world_to_cam,
    intrinsics,
    height: int,
    width: int,
    x0: int = 0,
    y0: int = 0,
    tile: int = 64,
    depth_threshold: float = 0.5,
    near: float = 0.05,
    budget: int = 1 << 24,
) -> Dict[str, np.ndarray]:
    """Render the window [x0, x0 + width) x [y0, y0 + height) of the camera image.

    Returns premultiplied `rgb` (H, W, 3) in 0..1, `alpha` (H, W) and
    median `depth` (H, W). Composite with `rgb + (1 - alpha) * background`.
    """
    device = gs.device
    rgb = torch.zeros(height, width, 3, device=device)
    alpha = torch.zeros(height, width, device=device)
    depth = torch.full((height, width), float("inf"), device=device)
    proj = project(gs, world_to_cam, intrinsics, near=near)
    if proj.depth.numel() == 0 or height <= 0 or width <= 0:
        return {"rgb": rgb.cpu().numpy(), "alpha": alpha.cpu().numpy(), "depth": depth.cpu().numpy()}
    cxs, cys, r = proj.center[:, 0], proj.center[:, 1], proj.radius
    for ty in range(0, height, tile):
        for tx in range(0, width, tile):
            h = min(tile, height - ty)
            w = min(tile, width - tx)
            gx0, gy0 = x0 + tx, y0 + ty
            gx1, gy1 = gx0 + w - 1, gy0 + h - 1
            overlap = (cxs + r >= gx0) & (cxs - r <= gx1) & (cys + r >= gy0) & (cys - r <= gy1)
            sel = torch.nonzero(overlap, as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                continue
            vs, us = torch.meshgrid(
                torch.arange(gy0, gy0 + h, device=device, dtype=torch.float32),
                torch.arange(gx0, gx0 + w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            pts = torch.stack([us.reshape(-1), vs.reshape(-1)], dim=1)
            color, acc, dep = _composite(proj, sel, pts, depth_threshold, budget)
            rgb[ty:ty + h, tx:tx + w] = color.view(h, w, 3)
            alpha[ty:ty + h, tx:tx + w] = acc.view(h, w)
            depth[ty:ty + h, tx:tx + w] = dep.view(h, w)
    return {"rgb": rgb.cpu().numpy(), "alpha": alpha.cpu().numpy(), "depth": depth.cpu().numpy()}


def footprint(gs: GaussianSet, world_to_cam, intrinsics, near: float = 0.05, min_opacity: float = 0.1):
    """Pixel bounding box (x0, y0, x1, y1) of the Gaussian centers plus their radii, or None."""
    proj = project(gs, world_to_cam, intrinsics, near=near)
    keep = proj.opacity >= min_opacity
    if not bool(keep.any()):
        return None
    c, r = proj.center[keep], proj.radius[keep]
    x0 = float((c[:, 0] - r).min())
    y0 = float((c[:, 1] - r).min())
    x1 = float((c[:, 0] + r).max())
    y1 = float((c[:, 1] + r).max())
    return x0, y0, x1, y1


def composite(background: np.ndarray, rendered: Dict[str, np.ndarray], x0: int = 0, y0: int = 0) -> np.ndarray:
    """Alpha-composite a rendered window onto a uint8 HWC image (copy)."""
    out = np.asarray(background).copy()
    h, w = rendered["alpha"].shape
    H, W = out.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + w, W), min(y0 + h, H)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    win = (slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0))
    region = out[sy0:sy1, sx0:sx1, :3].astype(np.float32) / 255.0
    a = rendered["alpha"][win][..., None]
    blended = rendered["rgb"][win] + (1.0 - a) * region
    out[sy0:sy1, sx0:sx1, :3] = np.clip(blended * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def look_at(cam_pos: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """OpenCV world-to-camera matrix for a camera at `cam_pos` looking at `target`."""
    cam = np.asarray(cam_pos, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - cam
    fwd /= max(np.linalg.norm(fwd), 1e-12)
    right = np.cross(fwd, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    rot = np.stack([right, down, fwd], axis=0)
    w2c = np.eye(4)
    w2c[:3, :3] = rot
    w2c[:3, 3] = -rot @ cam
    return w2c.astype(np.float32)
