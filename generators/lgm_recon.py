"""LGM reconstruction of four MV-Adapter views into 3D Gaussians.

Matches 3DTopia/LGM `infer.py` for the big checkpoint: four views at azimuths
0, 90, 180, 270 on a white background, ImageNet-normalized RGB concatenated
with Plücker ray embeddings, then `LGM.forward_gaussians`.

MV-Adapter renders on a flat gray background with an orthographic camera
whose frame spans about 1.1 object units. LGM was trained on white-background
renders where the object fills roughly 60% of the frame (`recenter(...,
border_ratio=0.2)` in LGM's own demo). `prepare_lgm_views` removes the gray
background and applies one common scale about the image center to all four
views, which keeps them consistent with each other.
"""

import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from utils.geometry import box_stats_from_points, select_views_by_azimuth
from utils.images import to_pil


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
LGM_FOVY_DEG = 49.1
LGM_CAM_RADIUS = 1.5


def foreground_alpha(image: np.ndarray, tol: float = 10.0, hole_tol: float = 6.0, min_hole: int = 24) -> np.ndarray:
    """Soft foreground mask of an object rendered on a flat background.

    The background color is the median of the image border. Pixels close to it
    and connected to the border are background. Enclosed patches of exactly
    that color (the gap between an arm and the torso) are background too.
    """
    from scipy import ndimage

    rgb = np.asarray(image)[..., :3].astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    dist = np.abs(rgb - bg).max(axis=2)
    near_bg = dist < tol
    labels, count = ndimage.label(near_bg)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, edge[edge > 0])
    holes, n_holes = ndimage.label((dist < hole_tol) & ~background)
    if n_holes:
        sizes = ndimage.sum(np.ones_like(dist), holes, index=np.arange(1, n_holes + 1))
        big = np.nonzero(sizes >= min_hole)[0] + 1
        background |= np.isin(holes, big)
    mask = (~background).astype(np.float32)
    mask = ndimage.binary_opening(mask > 0.5, iterations=1).astype(np.float32)
    return np.clip(ndimage.gaussian_filter(mask, 0.6), 0.0, 1.0)


def prepare_lgm_views(
    views: Sequence,
    size: int = 256,
    fill: float = 0.6,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """White-background views at LGM's object scale.

    Returns float RGB images (size, size, 3) in 0..1 and their alpha masks.
    One scale factor, chosen so the union of the four silhouettes spans `fill`
    of the frame, is applied about the image center of every view.
    """
    arrays = [np.asarray(to_pil(v).convert("RGB")) for v in views]
    alphas = [foreground_alpha(a) for a in arrays]
    h, w = arrays[0].shape[:2]
    extent = 0.0
    for alpha in alphas:
        ys, xs = np.nonzero(alpha > 0.5)
        if len(ys) == 0:
            continue
        # Distance from the image center, so that scaling about the center keeps the object in frame.
        extent = max(
            extent,
            2.0 * max(abs(ys.min() - h / 2.0), abs(ys.max() + 1 - h / 2.0)) / h,
            2.0 * max(abs(xs.min() - w / 2.0), abs(xs.max() + 1 - w / 2.0)) / w,
        )
    scale = fill / extent if extent > 0 else 1.0
    out_rgb, out_alpha = [], []
    for arr, alpha in zip(arrays, alphas):
        rgba = np.concatenate([arr.astype(np.float32) / 255.0, alpha[..., None]], axis=2)
        rgb = rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:])
        side = max(int(round(size * scale)), 1)
        small_rgb = np.asarray(
            Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((side, side), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        small_a = np.asarray(
            Image.fromarray((alpha * 255).round().astype(np.uint8)).resize((side, side), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        canvas = np.ones((size, size, 3), dtype=np.float32)
        mask = np.zeros((size, size), dtype=np.float32)
        off = (size - side) // 2
        src0 = max(-off, 0)
        dst0 = max(off, 0)
        n = min(side - src0, size - dst0)
        canvas[dst0:dst0 + n, dst0:dst0 + n] = small_rgb[src0:src0 + n, src0:src0 + n]
        mask[dst0:dst0 + n, dst0:dst0 + n] = small_a[src0:src0 + n, src0:src0 + n]
        out_rgb.append(np.clip(canvas, 0.0, 1.0))
        out_alpha.append(np.clip(mask, 0.0, 1.0))
    return out_rgb, out_alpha


def lgm_camera(azimuth_deg: float, size: int = 256, radius: float = LGM_CAM_RADIUS,
               fovy_deg: float = LGM_FOVY_DEG) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCV world-to-camera and K of LGM's input camera at `azimuth_deg` (elevation 0).

    Same pose as `kiui.cam.orbit_camera(0, azimuth, radius)`: azimuth 0 sits
    on +z, azimuth 90 on +x, y is up.
    """
    from render.gs_rasterizer import look_at

    a = np.deg2rad(azimuth_deg)
    cam = np.array([radius * np.sin(a), 0.0, radius * np.cos(a)])
    w2c = look_at(cam, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    f = 0.5 * size / np.tan(np.deg2rad(fovy_deg) / 2.0)
    k = np.array([[f, 0.0, (size - 1) / 2.0], [0.0, f, (size - 1) / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return w2c, k


def reprojection_psnr(gaussians, views: Sequence[np.ndarray], azimuths: Sequence[float] = (0, 90, 180, 270),
                      device: str = "cpu") -> Tuple[List[float], List[np.ndarray]]:
    """Render LGM Gaussians (LGM frame) at its four input cameras and compare with the inputs.

    A correct reconstruction reproduces its inputs; this catches a wrong view
    order, a wrong background, or a renderer bug. Returns PSNR per view and
    the rendered images.
    """
    from render.gs_rasterizer import GaussianSet, render

    gs = GaussianSet.from_packed(gaussians, device=device)
    psnrs, renders = [], []
    for az, ref in zip(azimuths, views):
        size = ref.shape[0]
        w2c, k = lgm_camera(az, size)
        out = render(gs, w2c, k, size, size)
        img = out["rgb"] + (1.0 - out["alpha"][..., None])
        mse = float(np.mean((np.clip(img, 0, 1) - ref) ** 2))
        psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-10)))
        renders.append(np.clip(img, 0.0, 1.0))
    return psnrs, renders


class LGMUnavailable(RuntimeError):
    """Raised when the LGM repo or its checkpoint cannot be loaded."""


def _allow_lgm_import_without_cuda_rasterizer() -> None:
    """LGM imports diff_gaussian_rasterization while building the module.

    Reconstruction only needs the UNet (`forward_gaussians`). The CUDA
    rasterizer is used for novel-view supervision, which this pipeline does
    not call. A stub lets the checkpoint load on a GPU node that does not
    have that extension compiled.
    """
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return
    except Exception:
        pass
    module = types.ModuleType("diff_gaussian_rasterization")

    class GaussianRasterizationSettings:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class GaussianRasterizer:
        def __init__(self, raster_settings=None):
            self.raster_settings = raster_settings

        def __call__(self, *args, **kwargs):
            raise RuntimeError("diff_gaussian_rasterization is not installed")

    module.GaussianRasterizationSettings = GaussianRasterizationSettings
    module.GaussianRasterizer = GaussianRasterizer
    sys.modules["diff_gaussian_rasterization"] = module


def _patch_kiui_typing() -> None:
    """kiui 0.3.5 uses ``Union`` in ``op.py`` but does not import it.

    ``from kiui.typing import *`` therefore raises NameError on Python 3.10
    when LGM asks for ``orbit_camera``. Later kiui releases define those names.
    """
    try:
        import kiui
    except Exception:
        return
    path = Path(kiui.__file__).resolve().parent / "typing.py"
    if not path.is_file():
        return
    text = path.read_text()
    if "Union" in text:
        return
    path.write_text(
        text.rstrip()
        + "\nfrom typing import Sequence, List, Tuple, Dict, Any, Optional, Union, Literal, Callable\n"
        + "from torch import Tensor\nfrom numpy import ndarray\n"
    )


def add_repo_to_path(repo: Optional[str]) -> None:
    if not repo:
        return
    path = str(Path(repo).expanduser().resolve())
    if Path(path).is_dir() and path not in sys.path:
        sys.path.insert(0, path)


def _normalized_stack(images: Sequence[np.ndarray], device: str) -> torch.Tensor:
    batch = torch.stack([torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1) for a in images], dim=0).float()
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
    return ((batch - mean) / std).to(device=device, dtype=torch.float32)


def metric_pedestrian(raw: torch.Tensor, target_height: float = 1.7, opacity_min: float = 0.05,
                      solid: float = 0.5) -> Tuple[torch.Tensor, float]:
    """Drop faint and stray Gaussians, then scale the object to `target_height` meters.

    Height and center come from Gaussians with opacity >= `solid`, using
    0.2/99.8 percentiles so that a few floaters do not set the scale. The
    result is y-up with the feet at y=0 and the body centered on x=z=0.
    """
    g = raw[raw[:, 3] > opacity_min].clone()
    if len(g) < 32:
        g = raw.clone()
    core = g[g[:, 3] >= solid]
    if len(core) < 32:
        core = g
    lo = torch.quantile(core[:, 0:3], 0.002, dim=0)
    hi = torch.quantile(core[:, 0:3], 0.998, dim=0)
    span = (hi - lo).clamp(min=1e-4)
    margin = 0.15 * span
    inside = ((g[:, 0:3] >= lo - margin) & (g[:, 0:3] <= hi + margin)).all(dim=1)
    g = g[inside]
    factor = float(target_height / span[1])
    center = 0.5 * (lo + hi)
    g[:, 0] = (g[:, 0] - center[0]) * factor
    g[:, 2] = (g[:, 2] - center[2]) * factor
    g[:, 1] = (g[:, 1] - lo[1]) * factor
    g[:, 4:7] = g[:, 4:7] * factor
    return g, factor


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        if "lpips_loss" in new_key:
            continue
        cleaned[new_key] = value
    return cleaned


class LGMReconstructor:
    """Load the official LGM UNet and turn four views into activated Gaussians."""

    def __init__(
        self,
        weights: str,
        repo: Optional[str] = None,
        device: str = "cpu",
        target_height: float = 1.7,
        opacity_min: float = 0.05,
        config_name: str = "big",
        fill: float = 0.6,
    ):
        self.weights = weights
        self.repo = repo
        self.device = device
        self.target_height = target_height
        self.opacity_min = opacity_min
        self.config_name = config_name
        self.fill = fill
        self.last: Dict = {}
        self.model = None
        self.opt = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights or not Path(self.weights).expanduser().is_file():
            raise LGMUnavailable(
                "LGM weights were not found. Download the fp16 checkpoint "
                "(https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors) "
                "and set paths.lgm_weights to that file."
            )
        add_repo_to_path(self.repo)
        _patch_kiui_typing()
        _allow_lgm_import_without_cuda_rasterizer()
        try:
            from core.models import LGM
            from core.options import config_defaults
        except Exception as exc:
            raise LGMUnavailable(
                "LGM is not importable. Clone https://github.com/3DTopia/LGM, install its "
                "requirements, and set paths.lgm_repo to that checkout. "
                f"Import error: {exc}"
            ) from exc

        opt = config_defaults[self.config_name]
        # LPIPS is only used in training. Skip it so inference does not pull VGG.
        opt.lambda_lpips = 0.0
        model = LGM(opt)
        state = _load_state_dict(str(Path(self.weights).expanduser()))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[LGM] Missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            print(f"[LGM] Unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
        model = model.to(self.device).eval()
        if self.device == "cpu":
            model = model.float()
        self.model = model
        self.opt = opt
        print(f"[LGM] Loaded {self.config_name} weights from {self.weights} on {self.device}")

    @torch.no_grad()
    def reconstruct(
        self,
        images: Sequence,
        azimuths: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Return activated Gaussians (N, 14), a metric point cloud (N, 3), and box stats.

        Gaussians are in the y-up metric frame used by the renderer: feet at y=0,
        height `target_height` meters, front facing +z. Channels are
        xyz, opacity, scale(3), rotation quaternion (w, x, y, z), rgb.
        `self.last` keeps the LGM-frame Gaussians and the prepared input views
        for `reprojection_psnr`.
        """
        self.load()
        views = select_views_by_azimuth(images, azimuths)
        size = int(self.opt.input_size)
        prepared, masks = prepare_lgm_views(views, size=size, fill=self.fill)
        rgb = _normalized_stack(prepared, self.device)
        rays = self.model.prepare_default_rays(self.device).float()
        inputs = torch.cat([rgb, rays], dim=1).unsqueeze(0)  # [1, 4, 9, H, W]
        if str(self.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gaussians = self.model.forward_gaussians(inputs)
        else:
            gaussians = self.model.forward_gaussians(inputs.float())
        raw = gaussians[0].detach().float().cpu()
        self.last = {"lgm_gaussians": raw, "views": prepared, "masks": masks}
        packed, _ = metric_pedestrian(raw, self.target_height, opacity_min=self.opacity_min)
        points = packed[:, 0:3].contiguous()
        return packed, points, box_stats_from_points(packed[packed[:, 3] >= 0.5, 0:3])
