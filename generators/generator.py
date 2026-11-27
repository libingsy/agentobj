"""3D object generators.

`DummyGenerator` keeps the prompt-search loop runnable without GPU weights.
`LGMGenerator` is the paper pipeline:

1. MV-Adapter T2MV(base, semantic) -> geometry views
2. MV-Adapter I2MV(geometry views, style) -> view-consistent appearance
3. LGM -> 3D Gaussians and a metric pedestrian point cloud
"""

import hashlib
from typing import Dict, Optional

import numpy as np
import torch

from generators.lgm_recon import LGMReconstructor, LGMUnavailable
from generators.mvadapter import DEFAULT_AZIMUTHS, MVAdapterRuntime, MVAdapterUnavailable
from generators.prompts import appearance_prompt, geometry_prompt
from utils.device import resolve_device
from utils.geometry import LGM_AZIMUTHS, box_stats_from_points
from utils.images import stack_uint8


class GeneratorNotReady(RuntimeError):
    """Real weights or source checkouts are missing."""


def _stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pedestrian_gaussians(prompt: str, n_body: int = 500, n_head: int = 120) -> torch.Tensor:
    """Activated Gaussians of a standing person, colored by the prompt.

    Layout matches LGM: xyz, opacity, scale xyz, quaternion, rgb. y is up.
    """
    rng = np.random.RandomState(_stable_seed(prompt))
    body = np.stack(
        [
            rng.normal(0.0, 0.12, n_body),
            rng.uniform(0.05, 1.35, n_body),
            rng.normal(0.0, 0.08, n_body),
        ],
        axis=1,
    )
    head = np.stack(
        [
            rng.normal(0.0, 0.07, n_head),
            rng.uniform(1.40, 1.70, n_head),
            rng.normal(0.0, 0.07, n_head),
        ],
        axis=1,
    )
    pts = np.concatenate([body, head], axis=0).astype(np.float32)
    tint = np.array(
        [
            0.25 + 0.65 * ((_stable_seed(prompt) >> 0) % 100) / 100.0,
            0.25 + 0.65 * ((_stable_seed(prompt) >> 8) % 100) / 100.0,
            0.25 + 0.65 * ((_stable_seed(prompt) >> 16) % 100) / 100.0,
        ],
        dtype=np.float32,
    )
    rgb = np.clip(tint + rng.normal(0.0, 0.03, (len(pts), 3)).astype(np.float32), 0.05, 0.95)
    opacity = np.full((len(pts), 1), 0.9, dtype=np.float32)
    scale = np.full((len(pts), 3), 0.035, dtype=np.float32)
    quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(pts), 1))
    packed = np.concatenate([pts, opacity, scale, quat, rgb], axis=1)
    return torch.from_numpy(packed)


class DummyGenerator:
    """Prompt-conditioned stand-in used when MV-Adapter and LGM are not loaded."""

    def generate(self, base: str, sem: str, sty: str) -> Dict:
        prompt = appearance_prompt(base, sem, sty)
        gaussians = _pedestrian_gaussians(prompt)
        points = gaussians[:, 0:3].contiguous()
        return {
            "prompt": prompt,
            "geometry_prompt": geometry_prompt(base, sem),
            "style_prompt": sty,
            "gaussians": gaussians,
            "point_cloud": points,
            "box_stats": box_stats_from_points(points),
            "backend": "dummy",
        }


class LGMGenerator:
    """T2MV + MV-Adapter personalization + LGM."""

    def __init__(
        self,
        lgm_weights: str,
        mv_adapter_path: Optional[str] = None,
        device: str = "auto",
        lgm_repo: Optional[str] = None,
        mvadapter_repo: Optional[str] = None,
        variant: str = "sdxl",
        base_model: Optional[str] = None,
        vae_model: Optional[str] = None,
        scheduler: Optional[str] = None,
        lora_model: Optional[str] = None,
        azimuth_deg=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        steps_t2mv: int = 50,
        steps_i2mv: int = 50,
        guidance_t2mv: float = 7.0,
        guidance_i2mv: float = 3.0,
        seed: int = 42,
        negative_prompt: Optional[str] = None,
        target_height: float = 1.7,
        fallback_to_dummy: bool = False,
        sequential: bool = True,
    ):
        self.device = resolve_device(device)
        self.fallback_to_dummy = fallback_to_dummy
        self.azimuth_deg = tuple(azimuth_deg or DEFAULT_AZIMUTHS)
        self.seed = seed
        mv_kwargs = dict(
            adapter_path=mv_adapter_path or "huanngzh/mv-adapter",
            variant=variant,
            repo=mvadapter_repo,
            base_model=base_model,
            vae_model=vae_model,
            scheduler=scheduler,
            lora_model=lora_model,
            device=self.device,
            num_inference_steps_t2mv=steps_t2mv,
            num_inference_steps_i2mv=steps_i2mv,
            guidance_scale_t2mv=guidance_t2mv,
            guidance_scale_i2mv=guidance_i2mv,
            height=height,
            width=width,
            seed=seed,
            sequential=sequential,
        )
        if negative_prompt:
            mv_kwargs["negative_prompt"] = negative_prompt
        self.mv = MVAdapterRuntime(**mv_kwargs)
        self.lgm = LGMReconstructor(
            weights=lgm_weights,
            repo=lgm_repo,
            device=self.device,
            target_height=target_height,
        )
        self._dummy = DummyGenerator()

    def generate(self, base: str, sem: str, sty: str) -> Dict:
        geo = geometry_prompt(base, sem)
        style = appearance_prompt(base, sem, sty)
        try:
            geometry_views = self.mv.text_to_multiview(geo, self.azimuth_deg)
            styled_views = self.mv.personalize(geometry_views, style, self.azimuth_deg)
            gaussians, points, box = self.lgm.reconstruct(styled_views, self.azimuth_deg)
        except (MVAdapterUnavailable, LGMUnavailable, FileNotFoundError, ImportError) as exc:
            if self.fallback_to_dummy:
                print(f"[LGMGenerator] Real pipeline unavailable ({exc}). Using DummyGenerator.")
                obj = self._dummy.generate(base, sem, sty)
                obj["backend_error"] = str(exc)
                return obj
            raise GeneratorNotReady(str(exc)) from exc

        return {
            "prompt": style,
            "geometry_prompt": geo,
            "style_prompt": sty,
            "gaussians": gaussians,
            "point_cloud": points,
            "box_stats": box,
            "mv_images": stack_uint8(styled_views),
            "geometry_images": stack_uint8(geometry_views),
            "mv_azimuths": [int(a) for a in self.azimuth_deg],
            "lgm_azimuths": list(LGM_AZIMUTHS),
            "backend": "mvadapter+lgm",
        }


def build_generator(cfg: dict, use_real: bool):
    """Construct the generator selected by the CLI and `configs/default.yaml`."""
    if not use_real:
        print("[Info] Using DummyGenerator")
        return DummyGenerator()
    paths = cfg.get("paths", {})
    gen = cfg.get("generator", {})
    device = resolve_device(cfg.get("experiment", {}).get("device", "auto"))
    print("[Info] Using MV-Adapter + LGM")
    return LGMGenerator(
        lgm_weights=paths.get("lgm_weights", ""),
        mv_adapter_path=paths.get("mv_adapter_weights"),
        device=device,
        lgm_repo=paths.get("lgm_repo") or None,
        mvadapter_repo=paths.get("mvadapter_repo") or None,
        variant=gen.get("variant", "sdxl"),
        base_model=gen.get("base_model") or None,
        vae_model=gen.get("vae_model") or None,
        scheduler=gen.get("scheduler") or None,
        lora_model=gen.get("lora_model") or None,
        azimuth_deg=gen.get("azimuth_deg"),
        height=gen.get("height"),
        width=gen.get("width"),
        steps_t2mv=gen.get("steps_t2mv", 50),
        steps_i2mv=gen.get("steps_i2mv", 50),
        guidance_t2mv=gen.get("guidance_t2mv", 7.0),
        guidance_i2mv=gen.get("guidance_i2mv", 3.0),
        seed=cfg.get("experiment", {}).get("seed", 42),
        negative_prompt=gen.get("negative_prompt"),
        target_height=gen.get("target_height", 1.7),
        fallback_to_dummy=bool(gen.get("fallback_to_dummy", False)),
        sequential=bool(gen.get("sequential_load", True)),
    )
