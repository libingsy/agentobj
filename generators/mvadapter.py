"""MV-Adapter runtime.

Follows the official inference scripts in https://github.com/huanngzh/MV-Adapter:

- scripts/inference_t2mv_sdxl.py and scripts/inference_t2mv_sd.py
- scripts/inference_i2mv_sdxl.py and scripts/inference_i2mv_sd.py

Paper mapping: T2MV instantiates canonical geometry from the base prompt and
the semantic prompt. I2MV is the multi-view personalization model. It is
conditioned on that geometry (the azimuth-0 view is the reference image the
official pipeline accepts) and on the style prompt, and it regenerates every
azimuth so the appearance stays view-consistent.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from utils.images import to_pil


VARIANTS = {
    "sdxl": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "vae_model": "madebyollin/sdxl-vae-fp16-fix",
        "t2mv_cls": "mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl.MVAdapterT2MVSDXLPipeline",
        "i2mv_cls": "mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl.MVAdapterI2MVSDXLPipeline",
        "t2mv_weight": "mvadapter_t2mv_sdxl.safetensors",
        "i2mv_weight": "mvadapter_i2mv_sdxl.safetensors",
        "height": 768,
        "width": 768,
        "sdxl": True,
    },
    "sd21": {
        "base_model": "stabilityai/stable-diffusion-2-1-base",
        "vae_model": None,
        "t2mv_cls": "mvadapter.pipelines.pipeline_mvadapter_t2mv_sd.MVAdapterT2MVSDPipeline",
        "i2mv_cls": "mvadapter.pipelines.pipeline_mvadapter_i2mv_sd.MVAdapterI2MVSDPipeline",
        "t2mv_weight": "mvadapter_t2mv_sd21.safetensors",
        "i2mv_weight": "mvadapter_i2mv_sd21.safetensors",
        "height": 512,
        "width": 512,
        "sdxl": False,
    },
}

DEFAULT_AZIMUTHS = (0, 45, 90, 180, 270, 315)
NEGATIVE_PROMPT = "watermark, ugly, deformed, noisy, blurry, low contrast"


class MVAdapterUnavailable(RuntimeError):
    """Raised when the MV-Adapter package or its weights cannot be loaded."""


def _stub_nvdiffrast() -> None:
    """Camera helpers import nvdiffrast at module level. Reconstruction does not rasterize meshes."""
    if "nvdiffrast.torch" in sys.modules:
        return
    try:
        import nvdiffrast.torch  # noqa: F401
        return
    except Exception:
        pass
    package = types.ModuleType("nvdiffrast")
    package.__path__ = []
    torch_mod = types.ModuleType("nvdiffrast.torch")

    class _Missing:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("nvdiffrast is not installed")

    for name in (
        "RasterizeGLContext",
        "RasterizeCudaContext",
        "rasterize",
        "interpolate",
        "texture",
        "antialias",
    ):
        setattr(torch_mod, name, _Missing)
    package.torch = torch_mod
    sys.modules["nvdiffrast"] = package
    sys.modules["nvdiffrast.torch"] = torch_mod


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    elif getattr(module, "__path__", None) is None:
        module.__path__ = [str(path)]
    return module


def _load_file(fullname: str, path: Path):
    loaded = sys.modules.get(fullname)
    if loaded is not None and hasattr(loaded, "__file__"):
        return loaded
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = fullname.rpartition(".")[0]
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _camera_helpers(repo: Optional[str]):
    """Load Plücker helpers without MV-Adapter's mesh package init.

    ``mvadapter.utils.mesh_utils`` imports nvdiffrast, a triton JIT blend, and
    pymeshlab. Control images only need the orthogonal camera and the Plücker
    embedding used by the official inference scripts.
    """
    if not repo:
        raise MVAdapterUnavailable("paths.mvadapter_repo is empty.")
    root = Path(repo) / "mvadapter"
    _stub_nvdiffrast()
    _ensure_package("mvadapter", root)
    _ensure_package("mvadapter.utils", root / "utils")
    _ensure_package("mvadapter.utils.mesh_utils", root / "utils" / "mesh_utils")
    camera = _load_file(
        "mvadapter.utils.mesh_utils.camera",
        root / "utils" / "mesh_utils" / "camera.py",
    )
    geometry = _load_file("mvadapter.utils.geometry", root / "utils" / "geometry.py")
    return camera.get_orthogonal_camera, geometry.get_plucker_embeds_from_cameras_ortho


def add_repo_to_path(repo: Optional[str]) -> None:
    if not repo:
        return
    path = str(Path(repo).expanduser().resolve())
    if Path(path).is_dir() and path not in sys.path:
        sys.path.insert(0, path)


def _import_class(qualname: str):
    module_name, class_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def preprocess_rgba(image: Image.Image, height: int, width: int) -> Image.Image:
    """Same crop-and-pad used by the official i2mv scripts for RGBA references."""
    image_np = np.array(image)
    alpha = image_np[..., 3] > 0
    if not alpha.any():
        return image.convert("RGB").resize((width, height), Image.BICUBIC)
    ys, xs = np.where(alpha)
    y0, y1 = max(int(ys.min()) - 1, 0), min(int(ys.max()) + 1, alpha.shape[0])
    x0, x1 = max(int(xs.min()) - 1, 0), min(int(xs.max()) + 1, alpha.shape[1])
    crop = image_np[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch > cw:
        cw = int(cw * (height * 0.9) / ch)
        ch = int(height * 0.9)
    else:
        ch = int(ch * (width * 0.9) / cw)
        cw = int(width * 0.9)
    crop = np.array(Image.fromarray(crop).resize((max(cw, 1), max(ch, 1))))
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    top = (height - crop.shape[0]) // 2
    left = (width - crop.shape[1]) // 2
    canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    rgb = canvas.astype(np.float32) / 255.0
    rgb = rgb[:, :, :3] * rgb[:, :, 3:4] + (1.0 - rgb[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))


def prepare_reference(image, height: int, width: int) -> Image.Image:
    pil = to_pil(image)
    if pil.mode == "RGBA":
        return preprocess_rgba(pil, height, width)
    return pil.convert("RGB").resize((width, height), Image.BICUBIC)


class MVAdapterRuntime:
    """Lazy loader for the text-to-multiview and image-to-multiview pipelines.

    The two pipelines are loaded one at a time so a single GPU does not have
    to hold both UNets at once. Set `sequential=False` to keep both resident.
    """

    def __init__(
        self,
        adapter_path: str = "huanngzh/mv-adapter",
        variant: str = "sdxl",
        repo: Optional[str] = None,
        base_model: Optional[str] = None,
        vae_model: Optional[str] = None,
        scheduler: Optional[str] = None,
        lora_model: Optional[str] = None,
        lora_scale: float = 1.0,
        device: str = "cpu",
        num_inference_steps_t2mv: int = 50,
        num_inference_steps_i2mv: int = 50,
        guidance_scale_t2mv: float = 7.0,
        guidance_scale_i2mv: float = 3.0,
        reference_conditioning_scale: float = 1.0,
        negative_prompt: str = NEGATIVE_PROMPT,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: int = 42,
        sequential: bool = True,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown MV-Adapter variant {variant!r}. Choose from {list(VARIANTS)}.")
        spec = VARIANTS[variant]
        self.variant = variant
        self.spec = spec
        self.adapter_path = adapter_path or "huanngzh/mv-adapter"
        self.repo = repo
        self.base_model = base_model or spec["base_model"]
        self.vae_model = vae_model if vae_model is not None else spec["vae_model"]
        self.scheduler = scheduler
        self.lora_model = lora_model
        self.lora_scale = lora_scale
        self.device = device
        self.steps_t2mv = num_inference_steps_t2mv
        self.steps_i2mv = num_inference_steps_i2mv
        self.guidance_t2mv = guidance_scale_t2mv
        self.guidance_i2mv = guidance_scale_i2mv
        self.reference_scale = reference_conditioning_scale
        self.negative_prompt = negative_prompt
        self.height = int(height or spec["height"])
        self.width = int(width or spec["width"])
        self.seed = seed
        self.sequential = sequential
        self._pipes = {}
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    def text_to_multiview(self, prompt: str, azimuth_deg: Sequence[int] = DEFAULT_AZIMUTHS) -> List[Image.Image]:
        """Geometry stage. `prompt` should be the base prompt plus the semantic prompt."""
        pipe = self._pipeline("t2mv", len(azimuth_deg))
        control = self._control_images(azimuth_deg)
        kwargs = self._common_kwargs(self.seed)
        if self.spec["sdxl"]:
            kwargs["max_sequence_length"] = 214
        images = pipe(
            prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps_t2mv,
            guidance_scale=self.guidance_t2mv,
            num_images_per_prompt=len(azimuth_deg),
            control_image=control,
            control_conditioning_scale=1.0,
            negative_prompt=self.negative_prompt,
            **kwargs,
        ).images
        if self.sequential:
            self._drop("t2mv")
        return list(images)

    def personalize(
        self,
        reference_views: Sequence,
        style_prompt: str,
        azimuth_deg: Sequence[int] = DEFAULT_AZIMUTHS,
    ) -> List[Image.Image]:
        """Appearance stage.

        `reference_views` is the geometry set. The official I2MV pipeline takes
        one reference image, so the azimuth-0 view (index 0 when azimuths start
        at 0) anchors identity and the style prompt controls appearance.
        """
        if not reference_views:
            raise ValueError("personalize() needs at least one geometry view.")
        azimuths = [int(a) for a in azimuth_deg]
        if 0 in azimuths:
            reference = reference_views[azimuths.index(0)]
        else:
            reference = reference_views[0]
        reference = prepare_reference(reference, self.height, self.width)
        pipe = self._pipeline("i2mv", len(azimuths))
        control = self._control_images(azimuths)
        images = pipe(
            style_prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps_i2mv,
            guidance_scale=self.guidance_i2mv,
            num_images_per_prompt=len(azimuths),
            control_image=control,
            control_conditioning_scale=1.0,
            reference_image=reference,
            reference_conditioning_scale=self.reference_scale,
            negative_prompt=self.negative_prompt,
            **self._common_kwargs(self.seed),
        ).images
        if self.sequential:
            self._drop("i2mv")
        return list(images)

    def _common_kwargs(self, seed: int) -> dict:
        kwargs = {}
        if seed is not None and int(seed) >= 0:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(int(seed))
        if self.lora_model:
            kwargs["cross_attention_kwargs"] = {"scale": self.lora_scale}
        return kwargs

    def _pipeline(self, kind: str, num_views: int):
        key = (kind, num_views)
        if key in self._pipes:
            return self._pipes[key]
        add_repo_to_path(self.repo)
        try:
            cls = _import_class(self.spec[f"{kind}_cls"])
            from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler
            from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
        except Exception as exc:
            raise MVAdapterUnavailable(
                "MV-Adapter is not importable. Clone https://github.com/huanngzh/MV-Adapter, "
                "install it (pip install -e .), and set paths.mvadapter_repo to that checkout. "
                f"Adapter weights default to the Hugging Face repo '{self.adapter_path}'. "
                f"Import error: {exc}"
            ) from exc

        pipe_kwargs = {"torch_dtype": self.dtype}
        if self.vae_model:
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.vae_model, torch_dtype=self.dtype
            )
        # fp16 siblings are about half the fp32 UNet. L40 inference casts to fp16 anyway.
        if self.dtype == torch.float16 and self.spec.get("sdxl"):
            pipe_kwargs["variant"] = "fp16"
        pipe = cls.from_pretrained(self.base_model, **pipe_kwargs)
        scheduler_class = None
        if self.scheduler == "ddpm":
            scheduler_class = DDPMScheduler
        elif self.scheduler == "lcm":
            scheduler_class = LCMScheduler
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class,
        )
        pipe.init_custom_adapter(num_views=num_views)
        pipe.load_custom_adapter(self.adapter_path, weight_name=self.spec[f"{kind}_weight"])
        pipe.to(device=self.device, dtype=self.dtype)
        pipe.cond_encoder.to(device=self.device, dtype=self.dtype)
        if self.lora_model:
            model_id, weight_name = self.lora_model.rsplit("/", 1)
            pipe.load_lora_weights(model_id, weight_name=weight_name)
        pipe.enable_vae_slicing()
        self._pipes[key] = pipe
        print(f"[MVAdapter] Loaded {kind} ({self.variant}, {num_views} views) on {self.device}")
        return pipe

    def _control_images(self, azimuth_deg: Sequence[int]):
        get_orthogonal_camera, get_plucker_embeds_from_cameras_ortho = _camera_helpers(self.repo)

        num_views = len(azimuth_deg)
        cameras = get_orthogonal_camera(
            elevation_deg=[0] * num_views,
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[int(x) - 90 for x in azimuth_deg],
            device=self.device,
        )
        plucker = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1] * num_views, self.width)
        return ((plucker + 1.0) / 2.0).clamp(0, 1)

    def _drop(self, kind: str) -> None:
        for key in list(self._pipes):
            if key[0] == kind:
                del self._pipes[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
