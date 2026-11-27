"""Detector wrappers: YOLOv10x (monocular 2D) and MambaFusion (camera-LiDAR, through its worker).

With real models, both detectors score the same composited nuScenes frame:
YOLOv10x the camera image, MambaFusion the image plus the stacked LiDAR
sweeps. Each returns the confidence of the detection that belongs to the
inserted object, not the highest score in the scene, which would come from
real pedestrians nearby.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from utils.device import resolve_device


class DummyDetector:
    """Prompt-keyword stand-in used by the CPU quick start."""

    def __init__(self, name: str = "dummy", seed: int = 42):
        self.name = name
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        obj: Any,
        image: Optional[np.ndarray] = None,
        lidar: Optional[Any] = None,
        viewpoint: int = 0,
    ) -> float:
        prompt = ""
        if isinstance(obj, dict):
            prompt = obj.get("prompt", "")
        base = 0.75
        keywords = ["backpack", "dark", "camouflage", "hat", "coat", "blurry", "vest"]
        for kw in keywords:
            if kw in prompt.lower():
                base -= 0.08
        noise = self.rng.uniform(-0.05, 0.05)
        view_factor = 0.02 * (viewpoint % 90) / 90.0
        return float(max(0.02, min(0.98, base - view_factor + noise)))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match_score(detections: List[Tuple[List[float], float]], target: Optional[Sequence[float]],
                min_iou: float = 0.3) -> Tuple[float, Optional[List[float]]]:
    """Highest confidence among detections overlapping `target` with IoU >= `min_iou`."""
    best, best_box = 0.0, None
    if target is None:
        return best, best_box
    for box, conf in detections:
        if box_iou(box, target) >= min_iou and conf > best:
            best, best_box = float(conf), list(box)
    return best, best_box


class YOLODetector:
    """YOLOv10x person detections. COCO class 0 is person."""

    def __init__(
        self,
        model_path: str = "yolov10x.pt",
        device: str = "auto",
        person_class: int = 0,
        required: bool = False,
        min_conf: float = 0.05,
    ):
        self.model = None
        self.device = resolve_device(device)
        self.person_class = person_class
        self.min_conf = min_conf
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[YOLO] Loaded {model_path}")
        except Exception as e:
            if required:
                raise RuntimeError(f"YOLOv10x failed to load from {model_path}") from e
            print(f"[YOLO] Load failed, using dummy: {e}")

    def detect(self, image: np.ndarray) -> List[Tuple[List[float], float]]:
        """Person boxes (x0, y0, x1, y1) and confidences in an RGB image."""
        # Ultralytics treats a numpy image as BGR.
        image_bgr = np.ascontiguousarray(np.asarray(image)[..., :3][..., ::-1])
        results = self.model(image_bgr, verbose=False, device=self.device, conf=self.min_conf)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                if int(box.cls[0]) == self.person_class:
                    out.append(([float(v) for v in box.xyxy[0].tolist()], float(box.conf[0])))
        return out

    def score_box(self, image: np.ndarray, target: Optional[Sequence[float]], min_iou: float = 0.3):
        """Confidence of the person detection on the inserted object, its box, and all person boxes."""
        dets = self.detect(image)
        score, box = match_score(dets, target, min_iou)
        return score, box, dets

    def __call__(
        self,
        obj: Any,
        image: Optional[np.ndarray] = None,
        lidar: Optional[Any] = None,
        viewpoint: int = 0,
    ) -> float:
        """Highest person confidence in an object-centric render (fallback without a scene)."""
        if self.model is None or image is None:
            return DummyDetector("yolo")(obj, image, lidar, viewpoint)
        dets = self.detect(image)
        return max((conf for _, conf in dets), default=0.0)


class FusionDetector:
    """MambaFusion through the resident worker (`scripts/mambafusion_worker.py`).

    MambaFusion needs Python 3.8 / torch 2.1, so it always runs in its own
    process. Without a worker the keyword stand-in is used, which is only meant
    for the CPU quick start.
    """

    def __init__(
        self,
        worker_python: Optional[str] = None,
        worker_script: Optional[str] = None,
        nuscenes_root: Optional[str] = None,
        mambafusion_ckpt: Optional[str] = None,
    ):
        self.worker_python = worker_python
        self.worker_script = worker_script
        self.nuscenes_root = nuscenes_root
        self.mambafusion_ckpt = mambafusion_ckpt
        self._client = None
        if self.available:
            print("[FusionDetector] MambaFusion scores come from the worker process.")
        else:
            print("[FusionDetector] No MambaFusion worker configured. Using the keyword stand-in.")

    @property
    def available(self) -> bool:
        return bool(
            self.worker_python and self.worker_script
            and Path(self.worker_python).expanduser().exists()
        )

    @property
    def client(self):
        if self._client is None:
            if not self.available:
                raise RuntimeError(
                    "MambaFusion worker is not configured: set paths.mambafusion_python to the "
                    "Python of the mambafusion environment"
                )
            from detectors.fusion_client import FusionWorkerClient

            env = {}
            if self.nuscenes_root:
                env["NUSCENES_ROOT"] = str(Path(self.nuscenes_root).resolve())
            if self.mambafusion_ckpt:
                env["MAMBAFUSION_CKPT"] = str(Path(self.mambafusion_ckpt).expanduser().resolve())
            self._client = FusionWorkerClient(self.worker_python, self.worker_script, env=env)
        return self._client

    def __call__(
        self,
        obj: Any,
        image: Optional[np.ndarray] = None,
        lidar: Optional[Any] = None,
        viewpoint: int = 0,
    ) -> float:
        return DummyDetector("fusion", seed=123)(obj, image, lidar, viewpoint)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def build_detectors(cfg: dict, use_real: bool):
    """2D YOLO and camera-LiDAR fusion detectors selected by the config."""
    device = resolve_device(cfg.get("experiment", {}).get("device", "auto"))
    det_cfg = cfg.get("detectors", {})
    paths = cfg.get("paths", {})
    if not use_real:
        return {"2d": DummyDetector("2d"), "fusion": DummyDetector("fusion")}
    det_2d = YOLODetector(
        det_cfg.get("yolo_model", "yolov10x.pt"),
        device=device,
        person_class=int(det_cfg.get("person_class", 0)),
        required=True,
    )
    det_fusion = FusionDetector(
        worker_python=paths.get("mambafusion_python") or None,
        worker_script=paths.get("mambafusion_worker") or None,
        nuscenes_root=paths.get("nuscenes_root") or None,
        mambafusion_ckpt=paths.get("mambafusion_ckpt") or None,
    )
    # Fail loudly instead of silently scoring fusion with the keyword stand-in.
    lambda_fusion = float(cfg.get("loss", {}).get("lambda_fusion", 0.0))
    if lambda_fusion > 0 and not det_fusion.available:
        raise RuntimeError(
            "--use_real_models with loss.lambda_fusion > 0 needs a MambaFusion worker: "
            "set paths.mambafusion_python to the mambafusion environment's Python "
            "(and paths.mambafusion_worker). Set loss.lambda_fusion: 0 to run the "
            "2D-only attack without the fusion detector."
        )
    return {"2d": det_2d, "fusion": det_fusion}
