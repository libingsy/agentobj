"""Resident MambaFusion scorer for the Agent-OBJ process.

The attack environment is torch 2.5. This process is the Python 3.8 / torch 2.1
environment that MambaFusion's CUDA kernels need. One JSON object per stdin
line, one JSON object per stdout line:

    {"op": "ping"}
    {"op": "select_samples", "scenes": 10, "keyframes": 5, "split": "val", "placement": {...}}
    {"op": "insert_score", "token": "...", "gaussians": "obj.npy", "yaws": [0, 90, 180, 270],
     "out_dir": "...", "placement": {...}}
    {"op": "score", "token": "..."}            # the unmodified frame
    {"op": "quit"}

`insert_score` places the object on the road in front of the ego vehicle
(`data/nuscenes_insert.py`), rasterizes it into the camera image, ray-casts it
into the keyframe and every stacked sweep, and runs MambaFusion once per yaw.
The fusion score of the object is the highest pedestrian score among predicted
boxes whose BEV center lies within `match_radius` meters of the object.

`--self-test` scores the first keyframe that has a pedestrian.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.forward_mambafusion import (  # noqa: E402
    CKPT,
    NUSC_ROOT,
    build_info,
    first_pedestrian_sample,
)

from nuscenes.nuscenes import NuScenes  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from data.nuscenes_insert import NuScenesScene, save_image  # noqa: E402

MAX_SWEEPS = int(os.environ.get("MAMBAFUSION_SWEEPS", "10"))
DEFAULT_PLACEMENT = {
    "distances": [8.0, 10.0, 12.0, 6.0],
    "lateral": [0.0, -1.0, 1.0],
    "match_radius": 1.5,
    "intensity": "auto",
    "intensity_fallback": 12.0,
    # LiDAR insertion: "geometric" ray-casts the body into every sweep and smears
    # the ten-sweep accumulation along the walk (a moving pedestrian). "shell" is
    # the legacy sparse single-sweep variant.
    "lidar_mode": "geometric",
    "walk_speed": 1.4,
    "walk_mode": "facing",
}


def _render_for_yaw(renders: dict, yaw: float):
    """The multi-view image closest to this yaw. The gray backdrop is removed at paste time."""
    from PIL import Image

    target = float(yaw) % 360.0

    def distance(key):
        delta = abs(float(key) - target) % 360.0
        return min(delta, 360.0 - delta)

    path = min(renders, key=distance)
    return np.array(Image.open(renders[path]).convert("RGB"))


class Scorer:
    def __init__(self):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        cfg_from_yaml_file("cfgs/mambafusion_models/mamba_fusion.yaml", cfg)
        cfg.DATA_CONFIG.DATA_PATH = str(NUSC_ROOT)
        cfg.DATA_CONFIG.VERSION = "v1.0-trainval"
        cfg.DATA_CONFIG.MAX_SWEEPS = MAX_SWEEPS
        self.cfg = cfg
        self.names = list(cfg.CLASS_NAMES)
        self.logger = common_utils.create_logger()
        self.nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)
        self.dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=NUSC_ROOT.parent,
            logger=self.logger,
        )
        self.model = None
        self._infos = {}
        self._scenes = {}
        self._clean = {}

    # ------------------------------------------------------------------ data
    def info(self, token: str) -> dict:
        if token not in self._infos:
            self._infos[token] = build_info(self.nusc, self.nusc.get("sample", token))
        return self._infos[token]

    def scene(self, token: str, camera: str = "CAM_FRONT") -> NuScenesScene:
        key = (token, camera)
        if key not in self._scenes:
            if len(self._scenes) > 48:
                self._scenes.pop(next(iter(self._scenes)))
            self._scenes[key] = NuScenesScene.load(self.nusc, token, camera=camera, max_sweeps=MAX_SWEEPS)
        return self._scenes[key]

    def _model(self):
        if self.model is None:
            model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.names), dataset=self.dataset)
            model.load_params_from_file(filename=str(CKPT), logger=self.logger, to_cpu=True)
            model.cuda().eval()
            self.model = model
        return self.model

    # ------------------------------------------------------------------ inference
    def predict(self, token: str, points: np.ndarray, camera_files=None) -> dict:
        """MambaFusion on a stacked (N, 5) cloud with optional replacement camera images."""
        import torch

        info = self.info(token)
        if camera_files:
            info = dict(info)
            info["cams"] = {name: dict(cam) for name, cam in info["cams"].items()}
            for name, path in camera_files.items():
                info["cams"][name]["data_path"] = str(path)
        sample = {
            "points": np.asarray(points, dtype=np.float32),
            "frame_id": Path(info["lidar_path"]).stem,
            "metadata": {"token": info["token"]},
        }
        sample = self.dataset.load_camera_info(sample, info)
        data = self.dataset.prepare_data(sample)
        data.pop("ori_imgs", None)
        batch = self.dataset.collate_batch([data])
        load_data_to_gpu(batch)
        with torch.no_grad():
            pred_dicts, _ = self._model()(batch)
        pred = pred_dicts[0]
        return {
            "boxes": pred["pred_boxes"].detach().cpu().numpy(),
            "scores": pred["pred_scores"].detach().cpu().numpy(),
            "labels": pred["pred_labels"].detach().cpu().numpy(),
        }

    def near(self, pred: dict, xy, radius: float) -> dict:
        """Scores of the predictions whose BEV center is within `radius` of `xy`."""
        ped = self.names.index("pedestrian") + 1
        out = {"pedestrian": 0.0, "any": 0.0, "any_class": None, "n_near": 0}
        if len(pred["scores"]) == 0:
            return out
        dist = np.linalg.norm(pred["boxes"][:, :2] - np.asarray(xy, dtype=np.float32)[:2], axis=1)
        sel = dist <= float(radius)
        out["n_near"] = int(sel.sum())
        if not sel.any():
            return out
        scores, labels = pred["scores"][sel], pred["labels"][sel]
        if (labels == ped).any():
            out["pedestrian"] = float(scores[labels == ped].max())
        best = int(scores.argmax())
        out["any"] = float(scores[best])
        out["any_class"] = self.names[int(labels[best]) - 1]
        return out

    def pedestrian_max(self, pred: dict) -> float:
        ped = self.names.index("pedestrian") + 1
        s = pred["scores"][pred["labels"] == ped]
        return float(s.max()) if len(s) else 0.0

    # ------------------------------------------------------------------ ops
    def score_clean(self, token: str, camera: str = "CAM_FRONT") -> dict:
        from data.nuscenes_insert import stack_sweeps

        if token not in self._clean:
            scene = self.scene(token, camera)
            self._clean[token] = self.predict(token, stack_sweeps(scene.sweeps))
        return self._clean[token]

    def insert_score(self, req: dict) -> dict:
        import torch

        token = req["token"]
        camera = req.get("camera", "CAM_FRONT")
        opts = dict(DEFAULT_PLACEMENT)
        opts.update(req.get("placement") or {})
        scene = self.scene(token, camera)
        place = scene.placement(distances=opts["distances"], lateral=opts["lateral"])
        if place is None:
            raise RuntimeError(f"no free spot in front of {camera} in sample {token}")
        if opts.get("intensity", "auto") == "auto":
            intensity = scene.intensity(float(opts["intensity_fallback"]))
        else:
            intensity = float(opts["intensity"])
        renders = req.get("renders") or {}
        gaussians = np.load(req["gaussians"]).astype(np.float32)
        points = np.load(req["points"]).astype(np.float32) if req.get("points") else gaussians[:, :3]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        xy = place["translation"][:2]
        radius = float(opts["match_radius"])
        clean = self.near(self.score_clean(token, camera), xy, radius)
        # Geometric motion insertion by default: ray-cast the body into every
        # sweep and smear the accumulation along the walk (a moving pedestrian).
        # "shell" is the legacy sparse variant and needs renders.
        from data.nuscenes_insert import walk_velocity

        use_shell = opts.get("lidar_mode") == "shell" and bool(renders)
        if use_shell:
            intensity = 35.0
        views = []
        for yaw in req.get("yaws", [0]):
            if use_shell:
                frame = scene.insert_shell(
                    points, _render_for_yaw(renders, yaw), place["translation"], float(yaw),
                    gaussians, intensity=intensity, device=device,
                )
            else:
                velocity = walk_velocity(
                    place["translation"], float(yaw),
                    speed=float(opts.get("walk_speed", 1.4)), mode=opts.get("walk_mode", "facing"),
                )
                frame = scene.insert(
                    gaussians, place["translation"], float(yaw), intensity,
                    velocity=velocity, device=device,
                )
            if frame["bbox"] is None:
                raise RuntimeError(f"object is not visible in {camera} of {token} at yaw {yaw}")
            image_path = save_image(frame["image"], out_dir / f"{camera.lower()}_yaw{int(yaw):03d}.png")
            pred = self.predict(token, frame["points"], camera_files={camera: image_path})
            near = self.near(pred, xy, radius)
            views.append({
                "yaw": float(yaw),
                "fusion": near["pedestrian"],
                "fusion_any": near["any"],
                "fusion_any_class": near["any_class"],
                "image": image_path,
                "bbox": frame["bbox"],
                "object_points": frame["stats"]["object_points"],
                "removed_points": frame["stats"]["removed"],
                "pedestrian_max": self.pedestrian_max(pred),
            })
            if req.get("save_points"):
                np.save(out_dir / f"lidar_yaw{int(yaw):03d}.npy", frame["points"])
        return {
            "ok": True,
            "token": token,
            "placement": place,
            "intensity": intensity,
            "clean_near": clean["pedestrian"],
            "views": views,
        }

    def select_samples(self, req: dict) -> dict:
        """Evenly spaced scenes of a split, and keyframes in each with a free spot in front of the camera."""
        from nuscenes.utils.splits import create_splits_scenes

        n_scenes = int(req.get("scenes", 10))
        n_keys = int(req.get("keyframes", 5))
        camera = req.get("camera", "CAM_FRONT")
        opts = dict(DEFAULT_PLACEMENT)
        opts.update(req.get("placement") or {})
        names = sorted(create_splits_scenes()[req.get("split", "val")])
        by_name = {s["name"]: s for s in self.nusc.scene}
        names = [n for n in names if n in by_name]
        if not names:
            raise RuntimeError("no scenes of that split in this nuScenes copy")
        stride = max(len(names) // max(n_scenes, 1), 1)
        order = []
        for offset in range(stride):
            order.extend(names[offset::stride])
        chosen = []
        for name in order:
            if len(chosen) >= n_scenes:
                break
            rec = by_name[name]
            tokens = []
            tok = rec["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.nusc.get("sample", tok)["next"]
            # Evenly spaced keyframes first, then the positions between them as fallbacks.
            spread = [int(i) for i in np.linspace(0, len(tokens) - 1, n_keys).round()]
            extra = [int(i) for i in np.linspace(0, len(tokens) - 1, 3 * n_keys).round() if int(i) not in spread]
            picks = []
            for i in spread + extra:
                if len(picks) >= n_keys:
                    break
                token = tokens[i]
                if token in picks:
                    continue
                scene = self.scene(token, camera)
                if scene.placement(distances=opts["distances"], lateral=opts["lateral"]) is not None:
                    picks.append(token)
                else:
                    self._scenes.pop((token, camera), None)
            if len(picks) == n_keys:
                picks.sort(key=tokens.index)
                chosen.append({"scene": name, "description": rec.get("description", ""), "tokens": picks})
        return {"ok": True, "split": req.get("split", "val"), "scenes": chosen}


def self_test() -> None:
    from data.nuscenes_insert import stack_sweeps

    scorer = Scorer()
    sample = first_pedestrian_sample(scorer.nusc)
    scene = scorer.scene(sample["token"])
    pred = scorer.predict(sample["token"], stack_sweeps(scene.sweeps))
    print(
        "MAMBAFUSION_BRIDGE_OK "
        f"token={sample['token']} sweeps={len(scene.sweeps)} "
        f"pedestrian_max={scorer.pedestrian_max(pred):.4f} n_det={len(pred['scores'])}"
    )


def _quiet(fn, *args, **kwargs):
    """MambaFusion prints tensor shapes to stdout. The protocol needs that stream."""
    real = sys.stdout
    sys.stdout = sys.stderr
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout = real


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def serve() -> None:
    try:
        scorer = _quiet(Scorer)
    except Exception as exc:
        _reply({"ok": False, "error": f"scorer init failed: {exc}"})
        return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                reply = {"ok": True, "ready": True, "max_sweeps": MAX_SWEEPS}
            elif op == "quit":
                _reply({"ok": True})
                return
            elif op == "score":
                pred = _quiet(scorer.score_clean, req["token"], req.get("camera", "CAM_FRONT"))
                reply = {"ok": True, "pedestrian_max": scorer.pedestrian_max(pred), "n_det": int(len(pred["scores"]))}
            elif op == "insert_score":
                reply = _quiet(scorer.insert_score, req)
            elif op == "select_samples":
                reply = _quiet(scorer.select_samples, req)
            else:
                reply = {"ok": False, "error": f"unknown op {op}"}
        except Exception as exc:
            import traceback

            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _reply(reply)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        serve()
