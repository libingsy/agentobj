"""Insert one pedestrian into nuScenes and score it with both detectors.

Runs in the Agent-OBJ environment. YOLOv10x scores the composited camera
image here. MambaFusion runs in its own process (`scripts/mambafusion_worker.py`),
which rasterizes the Gaussians into the image and ray-casts them into the
stacked sweeps. This is the benign baseline: a normal pedestrian, several val
keyframes, four yaws. Both confidences should be above 0.5 when the insertion
is right.

    python scripts/insert_and_score.py \
        --config configs/cluster.yaml \
        --object outputs/insert_object \
        --output outputs/benign_baseline \
        --scenes 4 --keyframes 1
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from detectors.detectors import build_detectors
from eval.evaluator import Evaluator
from eval.scene_eval import SceneEvaluator, crop_strip


def load_object(path: Path) -> dict:
    if path.is_dir():
        for name in ("object.pt", "object_gaussians.npy"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            raise SystemExit(f"no object.pt or object_gaussians.npy in {path}")
    folder = path if path.is_dir() else path.parent
    if path.suffix == ".npy":
        gaussians = np.load(path)
        obj = {"gaussians": gaussians, "prompt": path.stem, "backend": "npy"}
    else:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
    renders = {}
    for view in sorted(folder.glob("view_*.png")):
        try:
            yaw = int(view.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        renders[yaw] = str(view)
    if renders:
        obj["renders"] = renders
        print(f"[object] renders {sorted(renders)}")
    points_path = folder / "points.npy"
    if obj.get("point_cloud") is None and points_path.is_file():
        obj["point_cloud"] = np.load(points_path)
    gaussians = obj.get("gaussians")
    shape = tuple(getattr(gaussians, "shape", ()))
    if len(shape) != 2 or shape[1] < 14:
        raise SystemExit(f"expected gaussians (N, 14), got {shape} from {path}")
    print(f"[object] {path} gaussians={shape} backend={obj.get('backend')} prompt={obj.get('prompt')}")
    return obj


def parse_tokens(text: str):
    return [t for t in text.replace(",", " ").split() if t]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "cluster.yaml"))
    parser.add_argument("--object", default=str(ROOT / "outputs" / "insert_object"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "benign_baseline"))
    parser.add_argument("--scenes", type=int, default=4, help="val scenes when --tokens is empty")
    parser.add_argument("--keyframes", type=int, default=1, help="keyframes kept in each scene")
    parser.add_argument("--split", default="val")
    parser.add_argument("--tokens", default="", help="comma-separated sample tokens; skips scene selection")
    parser.add_argument("--theta", type=float, default=None, help="detection threshold, default loss.conf_threshold")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    obj = load_object(Path(args.object))
    detectors = build_detectors(cfg, use_real=True)
    fusion = detectors["fusion"]
    if not getattr(fusion, "available", False):
        raise SystemExit("MambaFusion worker is not configured (paths.mambafusion_python)")

    theta = float(args.theta if args.theta is not None else cfg.get("loss", {}).get("conf_threshold", 0.5))
    yaws = [float(y) for y in cfg.get("optimization", {}).get("viewpoints", [0, 90, 180, 270])]
    ev = cfg.setdefault("evaluation", {})
    placement = ev.get("placement") or {}
    camera = ev.get("camera", "CAM_FRONT")

    try:
        tokens = parse_tokens(args.tokens)
        scenes = []
        if tokens:
            scenes = [{"scene": "cli", "tokens": tokens}]
        else:
            scenes = fusion.client.select_samples(
                args.scenes, args.keyframes, split=args.split, placement=placement, camera=camera,
            )
            tokens = [t for scene in scenes for t in scene["tokens"]]
        if not tokens:
            raise SystemExit(f"no {args.split} keyframes with a free spot in front of {camera}")
        print(f"[select] {len(scenes)} scenes, {len(tokens)} keyframes, yaws={yaws}", flush=True)
        for scene in scenes:
            print(f"  {scene.get('scene')} {scene.get('tokens')}", flush=True)

        rows = SceneEvaluator(detectors["2d"], fusion, cfg).evaluate(obj, tokens, out, yaws=yaws)
        metrics = Evaluator(theta).view_metrics(rows)
    finally:
        fusion.close()

    strip = crop_strip(rows, out / "strip.jpg")
    both = bool(rows) and all(r["score_2d"] > theta and r["fusion"] > theta for r in rows)
    yolo_scores = [r["score_2d"] for r in rows]
    fusion_scores = [r["fusion"] for r in rows]
    summary = {
        "object": str(Path(args.object).resolve()),
        "prompt": obj.get("prompt"),
        "backend": obj.get("backend"),
        "theta": theta,
        "yaws": yaws,
        "scenes": scenes,
        "n_views": len(rows),
        "both_above_theta": both,
        "yolo_min": min(yolo_scores) if yolo_scores else None,
        "yolo_max": max(yolo_scores) if yolo_scores else None,
        "fusion_min": min(fusion_scores) if fusion_scores else None,
        "fusion_max": max(fusion_scores) if fusion_scores else None,
        "metrics": metrics,
        "strip": strip,
        "views": [
            {
                "token": r["token"],
                "yaw": r["yaw"],
                "score_2d": r["score_2d"],
                "fusion": r["fusion"],
                "object_points_total": r.get("object_points_total"),
                "image": r.get("image"),
            }
            for r in rows
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for r in rows:
        print(
            f"view token={r['token'][:12]} yaw={int(r['yaw']):03d} "
            f"yolo={r['score_2d']:.3f} fusion={r['fusion']:.3f} "
            f"points={r.get('object_points_total')}",
            flush=True,
        )
    print(
        "BENIGN_BASELINE "
        f"n_views={len(rows)} n_scenes={len(scenes)} theta={theta:.2f} "
        f"yolo_min={summary['yolo_min']:.4f} yolo_max={summary['yolo_max']:.4f} "
        f"fusion_min={summary['fusion_min']:.4f} fusion_max={summary['fusion_max']:.4f} "
        f"det_rate_2d={metrics.get('det_rate_2d', 0):.3f} "
        f"det_rate_fusion={metrics.get('det_rate_fusion', 0):.3f} "
        f"both_above_{theta:.2f}={both} strip={strip}",
        flush=True,
    )
    if not rows:
        raise SystemExit("no views were scored")


if __name__ == "__main__":
    main()
