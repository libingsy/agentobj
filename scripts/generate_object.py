#!/usr/bin/env python
"""Generate one pedestrian with MV-Adapter + LGM and save the views and Gaussians."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image

from generators.generator import build_generator


def main():
    parser = argparse.ArgumentParser(description="Run one MV-Adapter + LGM generation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--base", default=None)
    parser.add_argument("--semantic", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--output", default="outputs/object")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    prompts = cfg.get("prompts", {})
    base = args.base or prompts.get("base", "a realistic pedestrian standing on the road")
    semantic = args.semantic if args.semantic is not None else prompts.get("semantic_init", "")
    style = args.style if args.style is not None else prompts.get("style_init", "")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    obj = build_generator(cfg, use_real=True).generate(base, semantic, style)

    import torch
    torch.save(
        {
            "prompt": obj["prompt"],
            "gaussians": obj["gaussians"],
            "point_cloud": obj["point_cloud"],
            "box_stats": obj["box_stats"],
            "backend": obj["backend"],
            "mv_azimuths": obj.get("mv_azimuths"),
        },
        out / "object.pt",
    )
    import numpy as np
    cloud = obj["point_cloud"]
    if hasattr(cloud, "detach"):
        cloud = cloud.detach().float().cpu().numpy()
    np.save(out / "points.npy", np.asarray(cloud))
    for azimuth, image in zip(obj.get("mv_azimuths", []), obj.get("mv_images", [])):
        Image.fromarray(image).save(out / f"view_{int(azimuth):03d}.png")
    print(f"Saved {obj['backend']} object to {out}")
    print(f"Prompt: {obj['prompt']}")
    print(f"Box: {obj['box_stats']}")


if __name__ == "__main__":
    main()
