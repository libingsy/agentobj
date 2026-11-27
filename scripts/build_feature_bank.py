#!/usr/bin/env python
"""Build the pedestrian PointNet feature bank used by L_nat.

Point clouds are `.npy` or `.pt` files of shape (N, 3) in a directory. They MUST
be in the same y-up object frame the generator produces (up = +y, feet near
y = 0), because L_nat compares each bank cloud with the generated object's
point cloud; a bank in the LiDAR z-up frame would make the cosine distance
measure the axis difference instead of naturalness. Rotate real LiDAR pedestrian
clouds into that frame before building the bank. `--synthetic N` builds a bank
from the same standing-person clouds the dummy generator uses (already y-up), so
the regularizer can be exercised before real scans exist.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from generators.generator import _pedestrian_gaussians
from regularizers.naturalness import NaturalnessRegularizer


def _load_cloud(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".pt":
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, dict):
            arr = arr.get("point_cloud", arr.get("points"))
    else:
        raise ValueError(f"Unsupported cloud file {path}")
    if isinstance(arr, torch.Tensor):
        return arr.float()
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clouds", type=str, default="", help="Directory of .npy or .pt point clouds")
    parser.add_argument("--synthetic", type=int, default=0, help="Number of synthetic pedestrians to add")
    parser.add_argument("--output", type=str, default="data/pedestrian_feature_bank.pt")
    parser.add_argument("--pointnet_ckpt", type=str, default="",
                        help="Standard PointNet classification checkpoint (yanx27 format).")
    parser.add_argument("--allow-random-encoder", dest="allow_random_encoder", action="store_true",
                        help="Build the bank with a randomly initialised encoder (no --pointnet_ckpt)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.pointnet_ckpt and not args.allow_random_encoder:
        raise SystemExit(
            "No --pointnet_ckpt given. Pass a PointNet classification checkpoint "
            "(yanx27 format), or --allow-random-encoder to build the bank with a "
            "randomly initialised encoder on purpose."
        )

    clouds = []
    if args.clouds:
        root = Path(args.clouds)
        for path in sorted(root.iterdir()):
            if path.suffix in {".npy", ".pt"}:
                clouds.append(_load_cloud(path))
    for i in range(args.synthetic):
        prompt = f"synthetic pedestrian {i}"
        clouds.append(_pedestrian_gaussians(prompt)[:, 0:3])
    if not clouds:
        raise SystemExit("No clouds found. Pass --clouds or --synthetic N.")

    reg = NaturalnessRegularizer(device=args.device, pointnet_ckpt=args.pointnet_ckpt or None)
    reg.build_bank_from_points(clouds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    reg.save_bank(args.output)
    print(f"Wrote {reg.feature_bank.shape[0]} features to {args.output}")


if __name__ == "__main__":
    main()
