"""Run one real nuScenes keyframe through MambaFusion and print the pedestrian score.

Uses the mambafusion env. The frame is built from the nuScenes devkit and scored
with the keyframe only, so the official 10-sweep info pickle is not required.

Paths come from the environment:

    MAMBAFUSION_REPO  MambaFusion checkout       (default: third_party/MambaFusion)
    MAMBAFUSION_CKPT  detector checkpoint        (default: $MAMBAFUSION_REPO/ckpts/checkpoint_epoch_10.pth)
    NUSCENES_ROOT     nuScenes dataroot          (default: data/nuscenes)
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("MAMBAFUSION_REPO", PROJECT_ROOT / "third_party" / "MambaFusion")).resolve()
CKPT = Path(os.environ.get("MAMBAFUSION_CKPT", REPO / "ckpts" / "checkpoint_epoch_10.pth")).resolve()
NUSC_ROOT = Path(os.environ.get("NUSCENES_ROOT", PROJECT_ROOT / "data" / "nuscenes")).resolve()
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

os.chdir(REPO / "tools")
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from nuscenes.nuscenes import NuScenes
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.datasets.nuscenes.nuscenes_utils import obtain_sensor2top
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def first_pedestrian_sample(nusc):
    for sample in nusc.sample:
        for token in sample["anns"]:
            name = nusc.get("sample_annotation", token)["category_name"]
            if name.startswith("human.pedestrian"):
                return sample
    return nusc.sample[0]


def build_info(nusc, sample):
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    l2e_t = np.array(lidar_cs["translation"], dtype=np.float32).reshape(1, 3)
    l2e_r = Quaternion(lidar_cs["rotation"]).rotation_matrix
    e2g_t = np.array(lidar_pose["translation"], dtype=np.float32).reshape(1, 3)
    e2g_r = Quaternion(lidar_pose["rotation"]).rotation_matrix

    cams = {}
    for cam in CAMERAS:
        cam_token = sample["data"][cam]
        cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r, e2g_t, e2g_r, cam)
        sd = nusc.get("sample_data", cam_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        cam_info["camera_intrinsics"] = np.array(cs["camera_intrinsic"], dtype=np.float32)
        cams[cam] = cam_info
    return {
        "lidar_path": nusc.get_sample_data_path(lidar_token),
        "token": sample["token"],
        "sweeps": [],
        "cams": cams,
        "timestamp": sample["timestamp"],
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    print("gpu", torch.cuda.get_device_name(0))
    cfg_from_yaml_file("cfgs/mambafusion_models/mamba_fusion.yaml", cfg)
    cfg.DATA_CONFIG.DATA_PATH = str(NUSC_ROOT)
    cfg.DATA_CONFIG.VERSION = "v1.0-trainval"
    cfg.DATA_CONFIG.MAX_SWEEPS = 1
    logger = common_utils.create_logger()

    print("opening nuScenes tables")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)
    sample = first_pedestrian_sample(nusc)
    info = build_info(nusc, sample)
    print("sample", sample["token"], "lidar", info["lidar_path"])

    dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=NUSC_ROOT.parent,
        logger=logger,
    )
    dataset.infos = [info]
    data = dataset[0]
    print("points", tuple(data["points"].shape), "cameras", len(data["camera_imgs"]))
    batch = dataset.collate_batch([data])
    load_data_to_gpu(batch)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(
        filename=str(CKPT),
        logger=logger,
        to_cpu=True,
    )
    model.cuda().eval()
    with torch.no_grad():
        pred_dicts, _ = model(batch)
    pred = pred_dicts[0]
    labels = pred["pred_labels"].detach().cpu().numpy()
    scores = pred["pred_scores"].detach().cpu().numpy()
    names = cfg.CLASS_NAMES
    ped_index = names.index("pedestrian") + 1
    ped = scores[labels == ped_index]
    ped_max = float(ped.max()) if len(ped) else 0.0
    print("detections", len(scores))
    order = np.argsort(-scores)[:8]
    for i in order:
        print(f"  {names[int(labels[i]) - 1]:20s} {scores[i]:.4f}")
    print(f"MAMBAFUSION_FORWARD_OK pedestrian_max={ped_max:.4f} n_ped={len(ped)}")


if __name__ == "__main__":
    main()
