"""Load the official MambaFusion checkpoint. Run with the mambafusion env.

Working directory must be the repo's tools/ so the Hilbert templates resolve.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("MAMBAFUSION_REPO", PROJECT_ROOT / "third_party" / "MambaFusion")).resolve()
CKPT = Path(os.environ.get("MAMBAFUSION_CKPT", REPO / "ckpts" / "checkpoint_epoch_10.pth")).resolve()
os.chdir(REPO / "tools")
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils


class _EmptyDataset(DatasetTemplate):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
    cfg_from_yaml_file("cfgs/mambafusion_models/mamba_fusion.yaml", cfg)
    logger = common_utils.create_logger()
    dataset = _EmptyDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=REPO,
        logger=logger,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(CKPT), logger=logger, to_cpu=True)
    model.cuda().eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"MAMBAFUSION_LOAD_OK params={params} classes={list(cfg.CLASS_NAMES)}")


if __name__ == "__main__":
    main()
