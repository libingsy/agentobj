"""
Naturalness regularizer utilities.

L_nat is the cosine distance from a standard PointNet embedding of the object's
point cloud to a bank of real-pedestrian embeddings (plus a box-size prior).
The encoder is the standard PointNet backbone (`regularizers/pointnet.py`); its
1024-d global feature is the embedding.

Loading a classification checkpoint (`pointnet_ckpt`, e.g. the yanx27 ModelNet40
model) gives a trained encoder. The encoder is saved together with the bank and
reloaded at attack time, so the bank features and the query features always come
from the SAME weights. Point clouds are centred and scaled before encoding, so a
Y-up generated object and a real LiDAR cloud are compared in a common frame.

Without a checkpoint the backbone is randomly initialised (fixed seed);
`scripts/build_feature_bank.py` then requires the explicit --allow-random-encoder
flag. A standard checkpoint is trained on ModelNet40.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from regularizers.pointnet import PointNetEncoder, load_encoder_checkpoint
from utils.device import resolve_device

ENCODER_SEED = 1234
NUM_POINTS = 1024


def _normalize(pc: torch.Tensor) -> torch.Tensor:
    """Centre to the centroid and scale to unit radius (PointNet preprocessing).

    Invariant to absolute position and overall size, so a generated object and a
    reference cloud are measured in the same frame. It does not align
    orientation; the trained encoder supplies the shape prior.
    """
    pc = pc - pc.mean(dim=0, keepdim=True)
    scale = pc.norm(dim=1).max()
    if scale > 1e-6:
        pc = pc / scale
    return pc


class NaturalnessRegularizer:
    def __init__(
        self,
        device: str = "auto",
        bank_path: Optional[str] = None,
        pointnet_ckpt: Optional[str] = None,
    ):
        self.device = resolve_device(device)
        # Deterministic init so an untrained encoder is at least reproducible;
        # restore the global RNG so seeding it does not perturb the run.
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(ENCODER_SEED)
        self.encoder = PointNetEncoder(channel=3, feature_transform=True).to(self.device)
        torch.random.set_rng_state(rng_state)
        if pointnet_ckpt and Path(pointnet_ckpt).is_file():
            load_encoder_checkpoint(self.encoder, pointnet_ckpt)
            print(f"[Naturalness] Loaded PointNet checkpoint {pointnet_ckpt}")
        elif pointnet_ckpt:
            raise FileNotFoundError(f"pointnet_ckpt not found: {pointnet_ckpt}")
        else:
            print("[Naturalness] No PointNet checkpoint; encoder is randomly "
                  "initialised (set paths.pointnet_ckpt for trained weights).")
        self.encoder.eval()
        self.feature_bank: Optional[torch.Tensor] = None
        self.ref_box = {"w": 0.6, "h": 1.7, "l": 0.6}  # typical pedestrian

        if bank_path and Path(bank_path).is_file():
            self.load_bank(bank_path)

    def extract_feature(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Normalized 1024-d PointNet feature of a point cloud."""
        if point_cloud is None or len(point_cloud) < 10:
            return torch.zeros(self.encoder.feature_dim, device=self.device)
        pc = point_cloud.to(self.device).float()
        if pc.dim() == 2 and pc.shape[1] > 3:
            pc = pc[:, :3]
        if len(pc) >= NUM_POINTS:
            idx = torch.randperm(len(pc), device=pc.device)[:NUM_POINTS]
            pc = pc[idx]
        pc = _normalize(pc)
        x = pc.t().unsqueeze(0).contiguous()  # (1, 3, N)
        with torch.no_grad():
            feat = self.encoder(x).squeeze(0)
        return feat / (feat.norm() + 1e-6)

    def build_bank_from_points(self, point_clouds: List[torch.Tensor]):
        """Build the feature bank from a list of real pedestrian point clouds."""
        feats = []
        for pc in point_clouds:
            if not isinstance(pc, torch.Tensor):
                pc = torch.as_tensor(pc)
            feats.append(self.extract_feature(pc))
        if feats:
            self.feature_bank = torch.stack(feats).to(self.device)
            print(f"[Naturalness] Built feature bank with {len(feats)} samples")

    def min_cosine_distance(self, feat: torch.Tensor) -> float:
        if self.feature_bank is None or len(self.feature_bank) == 0:
            return 0.0
        cos = torch.nn.functional.cosine_similarity(
            feat.unsqueeze(0), self.feature_bank, dim=1
        )
        return float(1.0 - cos.max())

    def box_deviation(self, box_stats: Optional[Dict]) -> float:
        if not box_stats:
            return 0.0
        loss = 0.0
        for k in ["w", "h", "l"]:
            if k in box_stats and k in self.ref_box:
                loss += abs(box_stats[k] - self.ref_box[k])
        return loss

    def compute(self, point_cloud: Optional[torch.Tensor], box_stats: Optional[Dict] = None) -> float:
        feat = self.extract_feature(point_cloud) if point_cloud is not None else None
        feat_loss = self.min_cosine_distance(feat) if feat is not None else 0.0
        box_loss = self.box_deviation(box_stats)
        return feat_loss + 0.1 * box_loss

    def save_bank(self, path: str):
        """Save the bank together with the encoder weights that produced it."""
        if self.feature_bank is None:
            return
        torch.save(
            {
                "feature_bank": self.feature_bank.cpu(),
                "encoder_state_dict": self.encoder.state_dict(),
                "feature_dim": self.encoder.feature_dim,
                "arch": "pointnet",
            },
            path,
        )

    def load_bank(self, path: str):
        """Load the bank and the encoder that built it, so query == bank encoder."""
        try:
            obj = torch.load(path, map_location=self.device)
            if isinstance(obj, dict) and "feature_bank" in obj:
                state = obj.get("encoder_state_dict")
                if state is not None:
                    try:
                        self.encoder.load_state_dict(state)
                        self.encoder.to(self.device).eval()
                    except Exception as e:
                        print(f"[Naturalness] Bank encoder does not match PointNet "
                              f"({e}); rebuild the bank with this code.")
                self.feature_bank = obj["feature_bank"].to(self.device)
            else:
                print("[Naturalness] Bank stores no encoder; rebuild it so bank "
                      "and query share one encoder.")
                self.feature_bank = obj.to(self.device)
            print(f"[Naturalness] Loaded feature bank from {path}")
        except Exception as e:
            print(f"[Naturalness] Failed to load bank: {e}")
