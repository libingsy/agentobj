"""
Loss functions for Agent-OBJ
- Attack loss (joint 2D + Fusion confidence)
- Naturalness regularizer (feature bank)
- Multi-view consistency regularizer
"""

import torch
from typing import Dict, List, Optional

from utils.device import resolve_device


def threshold_aware_phi(confidence: float, theta: float = 0.5) -> float:
    """Surrogate that penalizes confidence above decision threshold."""
    return max(0.0, confidence - theta)


class AgentOBJLoss:
    def __init__(
        self,
        lambda_2d: float = 1.0,
        lambda_fusion: float = 1.0,
        alpha_nat: float = 0.1,
        beta_mv: float = 0.1,
        conf_threshold: float = 0.5,
        use_naturalness: bool = True,
        device: str = "auto",
    ):
        self.lambda_2d = lambda_2d
        self.lambda_fusion = lambda_fusion
        self.alpha_nat = alpha_nat
        self.beta_mv = beta_mv
        self.theta = conf_threshold
        self.use_naturalness = use_naturalness
        self.device = resolve_device(device)
        self.naturalness = None

    def attach_naturalness(self, regularizer) -> None:
        """Share one PointNet and feature bank with the optimizer."""
        self.naturalness = regularizer
        self.use_naturalness = regularizer is not None

    def attack_loss(
        self,
        scores_2d: List[float],
        scores_fusion: List[float],
    ) -> Dict[str, float]:
        """Compute per-view and aggregated attack loss."""
        per_view = []
        for s2d, sf in zip(scores_2d, scores_fusion):
            l = (
                self.lambda_2d * threshold_aware_phi(s2d, self.theta)
                + self.lambda_fusion * threshold_aware_phi(sf, self.theta)
            )
            per_view.append(l)
        # One view above 0.5 fails the attack. The mean hides that view
        # behind the other three zeros.
        l_atk = max(per_view) if per_view else 0.0
        return {
            "l_atk": l_atk,
            "per_view": per_view,
            "mean_2d": sum(scores_2d) / max(len(scores_2d), 1),
            "mean_fusion": sum(scores_fusion) / max(len(scores_fusion), 1),
        }

    def _regularizer(self):
        if self.naturalness is None:
            from regularizers.naturalness import NaturalnessRegularizer
            self.naturalness = NaturalnessRegularizer(device=self.device)
        return self.naturalness

    def naturalness_loss(
        self,
        point_cloud: torch.Tensor,
        feature_bank: Optional[torch.Tensor] = None,
        box_stats: Optional[Dict] = None,
        ref_box: Optional[Dict] = None,
    ) -> float:
        """
        L_nat = box deviation from a standing pedestrian + min cosine distance
        of a PointNet embedding to the pedestrian feature bank.
        """
        if not self.use_naturalness:
            return 0.0
        reg = self._regularizer()
        if feature_bank is not None:
            reg.feature_bank = feature_bank
        if ref_box is not None:
            reg.ref_box = ref_box
        return float(reg.compute(point_cloud, box_stats))

    def multi_view_loss(self, per_view_losses: List[float]) -> float:
        """L_mv = Std of per-view attack losses."""
        if len(per_view_losses) < 2:
            return 0.0
        t = torch.tensor(per_view_losses)
        return float(t.std())

    def total(
        self,
        scores_2d: List[float],
        scores_fusion: List[float],
        point_cloud: Optional[torch.Tensor] = None,
        feature_bank: Optional[torch.Tensor] = None,
        box_stats: Optional[Dict] = None,
        ref_box: Optional[Dict] = None,
    ) -> Dict[str, float]:
        atk = self.attack_loss(scores_2d, scores_fusion)
        l_nat = self.naturalness_loss(point_cloud, feature_bank, box_stats, ref_box)
        l_mv = self.multi_view_loss(atk["per_view"])
        total = atk["l_atk"] + self.alpha_nat * l_nat + self.beta_mv * l_mv
        return {
            "total": total,
            "l_atk": atk["l_atk"],
            "l_nat": l_nat,
            "l_mv": l_mv,
            "mean_2d": atk["mean_2d"],
            "mean_fusion": atk["mean_fusion"],
            "per_view": atk["per_view"],
        }
