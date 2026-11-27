"""Attack success rates.

A view is one (nuScenes keyframe, object yaw) pair. Following the paper, the
attack succeeds on a view when the object's confidence is below the threshold
theta for the 2D detector (ASR_2D), for the fusion detector (ASR_F), or for
both (ASR_joint). `asr_joint_all_yaws` additionally requires every yaw of a
keyframe to succeed, the stricter multi-view criterion that Algorithm 1 uses
to stop.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class Evaluator:
    def __init__(self, conf_threshold: float = 0.5):
        self.theta = conf_threshold

    def is_attack_success(
        self,
        scores_2d: List[float],
        scores_fusion: List[float],
        require_both: bool = True,
    ) -> bool:
        """Every view below theta: on both detectors (`require_both`) or on either."""
        success_2d = all(s < self.theta for s in scores_2d)
        success_fusion = all(s < self.theta for s in scores_fusion)
        if require_both:
            return success_2d and success_fusion
        return success_2d or success_fusion

    def view_metrics(self, rows: List[Dict]) -> Dict:
        """ASR over views. Each row has `token`, `yaw`, `score_2d` and `fusion`."""
        if not rows:
            return {"n_views": 0}
        n = len(rows)
        ok_2d = [r["score_2d"] < self.theta for r in rows]
        ok_f = [r["fusion"] < self.theta for r in rows]
        joint = [a and b for a, b in zip(ok_2d, ok_f)]
        per_key = defaultdict(list)
        for r, j in zip(rows, joint):
            per_key[r["token"]].append(j)
        return {
            "n_views": n,
            "n_keyframes": len(per_key),
            "asr_2d": sum(ok_2d) / n,
            "asr_fusion": sum(ok_f) / n,
            "asr_joint": sum(joint) / n,
            "asr_joint_all_yaws": sum(all(v) for v in per_key.values()) / len(per_key),
            "mean_score_2d": sum(r["score_2d"] for r in rows) / n,
            "mean_fusion": sum(r["fusion"] for r in rows) / n,
            # Detection rates of the same views, the complement used for the benign object.
            "det_rate_2d": 1.0 - sum(ok_2d) / n,
            "det_rate_fusion": 1.0 - sum(ok_f) / n,
        }

    def compute_asr(self, results: List[Dict], require_both: bool = True) -> Dict:
        """Fraction of objects whose every view succeeds. Each result has `scores_2d` and `scores_fusion`."""
        if not results:
            return {"asr": 0.0, "n": 0}
        successes = sum(
            1 for r in results
            if self.is_attack_success(r.get("scores_2d", []), r.get("scores_fusion", []), require_both)
        )
        return {
            "asr": successes / len(results),
            "successes": successes,
            "total": len(results),
            "require_both": require_both,
        }

    def summarize_history(self, history: List[Dict]) -> Dict:
        if not history:
            return {}
        losses = [h["loss"] for h in history if h.get("loss") is not None]
        decisions = [h for h in history if h.get("iter", 0) != 0]
        accepted = sum(1 for h in decisions if h.get("accepted", False))
        return {
            "num_iters": len(history),
            "best_loss": min(losses) if losses else None,
            "final_loss": losses[-1] if losses else None,
            "accepted_rate": accepted / len(decisions) if decisions else 0.0,
            "mean_2d_final": history[-1].get("mean_2d"),
            "mean_fusion_final": history[-1].get("mean_fusion"),
        }

    def save_report(self, metrics: Dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Eval] Report saved to {path}")
