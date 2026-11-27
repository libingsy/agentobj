"""
Algorithm 1. Each candidate is scored once.

When a MambaFusion worker and at least one nuScenes token are configured,
that score is `SceneEvaluator.evaluate`: every keyframe and every yaw, both
detectors, on the inserted object. `Evaluator.view_metrics` then supplies the
attack rates, and the search stops when `asr_joint_all_yaws` is 1 (every yaw
of every keyframe is below theta on both detectors). Without a worker the
orbit renders and the keyword stand-in are used, so the CPU tests still run.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from agent.improved_edit_agent import ImprovedEditAgent, EditAction
from regularizers.losses import AgentOBJLoss
from regularizers.naturalness import NaturalnessRegularizer
from render.scene_renderer import SceneRenderer
from eval.evaluator import Evaluator
from utils.device import resolve_device


_LOSS_KEYS = ("l_atk", "l_nat", "l_mv", "mean_2d", "mean_fusion")


class FullAgentOBJOptimizer:
    def __init__(
        self,
        generator,
        detectors: Dict,
        edit_agent: ImprovedEditAgent,
        loss_fn: AgentOBJLoss,
        config: Dict,
        output_dir: str = "./outputs",
        use_naturalness: bool = True,
    ):
        self.generator = generator
        self.detectors = detectors
        self.edit_agent = edit_agent
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = resolve_device(config.get("experiment", {}).get("device", "auto"))
        render_cfg = config.get("render", {})
        resolution = int(render_cfg.get("resolution", 256))
        opt = config.get("optimization", {})
        self.renderer = SceneRenderer(
            device=device,
            viewpoints=opt.get("viewpoints", [0, 90, 180, 270]),
            resolution=(resolution, resolution),
            fovy_deg=float(render_cfg.get("fovy_deg", 40.0)),
            cam_distance=float(render_cfg.get("cam_distance", 6.0)),
        )
        bank = config.get("paths", {}).get("feature_bank")
        if not (bank and Path(bank).is_file()):
            bank = None
        self.naturalness = NaturalnessRegularizer(device=device, bank_path=bank) if use_naturalness else None
        if use_naturalness:
            self.loss_fn.attach_naturalness(self.naturalness)
        else:
            self.loss_fn.use_naturalness = False
        self.evaluator = Evaluator(conf_threshold=loss_fn.theta)
        self._token_list: Optional[List[str]] = None
        self.queries = 0

        self.history: List[Dict] = []
        self.best_object = None
        self.best_loss = float("inf")
        self.best_prompts: Dict = {}
        self.best_scores: Dict = {}
        self.best_views: Optional[Dict] = None
        self.best_rows: Optional[List[Dict]] = None
        self.best_metrics: Dict = {}
        # best_loss above is the minimum loss ever seen (honest). The result
        # candidate we return prefers an attack success over a merely lower loss.
        self.result_loss = float("inf")
        self.result_success = False

    def _fusion_ready(self) -> bool:
        fusion = self.detectors.get("fusion")
        return bool(getattr(fusion, "available", False))

    def _tokens(self) -> List[str]:
        """Keyframes scored inside nuScenes. Explicit tokens win over a scene search."""
        if self._token_list is not None:
            return self._token_list
        ev = self.config.get("evaluation", {}) or {}
        tokens = [str(t) for t in (ev.get("sample_tokens") or []) if t]
        if not tokens and ev.get("sample_token"):
            tokens = [str(ev["sample_token"])]
        n_scenes = int(ev.get("scenes") or 0)
        if not tokens and n_scenes and self._fusion_ready():
            chosen = self.detectors["fusion"].client.select_samples(
                n_scenes,
                int(ev.get("keyframes", 1)),
                split=ev.get("split", "val"),
                placement=ev.get("placement") or {},
                camera=ev.get("camera", "CAM_FRONT"),
            )
            tokens = [t for scene in chosen for t in scene["tokens"]]
        self._token_list = tokens
        return tokens

    def _use_scene(self) -> bool:
        """Real insertion scoring. The keyword stand-in never takes this path."""
        if not self._fusion_ready():
            return False
        if self._tokens():
            return True
        weight = float(self.config.get("loss", {}).get("lambda_fusion", 0.0))
        if weight > 0:
            raise RuntimeError(
                "MambaFusion worker is configured but evaluation.sample_token is empty"
            )
        return False

    def _loss(self, obj: Dict, scores_2d: List[float], scores_fusion: List[float]) -> Dict:
        return self.loss_fn.total(
            scores_2d,
            scores_fusion,
            point_cloud=obj.get("point_cloud"),
            feature_bank=self.naturalness.feature_bank if self.naturalness else None,
            box_stats=obj.get("box_stats"),
        )

    def _evaluate_renders(self, obj: Dict) -> Dict:
        """Orbit renders. Used only when no nuScenes worker is configured."""
        views = self.renderer.render_all_views(obj)
        scores_2d = [
            float(self.detectors["2d"](obj, image=view["image"], viewpoint=yaw))
            for yaw, view in views.items()
        ]
        fusion = self.detectors["fusion"]
        scores_fusion = [
            float(fusion(obj, image=view["image"], viewpoint=yaw))
            for yaw, view in views.items()
        ]
        return {
            "scores_2d": scores_2d,
            "scores_fusion": scores_fusion,
            "loss_dict": self._loss(obj, scores_2d, scores_fusion),
            "views": views,
            "rows": None,
            "metrics": {},
        }

    def _failed_scene(self, reason: str) -> Dict:
        """An object that cannot be inserted is not an attack success."""
        yaws = self.config.get("optimization", {}).get("viewpoints", [0, 90, 180, 270])
        tokens = self._tokens() or ["none"]
        rows = [
            {"token": token, "yaw": float(yaw), "score_2d": 1.0, "fusion": 1.0, "error": reason}
            for token in tokens
            for yaw in yaws
        ]
        scores_2d = [1.0] * len(rows)
        scores_fusion = [1.0] * len(rows)
        return {
            "scores_2d": scores_2d,
            "scores_fusion": scores_fusion,
            "loss_dict": {},
            "views": None,
            "rows": rows,
            "metrics": self.evaluator.view_metrics(rows),
        }

    def _evaluate_in_scene(self, obj: Dict, tag: str) -> Dict:
        from eval.scene_eval import SceneEvaluator

        scene = SceneEvaluator(self.detectors["2d"], self.detectors["fusion"], self.config)
        out = self.output_dir / "candidates" / tag
        try:
            rows = scene.evaluate(obj, self._tokens(), out)
        except RuntimeError as exc:
            print(f"[scene] {tag} was not inserted: {exc}")
            failed = self._failed_scene(str(exc))
            failed["loss_dict"] = self._loss(obj, failed["scores_2d"], failed["scores_fusion"])
            return failed
        scores_2d = [float(r["score_2d"]) for r in rows]
        scores_fusion = [float(r["fusion"]) for r in rows]
        return {
            "scores_2d": scores_2d,
            "scores_fusion": scores_fusion,
            "loss_dict": self._loss(obj, scores_2d, scores_fusion),
            "views": None,
            "rows": rows,
            "metrics": self.evaluator.view_metrics(rows),
        }

    def _evaluate_object(self, obj: Dict, tag: Optional[str] = None) -> Dict:
        """One candidate. Scene insertion when the worker is configured, else orbit renders."""
        if self._use_scene():
            return self._evaluate_in_scene(obj, tag or f"iter_{len(self.history):03d}")
        return self._evaluate_renders(obj)

    def _attack_done(self, metrics: Optional[Dict]) -> bool:
        if not metrics:
            return False
        return float(metrics.get("asr_joint_all_yaws", 0.0)) >= 1.0 - 1e-9

    def _consider(self, obj, sem, sty, loss, ev, success: bool) -> None:
        """Track the honest minimum loss, and keep the result candidate.

        A joint miss on every yaw (`success`) is the goal and beats any
        lower-loss non-success candidate -- naturalness can make a success's
        scalar loss higher than a partial miss. Among the same success status,
        the lower loss wins. `best_loss` stays the minimum loss ever seen, so it
        is never inflated by accepting a successful but higher-loss candidate.
        """
        if loss is not None:
            self.best_loss = min(self.best_loss, loss)
        cand_loss = float("inf") if loss is None else loss
        better = (success and not self.result_success) or (
            success == self.result_success and cand_loss < self.result_loss
        )
        if self.best_object is None or better:
            self.result_loss = cand_loss
            self.result_success = success
            self.best_object = copy.deepcopy(obj)
            self.best_prompts = {"sem": sem, "sty": sty}
            self.best_scores = {"2d": ev["scores_2d"], "fusion": ev["scores_fusion"]}
            self.best_views = ev.get("views")
            self.best_rows = ev.get("rows")
            self.best_metrics = dict(ev.get("metrics") or {})

    def _record(self, iteration: int, sem: str, sty: str, ev: Dict, loss: float,
                accepted: bool, edit: Optional[EditAction] = None) -> Dict:
        self.queries += len(ev["scores_2d"])
        record = {
            "iter": iteration,
            "sem": sem,
            "sty": sty,
            "loss": loss,
            "accepted": accepted,
            **{k: ev["loss_dict"].get(k) for k in _LOSS_KEYS},
            "scores_2d": ev["scores_2d"],
            "scores_fusion": ev["scores_fusion"],
            "metrics": ev.get("metrics") or {},
            "queries": self.queries,
        }
        if ev.get("error"):
            record["error"] = ev["error"]
        if edit is not None:
            record["edit"] = {
                "target": edit.target,
                "action": edit.action,
                "phrase": edit.phrase,
                "reason": edit.reason,
            }
        self.history.append(record)
        return record

    def optimize(self, base_prompt: str, init_sem: str, init_sty: str) -> Dict:
        opt = self.config.get("optimization", {})
        max_iter = int(opt.get("max_iterations", 50))
        budget = int(opt.get("query_budget") or 0)

        current_sem, current_sty = init_sem, init_sty
        try:
            obj = self.generator.generate(base_prompt, current_sem, current_sty)
            ev = self._evaluate_object(obj, "init")
            current_loss = ev["loss_dict"]["total"]
            self._consider(obj, current_sem, current_sty, current_loss, ev,
                           self._attack_done(ev.get("metrics")))
            self._record(0, current_sem, current_sty, ev, current_loss, True)

            if not self._attack_done(ev.get("metrics")):
                pbar = tqdm(range(1, max_iter + 1), desc="Agent-OBJ")
                for t in pbar:
                    if budget and self.queries >= budget:
                        print(f"[Agent-OBJ] stop: query budget {budget}")
                        break
                    edit = self.edit_agent.propose(current_sem, current_sty, self.history)
                    new_sem, new_sty = current_sem, current_sty
                    if edit.target == "semantic":
                        new_sem = self.edit_agent.apply(current_sem, edit)
                    else:
                        new_sty = self.edit_agent.apply(current_sty, edit)
                    try:
                        new_obj = self.generator.generate(base_prompt, new_sem, new_sty)
                    except Exception as exc:
                        print(f"[Iter {t}] Generation error: {exc}")
                        fail_ev = {
                            "scores_2d": [], "scores_fusion": [], "loss_dict": {},
                            "views": None, "rows": None, "metrics": {}, "error": str(exc),
                        }
                        self._record(t, new_sem, new_sty, fail_ev, None, accepted=False, edit=edit)
                        continue
                    new_ev = self._evaluate_object(new_obj, f"iter_{t:03d}")
                    new_loss = new_ev["loss_dict"]["total"]
                    succeeded = self._attack_done(new_ev.get("metrics"))
                    # A joint miss on every yaw is the goal, even when naturalness
                    # makes its scalar loss a little higher than a partial miss.
                    accepted = new_loss < current_loss or succeeded
                    if accepted:
                        current_sem, current_sty = new_sem, new_sty
                        obj = new_obj
                        current_loss = new_loss
                    self._consider(new_obj, new_sem, new_sty, new_loss, new_ev, succeeded)
                    self._record(t, new_sem, new_sty, new_ev, new_loss, accepted, edit)
                    pbar.set_postfix({
                        "best": f"{self.best_loss:.3f}",
                        "cur": f"{current_loss:.3f}",
                        "acc": accepted,
                        "asr": f"{(new_ev.get('metrics') or {}).get('asr_joint_all_yaws', 0):.2f}",
                    })
                    if succeeded:
                        print("[Agent-OBJ] stop: asr_joint_all_yaws = 1")
                        break
        finally:
            self.close()

        require_both = bool(self.config.get("evaluation", {}).get("asr_require_both", True))
        summary = self.evaluator.summarize_history(self.history)
        summary["queries"] = self.queries
        summary["best_prompts"] = self.best_prompts
        summary["asr_require_both"] = require_both
        summary["attack_success"] = bool(self.result_success)
        summary["result_loss"] = None if self.result_loss == float("inf") else self.result_loss
        if self.best_metrics:
            summary.update(self.best_metrics)
            summary["asr"] = float(self.best_metrics.get("asr_joint_all_yaws", 0.0))
        else:
            asr_info = self.evaluator.compute_asr([{
                "scores_2d": self.best_scores.get("2d", []),
                "scores_fusion": self.best_scores.get("fusion", []),
            }], require_both=require_both)
            summary["asr"] = asr_info["asr"]
        self._save(summary)
        self._save_best()
        return {
            "best_loss": self.best_loss,
            "best_prompts": self.best_prompts,
            "best_scores": self.best_scores,
            "history": self.history,
            "summary": summary,
            "best_object": self.best_object,
            "metrics": self.best_metrics,
        }

    def _save_best(self) -> None:
        if self.best_rows:
            from eval.scene_eval import crop_strip
            crop_strip(self.best_rows, self.output_dir / "best_views.jpg")
        if self.best_views:
            self._save_views(self.best_views, "best_views")

    def _save_views(self, views: Optional[Dict], folder: str) -> None:
        if not views:
            return
        from PIL import Image

        dest = self.output_dir / folder
        dest.mkdir(parents=True, exist_ok=True)
        for yaw, view in views.items():
            Image.fromarray(view["image"]).save(dest / f"yaw_{int(yaw):03d}.png")
        print(f"Saved views to {dest}")

    def _save(self, summary: Dict):
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {self.output_dir}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Best prompts: {self.best_prompts}")
        print(f"Summary: {summary}")

    def close(self) -> None:
        fusion = self.detectors.get("fusion")
        close = getattr(fusion, "close", None)
        if close:
            close()
