#!/usr/bin/env python
"""Full main entry for Agent-OBJ.

Dummy mode runs the prompt search with a pedestrian-shaped stand-in.
`--use_real_models` loads MV-Adapter, LGM, YOLOv10x, and MambaFusion.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from agent.improved_edit_agent import ImprovedEditAgent
from agent.full_optimizer import FullAgentOBJOptimizer
from detectors.detectors import build_detectors
from generators.generator import build_generator
from regularizers.losses import AgentOBJLoss
from utils.seed import seed_everything


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Agent-OBJ Attack Optimization")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--use_real_models", action="store_true")
    parser.add_argument("--use_llm", action="store_true", help="Use Qwen2.5 for editing")
    parser.add_argument("--allow-sampled-edits", dest="allow_sampled_edits", action="store_true",
                        help="Run --use_real_models without --use_llm, sampling edits from the pools")
    parser.add_argument("--max_iters", type=int, default=None)
    args = parser.parse_args()

    if args.use_real_models and not args.use_llm and not args.allow_sampled_edits:
        raise SystemExit(
            "A real attack run uses the paper's LLM EditAgent: pass --use_llm. "
            "Without it, edits are sampled from the phrase pools; pass "
            "--allow-sampled-edits to run that way on purpose (e.g. an ablation)."
        )

    cfg = load_config(args.config)
    if args.max_iters:
        cfg.setdefault("optimization", {})["max_iterations"] = args.max_iters
    seed_everything(int(cfg.get("experiment", {}).get("seed", 42)))

    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = build_generator(cfg, use_real=args.use_real_models)
    detectors = build_detectors(cfg, use_real=args.use_real_models)

    pools = cfg.get("edit_pools", {})
    agent = ImprovedEditAgent(
        semantic_pool=pools.get("semantic", []),
        style_pool=pools.get("style", []),
        use_llm=args.use_llm,
        device=cfg.get("experiment", {}).get("device", "auto"),
    )

    loss_cfg = cfg.get("loss", {})
    loss_fn = AgentOBJLoss(
        lambda_2d=loss_cfg.get("lambda_2d", 1.0),
        lambda_fusion=loss_cfg.get("lambda_fusion", 1.0),
        alpha_nat=loss_cfg.get("alpha_nat", 0.1),
        beta_mv=loss_cfg.get("beta_mv", 0.1),
        conf_threshold=loss_cfg.get("conf_threshold", 0.5),
        device=cfg.get("experiment", {}).get("device", "auto"),
    )

    optimizer = FullAgentOBJOptimizer(
        generator=gen,
        detectors=detectors,
        edit_agent=agent,
        loss_fn=loss_fn,
        config=cfg,
        output_dir=str(out_dir),
        use_naturalness=True,
    )

    prompts = cfg.get("prompts", {})
    result = optimizer.optimize(
        base_prompt=prompts.get("base", "a realistic pedestrian standing on the road"),
        init_sem=prompts.get("semantic_init", "carrying a backpack"),
        init_sty=prompts.get("style_init", "wearing a dark jacket"),
    )

    print("\n========== Done ==========")
    print(f"Best loss     : {result['best_loss']:.4f}")
    print(f"Best prompts  : {result['best_prompts']}")
    print(f"Best 2D scores: {result['best_scores']['2d']}")
    print(f"Best Fusion   : {result['best_scores']['fusion']}")
    print(f"Summary       : {result['summary']}")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
