"""
EditAgent for Agent-OBJ.

The paper's agent is an LLM (Qwen2.5). It reads the optimization history --
including each detector's confidence -- and proposes one discrete edit
(add / replace / remove a phrase) to the semantic or style prompt.

When no LLM is loaded, edits are sampled from the phrase pools instead, so the
loop runs without a GPU/LLM. A real run (`--use_real_models`) requires
`--use_llm` unless `--allow-sampled-edits` is passed.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EditAction:
    target: str          # "semantic" or "style"
    action: str          # "add", "replace", "remove"
    phrase: str
    reason: str = ""


def _max(scores) -> Optional[float]:
    return max(scores) if scores else None


class ImprovedEditAgent:
    def __init__(
        self,
        semantic_pool: Optional[List[str]] = None,
        style_pool: Optional[List[str]] = None,
        use_llm: bool = False,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        temperature: float = 0.7,
    ):
        self.semantic_pool = semantic_pool or []
        self.style_pool = style_pool or []
        self.use_llm = use_llm
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

        if use_llm:
            self._load_llm(model_name, device)

    def _load_llm(self, model_name: str, device: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"[EditAgent] Loaded LLM: {model_name}")
        except Exception as e:
            print(f"[EditAgent] LLM load failed, sampling edits from the pools: {e}")
            self.use_llm = False

    def propose(
        self,
        current_sem: str,
        current_sty: str,
        history: List[Dict],
        prefer_semantic: bool = False,
    ) -> EditAction:
        del prefer_semantic
        if self.use_llm and self.model is not None and len(history) >= 1:
            try:
                edit = self._llm_propose(current_sem, current_sty, history)
                if edit is not None:
                    return edit
                print("[EditAgent] LLM reply could not be parsed; sampling an edit from the pools.")
            except Exception as e:
                print(f"[EditAgent] LLM propose failed: {e}")
        return self._random_propose(current_sem, current_sty)

    def _random_propose(self, sem: str, sty: str) -> EditAction:
        """Sample an edit from the phrase pools: add or remove a random phrase.
        Used when no LLM is loaded."""
        target = random.choice(["semantic", "style"])
        pool = self.semantic_pool if target == "semantic" else self.style_pool
        text = sem if target == "semantic" else sty
        if not pool:
            return EditAction(target, "add", "wearing a hood", reason="random-stub")
        phrase = random.choice(pool)
        action = "remove" if phrase.lower() in text.lower() else "add"
        return EditAction(target, action, phrase, reason="random-stub")

    def _llm_propose(self, sem: str, sty: str, history: List[Dict]) -> Optional[EditAction]:
        prompt = self._build_llm_prompt(sem, sty, history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse(text)

    def _build_llm_prompt(self, sem: str, sty: str, history: List[Dict]) -> str:
        recent = history[-4:]
        lines = []
        for h in recent:
            edit = h.get("edit", {}) or {}
            max2d = _max(h.get("scores_2d"))
            maxf = _max(h.get("scores_fusion"))
            f2d = f"{max2d:.3f}" if max2d is not None else "n/a"
            ff = f"{maxf:.3f}" if maxf is not None else "n/a"
            lines.append(
                f"Iter {h.get('iter')}: {edit.get('target')}/{edit.get('action')} "
                f"'{edit.get('phrase')}' -> 2D_conf={f2d}, fusion_conf={ff}, "
                f"loss={h.get('loss', 0) or 0:.3f}, accepted={h.get('accepted')}"
            )
        hist_text = "\n".join(lines) if lines else "None"

        return f"""You are an expert adversarial prompt engineer for 3D pedestrian objects.
Goal: push BOTH detector confidences below 0.5 while keeping the object natural.
2D_conf is the monocular detector's confidence and fusion_conf the Camera-LiDAR
detector's; lower is better for both.

Current semantic prompt: {sem}
Current style prompt: {sty}

Recent optimization history (per-detector confidence and total loss):
{hist_text}

Attack whichever detector is still above 0.5. Propose exactly ONE edit.
Reply ONLY in this format:
TARGET|ACTION|PHRASE

TARGET must be "semantic" or "style".
ACTION must be "add", "replace", or "remove".
PHRASE must be a short attribute (e.g. "carrying a large backpack").

Example: semantic|add|wearing a wide-brimmed hat
"""

    def _parse(self, text: str) -> Optional[EditAction]:
        """Parse the LLM reply into an edit, or None if it does not match."""
        for line in text.strip().split("\n"):
            line = line.strip()
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    target, action, phrase = parts[0].lower(), parts[1].lower(), parts[2]
                    if target in ("semantic", "style") and action in ("add", "replace", "remove"):
                        return EditAction(target, action, phrase, reason="llm")
        return None

    def apply(self, current: str, edit: EditAction) -> str:
        phrase = edit.phrase.strip()
        if edit.action == "add":
            if not current:
                return phrase
            if phrase.lower() in current.lower():
                return current
            return f"{current}, {phrase}"
        elif edit.action == "replace":
            return phrase
        elif edit.action == "remove":
            result = current.replace(phrase, "")
            result = result.replace(", ,", ",").replace("  ", " ").strip(" ,")
            return result if result else current
        return current
