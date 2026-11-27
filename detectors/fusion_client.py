"""Talk to the MambaFusion worker without importing its torch build."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence


class FusionWorkerClient:
    """One long-lived scorer process. The model stays loaded between calls."""

    def __init__(self, python: str, worker: str, cwd: Optional[str] = None, env: Optional[dict] = None):
        child_env = dict(os.environ)
        child_env.update({k: str(v) for k, v in (env or {}).items() if v})
        self.proc = subprocess.Popen(
            [os.path.expanduser(python), os.path.expanduser(worker)],
            cwd=cwd or str(Path(worker).expanduser().resolve().parents[1]),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        ready = self._request({"op": "ping"})
        if not ready.get("ok"):
            raise RuntimeError(f"MambaFusion worker failed to start: {ready}")

    def _drain_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            print(f"[mambafusion] {line.rstrip()}", file=sys.stderr)

    def score(self, token: str) -> float:
        """Highest pedestrian score in the unmodified frame."""
        return float(self._ok({"op": "score", "token": token})["pedestrian_max"])

    def insert_score(
        self,
        token: str,
        gaussians_path: str,
        yaws: Sequence[float],
        out_dir: str,
        placement: Optional[dict] = None,
        camera: str = "CAM_FRONT",
        save_points: bool = False,
        points_path: Optional[str] = None,
        renders: Optional[dict] = None,
    ) -> dict:
        """Insert the object once per yaw and score each frame.

        When `points_path` and `renders` are set, the worker pastes the render
        for that yaw and appends a 40-point pedestrian shell. Otherwise it
        ray-casts the Gaussian volume.
        """
        payload = {
            "op": "insert_score",
            "token": token,
            "gaussians": str(Path(gaussians_path).resolve()) if gaussians_path else "",
            "yaws": [float(y) for y in yaws],
            "out_dir": str(Path(out_dir).resolve()),
            "placement": placement or {},
            "camera": camera,
            "save_points": bool(save_points),
        }
        if points_path:
            payload["points"] = str(Path(points_path).resolve())
        if renders:
            payload["renders"] = {str(k): str(Path(v).resolve()) for k, v in renders.items()}
        return self._ok(payload)

    def select_samples(self, scenes: int, keyframes: int, split: str = "val",
                       placement: Optional[dict] = None, camera: str = "CAM_FRONT") -> List[dict]:
        reply = self._ok({
            "op": "select_samples",
            "scenes": int(scenes),
            "keyframes": int(keyframes),
            "split": split,
            "placement": placement or {},
            "camera": camera,
        })
        return reply["scenes"]

    def _ok(self, payload: dict) -> dict:
        reply = self._request(payload)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", reply))
        return reply

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self._request({"op": "quit"})
        except Exception:
            self.proc.kill()
        self.proc.wait(timeout=30)

    def _request(self, payload: dict) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"MambaFusion worker exited {self.proc.returncode}")
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        # A stray print from the model must not be parsed as the reply.
        for _ in range(100):
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"MambaFusion worker closed its output, code {self.proc.poll()}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError("MambaFusion worker produced no JSON reply")
