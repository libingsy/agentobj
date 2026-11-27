"""Score one generated object inside nuScenes keyframes with both detectors.

For every keyframe and every yaw, the MambaFusion worker inserts the object
into the camera image and the stacked LiDAR sweeps and returns the fusion
score of the object. YOLOv10x then reads the same composited camera image, and
its score is the confidence of the person box that matches the object's
rendered silhouette box (IoU >= `match_iou_2d`).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


def save_gaussians(obj: Dict, path: Path) -> str:
    g = obj["gaussians"]
    arr = g.detach().float().cpu().numpy() if hasattr(g, "detach") else np.asarray(g, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))
    return str(path)


def write_points(obj: Dict, path: Path) -> str:
    """Y-up body points. The Gaussian means are that cloud when no point cloud was stored."""
    pts = obj.get("point_cloud")
    if pts is None:
        g = obj["gaussians"]
        pts = g.detach().float().cpu().numpy()[:, :3] if hasattr(g, "detach") else np.asarray(g)[:, :3]
    elif hasattr(pts, "detach"):
        pts = pts.detach().float().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(pts, dtype=np.float32)[:, :3])
    return str(path)


def _as_uint8(image) -> np.ndarray:
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if float(np.max(arr)) > 1 else np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return arr[..., :3]


def save_renders(obj: Dict, directory: Path) -> Dict[str, str]:
    """Yaw (degrees) to a PNG of that side of the person. Existing paths are reused."""
    ready = obj.get("renders") or {}
    if ready:
        return {str(int(float(k))): str(v) for k, v in ready.items()}
    images = obj.get("mv_images")
    azimuths = obj.get("mv_azimuths") or obj.get("lgm_azimuths")
    if images is None or azimuths is None:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    for azimuth, image in zip(azimuths, images):
        path = directory / f"view_{int(float(azimuth)):03d}.png"
        Image.fromarray(_as_uint8(image)).save(path)
        saved[str(int(float(azimuth)))] = str(path)
    return saved


class SceneEvaluator:
    def __init__(self, yolo, fusion, config: Dict):
        ev = config.get("evaluation", {})
        self.yolo = yolo
        self.fusion = fusion
        self.camera = ev.get("camera", "CAM_FRONT")
        self.placement = dict(ev.get("placement", {}) or {})
        self.min_iou = float(ev.get("match_iou_2d", 0.3))
        self.yaws = [float(y) for y in config.get("optimization", {}).get("viewpoints", [0, 90, 180, 270])]

    def evaluate(self, obj: Dict, tokens: Sequence[str], out_dir, yaws: Optional[Sequence[float]] = None,
                 save_points: bool = False) -> List[Dict]:
        out_dir = Path(out_dir)
        gpath = save_gaussians(obj, out_dir / "object_gaussians.npy")
        ppath = write_points(obj, out_dir / "object_points.npy")
        renders = save_renders(obj, out_dir / "renders")
        rows = []
        for token in tokens:
            reply = self.fusion.client.insert_score(
                token, gpath, yaws or self.yaws, out_dir / token[:12],
                placement=self.placement, camera=self.camera, save_points=save_points,
                points_path=ppath, renders=renders,
            )
            for view in reply["views"]:
                image = np.asarray(Image.open(view["image"]).convert("RGB"))
                score_2d, box_2d, dets = self.yolo.score_box(image, view["bbox"], self.min_iou)
                rows.append({
                    "token": token,
                    "yaw": view["yaw"],
                    "score_2d": float(score_2d),
                    "fusion": float(view["fusion"]),
                    "fusion_any": view.get("fusion_any"),
                    "fusion_any_class": view.get("fusion_any_class"),
                    "object_box": view["bbox"],
                    "yolo_box": box_2d,
                    "n_person_boxes": len(dets),
                    "object_points_keyframe": view["object_points"][0] if view["object_points"] else 0,
                    "object_points_total": int(sum(view["object_points"])),
                    "image": view["image"],
                    "placement": reply["placement"],
                    "clean_near": reply["clean_near"],
                    "intensity": reply["intensity"],
                })
        with open(out_dir / "views.json", "w") as f:
            json.dump(rows, f, indent=2)
        return rows


def crop_strip(rows: List[Dict], dest, pad: float = 0.6, height: int = 320) -> Optional[str]:
    """One image: a crop around the inserted object for each view, with both scores drawn on it."""
    from PIL import ImageDraw

    tiles = []
    for r in rows:
        if not r.get("object_box"):
            continue
        im = Image.open(r["image"]).convert("RGB")
        x0, y0, x1, y1 = r["object_box"]
        w, h = x1 - x0, y1 - y0
        side = max(w, h) * (1 + pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
        tile = im.crop(box).resize((int(height * 0.75), height))
        d = ImageDraw.Draw(tile)
        d.rectangle([0, 0, tile.width, 34], fill=(0, 0, 0))
        d.text((4, 2), f"yaw {int(r['yaw'])}  {r['token'][:6]}", fill=(255, 255, 255))
        d.text((4, 18), f"YOLO {r['score_2d']:.2f}  MF {r['fusion']:.2f}", fill=(255, 255, 255))
        tiles.append(tile)
    if not tiles:
        return None
    canvas = Image.new("RGB", (sum(t.width for t in tiles), height), (0, 0, 0))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return str(dest)
