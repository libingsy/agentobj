"""Load one nuScenes sample and insert an object into it.

`NuScenesScene` gathers the keyframe, the stacked sweeps, the calibration and
the camera image, finds a free spot on the road in front of the ego vehicle,
and composites the object through `data.object_insert`. `load` needs the
nuScenes devkit; everything else is plain numpy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.geometry import project_lidar_to_image, subsample_pedestrian

from data.lidar_sensor import UP, BeamModel, LidarSweep
from data.object_insert import (
    insert_camera,
    insert_into_sweep,
    insert_lidar,
    paste_render,
    place_body,
    place_gaussians,
    squeeze_standing,
)


def road_height(lidar_points: np.ndarray, xy: Sequence[float], fallback: float, radius: float = 2.5) -> float:
    """Height of the road under `xy`, from low LiDAR returns around that spot."""
    pts = np.asarray(lidar_points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 50:
        return float(fallback)
    dist = np.linalg.norm(pts[:, :2] - np.asarray(xy[:2], dtype=np.float64), axis=1)
    near = pts[(dist > 0.3) & (dist < radius) & (pts[:, 2] < fallback + 1.0)]
    if len(near) < 20:
        return float(fallback)
    return float(np.percentile(near[:, 2], 10))


def camera_forward(lidar_to_cam) -> np.ndarray:
    """Horizontal unit vector, in the LiDAR frame, along the camera's optical axis."""
    fwd = np.asarray(lidar_to_cam, dtype=np.float64)[2, :3].copy()
    fwd[2] = 0.0
    n = float(np.linalg.norm(fwd))
    return fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def find_placement(
    lidar_points: np.ndarray,
    lidar_to_cam,
    intrinsics,
    image_size: Tuple[int, int],
    sensor_height: float,
    distances: Sequence[float] = (8.0, 10.0, 12.0, 6.0),
    lateral: Sequence[float] = (0.0, -1.0, 1.0),
    radius: float = 0.7,
    height: float = 1.7,
    boxes_xy: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """First free, visible spot in front of the camera.

    A spot is rejected if LiDAR returns occupy a cylinder of `radius` meters
    above the ground there, if returns between the sensor and the spot block
    the line of sight, if an annotated object stands within 1.5 m, or if the
    object would leave the image.
    """
    pts = np.asarray(lidar_points, dtype=np.float64)
    fwd = camera_forward(lidar_to_cam)
    left = np.cross(UP, fwd)
    w, h = int(image_size[0]), int(image_size[1])
    for d in distances:
        for off in lateral:
            xy = fwd[:2] * float(d) + left[:2] * float(off)
            ground = road_height(pts, xy, -float(sensor_height))
            rel = pts[:, :2] - xy
            horiz = np.linalg.norm(rel, axis=1)
            body = (horiz < radius) & (pts[:, 2] > ground + 0.2) & (pts[:, 2] < ground + height + 0.3)
            if int(body.sum()) > 2:
                continue
            # Line of sight: returns inside the object's silhouette cone, closer than the object.
            rng_xy = np.linalg.norm(pts[:, :2], axis=1)
            spot = float(np.linalg.norm(xy))
            az = np.arctan2(pts[:, 1], pts[:, 0])
            half = np.arctan2(0.45, spot)
            daz = np.abs(np.angle(np.exp(1j * (az - np.arctan2(xy[1], xy[0])))))
            el = np.arctan2(pts[:, 2], np.maximum(rng_xy, 1e-6))
            el_lo, el_hi = np.arctan2(ground + 0.3, spot), np.arctan2(ground + height, spot)
            block = (daz < half) & (rng_xy > 1.5) & (rng_xy < spot - radius) & (el > el_lo) & (el < el_hi)
            if int(block.sum()) > 2:
                continue
            if boxes_xy is not None and len(boxes_xy):
                if float(np.min(np.linalg.norm(np.asarray(boxes_xy)[:, :2] - xy, axis=1))) < 1.5:
                    continue
            corners = np.array([
                [xy[0] + 0.4 * left[0], xy[1] + 0.4 * left[1], ground],
                [xy[0] - 0.4 * left[0], xy[1] - 0.4 * left[1], ground],
                [xy[0], xy[1], ground + height],
            ])
            cam = (np.asarray(lidar_to_cam, dtype=np.float64) @ np.c_[corners, np.ones(3)].T).T
            if np.any(cam[:, 2] < 1.0):
                continue
            uv = (np.asarray(intrinsics, dtype=np.float64) @ cam[:, :3].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            if uv[:, 0].min() < 10 or uv[:, 0].max() > w - 10 or uv[:, 1].min() < 10 or uv[:, 1].max() > h - 10:
                continue
            return {
                "translation": [float(xy[0]), float(xy[1]), float(ground)],
                "distance": float(d),
                "lateral": float(off),
            }
    return None


def pedestrian_intensity(points: np.ndarray, boxes: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                         fallback: float) -> float:
    """Median intensity of returns inside annotated pedestrian boxes (center, size wlh, rotation)."""
    pts = np.asarray(points, dtype=np.float64)
    vals = []
    for center, wlh, rot in boxes:
        local = (pts[:, :3] - center) @ rot
        inside = (np.abs(local[:, 0]) <= wlh[1] / 2) & (np.abs(local[:, 1]) <= wlh[0] / 2) & (np.abs(local[:, 2]) <= wlh[2] / 2)
        vals.append(pts[inside, 3])
    vals = np.concatenate(vals) if vals else np.zeros(0)
    return float(np.median(vals)) if len(vals) >= 20 else float(fallback)


# --------------------------------------------------------------------------- nuScenes

class NuScenesScene:
    """Keyframe, sweeps, calibration and camera image of one nuScenes sample.

    `load` needs the nuScenes devkit. The sweep list follows OpenPCDet's
    `fill_trainval_infos`: up to `max_sweeps - 1` previous sweeps, the last one
    repeated when the scene starts, ego points (|x|, |y| < 1 m) removed from
    sweeps, time lag in seconds.
    """

    def __init__(self, token: str, image: np.ndarray, image_path: str, camera: str,
                 lidar_to_cam: np.ndarray, intrinsics: np.ndarray, sweeps: List[LidarSweep],
                 beams: BeamModel, sensor_height: float, pedestrian_boxes, other_boxes_xy: np.ndarray,
                 info: Optional[Dict] = None):
        self.token = token
        self.image = image
        self.image_path = image_path
        self.camera = camera
        self.lidar_to_cam = lidar_to_cam
        self.intrinsics = intrinsics
        self.sweeps = sweeps
        self.beams = beams
        self.sensor_height = sensor_height
        self.pedestrian_boxes = pedestrian_boxes
        self.boxes_xy = other_boxes_xy
        self.info = info or {}

    @classmethod
    def load(cls, nusc, token: str, camera: str = "CAM_FRONT", max_sweeps: int = 10) -> "NuScenesScene":
        from nuscenes.utils.geometry_utils import transform_matrix
        from PIL import Image
        from pyquaternion import Quaternion

        sample = nusc.get("sample", token)
        ref_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cam_sd = nusc.get("sample_data", sample["data"][camera])
        ref_cs = nusc.get("calibrated_sensor", ref_sd["calibrated_sensor_token"])
        ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
        ref_from_car = transform_matrix(ref_cs["translation"], Quaternion(ref_cs["rotation"]), inverse=True)
        car_from_global = transform_matrix(ref_pose["translation"], Quaternion(ref_pose["rotation"]), inverse=True)
        ref_time = 1e-6 * ref_sd["timestamp"]

        def read(path):
            raw = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
            return raw[:, :4].astype(np.float32), raw[:, 4].astype(np.int64)

        ref_path = nusc.get_sample_data_path(ref_sd["token"])
        key_pts, key_ring = read(ref_path)
        sweeps = [LidarSweep(points=key_pts, ring=key_ring, time_lag=0.0)]
        curr = ref_sd
        previous = []
        while len(previous) < max_sweeps - 1:
            if curr["prev"] == "":
                if not previous:
                    previous.append((ref_path, np.eye(4), 0.0, True))
                else:
                    previous.append(previous[-1])
                continue
            curr = nusc.get("sample_data", curr["prev"])
            pose = nusc.get("ego_pose", curr["ego_pose_token"])
            cs = nusc.get("calibrated_sensor", curr["calibrated_sensor_token"])
            global_from_car = transform_matrix(pose["translation"], Quaternion(pose["rotation"]), inverse=False)
            car_from_current = transform_matrix(cs["translation"], Quaternion(cs["rotation"]), inverse=False)
            tm = ref_from_car @ car_from_global @ global_from_car @ car_from_current
            previous.append((nusc.get_sample_data_path(curr["token"]), tm, ref_time - 1e-6 * curr["timestamp"], False))
        for path, tm, lag, first in previous:
            pts, ring = read(path)
            ego = (np.abs(pts[:, 0]) < 1.0) & (np.abs(pts[:, 1]) < 1.0)
            pts, ring = pts[~ego], ring[~ego]
            if not first:
                pts = pts.copy()
                pts[:, :3] = pts[:, :3] @ tm[:3, :3].T.astype(np.float32) + tm[:3, 3].astype(np.float32)
            sweeps.append(LidarSweep(points=pts, ring=ring, origin=tm[:3, 3].astype(np.float64),
                                     rotation=tm[:3, :3].astype(np.float64), time_lag=float(lag)))

        lidar_to_cam, intrinsics = lidar_to_camera(nusc, ref_sd, cam_sd)
        image_path = nusc.get_sample_data_path(cam_sd["token"])
        image = np.array(Image.open(image_path).convert("RGB"))

        _, boxes, _ = nusc.get_sample_data(ref_sd["token"])
        peds, others = [], []
        for box in boxes:
            others.append(box.center[:2])
            if box.name.startswith("human.pedestrian"):
                peds.append((box.center, box.wlh, box.rotation_matrix))
        beams = BeamModel.measure(key_pts[:, :3], key_ring)
        return cls(
            token=token,
            image=image,
            image_path=image_path,
            camera=camera,
            lidar_to_cam=lidar_to_cam,
            intrinsics=intrinsics,
            sweeps=sweeps,
            beams=beams,
            sensor_height=float(ref_cs["translation"][2]),
            pedestrian_boxes=peds,
            other_boxes_xy=np.asarray(others, dtype=np.float64).reshape(-1, 2),
        )

    def placement(self, **kwargs) -> Optional[Dict]:
        h, w = self.image.shape[:2]
        return find_placement(
            self.sweeps[0].points, self.lidar_to_cam, self.intrinsics, (w, h), self.sensor_height,
            boxes_xy=self.boxes_xy, **kwargs,
        )

    def intensity(self, fallback: float) -> float:
        return pedestrian_intensity(self.sweeps[0].points, self.pedestrian_boxes, fallback)

    def insert(self, gaussians, translation: Sequence[float], yaw_deg: float, intensity: float,
               velocity: Optional[Sequence[float]] = None, device: str = "cpu") -> Dict:
        """Camera image, stacked LiDAR cloud and object box for one yaw.

        `velocity` (m/s, keyframe LiDAR frame) smears the LiDAR accumulation
        along the walk; the camera image stays at the keyframe pose.
        """
        # Squeeze to standing-pedestrian proportions (<=0.6 m wide) so the object
        # is a person and not the ~1.7 m-wide LGM slab in both modalities.
        gs = place_gaussians(squeeze_standing(gaussians), translation, yaw_deg, device=device)
        image, box = insert_camera(self.image, gs, self.lidar_to_cam, self.intrinsics)
        points, stats = insert_lidar(gs, self.sweeps, self.beams, intensity, velocity=velocity)
        return {"image": image, "bbox": box, "points": points, "stats": stats}

    def insert_shell(self, points_yup, render, translation: Sequence[float], yaw_deg: float,
                     gaussians, intensity: float = 35.0, n_keep: int = 40, device: str = "cpu") -> Dict:
        """Legacy variant: paste one render, ray-cast the standing body, and keep
        `n_keep` returns on the keyframe spread over the height (a sparse,
        single-sweep insertion). The default path is the geometric motion
        insertion in `insert` / `insert_lidar`; this one is kept for comparison.
        """
        body = place_body(points_yup, translation, yaw_deg)
        image, box = paste_render(self.image, body, render, self.lidar_to_cam, self.intrinsics)
        gs = place_gaussians(squeeze_standing(gaussians), translation, yaw_deg, device=device)
        stacked = []
        stats = {"object_points": [], "removed": []}
        for index, sweep in enumerate(self.sweeps):
            pts = sweep.points
            added = 0
            if index == 0:
                obj, _keep = insert_into_sweep(gs, sweep, self.beams, intensity)
                shell = subsample_pedestrian(obj[:, :3], n_keep) if len(obj) else obj.reshape(0, 3)
                if len(shell):
                    extra = np.zeros((len(shell), pts.shape[1]), dtype=np.float32)
                    extra[:, :3] = shell[:, :3]
                    extra[:, 3] = float(intensity)
                    pts = np.concatenate([pts, extra], axis=0)
                    added = int(len(shell))
            lag = np.full((len(pts), 1), float(sweep.time_lag), dtype=np.float32)
            stacked.append(np.concatenate([pts.astype(np.float32), lag], axis=1))
            stats["object_points"].append(added)
            stats["removed"].append(0)
        cloud = np.concatenate(stacked, axis=0) if stacked else np.zeros((0, 5), dtype=np.float32)
        return {"image": image, "bbox": box, "points": cloud, "stats": stats}


def lidar_to_camera(nusc, lidar_sd, cam_sd) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 matrix taking keyframe LiDAR points into the camera frame (through the global frame), and K."""
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    lidar_ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    cam_ego = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    lidar_to_ego = transform_matrix(lidar_cs["translation"], Quaternion(lidar_cs["rotation"]))
    ego_to_global = transform_matrix(lidar_ego["translation"], Quaternion(lidar_ego["rotation"]))
    global_to_ego = transform_matrix(cam_ego["translation"], Quaternion(cam_ego["rotation"]), inverse=True)
    ego_to_cam = transform_matrix(cam_cs["translation"], Quaternion(cam_cs["rotation"]), inverse=True)
    lidar_to_cam = ego_to_cam @ global_to_ego @ ego_to_global @ lidar_to_ego
    intrinsics = np.asarray(cam_cs["camera_intrinsic"], dtype=np.float32)
    return lidar_to_cam.astype(np.float32), intrinsics


def save_image(image: np.ndarray, path) -> str:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(str(path))
    return str(path)
