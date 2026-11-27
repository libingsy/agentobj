"""Insert a generated 3D Gaussian pedestrian into a nuScenes keyframe.

This module is now a thin facade. The implementation is split by concern:

- `data.lidar_sensor`   LiDAR sweep/beam model and ray-casting
- `data.object_insert`  object placement, LiDAR insertion, camera insertion
- `data.nuscenes_scene` loading a nuScenes sample and placing the object in it

The names below are re-exported so existing imports keep working (including the
Python 3.8 MambaFusion worker).
"""

from data.lidar_sensor import UP, BeamModel, LidarSweep, _ray_frame, cast_rays
from data.object_insert import (
    insert_camera,
    insert_into_sweep,
    insert_lidar,
    object_rotation,
    paste_render,
    pedestrian_shell,
    place_body,
    place_gaussians,
    squeeze_standing,
    stack_sweeps,
    trim_background,
    walk_velocity,
)
from data.nuscenes_scene import (
    NuScenesScene,
    camera_forward,
    find_placement,
    lidar_to_camera,
    pedestrian_intensity,
    road_height,
    save_image,
)

__all__ = [
    "UP", "BeamModel", "LidarSweep", "_ray_frame", "cast_rays",
    "insert_camera", "insert_into_sweep", "insert_lidar", "object_rotation",
    "paste_render", "pedestrian_shell", "place_body", "place_gaussians",
    "squeeze_standing", "stack_sweeps", "trim_background", "walk_velocity",
    "NuScenesScene", "camera_forward", "find_placement", "lidar_to_camera",
    "pedestrian_intensity", "road_height", "save_image",
]
