"""2D and camera-LiDAR detectors."""

from detectors.detectors import DummyDetector, FusionDetector, YOLODetector, build_detectors

__all__ = ["DummyDetector", "FusionDetector", "YOLODetector", "build_detectors"]
