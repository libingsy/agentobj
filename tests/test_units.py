"""Unit tests for the completed MV-Adapter pipeline pieces. No GPU weights required."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.nuscenes_insert import (
    BeamModel,
    LidarSweep,
    camera_forward,
    find_placement,
    insert_camera,
    insert_into_sweep,
    insert_lidar,
    object_rotation,
    pedestrian_intensity,
    place_gaussians,
    road_height,
    squeeze_standing,
    trim_background,
)
from render.gs_rasterizer import GaussianSet
from detectors.detectors import box_iou, match_score
from generators.generator import DummyGenerator, GeneratorNotReady, LGMGenerator
from generators.mvadapter import VARIANTS, prepare_reference
from generators.prompts import appearance_prompt, geometry_prompt
from regularizers.losses import AgentOBJLoss
from regularizers.naturalness import NaturalnessRegularizer
from render.gaussian_renderer import render_gaussians
from utils.geometry import (
    fit_standing_body,
    project_lidar_to_image,
    select_views_by_azimuth,
    subsample_pedestrian,
    visible_lidar_returns,
)
from utils.images import to_pil


class PromptAndViewTests(unittest.TestCase):
    def test_prompts_keep_style_off_the_geometry_stage(self):
        geo = geometry_prompt("a pedestrian", "carrying a backpack")
        styled = appearance_prompt("a pedestrian", "carrying a backpack", "dark clothing")
        self.assertEqual(geo, "a pedestrian, carrying a backpack")
        self.assertEqual(styled, "a pedestrian, carrying a backpack, dark clothing")
        self.assertNotIn("dark clothing", geo)

    def test_lgm_view_order(self):
        images = ["a", "b", "c", "d", "e", "f"]
        azimuths = [0, 45, 90, 180, 270, 315]
        picked = select_views_by_azimuth(images, azimuths)
        self.assertEqual(picked, ["a", "c", "d", "e"])

    def test_official_weight_names(self):
        self.assertEqual(VARIANTS["sdxl"]["t2mv_weight"], "mvadapter_t2mv_sdxl.safetensors")
        self.assertEqual(VARIANTS["sdxl"]["i2mv_weight"], "mvadapter_i2mv_sdxl.safetensors")
        self.assertEqual(VARIANTS["sd21"]["t2mv_weight"], "mvadapter_t2mv_sd21.safetensors")
        self.assertEqual(VARIANTS["sd21"]["i2mv_weight"], "mvadapter_i2mv_sd21.safetensors")
        self.assertEqual(VARIANTS["sdxl"]["height"], 768)
        self.assertEqual(VARIANTS["sd21"]["height"], 512)

    def test_rgba_reference_is_rgb(self):
        image = to_pil(np.zeros((32, 32, 4), dtype=np.uint8))
        image.putalpha(255)
        prepared = prepare_reference(image, 64, 64)
        self.assertEqual(prepared.mode, "RGB")
        self.assertEqual(prepared.size, (64, 64))


def _packed_column(n: int = 6) -> np.ndarray:
    """A few opaque red Gaussians of a standing body, y up, front +z."""
    rows = []
    for y in np.linspace(0.25, 1.55, n):
        row = np.zeros(14, dtype=np.float32)
        row[1] = y
        row[3] = 0.95
        row[4:7] = 0.08
        row[7] = 1.0
        row[11:14] = (0.85, 0.15, 0.15)
        rows.append(row)
    return np.stack(rows)


def _looking_along_y() -> np.ndarray:
    """Camera at the LiDAR origin, optical axis +Y, no pitch. y_cam = 1.84 - z_lidar."""
    return np.array(
        [[1, 0, 0, 0], [0, 0, -1, 1.84], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )


class GeometryTests(unittest.TestCase):
    def test_red_gaussian_lands_on_the_lookat_pixel(self):
        g = torch.zeros(1, 14)
        g[0, 1] = 0.85
        g[0, 3] = 1.0
        g[0, 4:7] = 0.5
        g[0, 7] = 1.0
        g[0, 11] = 1.0
        image = render_gaussians(g, yaw_deg=0.0, resolution=(128, 128), fovy_deg=40.0)
        window = image[48:80, 48:80]
        pixel = window.reshape(-1, 3)[int(window[:, :, 0].argmax())]
        self.assertGreater(int(pixel[0]), 200)
        self.assertLess(int(pixel[1]), 40)
        self.assertLess(int(pixel[2]), 40)

    def test_object_front_faces_the_sensor(self):
        rotation = object_rotation((0.0, 8.0, -1.7), 0.0)
        np.testing.assert_allclose(rotation[:, 2], [0.0, -1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(rotation[:, 1], [0.0, 0.0, 1.0], atol=1e-5)
        placed = place_gaussians(_packed_column(1), (0.0, 8.0, -1.7), 0.0)
        # The body's +z (front) is the third column, so a point at object z=1 lands closer to the origin.
        row = np.zeros(14, dtype=np.float32)
        row[2] = 1.0
        row[3] = 1.0
        row[4:7] = 0.02
        row[7] = 1.0
        moved = place_gaussians(row.reshape(1, 14), (0.0, 8.0, -1.7), 0.0)
        np.testing.assert_allclose(moved.means[0].numpy(), [0.0, 7.0, -1.7], atol=1e-4)
        self.assertAlmostEqual(float(placed.means[0, 1]), 8.0, places=3)

    def test_forward_lidar_point_projects_to_the_image_center(self):
        # LiDAR +x -> camera +z, LiDAR +y -> camera -x, LiDAR +z -> camera -y.
        lidar_to_cam = np.array(
            [
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsics = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float32)
        uv, depth = project_lidar_to_image(np.array([[10.0, 0.0, 0.0]]), lidar_to_cam, intrinsics)
        self.assertAlmostEqual(float(depth[0]), 10.0, places=4)
        np.testing.assert_allclose(uv[0], [50.0, 50.0], atol=1e-4)

    def test_camera_forward_follows_the_optical_axis(self):
        looking_along_y = _looking_along_y()
        looking_along_x = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(camera_forward(looking_along_y), [0, 1, 0], atol=1e-5)
        np.testing.assert_allclose(camera_forward(looking_along_x), [1, 0, 0], atol=1e-5)

    def test_road_height_uses_nearby_returns(self):
        road = np.zeros((200, 4), dtype=np.float32)
        road[:, 0] = 0.2
        road[:, 1] = np.linspace(6.0, 10.0, 200)
        road[:, 2] = -1.55
        self.assertAlmostEqual(road_height(road, (0.0, 8.0), -1.84), -1.55, places=2)

    def test_placement_is_in_front_of_the_camera_and_on_the_road(self):
        rng = np.random.default_rng(0)
        n = 800
        xy = rng.uniform([-4, 3], [4, 16], size=(n, 2))
        road = np.column_stack([xy, np.full(n, -1.70), np.full(n, 12.0)]).astype(np.float32)
        intrinsics = np.array([[220, 0, 200], [0, 220, 40], [0, 0, 1]], dtype=np.float32)
        place = find_placement(
            road, _looking_along_y(), intrinsics, (400, 300), 1.84,
            distances=(8.0,), lateral=(0.0,),
        )
        self.assertIsNotNone(place)
        self.assertAlmostEqual(place["translation"][1], 8.0, places=3)
        self.assertAlmostEqual(place["translation"][2], -1.70, places=2)
        # A body already standing on that spot is not a free placement.
        blocked = road.copy()
        blocked[:30, 0] = 0.0
        blocked[:30, 1] = 8.0
        blocked[:30, 2] = -0.8
        none = find_placement(
            blocked, _looking_along_y(), intrinsics, (400, 300), 1.84,
            distances=(8.0,), lateral=(0.0,),
        )
        self.assertIsNone(none)

    def test_insert_camera_paints_the_object_and_lidar_adds_returns(self):
        gs = place_gaussians(_packed_column(), (0.0, 8.0, -1.70), 0.0)
        intrinsics = np.array([[220, 0, 200], [0, 220, 40], [0, 0, 1]], dtype=np.float32)
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        painted, box = insert_camera(image, gs, _looking_along_y(), intrinsics)
        self.assertIsNotNone(box)
        self.assertEqual(int(painted[0, 0].sum()), 0)
        self.assertGreater(int(painted[..., 0].max()), 150)

        elev = float(np.arctan2(-0.9, 8.0))
        beams = BeamModel(elevation=np.array([elev - 0.02, elev, elev + 0.02]), azimuth_step=np.deg2rad(1.0))
        ring = np.array([0, 1, 2] * 40, dtype=np.int64)
        az = np.linspace(-0.4, 0.4, len(ring)) + np.pi / 2
        far = np.stack([20 * np.cos(az), 20 * np.sin(az), np.full(len(ring), -0.9), np.full(len(ring), 8.0)], axis=1)
        sweep = LidarSweep(points=far.astype(np.float32), ring=ring)
        cloud, stats = insert_lidar(gs, [sweep], beams, intensity=35.0)
        self.assertEqual(cloud.shape[1], 5)
        self.assertGreater(sum(stats["object_points"]), 0)
        self.assertGreater(float(cloud[:, 3].max()), 30.0)

    def test_trim_drops_the_gray_cube_and_keeps_the_colored_body(self):
        body = _packed_column(8)
        gray = np.zeros((1000, 14), dtype=np.float32)
        gray[:, 0] = 1.1
        gray[:, 3] = 0.7
        gray[:, 4:7] = 0.05
        gray[:, 7] = 1.0
        gray[:, 11:14] = 0.53
        kept = trim_background(np.concatenate([body, gray], axis=0), min_keep=4)
        self.assertEqual(len(kept), len(body))
        colored = body.copy()
        colored[:, 0] = 1.05
        colored[:, 11:14] = (0.2, 0.3, 0.8)
        kept_color = trim_background(np.concatenate([colored, gray], axis=0), min_keep=4)
        self.assertEqual(len(kept_color), len(colored))

    def test_standing_body_is_a_thin_shell_of_forty_points(self):
        wide = np.array([[-1.0, 0.0, -1.0], [1.0, 1.7, 1.0], [0.0, 0.8, 0.0]], dtype=np.float32)
        body = fit_standing_body(wide, max_width=0.6)
        self.assertAlmostEqual(float(body[:, 1].max() - body[:, 1].min()), 1.7, places=4)
        self.assertLessEqual(float(body[:, 0].max() - body[:, 0].min()), 0.6 + 1e-4)
        self.assertLessEqual(float(body[:, 2].max() - body[:, 2].min()), 0.6 + 1e-4)
        rng = np.random.default_rng(0)
        face = np.column_stack([
            np.full(400, 8.0),
            rng.uniform(-0.25, 0.25, 400),
            rng.uniform(-1.5, 0.1, 400),
        ])
        interior = face.copy()
        interior[:, 0] = 8.4
        shell = visible_lidar_returns(np.vstack([face, interior]))
        self.assertGreater(len(shell), 10)
        self.assertLess(len(shell), 800)
        self.assertLess(float(shell[:, 0].min()), 8.05)
        sparse = subsample_pedestrian(shell, 40)
        self.assertEqual(len(sparse), 40)
        self.assertAlmostEqual(float(sparse[:, 2].min()), float(shell[:, 2].min()), places=4)
        self.assertAlmostEqual(float(sparse[:, 2].max()), float(shell[:, 2].max()), places=4)

    def test_beam_hits_follow_distance(self):
        wide = np.zeros((4, 14), dtype=np.float32)
        wide[:, 0] = [-1.2, 1.2, 0.0, 0.0]
        wide[:, 2] = [0.0, 0.0, -1.2, 1.2]
        wide[:, 1] = 0.8
        wide[:, 3] = 0.9
        wide[:, 4:7] = 0.05
        wide[:, 7] = 1.0
        squeezed = squeeze_standing(wide, max_width=0.6)
        self.assertLessEqual(float(squeezed[:, 0].max() - squeezed[:, 0].min()), 0.6 + 1e-4)
        self.assertLessEqual(float(squeezed[:, 2].max() - squeezed[:, 2].min()), 0.6 + 1e-4)

        def hits(distance):
            elev = np.arctan2(np.linspace(-1.4, 0.0, 8), distance)
            beams = BeamModel(elevation=elev, azimuth_step=np.deg2rad(1.0))
            ring = np.repeat(np.arange(8), 40).astype(np.int64)
            az = np.tile(np.linspace(-0.6, 0.6, 40), 8)
            far = np.stack([
                25.0 * np.sin(az), 25.0 * np.cos(az), np.zeros(len(az)), np.full(len(az), 4.0),
            ], axis=1).astype(np.float32)
            sweep = LidarSweep(points=far, ring=ring)
            rows = []
            for x in np.linspace(-0.3, 0.3, 5):
                for z in np.linspace(-1.4, 0.0, 8):
                    row = np.zeros(14, dtype=np.float32)
                    row[0], row[1], row[2] = x, distance, z
                    row[3] = 0.95
                    row[4:7] = 0.08
                    row[7] = 1.0
                    rows.append(row)
            gs = GaussianSet.from_packed(np.stack(rows))
            obj, _ = insert_into_sweep(gs, sweep, beams, 35.0)
            return len(obj)

        near, far = hits(6.0), hits(12.0)
        self.assertGreater(near, far)
        self.assertGreater(far, 0)
        self.assertNotEqual(near, 40)

    def test_shell_keeps_forty_keyframe_hits(self):
        from data.nuscenes_insert import NuScenesScene

        distance = 8.0
        elev = np.arctan2(np.linspace(-1.5, 0.2, 16), distance)
        beams = BeamModel(elevation=elev, azimuth_step=np.deg2rad(0.4))
        ring = np.repeat(np.arange(16), 60).astype(np.int64)
        az = np.tile(np.linspace(-0.8, 0.8, 60), 16)
        far = np.stack([
            25.0 * np.sin(az), 25.0 * np.cos(az), np.zeros(len(az)), np.full(len(az), 4.0),
        ], axis=1).astype(np.float32)
        sweeps = [
            LidarSweep(points=far, ring=ring, time_lag=0.0),
            LidarSweep(points=far.copy(), ring=ring.copy(), time_lag=0.05),
        ]
        rows = []
        for x in np.linspace(-0.25, 0.25, 6):
            for y in np.linspace(0.2, 1.6, 12):
                row = np.zeros(14, dtype=np.float32)
                row[0], row[1], row[2] = x, y, 0.0
                row[3] = 0.95
                row[4:7] = 0.06
                row[7] = 1.0
                row[11:14] = (0.2, 0.4, 0.8)
                rows.append(row)
        gaussians = np.stack(rows)
        points = gaussians[:, :3].copy()
        render = np.full((40, 24, 3), 128, dtype=np.uint8)
        render[8:32, 6:18] = (30, 70, 160)
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        intrinsics = np.array([[220, 0, 200], [0, 220, 40], [0, 0, 1]], dtype=np.float32)
        scene = NuScenesScene(
            "tok", image, "", "CAM_FRONT", _looking_along_y(), intrinsics, sweeps, beams,
            1.84, [], np.zeros((0, 2)),
        )
        frame = scene.insert_shell(points, render, (0.0, distance, -1.7), 0.0, gaussians, n_keep=40)
        self.assertIsNotNone(frame["bbox"])
        self.assertGreater(frame["stats"]["object_points"][0], 0)
        self.assertLessEqual(frame["stats"]["object_points"][0], 40)
        self.assertEqual(frame["stats"]["object_points"][1], 0)

    def test_paste_keeps_the_gray_backdrop_transparent(self):
        from data.nuscenes_insert import paste_render, place_body

        points = np.array([[0.0, 0.1, 0.0], [0.0, 1.6, 0.0], [0.2, 0.8, 0.1], [-0.2, 0.8, -0.1]], dtype=np.float32)
        body = place_body(points, (0.0, 8.0, -1.7), 0.0)
        render = np.full((32, 24, 3), 128, dtype=np.uint8)
        render[8:24, 6:18] = (40, 80, 160)
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        intrinsics = np.array([[220, 0, 200], [0, 220, 40], [0, 0, 1]], dtype=np.float32)
        painted, box = paste_render(image, body, render, _looking_along_y(), intrinsics)
        self.assertIsNotNone(box)
        self.assertEqual(int(painted[0, 0].sum()), 0)
        self.assertGreater(int(painted[..., 2].max()), 100)
        np.testing.assert_allclose(body[1], [0.0, 8.0, -0.1], atol=1e-3)

    def test_pedestrian_intensity_uses_points_inside_the_box(self):
        pts = np.zeros((40, 4), dtype=np.float64)
        pts[:, 3] = 4.0
        pts[:25, 3] = 36.0
        center = np.zeros(3)
        wlh = np.array([0.6, 0.6, 1.7])
        rot = np.eye(3)
        # Points are at the origin, so they sit inside the box. The tail is moved outside.
        pts[25:, 0] = 10.0
        got = pedestrian_intensity(pts, [(center, wlh, rot)], fallback=12.0)
        self.assertAlmostEqual(got, 36.0, places=3)


class LossAndGeneratorTests(unittest.TestCase):
    def test_naturalness_uses_the_feature_bank(self):
        reg = NaturalnessRegularizer(device="cpu")
        bank = []
        for i in range(4):
            cloud = torch.randn(128, 3)
            cloud[:, 1] = torch.linspace(0, 1.7, 128)
            bank.append(cloud)
        reg.build_bank_from_points(bank)
        near = reg.compute(bank[0], {"w": 0.6, "h": 1.7, "l": 0.6})
        far = reg.compute(torch.randn(128, 3) * 5, {"w": 3.0, "h": 0.2, "l": 3.0})
        self.assertNotAlmostEqual(near, 0.05, places=5)
        self.assertGreater(far, near)

    def test_attack_loss_uses_the_hottest_view(self):
        loss = AgentOBJLoss(lambda_2d=1.0, lambda_fusion=0.0, device="cpu")
        one_hot = loss.attack_loss([0.8, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        spread = loss.attack_loss([0.6, 0.6, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0])
        self.assertGreater(one_hot["l_atk"], spread["l_atk"])

    def test_random_stub_proposes_a_valid_edit(self):
        """Without an LLM the stand-in proposes a syntactically valid edit from the pools."""
        from agent.improved_edit_agent import ImprovedEditAgent

        agent = ImprovedEditAgent(
            semantic_pool=["carrying a suitcase"],
            style_pool=["matte fabric", "dark clothing"],
        )
        for _ in range(20):
            edit = agent.propose("carrying a backpack", "dark clothing", history=[])
            self.assertIn(edit.target, ("semantic", "style"))
            self.assertIn(edit.action, ("add", "replace", "remove"))
            pool = agent.semantic_pool if edit.target == "semantic" else agent.style_pool
            self.assertIn(edit.phrase, pool)

    def test_llm_reply_parsing(self):
        """A valid reply parses; an unparseable one returns None (no rule-based fallback)."""
        from agent.improved_edit_agent import ImprovedEditAgent

        agent = ImprovedEditAgent()
        ok = agent._parse("style|add|wearing a hood")
        self.assertIsNotNone(ok)
        self.assertEqual((ok.target, ok.action, ok.phrase), ("style", "add", "wearing a hood"))
        self.assertIsNone(agent._parse("no pipe here"))
        self.assertIsNone(agent._parse("bad|action|x"))

    def test_attack_loss_includes_naturalness(self):
        loss = AgentOBJLoss(device="cpu")
        points = torch.zeros(64, 3)
        points[:, 1] = torch.linspace(0, 1.7, 64)
        out = loss.total([0.9, 0.2], [0.8, 0.1], point_cloud=points, box_stats={"w": 0.6, "h": 1.0, "l": 0.6})
        self.assertGreater(out["l_nat"], 0.0)
        self.assertGreater(out["total"], out["l_atk"])

    def test_dummy_object_is_a_standing_person(self):
        obj = DummyGenerator().generate("a pedestrian", "carrying a backpack", "dark clothing")
        self.assertEqual(obj["backend"], "dummy")
        self.assertEqual(obj["gaussians"].shape[1], 14)
        self.assertAlmostEqual(obj["box_stats"]["h"], 1.7, delta=0.15)
        self.assertIn("dark clothing", obj["prompt"])

    def _missing_mvadapter(self, gen):
        from generators.mvadapter import MVAdapterUnavailable

        def _raise(*args, **kwargs):
            raise MVAdapterUnavailable("mvadapter is not installed")

        gen.mv.text_to_multiview = _raise

    def test_real_generator_reports_a_missing_install(self):
        gen = LGMGenerator(lgm_weights="", device="cpu", fallback_to_dummy=False)
        self._missing_mvadapter(gen)
        with self.assertRaises(GeneratorNotReady):
            gen.generate("a pedestrian", "with a hat", "dark clothing")

    def test_real_generator_can_fall_back(self):
        gen = LGMGenerator(lgm_weights="", device="cpu", fallback_to_dummy=True)
        self._missing_mvadapter(gen)
        obj = gen.generate("a pedestrian", "with a hat", "dark clothing")
        self.assertEqual(obj["backend"], "dummy")
        self.assertIn("backend_error", obj)

    def test_person_box_match_ignores_a_distant_detection(self):
        target = [10.0, 10.0, 40.0, 80.0]
        score, box = match_score([([10, 12, 38, 78], 0.7), ([200, 200, 240, 280], 0.95)], target, 0.3)
        self.assertAlmostEqual(score, 0.7)
        self.assertEqual(box, [10, 12, 38, 78])
        self.assertAlmostEqual(box_iou(target, [200, 200, 240, 280]), 0.0)
        missed, _ = match_score([([200, 200, 240, 280], 0.95)], target, 0.3)
        self.assertEqual(missed, 0.0)

    def test_one_optimization_step_writes_history(self):
        from agent.full_optimizer import FullAgentOBJOptimizer
        from agent.improved_edit_agent import ImprovedEditAgent
        from detectors.detectors import DummyDetector

        cfg = {
            "experiment": {"device": "cpu", "seed": 0},
            "optimization": {"max_iterations": 1, "viewpoints": [0, 180]},
            "render": {"resolution": 96},
            "paths": {},
            "evaluation": {},
        }
        loss = AgentOBJLoss(device="cpu")
        optimizer = FullAgentOBJOptimizer(
            generator=DummyGenerator(),
            detectors={"2d": DummyDetector("2d"), "fusion": DummyDetector("fusion")},
            edit_agent=ImprovedEditAgent(semantic_pool=["wearing a cap"], style_pool=["dark clothing"]),
            loss_fn=loss,
            config=cfg,
            output_dir=tempfile.mkdtemp(prefix="agentobj_"),
            use_naturalness=True,
        )
        result = optimizer.optimize("a pedestrian", "carrying a backpack", "dark clothing")
        self.assertEqual(len(result["history"]), 2)
        self.assertLessEqual(result["summary"]["accepted_rate"], 1.0)
        self.assertTrue((Path(optimizer.output_dir) / "history.json").is_file())
        self.assertGreater(result["best_object"]["gaussians"].shape[0], 10)
        self.assertEqual(result["summary"]["asr"], 0.0)


class ViewMetricTests(unittest.TestCase):
    def test_all_yaws_must_succeed_before_the_search_stops(self):
        from eval.evaluator import Evaluator

        rows = [
            {"token": "a", "yaw": 0, "score_2d": 0.1, "fusion": 0.1},
            {"token": "a", "yaw": 90, "score_2d": 0.9, "fusion": 0.1},
            {"token": "b", "yaw": 0, "score_2d": 0.1, "fusion": 0.1},
            {"token": "b", "yaw": 90, "score_2d": 0.1, "fusion": 0.2},
        ]
        metrics = Evaluator(0.5).view_metrics(rows)
        self.assertAlmostEqual(metrics["asr_joint"], 0.75)
        self.assertAlmostEqual(metrics["asr_joint_all_yaws"], 0.5)
        self.assertAlmostEqual(metrics["det_rate_2d"], 0.25)
        self.assertAlmostEqual(metrics["det_rate_fusion"], 0.0)

    def test_scene_evaluator_scores_each_yaw_from_one_worker_call(self):
        from PIL import Image

        from eval.scene_eval import SceneEvaluator

        class _YOLO:
            def score_box(self, image, target, min_iou=0.3):
                return 0.82, [2, 2, 20, 30], [([2, 2, 20, 30], 0.82)]

        class _Fusion:
            available = True

            def __init__(self):
                self.client = self
                self.calls = []

            def insert_score(self, token, gaussians_path, yaws, out_dir, placement=None,
                             camera="CAM_FRONT", save_points=False, points_path=None, renders=None):
                self.calls.append(token)
                out = Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                views = []
                for yaw in yaws:
                    image = out / f"yaw{int(yaw)}.png"
                    Image.new("RGB", (24, 32), (180, 40, 40)).save(image)
                    views.append({
                        "yaw": float(yaw),
                        "fusion": 0.66,
                        "image": str(image),
                        "bbox": [4.0, 4.0, 18.0, 28.0],
                        "object_points": [9],
                    })
                return {
                    "views": views,
                    "placement": {"translation": [0.0, 8.0, -1.6]},
                    "intensity": 32.0,
                    "clean_near": 0.4,
                }

        fusion = _Fusion()
        cfg = {"evaluation": {"camera": "CAM_FRONT"}, "optimization": {"viewpoints": [0, 180]}}
        with tempfile.TemporaryDirectory() as tmp:
            rows = SceneEvaluator(_YOLO(), fusion, cfg).evaluate(
                {"gaussians": _packed_column(2)}, ["tok-aaa", "tok-bbb"], tmp,
            )
        self.assertEqual(fusion.calls, ["tok-aaa", "tok-bbb"])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["score_2d"] == 0.82 and r["fusion"] == 0.66 for r in rows))
        self.assertEqual(rows[0]["object_points_total"], 9)


class SceneLoopTests(unittest.TestCase):
    def _optimizer(self, yolo_score, fusion_score, max_iterations):
        from PIL import Image

        from agent.full_optimizer import FullAgentOBJOptimizer
        from agent.improved_edit_agent import ImprovedEditAgent

        class _YOLO:
            def score_box(self, image, target, min_iou=0.3):
                return yolo_score, [1, 1, 10, 20], []

            def __call__(self, *args, **kwargs):
                return yolo_score

        class _Fusion:
            available = True

            def __init__(self):
                self.client = self
                self.calls = 0

            def insert_score(self, token, gaussians_path, yaws, out_dir, placement=None,
                             camera="CAM_FRONT", save_points=False, points_path=None, renders=None):
                self.calls += 1
                out = Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                views = []
                for yaw in yaws:
                    image = out / f"{token[:6]}_{int(yaw)}.png"
                    Image.new("RGB", (16, 16), (20, 20, 20)).save(image)
                    views.append({
                        "yaw": float(yaw),
                        "fusion": fusion_score,
                        "image": str(image),
                        "bbox": [1.0, 1.0, 14.0, 14.0],
                        "object_points": [4],
                    })
                return {"views": views, "placement": {"translation": [0, 8, -1.6]},
                        "intensity": 30.0, "clean_near": 0.0}

            def close(self):
                pass

            def __call__(self, *args, **kwargs):
                return fusion_score

        cfg = {
            "experiment": {"device": "cpu"},
            "optimization": {"max_iterations": max_iterations, "viewpoints": [0, 180]},
            "render": {"resolution": 32},
            "paths": {},
            "loss": {"lambda_fusion": 1.0, "conf_threshold": 0.5},
            "evaluation": {"sample_tokens": ["frame-a", "frame-b"], "asr_require_both": True},
        }
        fusion = _Fusion()
        optimizer = FullAgentOBJOptimizer(
            generator=DummyGenerator(),
            detectors={"2d": _YOLO(), "fusion": fusion},
            edit_agent=ImprovedEditAgent(semantic_pool=["wearing a cap"], style_pool=["dark clothing"]),
            loss_fn=AgentOBJLoss(device="cpu", conf_threshold=0.5),
            config=cfg,
            output_dir=tempfile.mkdtemp(prefix="agentobj_scene_"),
            use_naturalness=False,
        )
        return optimizer, fusion

    def test_search_stops_when_every_yaw_is_below_the_threshold(self):
        optimizer, fusion = self._optimizer(0.1, 0.1, max_iterations=5)
        result = optimizer.optimize("a pedestrian", "carrying a backpack", "dark clothing")
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(fusion.calls, 2)
        self.assertAlmostEqual(result["summary"]["asr_joint_all_yaws"], 1.0)
        self.assertAlmostEqual(result["summary"]["asr"], 1.0)

    def test_a_detected_object_keeps_searching(self):
        optimizer, fusion = self._optimizer(0.9, 0.8, max_iterations=1)
        result = optimizer.optimize("a pedestrian", "carrying a backpack", "dark clothing")
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(fusion.calls, 4)
        self.assertAlmostEqual(result["summary"]["asr_joint_all_yaws"], 0.0)
        self.assertEqual(result["history"][0]["metrics"]["n_views"], 4)


if __name__ == "__main__":
    unittest.main()
