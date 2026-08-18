import unittest

import cv2
import numpy as np

from visio_camera_analyzer.app import (
    BodyPart,
    analyze_frame,
    build_distance_references,
    build_hand_rois,
    classify_eye_color,
    classify_hair_color,
    count_visible_fingers,
    estimate_object_distance,
    mode_from_key,
    next_mode,
    parse_mobilenet_detections,
    parse_openpose_hand_output,
    parse_pose_output,
    parse_screen_region,
    parse_yolo_detections,
)


class FakeDetector:
    def detectMultiScale(self, *_args, **_kwargs):
        return np.array([[10, 20, 30, 40]])


class AnalyzeFrameTest(unittest.TestCase):
    def test_scales_face_boxes_back_to_original_frame(self):
        frame = np.full((120, 200, 3), 120, dtype=np.uint8)

        analysis = analyze_frame(frame, FakeDetector(), scale=0.5)

        self.assertEqual(analysis.faces, ((20, 40, 60, 80),))
        self.assertAlmostEqual(analysis.brightness, 120.0)
        self.assertEqual(analysis.sharpness, 0.0)

    def test_parses_mobilenet_detections_above_threshold(self):
        raw = np.zeros((1, 1, 2, 7), dtype=np.float32)
        raw[0, 0, 0] = [0, 15, 0.91, 0.10, 0.20, 0.60, 0.80]
        raw[0, 0, 1] = [0, 7, 0.20, 0.00, 0.00, 1.00, 1.00]

        objects = parse_mobilenet_detections(
            raw,
            frame_width=200,
            frame_height=100,
            confidence_threshold=0.45,
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].label, "personne")
        self.assertAlmostEqual(objects[0].confidence, 0.91, places=5)
        self.assertEqual(objects[0].box, (20, 20, 100, 60))
        self.assertAlmostEqual(objects[0].distance_m, 25.5)

    def test_parses_yolo_detections_and_applies_nms(self):
        detection = np.zeros(85, dtype=np.float32)
        detection[0:4] = [0.50, 0.50, 0.40, 0.20]
        detection[4] = 1.0
        detection[5 + 2] = 0.90

        objects = parse_yolo_detections(
            (np.array([detection]),),
            labels=("person", "bicycle", "car"),
            frame_width=200,
            frame_height=100,
            confidence_threshold=0.45,
            nms_threshold=0.40,
        )

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].label, "voiture")
        self.assertAlmostEqual(objects[0].confidence, 0.90, places=5)
        self.assertEqual(objects[0].box, (60, 40, 80, 20))
        self.assertAlmostEqual(objects[0].distance_m, 20.25)

    def test_parses_pose_heatmaps(self):
        output = np.zeros((1, 18, 4, 4), dtype=np.float32)
        output[0, 0, 2, 1] = 0.80
        output[0, 1, 1, 3] = 0.90
        output[0, 2, 0, 0] = 0.05

        body_parts = parse_pose_output(
            output,
            frame_width=400,
            frame_height=200,
            confidence_threshold=0.12,
        )

        self.assertEqual(len(body_parts), 2)
        self.assertEqual(body_parts[0].label, "nez")
        self.assertEqual(body_parts[0].point, (100, 100))
        self.assertEqual(body_parts[1].label, "cou")
        self.assertEqual(body_parts[1].point, (300, 50))

    def test_parses_screen_region(self):
        self.assertEqual(parse_screen_region("10,20,300,400"), (10, 20, 300, 400))
        self.assertIsNone(parse_screen_region(None))

    def test_keyboard_mode_helpers(self):
        self.assertEqual(next_mode("faces"), "objects")
        self.assertEqual(mode_from_key(ord("6")), "all")
        self.assertIsNone(mode_from_key(ord("x")))

    def test_estimates_object_distance_from_reference_size(self):
        distance = estimate_object_distance(
            "bottle",
            (0, 0, 50, 100),
            focal_length_px=800,
            references={"bottle": ("height", 0.25)},
        )

        self.assertAlmostEqual(distance, 2.0)

    def test_builds_distance_reference_overrides(self):
        references = build_distance_references(("bouteille=hauteur:0.30",))

        self.assertEqual(references["bottle"], ("height", 0.30))

    def test_classifies_simple_hair_and_eye_colors(self):
        black_patch = np.full((16, 16, 3), (20, 20, 20), dtype=np.uint8)
        blue_eye_patch = np.full((24, 32, 3), (245, 245, 245), dtype=np.uint8)
        blue_eye_patch[8:16, 10:22] = (180, 80, 25)
        blue_eye_patch[10:14, 14:18] = (10, 10, 10)
        blue_eye_column = blue_eye_patch.reshape(-1, 1, 3)
        light_brown_hsv = np.full((16, 16, 3), (14, 80, 150), dtype=np.uint8)
        red_hair_hsv = np.full((16, 16, 3), (12, 160, 170), dtype=np.uint8)
        mixed_hair_hsv = np.full((24, 24, 3), (14, 80, 170), dtype=np.uint8)
        mixed_hair_hsv[:12, :] = (14, 120, 110)

        self.assertEqual(classify_hair_color(black_patch), "noir")
        self.assertEqual(
            classify_hair_color(cv2.cvtColor(light_brown_hsv, cv2.COLOR_HSV2BGR)),
            "chatain clair",
        )
        self.assertEqual(
            classify_hair_color(cv2.cvtColor(red_hair_hsv, cv2.COLOR_HSV2BGR)),
            "roux",
        )
        self.assertEqual(
            classify_hair_color(cv2.cvtColor(mixed_hair_hsv, cv2.COLOR_HSV2BGR)),
            "chatain",
        )
        self.assertEqual(classify_eye_color(blue_eye_patch), "bleu")
        self.assertEqual(classify_eye_color(blue_eye_column), "bleu")

    def test_counts_visible_fingers_from_hand_landmarks(self):
        landmarks = [(0.5, 0.90)] * 21
        landmarks[3] = (0.45, 0.76)
        landmarks[4] = (0.33, 0.58)
        landmarks[6] = (0.47, 0.64)
        landmarks[8] = (0.43, 0.34)
        landmarks[10] = (0.50, 0.62)
        landmarks[12] = (0.50, 0.28)
        landmarks[14] = (0.53, 0.64)
        landmarks[16] = (0.58, 0.34)
        landmarks[18] = (0.56, 0.70)
        landmarks[20] = (0.67, 0.50)

        self.assertEqual(count_visible_fingers(tuple(landmarks)), 5)

        folded = list(landmarks)
        folded[4] = (0.46, 0.82)
        folded[8] = (0.48, 0.74)

        self.assertEqual(count_visible_fingers(tuple(folded)), 3)

    def test_builds_hand_rois_from_pose_wrists(self):
        body_parts = (
            BodyPart(index=3, label="coude droit", confidence=0.9, point=(100, 100)),
            BodyPart(index=4, label="poignet droit", confidence=0.9, point=(150, 100)),
            BodyPart(index=6, label="coude gauche", confidence=0.9, point=(300, 100)),
            BodyPart(index=7, label="poignet gauche", confidence=0.9, point=(250, 100)),
        )

        rois = build_hand_rois(
            body_parts,
            frame_width=400,
            frame_height=300,
            max_hands=2,
            roi_scale=2.0,
        )

        self.assertEqual(rois[0], ("droite", (115, 50, 100, 100)))
        self.assertEqual(rois[1], ("gauche", (185, 50, 100, 100)))

    def test_parses_openpose_hand_heatmaps(self):
        output = np.zeros((1, 21, 8, 8), dtype=np.float32)
        heatmap_points = {
            0: (4, 7),
            3: (3, 6),
            4: (1, 3),
            6: (4, 5),
            8: (3, 2),
            10: (4, 5),
            12: (4, 1),
            14: (5, 5),
            16: (6, 2),
            18: (5, 6),
            20: (7, 3),
        }

        for part_index, (x, y) in heatmap_points.items():
            output[0, part_index, y, x] = 0.9

        hand = parse_openpose_hand_output(
            output,
            roi=(10, 20, 80, 80),
            confidence_threshold=0.1,
            handedness="droite",
        )

        self.assertIsNotNone(hand)
        self.assertEqual(hand.handedness, "droite")
        self.assertEqual(hand.finger_count, 5)


if __name__ == "__main__":
    unittest.main()
