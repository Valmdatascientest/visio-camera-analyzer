from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import cv2
import numpy as np


WINDOW_NAME = "Visio Camera Analyzer"
MODEL_DIR = Path("models")
DEFAULT_FOCAL_LENGTH_PX = 900.0
MODES = ("faces", "objects", "pose", "both", "objects-pose", "all")
MODE_LABELS = {
    "faces": "visages",
    "objects": "objets",
    "pose": "corps",
    "both": "visages + objets",
    "objects-pose": "objets + corps",
    "all": "tout",
}

DEFAULT_MOBILENET_PROTOTXT = MODEL_DIR / "deploy.prototxt"
DEFAULT_MOBILENET_MODEL = MODEL_DIR / "mobilenet_iter_73000.caffemodel"
MOBILENET_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
)
MOBILENET_MODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/"
    "mobilenet_iter_73000.caffemodel"
)
MOBILENET_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

DEFAULT_YOLO_CFG = MODEL_DIR / "yolov4-tiny.cfg"
DEFAULT_YOLO_WEIGHTS = MODEL_DIR / "yolov4-tiny.weights"
DEFAULT_YOLO_NAMES = MODEL_DIR / "coco.names"
YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = (
    "https://github.com/AlexeyAB/darknet/releases/download/"
    "darknet_yolo_v4_pre/yolov4-tiny.weights"
)
YOLO_NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"

DEFAULT_POSE_PROTOTXT = MODEL_DIR / "openpose_pose_coco.prototxt"
DEFAULT_POSE_MODEL = MODEL_DIR / "pose_iter_440000.caffemodel"
DEFAULT_HAND_PROTOTXT = MODEL_DIR / "openpose_hand_pose_deploy.prototxt"
DEFAULT_HAND_MODEL = MODEL_DIR / "openpose_hand_pose_iter_102000.caffemodel"
POSE_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/"
    "openpose_pose_coco.prototxt"
)
POSE_MODEL_URLS = (
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/"
    "pose_iter_440000.caffemodel",
    "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/"
    "pose_iter_440000.caffemodel?download=true",
)
HAND_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/"
    "master/models/hand/pose_deploy.prototxt"
)
HAND_MODEL_URLS = (
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/"
    "pose_iter_102000.caffemodel",
    "https://huggingface.co/camenduru/openpose/resolve/main/models/hand/"
    "pose_iter_102000.caffemodel?download=true",
)
COCO_BODY_PARTS = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
COCO_POSE_PAIRS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
FRENCH_LABELS = {
    "background": "fond",
    "person": "personne",
    "bicycle": "velo",
    "car": "voiture",
    "motorbike": "moto",
    "motorcycle": "moto",
    "aeroplane": "avion",
    "airplane": "avion",
    "bus": "bus",
    "train": "train",
    "truck": "camion",
    "boat": "bateau",
    "traffic light": "feu tricolore",
    "fire hydrant": "bouche incendie",
    "stop sign": "panneau stop",
    "parking meter": "parcmetre",
    "bench": "banc",
    "bird": "oiseau",
    "cat": "chat",
    "dog": "chien",
    "horse": "cheval",
    "sheep": "mouton",
    "cow": "vache",
    "elephant": "elephant",
    "bear": "ours",
    "zebra": "zebre",
    "giraffe": "girafe",
    "backpack": "sac a dos",
    "umbrella": "parapluie",
    "handbag": "sac a main",
    "tie": "cravate",
    "suitcase": "valise",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "ballon",
    "kite": "cerf-volant",
    "baseball bat": "batte de baseball",
    "baseball glove": "gant de baseball",
    "skateboard": "skateboard",
    "surfboard": "planche de surf",
    "tennis racket": "raquette de tennis",
    "bottle": "bouteille",
    "wine glass": "verre a vin",
    "cup": "tasse",
    "fork": "fourchette",
    "knife": "couteau",
    "spoon": "cuillere",
    "bowl": "bol",
    "banana": "banane",
    "apple": "pomme",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "brocoli",
    "carrot": "carotte",
    "hot dog": "hot-dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "gateau",
    "chair": "chaise",
    "sofa": "canape",
    "couch": "canape",
    "pottedplant": "plante",
    "potted plant": "plante",
    "bed": "lit",
    "diningtable": "table",
    "dining table": "table",
    "toilet": "toilettes",
    "tvmonitor": "television",
    "tv": "television",
    "laptop": "ordinateur portable",
    "mouse": "souris",
    "remote": "telecommande",
    "keyboard": "clavier",
    "cell phone": "telephone",
    "microwave": "micro-ondes",
    "oven": "four",
    "toaster": "grille-pain",
    "sink": "evier",
    "refrigerator": "refrigerateur",
    "book": "livre",
    "clock": "horloge",
    "vase": "vase",
    "scissors": "ciseaux",
    "teddy bear": "ours en peluche",
    "hair drier": "seche-cheveux",
    "toothbrush": "brosse a dents",
}
FRENCH_BODY_PARTS = {
    "nose": "nez",
    "neck": "cou",
    "right_shoulder": "epaule droite",
    "right_elbow": "coude droit",
    "right_wrist": "poignet droit",
    "left_shoulder": "epaule gauche",
    "left_elbow": "coude gauche",
    "left_wrist": "poignet gauche",
    "right_hip": "hanche droite",
    "right_knee": "genou droit",
    "right_ankle": "cheville droite",
    "left_hip": "hanche gauche",
    "left_knee": "genou gauche",
    "left_ankle": "cheville gauche",
    "right_eye": "oeil droit",
    "left_eye": "oeil gauche",
    "right_ear": "oreille droite",
    "left_ear": "oreille gauche",
}
REFERENCE_OBJECT_SIZES_M = {
    "person": ("height", 1.70),
    "bicycle": ("width", 1.70),
    "car": ("width", 1.80),
    "motorbike": ("width", 0.80),
    "motorcycle": ("width", 0.80),
    "aeroplane": ("width", 11.00),
    "airplane": ("width", 11.00),
    "bus": ("width", 2.50),
    "train": ("width", 3.00),
    "truck": ("width", 2.50),
    "boat": ("width", 2.00),
    "traffic light": ("height", 0.90),
    "fire hydrant": ("height", 0.75),
    "stop sign": ("width", 0.75),
    "parking meter": ("height", 1.40),
    "bench": ("width", 1.50),
    "bird": ("height", 0.20),
    "cat": ("height", 0.25),
    "dog": ("height", 0.50),
    "horse": ("height", 1.60),
    "sheep": ("height", 0.90),
    "cow": ("height", 1.40),
    "elephant": ("height", 3.00),
    "bear": ("height", 1.40),
    "zebra": ("height", 1.40),
    "giraffe": ("height", 4.50),
    "backpack": ("height", 0.45),
    "umbrella": ("width", 1.00),
    "handbag": ("width", 0.35),
    "tie": ("height", 1.40),
    "suitcase": ("height", 0.60),
    "frisbee": ("width", 0.27),
    "skis": ("height", 1.70),
    "snowboard": ("height", 1.50),
    "sports ball": ("width", 0.22),
    "kite": ("width", 1.00),
    "baseball bat": ("height", 0.85),
    "baseball glove": ("width", 0.25),
    "skateboard": ("width", 0.80),
    "surfboard": ("height", 1.80),
    "tennis racket": ("height", 0.70),
    "bottle": ("height", 0.25),
    "wine glass": ("height", 0.18),
    "cup": ("height", 0.12),
    "fork": ("height", 0.20),
    "knife": ("height", 0.22),
    "spoon": ("height", 0.18),
    "bowl": ("width", 0.18),
    "banana": ("height", 0.18),
    "apple": ("width", 0.08),
    "sandwich": ("width", 0.13),
    "orange": ("width", 0.08),
    "broccoli": ("height", 0.18),
    "carrot": ("height", 0.18),
    "hot dog": ("width", 0.16),
    "pizza": ("width", 0.30),
    "donut": ("width", 0.09),
    "cake": ("width", 0.25),
    "chair": ("height", 0.90),
    "sofa": ("width", 1.80),
    "couch": ("width", 1.80),
    "pottedplant": ("height", 0.45),
    "potted plant": ("height", 0.45),
    "bed": ("width", 1.40),
    "diningtable": ("width", 1.20),
    "dining table": ("width", 1.20),
    "toilet": ("height", 0.75),
    "tvmonitor": ("width", 0.80),
    "tv": ("width", 0.80),
    "laptop": ("width", 0.34),
    "mouse": ("width", 0.06),
    "remote": ("height", 0.18),
    "keyboard": ("width", 0.44),
    "cell phone": ("height", 0.15),
    "microwave": ("width", 0.50),
    "oven": ("width", 0.60),
    "toaster": ("width", 0.28),
    "sink": ("width", 0.55),
    "refrigerator": ("height", 1.70),
    "book": ("height", 0.24),
    "clock": ("width", 0.30),
    "vase": ("height", 0.30),
    "scissors": ("height", 0.18),
    "teddy bear": ("height", 0.35),
    "hair drier": ("width", 0.22),
    "toothbrush": ("height", 0.18),
}


@dataclass(frozen=True)
class FaceAppearance:
    box: tuple[int, int, int, int]
    hair_color: str
    eye_color: str
    eye_sample_boxes: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class FrameAnalysis:
    faces: tuple[tuple[int, int, int, int], ...]
    brightness: float
    sharpness: float
    appearances: tuple[FaceAppearance, ...] = ()


@dataclass(frozen=True)
class DetectedObject:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    raw_label: str = ""
    distance_m: float | None = None


@dataclass(frozen=True)
class BodyPart:
    index: int
    label: str
    confidence: float
    point: tuple[int, int]


@dataclass(frozen=True)
class DetectedHand:
    handedness: str
    confidence: float
    finger_count: int
    landmarks: tuple[tuple[int, int] | None, ...]
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    owner: str
    title: str
    bounds: tuple[int, int, int, int]


class FrameSource:
    name = "source"
    is_static = False
    ends_cleanly = False
    wait_ms = 1

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def release(self) -> None:
        return


class CameraFrameSource(FrameSource):
    def __init__(self, camera_index: int) -> None:
        self.name = f"camera {camera_index}"
        self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(
                "Impossible d'ouvrir la camera. Verifie les permissions macOS et "
                "essaie un autre --camera-index."
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class ImageFrameSource(FrameSource):
    is_static = True
    ends_cleanly = True
    wait_ms = 30

    def __init__(self, path: Path) -> None:
        self.name = f"image {path.name}"
        self.frame = cv2.imread(str(path))

        if self.frame is None:
            raise RuntimeError(f"Impossible de lire l'image: {path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        return True, self.frame.copy()


class VideoFrameSource(FrameSource):
    ends_cleanly = True

    def __init__(self, path: Path, loop: bool) -> None:
        self.name = f"video {path.name}"
        self.loop = loop
        self.capture = cv2.VideoCapture(str(path))

        if not self.capture.isOpened():
            raise RuntimeError(f"Impossible de lire la video: {path}")

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.wait_ms = int(1000 / fps) if 1 <= fps <= 120 else 1

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()

        if ok:
            return True, frame

        if not self.loop:
            return False, None

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class StreamFrameSource(FrameSource):
    def __init__(self, url: str, fps: float) -> None:
        self.name = f"flux {url}"
        self.capture = cv2.VideoCapture(url)

        if not self.capture.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir le flux video: {url}")

        stream_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.wait_ms = fps_to_wait_ms(stream_fps if 1 <= stream_fps <= 120 else fps)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class ScreenFrameSource(FrameSource):
    def __init__(
        self,
        screen_index: int,
        region: tuple[int, int, int, int] | None,
        fps: float,
    ) -> None:
        try:
            import mss
        except ImportError as error:
            raise RuntimeError("Capture ecran indisponible. Reinstalle le projet.") from error

        self.name = f"ecran {screen_index}"
        self.sct = mss.mss()
        self.wait_ms = fps_to_wait_ms(fps)

        if region is not None:
            x, y, width, height = region
            self.monitor = {"left": x, "top": y, "width": width, "height": height}
            self.name = f"ecran zone {x},{y},{width},{height}"
            return

        if screen_index < 0 or screen_index >= len(self.sct.monitors):
            raise RuntimeError(
                f"Ecran introuvable: {screen_index}. Utilise un index entre 0 "
                f"et {len(self.sct.monitors) - 1}."
            )

        self.monitor = self.sct.monitors[screen_index]

    def read(self) -> tuple[bool, np.ndarray | None]:
        screenshot = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self) -> None:
        self.sct.close()


class MacWindowFrameSource(FrameSource):
    def __init__(self, window_id: int, fps: float) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("La capture de fenetre est disponible uniquement sur macOS.")

        self.window_id = window_id
        self.name = f"fenetre {window_id}"
        self.wait_ms = fps_to_wait_ms(fps)
        self.capture_path = Path(tempfile.gettempdir()) / (
            f"visio-window-{window_id}.png"
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        result = subprocess.run(
            [
                "/usr/sbin/screencapture",
                "-x",
                "-l",
                str(self.window_id),
                str(self.capture_path),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            message = result.stderr.strip() or "capture fenetre impossible"
            raise RuntimeError(message)

        frame = cv2.imread(str(self.capture_path))
        return frame is not None, frame


class MobileNetObjectDetector:
    def __init__(
        self,
        prototxt_path: Path,
        model_path: Path,
        confidence_threshold: float,
        focal_length_px: float,
        distance_references: dict[str, tuple[str, float]],
    ) -> None:
        require_files(
            (prototxt_path, model_path),
            "Modele MobileNet-SSD introuvable. Lance: "
            "visio-camera --object-detector mobilenet --download-object-model",
        )
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        self.confidence_threshold = confidence_threshold
        self.focal_length_px = focal_length_px
        self.distance_references = distance_references

    def detect(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )

        self.net.setInput(blob)
        raw_detections = self.net.forward()
        return parse_mobilenet_detections(
            raw_detections,
            frame_width=width,
            frame_height=height,
            confidence_threshold=self.confidence_threshold,
            focal_length_px=self.focal_length_px,
            distance_references=self.distance_references,
        )


class YoloObjectDetector:
    def __init__(
        self,
        cfg_path: Path,
        weights_path: Path,
        names_path: Path,
        confidence_threshold: float,
        nms_threshold: float,
        focal_length_px: float,
        distance_references: dict[str, tuple[str, float]],
    ) -> None:
        require_files(
            (cfg_path, weights_path, names_path),
            "Modele YOLO introuvable. Lance: visio-camera --download-object-model",
        )
        self.labels = load_labels(names_path)
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.focal_length_px = focal_length_px
        self.distance_references = distance_references

    def detect(self, frame: np.ndarray) -> tuple[DetectedObject, ...]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        raw_outputs = self.net.forward(self.output_layer_names)
        return parse_yolo_detections(
            raw_outputs,
            labels=self.labels,
            frame_width=width,
            frame_height=height,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            focal_length_px=self.focal_length_px,
            distance_references=self.distance_references,
        )


class PoseDetector:
    def __init__(
        self,
        prototxt_path: Path,
        model_path: Path,
        confidence_threshold: float,
        input_height: int,
    ) -> None:
        require_files(
            (prototxt_path, model_path),
            "Modele OpenPose introuvable. Lance: visio-camera --download-pose-model",
        )
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        self.confidence_threshold = confidence_threshold
        self.input_height = input_height

    def detect(self, frame: np.ndarray) -> tuple[BodyPart, ...]:
        height, width = frame.shape[:2]
        input_height = max(128, self.input_height)
        input_width = max(128, int((input_height / height) * width))
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255,
            size=(input_width, input_height),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()
        return parse_pose_output(
            output,
            frame_width=width,
            frame_height=height,
            confidence_threshold=self.confidence_threshold,
        )


class HandDetector:
    def __init__(
        self,
        prototxt_path: Path,
        model_path: Path,
        max_hands: int,
        confidence_threshold: float,
        input_size: int,
        roi_scale: float,
    ) -> None:
        require_files(
            (prototxt_path, model_path),
            "Modele main OpenPose introuvable. Lance: visio-camera --download-hand-model",
        )
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
        self.max_hands = max(1, max_hands)
        self.confidence_threshold = confidence_threshold
        self.input_size = max(96, input_size)
        self.roi_scale = max(1.0, roi_scale)

    def detect(
        self,
        frame: np.ndarray,
        body_parts: tuple[BodyPart, ...],
    ) -> tuple[DetectedHand, ...]:
        height, width = frame.shape[:2]
        rois = build_hand_rois(
            body_parts,
            width,
            height,
            self.max_hands,
            self.roi_scale,
        )
        hands: list[DetectedHand] = []

        for handedness, roi in rois:
            x, y, roi_width, roi_height = roi
            crop = frame[y : y + roi_height, x : x + roi_width]

            if crop.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(
                crop,
                scalefactor=1.0 / 255,
                size=(self.input_size, self.input_size),
                mean=(0, 0, 0),
                swapRB=False,
                crop=False,
            )
            self.net.setInput(blob)
            output = self.net.forward()
            hand = parse_openpose_hand_output(
                output,
                roi,
                confidence_threshold=self.confidence_threshold,
                handedness=handedness,
            )

            if hand is not None:
                hands.append(hand)

        return tuple(hands[: self.max_hands])

    def close(self) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse locale de la camera avec detection de visage, "
            "reconnaissance d'objets et points du corps."
        )
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="faces",
        help=(
            "Type d'analyse. both = visages + objets, all = visages + objets + pose."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("camera", "image", "video", "stream", "screen", "window"),
        default="camera",
        help="Source a analyser: camera, image, video, stream, ecran ou fenetre macOS.",
    )
    parser.add_argument(
        "--input",
        help="Chemin image/video ou URL quand --source stream est utilise.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index OpenCV de la camera a ouvrir, generalement 0 sur Mac.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Relance la video au debut quand elle arrive a la fin.",
    )
    parser.add_argument(
        "--screen-index",
        type=int,
        default=1,
        help="Index de l'ecran pour --source screen. 0 = tous les ecrans avec mss.",
    )
    parser.add_argument(
        "--screen-region",
        help="Zone d'ecran a capturer au format x,y,largeur,hauteur.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=15.0,
        help="Frequence cible pour les sources ecran/fenetre.",
    )
    parser.add_argument(
        "--window-id",
        type=int,
        help="Identifiant de fenetre macOS a capturer avec --source window.",
    )
    parser.add_argument(
        "--window-title",
        help="Texte a chercher dans le titre/proprietaire d'une fenetre macOS.",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="Liste les fenetres macOS visibles puis quitte.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Facteur de reduction pour accelerer la detection visage, entre 0.2 et 1.0.",
    )
    parser.add_argument(
        "--face-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estime les couleurs de cheveux et des yeux en mode visage.",
    )
    parser.add_argument(
        "--eye-debug",
        action="store_true",
        help="Affiche les petites zones utilisees pour estimer la couleur des yeux.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Ouvre la camera et affiche l'analyse dans le terminal sans fenetre.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Nombre maximal d'images a analyser avant de quitter. 0 = illimite.",
    )
    parser.add_argument(
        "--object-detector",
        choices=("yolo", "mobilenet"),
        default="yolo",
        help="Modele objet a utiliser. yolo reconnait les 80 classes COCO.",
    )
    parser.add_argument(
        "--object-confidence",
        type=float,
        default=0.35,
        help="Score minimum pour afficher un objet detecte, entre 0.0 et 1.0.",
    )
    parser.add_argument(
        "--object-nms",
        type=float,
        default=0.40,
        help="Seuil NMS YOLO pour fusionner les boites redondantes.",
    )
    parser.add_argument(
        "--distance-estimates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Affiche une estimation de distance pour les objets avec taille connue.",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=DEFAULT_FOCAL_LENGTH_PX,
        help="Focale estimee en pixels pour la distance. A calibrer selon la source.",
    )
    parser.add_argument(
        "--distance-reference",
        action="append",
        default=[],
        metavar="LABEL=AXIS:METERS",
        help=(
            "Taille reelle de reference, ex: bottle=height:0.28 "
            "ou personne=height:1.75."
        ),
    )
    parser.add_argument(
        "--object-prototxt",
        type=Path,
        default=DEFAULT_MOBILENET_PROTOTXT,
        help="Chemin du fichier prototxt MobileNet-SSD.",
    )
    parser.add_argument(
        "--object-model",
        type=Path,
        default=DEFAULT_MOBILENET_MODEL,
        help="Chemin du fichier caffemodel MobileNet-SSD.",
    )
    parser.add_argument(
        "--yolo-cfg",
        type=Path,
        default=DEFAULT_YOLO_CFG,
        help="Chemin du fichier cfg YOLO.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=DEFAULT_YOLO_WEIGHTS,
        help="Chemin du fichier weights YOLO.",
    )
    parser.add_argument(
        "--yolo-names",
        type=Path,
        default=DEFAULT_YOLO_NAMES,
        help="Chemin du fichier de classes COCO.",
    )
    parser.add_argument(
        "--pose-prototxt",
        type=Path,
        default=DEFAULT_POSE_PROTOTXT,
        help="Chemin du fichier prototxt OpenPose COCO.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=DEFAULT_POSE_MODEL,
        help="Chemin du fichier caffemodel OpenPose COCO.",
    )
    parser.add_argument(
        "--pose-confidence",
        type=float,
        default=0.12,
        help="Score minimum pour afficher un point du corps.",
    )
    parser.add_argument(
        "--pose-height",
        type=int,
        default=256,
        help="Hauteur d'entree OpenPose. Plus haut = plus precis mais plus lent.",
    )
    parser.add_argument(
        "--hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detecte les mains et compte les doigts visibles dans les modes pose.",
    )
    parser.add_argument(
        "--hand-prototxt",
        type=Path,
        default=DEFAULT_HAND_PROTOTXT,
        help="Chemin du fichier prototxt OpenPose main.",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=DEFAULT_HAND_MODEL,
        help="Chemin du fichier caffemodel OpenPose main.",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=4,
        help="Nombre maximal de mains a detecter.",
    )
    parser.add_argument(
        "--hand-confidence",
        type=float,
        default=0.10,
        help="Score minimum pour afficher un point de main.",
    )
    parser.add_argument(
        "--hand-input-size",
        type=int,
        default=256,
        help="Taille d'entree OpenPose main. Plus haut = plus precis mais plus lent.",
    )
    parser.add_argument(
        "--hand-roi-scale",
        type=float,
        default=2.0,
        help="Echelle de la zone main autour du poignet detecte.",
    )
    parser.add_argument(
        "--download-object-model",
        action="store_true",
        help="Telecharge le modele objet selectionne puis quitte.",
    )
    parser.add_argument(
        "--download-pose-model",
        action="store_true",
        help="Telecharge le modele OpenPose COCO puis quitte.",
    )
    parser.add_argument(
        "--download-hand-model",
        action="store_true",
        help="Telecharge le modele OpenPose main puis quitte.",
    )
    parser.add_argument(
        "--download-all-models",
        action="store_true",
        help="Telecharge YOLO/COCO, OpenPose/COCO et OpenPose mains puis quitte.",
    )
    return parser


def require_files(paths: tuple[Path, ...], message: str) -> None:
    if any(not path.exists() for path in paths):
        raise RuntimeError(message)


def load_labels(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as labels_file:
        return tuple(line.strip() for line in labels_file if line.strip())


def fps_to_wait_ms(fps: float) -> int:
    if fps <= 0:
        return 1

    return max(1, int(1000 / fps))


def parse_screen_region(region: str | None) -> tuple[int, int, int, int] | None:
    if region is None:
        return None

    try:
        x, y, width, height = (int(part.strip()) for part in region.split(","))
    except ValueError as error:
        raise RuntimeError(
            "La zone d'ecran doit etre au format x,y,largeur,hauteur."
        ) from error

    if width <= 0 or height <= 0:
        raise RuntimeError("La largeur et la hauteur de --screen-region doivent etre positives.")

    return x, y, width, height


def list_macos_windows() -> tuple[WindowInfo, ...]:
    if sys.platform != "darwin":
        raise RuntimeError("La liste des fenetres est disponible uniquement sur macOS.")

    try:
        import Quartz
    except ImportError as error:
        raise RuntimeError("Quartz est indisponible. Reinstalle le projet.") from error

    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements
    )
    raw_windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
    windows: list[WindowInfo] = []

    for raw in raw_windows or []:
        bounds = raw.get("kCGWindowBounds", {})
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))
        layer = int(raw.get("kCGWindowLayer", 0))
        window_id = int(raw.get("kCGWindowNumber", 0))
        owner = str(raw.get("kCGWindowOwnerName", "") or "")
        title = str(raw.get("kCGWindowName", "") or "")

        if layer != 0 or window_id <= 0 or width < 80 or height < 60:
            continue

        windows.append(
            WindowInfo(
                window_id=window_id,
                owner=owner,
                title=title,
                bounds=(
                    int(bounds.get("X", 0)),
                    int(bounds.get("Y", 0)),
                    width,
                    height,
                ),
            )
        )

    return tuple(windows)


def print_window_list() -> None:
    windows = list_macos_windows()

    if not windows:
        print("Aucune fenetre visible trouvee.")
        return

    for window in windows:
        x, y, width, height = window.bounds
        title = f" - {window.title}" if window.title else ""
        print(
            f"{window.window_id:>10}  {window.owner}{title}  "
            f"({x},{y},{width},{height})"
        )


def resolve_window_id(window_id: int | None, window_title: str | None) -> int:
    if window_id is not None:
        return window_id

    if not window_title:
        raise RuntimeError(
            "Pour --source window, indique --window-id ou --window-title. "
            "Utilise --list-windows pour voir les identifiants."
        )

    needle = window_title.casefold()
    for window in list_macos_windows():
        haystack = f"{window.owner} {window.title}".casefold()

        if needle in haystack:
            return window.window_id

    raise RuntimeError(f"Aucune fenetre ne correspond a: {window_title}")


def require_input_path(path: str | Path | None, source: str) -> Path:
    if path is None:
        raise RuntimeError(f"--source {source} demande un chemin via --input.")

    path_obj = Path(path)

    if not path_obj.exists():
        raise RuntimeError(f"Fichier introuvable: {path_obj}")

    return path_obj


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "camera":
        return CameraFrameSource(args.camera_index)

    if args.source == "image":
        return ImageFrameSource(require_input_path(args.input, "image"))

    if args.source == "video":
        return VideoFrameSource(require_input_path(args.input, "video"), args.loop)

    if args.source == "stream":
        if args.input is None:
            raise RuntimeError("--source stream demande une URL via --input.")

        return StreamFrameSource(str(args.input), args.source_fps)

    if args.source == "screen":
        return ScreenFrameSource(
            args.screen_index,
            parse_screen_region(args.screen_region),
            args.source_fps,
        )

    if args.source == "window":
        return MacWindowFrameSource(
            resolve_window_id(args.window_id, args.window_title),
            args.source_fps,
        )

    raise RuntimeError(f"Source inconnue: {args.source}")


def translate_label(label: str) -> str:
    return FRENCH_LABELS.get(label, label.replace("_", " "))


def translate_body_part(label: str) -> str:
    return FRENCH_BODY_PARTS.get(label, label.replace("_", " "))


def normalize_reference_label(label: str) -> str:
    normalized = label.strip().casefold().replace("_", " ")

    for raw_label, french_label in FRENCH_LABELS.items():
        if normalized in (raw_label.casefold(), french_label.casefold()):
            return raw_label

    return normalized


def parse_distance_reference(
    value: str,
) -> tuple[str, tuple[str, float]]:
    try:
        label, reference = value.split("=", 1)
        axis, size = reference.split(":", 1)
    except ValueError as error:
        raise RuntimeError(
            "Format --distance-reference invalide. Utilise LABEL=AXIS:METERS."
        ) from error

    axis = axis.strip().casefold()
    axis_aliases = {
        "w": "width",
        "width": "width",
        "largeur": "width",
        "h": "height",
        "height": "height",
        "hauteur": "height",
    }
    normalized_axis = axis_aliases.get(axis)

    if normalized_axis is None:
        raise RuntimeError("AXIS doit etre width/height ou largeur/hauteur.")

    try:
        size_m = float(size.strip())
    except ValueError as error:
        raise RuntimeError("METERS doit etre un nombre, par exemple 0.28.") from error

    if size_m <= 0:
        raise RuntimeError("La taille de reference doit etre positive.")

    return normalize_reference_label(label), (normalized_axis, size_m)


def build_distance_references(
    overrides: list[str] | tuple[str, ...],
) -> dict[str, tuple[str, float]]:
    references = dict(REFERENCE_OBJECT_SIZES_M)

    for override in overrides:
        label, reference = parse_distance_reference(override)
        references[label] = reference

    return references


def estimate_object_distance(
    raw_label: str,
    box: tuple[int, int, int, int],
    focal_length_px: float,
    references: dict[str, tuple[str, float]],
) -> float | None:
    if focal_length_px <= 0:
        return None

    reference = references.get(raw_label)

    if reference is None:
        return None

    axis, real_size_m = reference
    _x, _y, box_width, box_height = box
    apparent_size_px = box_width if axis == "width" else box_height

    if apparent_size_px <= 0:
        return None

    return (focal_length_px * real_size_m) / apparent_size_px


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def next_mode(mode: str) -> str:
    index = MODES.index(mode) if mode in MODES else 0
    return MODES[(index + 1) % len(MODES)]


def mode_from_key(key: int) -> str | None:
    key_to_mode = {
        ord("1"): "faces",
        ord("2"): "objects",
        ord("3"): "pose",
        ord("4"): "both",
        ord("5"): "objects-pose",
        ord("6"): "all",
    }
    return key_to_mode.get(key)


def mode_uses_faces(mode: str) -> bool:
    return mode in ("faces", "both", "all")


def mode_uses_objects(mode: str) -> bool:
    return mode in ("objects", "both", "objects-pose", "all")


def mode_uses_pose(mode: str) -> bool:
    return mode in ("pose", "objects-pose", "all")


def mode_uses_hands(mode: str) -> bool:
    return mode_uses_pose(mode)


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError(f"Impossible de charger le modele OpenCV: {cascade_path}")

    return detector


def load_eye_detector() -> cv2.CascadeClassifier:
    cascade_paths = (
        Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml",
        Path(cv2.data.haarcascades) / "haarcascade_eye.xml",
    )

    for cascade_path in cascade_paths:
        detector = cv2.CascadeClassifier(str(cascade_path))

        if not detector.empty():
            return detector

    raise RuntimeError("Impossible de charger un modele OpenCV de detection des yeux.")


def load_object_detector(args: argparse.Namespace) -> MobileNetObjectDetector | YoloObjectDetector:
    confidence_threshold = max(0.0, min(args.object_confidence, 1.0))
    focal_length_px = max(0.0, args.focal_length_px)
    distance_references = build_distance_references(args.distance_reference)

    if args.object_detector == "mobilenet":
        return MobileNetObjectDetector(
            args.object_prototxt,
            args.object_model,
            confidence_threshold,
            focal_length_px,
            distance_references,
        )

    return YoloObjectDetector(
        args.yolo_cfg,
        args.yolo_weights,
        args.yolo_names,
        confidence_threshold,
        max(0.0, min(args.object_nms, 1.0)),
        focal_length_px,
        distance_references,
    )


def load_hand_detector(args: argparse.Namespace) -> HandDetector:
    return HandDetector(
        args.hand_prototxt,
        args.hand_model,
        args.max_hands,
        max(0.0, min(args.hand_confidence, 1.0)),
        args.hand_input_size,
        args.hand_roi_scale,
    )


def crop_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = box
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(frame_width, x))
    y1 = max(0, min(frame_height, y))
    x2 = max(0, min(frame_width, x + width))
    y2 = max(0, min(frame_height, y + height))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)

    return frame[y1:y2, x1:x2]


def median_hsv(region: np.ndarray) -> tuple[float, float, float] | None:
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    if pixels.shape[0] < 4:
        return None

    saturation = pixels[:, 1]
    value = pixels[:, 2]
    usable = pixels[(value > 20) & ~((saturation < 25) & (value > 235))]

    if usable.size == 0:
        usable = pixels

    hue, sat, val = np.median(usable, axis=0)
    return float(hue), float(sat), float(val)


def median_hair_hsv(region: np.ndarray) -> tuple[float, float, float] | None:
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    if pixels.shape[0] < 4:
        return None

    hue = pixels[:, 0]
    saturation = pixels[:, 1]
    value = pixels[:, 2]

    usable = pixels[
        (value > 18)
        & (value < 230)
        & ~((hue <= 22) & (saturation < 115) & (value > 120))
    ]

    if usable.shape[0] < max(8, int(pixels.shape[0] * 0.08)):
        usable = pixels[(value > 18) & (value < 230)]

    if usable.size == 0:
        usable = pixels

    usable_value = usable[:, 2]
    cutoff = np.percentile(usable_value, 55)
    darker = usable[usable_value <= cutoff]

    if darker.shape[0] >= max(8, int(usable.shape[0] * 0.10)):
        usable = darker

    hue, sat, val = np.median(usable, axis=0)
    return float(hue), float(sat), float(val)


def classify_hair_color(region: np.ndarray) -> str:
    color = median_hair_hsv(region)

    if color is None:
        return "inconnu"

    hue, saturation, value = color

    if value < 45:
        return "noir"

    if saturation < 35:
        if value > 190:
            return "blanc/gris clair"
        if value > 120:
            return "gris"
        return "brun sombre"

    if 4 <= hue <= 16 and saturation >= 145 and value >= 95:
        return "roux"

    if 18 < hue <= 38 and saturation < 120 and value > 150:
        return "blond"

    if 5 <= hue <= 28 and saturation < 155 and value >= 115:
        return "chatain clair"

    if 5 <= hue <= 24 and value >= 92:
        return "chatain"

    if value >= 105:
        return "chatain"

    if value < 95:
        return "brun fonce"

    return "brun"


def classify_eye_color(region: np.ndarray) -> str:
    if region.size == 0:
        return "inconnu"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    if pixels.shape[0] < 4:
        return "inconnu"

    hue = pixels[:, 0]
    saturation = pixels[:, 1]
    value = pixels[:, 2]

    usable = pixels[
        (value > 25)
        & (value < 245)
        & ~((saturation < 35) & (value > 155))
    ]

    colored = usable[usable[:, 1] > 28]

    if colored.shape[0] >= max(6, int(pixels.shape[0] * 0.06)):
        color = tuple(float(value) for value in np.median(colored, axis=0))
    else:
        color = median_hsv(region)

    if color is None:
        return "inconnu"

    hue, saturation, value = color

    if value < 42:
        return "marron fonce"

    if saturation < 22:
        return "gris"

    if 85 <= hue <= 140:
        return "bleu"

    if 38 <= hue < 85:
        return "vert"

    if 22 <= hue < 38:
        return "noisette"

    if hue < 22 or hue >= 150:
        return "marron"

    return "noisette"


def estimate_hair_color(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> str:
    x, y, width, height = face_box
    hair_box = (
        x + int(width * 0.12),
        y - int(height * 0.12),
        int(width * 0.76),
        int(height * 0.32),
    )
    hair_region = crop_box(frame, hair_box)

    if hair_region.size == 0:
        hair_box = (
            x + int(width * 0.15),
            y,
            int(width * 0.70),
            int(height * 0.22),
        )
        hair_region = crop_box(frame, hair_box)

    return classify_hair_color(hair_region)


def estimate_eye_color(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    eye_detector: cv2.CascadeClassifier | None,
) -> tuple[str, tuple[tuple[int, int, int, int], ...]]:
    x, y, width, height = face_box
    upper_face = crop_box(frame, (x, y, width, int(height * 0.58)))

    if upper_face.size == 0:
        return "inconnu", ()

    eye_regions: list[np.ndarray] = []
    sample_boxes: list[tuple[int, int, int, int]] = []

    if eye_detector is not None:
        gray = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
        detected_eyes = eye_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(max(12, width // 10), max(8, height // 14)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        plausible_eyes = [
            box
            for box in detected_eyes
            if int(height * 0.10) <= box[1] <= int(height * 0.45)
        ]
        detected_eyes = sorted(
            plausible_eyes or detected_eyes,
            key=lambda box: box[2] * box[3],
            reverse=True,
        )

        for eye_x, eye_y, eye_w, eye_h in detected_eyes[:2]:
            sample_w = max(4, int(eye_w * 0.18))
            sample_h = max(4, int(eye_h * 0.26))
            center_y = eye_y + int(eye_h * 0.54)

            for center_ratio in (0.42, 0.50, 0.58):
                center_x = eye_x + int(eye_w * center_ratio)
                iris_box = (
                    center_x - sample_w // 2,
                    center_y - sample_h // 2,
                    sample_w,
                    sample_h,
                )
                eye_regions.append(crop_box(upper_face, iris_box))
                sample_boxes.append((x + iris_box[0], y + iris_box[1], sample_w, sample_h))

    if not eye_regions:
        fallback_boxes = (
            (
                int(width * 0.26),
                int(height * 0.24),
                int(width * 0.09),
                int(height * 0.055),
            ),
            (
                int(width * 0.65),
                int(height * 0.24),
                int(width * 0.09),
                int(height * 0.055),
            ),
        )
        eye_regions = [crop_box(upper_face, box) for box in fallback_boxes]
        sample_boxes = [(x + box[0], y + box[1], box[2], box[3]) for box in fallback_boxes]

    usable_regions = [region for region in eye_regions if region.size > 0]

    if not usable_regions:
        return "inconnu", ()

    combined = np.concatenate(
        [region.reshape(-1, 3) for region in usable_regions],
        axis=0,
    ).reshape(-1, 1, 3)
    return classify_eye_color(combined), tuple(sample_boxes)


def estimate_face_appearance(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    eye_detector: cv2.CascadeClassifier | None,
) -> FaceAppearance:
    eye_color, eye_sample_boxes = estimate_eye_color(frame, face_box, eye_detector)

    return FaceAppearance(
        box=face_box,
        hair_color=estimate_hair_color(frame, face_box),
        eye_color=eye_color,
        eye_sample_boxes=eye_sample_boxes,
    )


def analyze_frame(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    scale: float,
    eye_detector: cv2.CascadeClassifier | None = None,
    estimate_appearance: bool = True,
) -> FrameAnalysis:
    scale = max(0.2, min(scale, 1.0))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, None, fx=scale, fy=scale)

    detected_faces = detector.detectMultiScale(
        resized_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    inv_scale = 1.0 / scale
    faces = tuple(
        (
            int(x * inv_scale),
            int(y * inv_scale),
            int(w * inv_scale),
            int(h * inv_scale),
        )
        for (x, y, w, h) in detected_faces
    )

    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    appearances = (
        tuple(
            estimate_face_appearance(frame, face_box, eye_detector)
            for face_box in faces
        )
        if estimate_appearance
        else ()
    )

    return FrameAnalysis(
        faces=faces,
        brightness=brightness,
        sharpness=sharpness,
        appearances=appearances,
    )


def parse_mobilenet_detections(
    raw_detections: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    distance_references: dict[str, tuple[str, float]] | None = None,
) -> tuple[DetectedObject, ...]:
    objects: list[DetectedObject] = []
    references = distance_references or REFERENCE_OBJECT_SIZES_M

    for i in range(raw_detections.shape[2]):
        confidence = float(raw_detections[0, 0, i, 2])

        if confidence < confidence_threshold:
            continue

        class_id = int(raw_detections[0, 0, i, 1])

        if class_id <= 0 or class_id >= len(MOBILENET_CLASSES):
            continue

        x1, y1, x2, y2 = (
            raw_detections[0, 0, i, 3:7]
            * np.array([frame_width, frame_height, frame_width, frame_height])
        ).astype(int)
        box = clamp_box(x1, y1, x2, y2, frame_width, frame_height)

        if box is None:
            continue

        raw_label = MOBILENET_CLASSES[class_id]
        objects.append(
            DetectedObject(
                label=translate_label(raw_label),
                confidence=confidence,
                box=box,
                raw_label=raw_label,
                distance_m=estimate_object_distance(
                    raw_label,
                    box,
                    focal_length_px,
                    references,
                ),
            )
        )

    return tuple(objects)


def parse_yolo_detections(
    raw_outputs: list[np.ndarray] | tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    nms_threshold: float,
    focal_length_px: float = DEFAULT_FOCAL_LENGTH_PX,
    distance_references: dict[str, tuple[str, float]] | None = None,
) -> tuple[DetectedObject, ...]:
    boxes: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    class_ids: list[int] = []
    references = distance_references or REFERENCE_OBJECT_SIZES_M

    for output in raw_outputs:
        for detection in output:
            scores = detection[5:]

            if scores.size == 0:
                continue

            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            objectness = float(detection[4])
            confidence = objectness * class_score

            if confidence < confidence_threshold or class_id >= len(labels):
                continue

            center_x = int(detection[0] * frame_width)
            center_y = int(detection[1] * frame_height)
            width = int(detection[2] * frame_width)
            height = int(detection[3] * frame_height)
            x = center_x - width // 2
            y = center_y - height // 2
            boxes.append((x, y, width, height))
            confidences.append(confidence)
            class_ids.append(class_id)

    selected = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        score_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
    )

    if len(selected) == 0:
        return ()

    objects: list[DetectedObject] = []
    for index in np.array(selected).flatten():
        x, y, width, height = boxes[int(index)]
        box = clamp_box(x, y, x + width, y + height, frame_width, frame_height)

        if box is None:
            continue

        class_id = class_ids[int(index)]
        raw_label = labels[class_id]
        objects.append(
            DetectedObject(
                label=translate_label(raw_label),
                confidence=confidences[int(index)],
                box=box,
                raw_label=raw_label,
                distance_m=estimate_object_distance(
                    raw_label,
                    box,
                    focal_length_px,
                    references,
                ),
            )
        )

    return tuple(objects)


def parse_pose_output(
    output: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
) -> tuple[BodyPart, ...]:
    body_parts: list[BodyPart] = []

    for part_index, label in enumerate(COCO_BODY_PARTS):
        heatmap = output[0, part_index, :, :]
        _min_value, confidence, _min_location, point = cv2.minMaxLoc(heatmap)

        if confidence < confidence_threshold:
            continue

        x = int((frame_width * point[0]) / heatmap.shape[1])
        y = int((frame_height * point[1]) / heatmap.shape[0])
        body_parts.append(
            BodyPart(
                index=part_index,
                label=translate_body_part(label),
                confidence=float(confidence),
                point=(x, y),
            )
        )

    return tuple(body_parts)


def point_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def count_visible_fingers(
    landmarks: tuple[tuple[float, float] | None, ...],
) -> int:
    if len(landmarks) < 21 or landmarks[0] is None:
        return 0

    wrist = landmarks[0]
    finger_joints = (
        (4, 3),
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    )
    count = 0

    for tip_index, joint_index in finger_joints:
        tip = landmarks[tip_index]
        joint = landmarks[joint_index]

        if tip is None or joint is None:
            continue

        tip_distance = point_distance(tip, wrist)
        joint_distance = point_distance(joint, wrist)

        if tip_distance > joint_distance * 1.08:
            count += 1

    return count


def build_hand_rois(
    body_parts: tuple[BodyPart, ...],
    frame_width: int,
    frame_height: int,
    max_hands: int,
    roi_scale: float,
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    parts_by_index = {part.index: part for part in body_parts}
    hand_specs = (
        ("droite", 4, 3),
        ("gauche", 7, 6),
    )
    rois: list[tuple[str, tuple[int, int, int, int]]] = []
    min_size = max(72, int(min(frame_width, frame_height) * 0.12))
    max_size = max(min_size, int(min(frame_width, frame_height) * 0.50))

    for handedness, wrist_index, elbow_index in hand_specs:
        wrist = parts_by_index.get(wrist_index)

        if wrist is None:
            continue

        wrist_x, wrist_y = wrist.point
        elbow = parts_by_index.get(elbow_index)

        if elbow is not None:
            elbow_x, elbow_y = elbow.point
            vector_x = wrist_x - elbow_x
            vector_y = wrist_y - elbow_y
            forearm_length = float(np.hypot(vector_x, vector_y))
            size = int(max(min_size, min(max_size, forearm_length * roi_scale)))
            center_x = wrist_x + vector_x * 0.30
            center_y = wrist_y + vector_y * 0.30
        else:
            estimated_size = min(frame_width, frame_height) * 0.22
            size = int(max(min_size, min(max_size, estimated_size)))
            center_x = float(wrist_x)
            center_y = float(wrist_y)

        box = square_box_around_point(
            center_x,
            center_y,
            size,
            frame_width,
            frame_height,
        )

        if box is not None:
            rois.append((handedness, box))

        if len(rois) >= max_hands:
            break

    if not rois and frame_width > 0 and frame_height > 0 and max_hands > 0:
        rois.append(("main", (0, 0, frame_width, frame_height)))

    return tuple(rois[:max_hands])


def square_box_around_point(
    center_x: float,
    center_y: float,
    size: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    half_size = max(1, size) // 2
    x1 = int(round(center_x - half_size))
    y1 = int(round(center_y - half_size))
    x2 = int(round(center_x + half_size))
    y2 = int(round(center_y + half_size))
    return clamp_box(x1, y1, x2, y2, frame_width, frame_height)


def parse_openpose_hand_output(
    output: np.ndarray,
    roi: tuple[int, int, int, int],
    confidence_threshold: float,
    handedness: str,
) -> DetectedHand | None:
    if output.ndim != 4 or output.shape[1] < 21:
        return None

    roi_x, roi_y, roi_width, roi_height = roi
    heatmap_height, heatmap_width = output.shape[2:4]
    landmarks: list[tuple[int, int] | None] = []
    scores: list[float] = []

    for part_index in range(21):
        prob_map = output[0, part_index]
        _, confidence, _, max_location = cv2.minMaxLoc(prob_map)

        if confidence <= confidence_threshold:
            landmarks.append(None)
            continue

        point_x = roi_x + int((max_location[0] / heatmap_width) * roi_width)
        point_y = roi_y + int((max_location[1] / heatmap_height) * roi_height)
        landmarks.append((point_x, point_y))
        scores.append(float(confidence))

    visible_points = [point for point in landmarks if point is not None]

    if len(visible_points) < 4:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    box = (
        min(xs),
        min(ys),
        max(1, max(xs) - min(xs)),
        max(1, max(ys) - min(ys)),
    )

    return DetectedHand(
        handedness=handedness,
        confidence=float(np.mean(scores)) if scores else 0.0,
        finger_count=count_visible_fingers(tuple(landmarks)),
        landmarks=tuple(landmarks),
        box=box,
    )


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(0, min(frame_width - 1, int(x2)))
    y2 = max(0, min(frame_height - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def blur_region(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    face = frame[y : y + h, x : x + w]

    if face.size == 0:
        return

    kernel = max(31, (min(w, h) // 2) | 1)
    frame[y : y + h, x : x + w] = cv2.GaussianBlur(face, (kernel, kernel), 0)


def draw_text_lines(frame: np.ndarray, lines: tuple[str, ...]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26


def draw_box_label(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    text_size, _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_width, label_height = text_size
    label_y = max(0, y - label_height - 8)

    cv2.rectangle(
        frame,
        (x, label_y),
        (x + label_width + 10, label_y + label_height + 8),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 5, label_y + label_height + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def format_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return ""

    if distance_m >= 10:
        return f" ~{distance_m:.0f}m"

    return f" ~{distance_m:.1f}m"


def format_detected_object(
    detected: DetectedObject,
    show_distances: bool,
) -> str:
    distance = format_distance(detected.distance_m) if show_distances else ""
    return f"{detected.label} {detected.confidence:.0%}{distance}"


def format_face_label(
    appearance: FaceAppearance | None,
    show_face_appearance: bool,
) -> str:
    if appearance is None or not show_face_appearance:
        return "visage"

    return f"visage | cheveux: {appearance.hair_color} | yeux: {appearance.eye_color}"


def format_face_appearance_summary(
    appearances: tuple[FaceAppearance, ...],
    show_face_appearance: bool = True,
) -> str:
    if not show_face_appearance or not appearances:
        return "aucun"

    return ", ".join(
        f"cheveux={appearance.hair_color} yeux={appearance.eye_color}"
        for appearance in appearances[:4]
    )


def draw_pose(frame: np.ndarray, body_parts: tuple[BodyPart, ...]) -> None:
    points_by_index = {part.index: part.point for part in body_parts}
    color = (249, 115, 22)

    for start, end in COCO_POSE_PAIRS:
        if start not in points_by_index or end not in points_by_index:
            continue

        cv2.line(frame, points_by_index[start], points_by_index[end], color, 2)

    for part in body_parts:
        cv2.circle(frame, part.point, 5, color, -1)
        cv2.putText(
            frame,
            part.label,
            (part.point[0] + 6, part.point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            part.label,
            (part.point[0] + 6, part.point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_hands(frame: np.ndarray, hands: tuple[DetectedHand, ...]) -> None:
    color = (236, 72, 153)

    for hand in hands:
        for start, end in HAND_CONNECTIONS:
            if start >= len(hand.landmarks) or end >= len(hand.landmarks):
                continue

            start_point = hand.landmarks[start]
            end_point = hand.landmarks[end]

            if start_point is None or end_point is None:
                continue

            cv2.line(frame, start_point, end_point, color, 2)

        for point in hand.landmarks:
            if point is None:
                continue

            cv2.circle(frame, point, 3, color, -1)

        hand_name = (
            f"main {hand.handedness}"
            if hand.handedness in {"gauche", "droite"}
            else "main"
        )
        label = f"{hand_name} | doigts: {hand.finger_count}"
        draw_box_label(frame, hand.box, label, color)


def draw_overlay(
    frame: np.ndarray,
    current_mode: str,
    source_name: str,
    analysis: FrameAnalysis | None,
    objects: tuple[DetectedObject, ...],
    body_parts: tuple[BodyPart, ...],
    hands: tuple[DetectedHand, ...],
    show_hands: bool,
    blur_faces: bool,
    show_help: bool,
    show_distances: bool,
    show_face_appearance: bool,
    show_eye_debug: bool,
) -> None:
    lines = [
        f"Source: {source_name}",
        f"Mode: {mode_label(current_mode)}",
        f"Visages: {len(analysis.faces) if analysis else 0}",
        f"Objets: {len(objects)}",
        f"Corps: {len(body_parts)} points",
        (
            f"Mains: {len(hands)} | doigts: {sum(hand.finger_count for hand in hands)}"
            if show_hands
            else "Mains: OFF"
        ),
    ]

    if analysis is not None:
        appearances_by_box = {
            appearance.box: appearance for appearance in analysis.appearances
        }

        for box in analysis.faces:
            if blur_faces:
                blur_region(frame, box)

            draw_box_label(
                frame,
                box,
                format_face_label(appearances_by_box.get(box), show_face_appearance),
                (34, 197, 94),
            )

        if show_eye_debug:
            for appearance in analysis.appearances:
                for sample_box in appearance.eye_sample_boxes:
                    x, y, width, height = sample_box
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (255, 255, 0),
                        1,
                    )

        lines.extend(
            (
                f"Luminosite: {analysis.brightness:5.1f}/255",
                f"Nettete: {analysis.sharpness:7.1f}",
                f"Apparence visage: {'ON' if show_face_appearance else 'OFF'}",
                f"Debug yeux: {'ON' if show_eye_debug else 'OFF'}",
                f"Floutage: {'ON' if blur_faces else 'OFF'}",
            )
        )

    draw_pose(frame, body_parts)
    draw_hands(frame, hands)

    for detected in objects:
        label = format_detected_object(detected, show_distances)
        draw_box_label(frame, detected.box, label, (59, 130, 246))

    if show_help:
        lines.extend(
            (
                "m mode suivant | 1 visages | 2 objets | 3 corps",
                "4 visages+objets | 5 objets+corps | 6 tout",
                "h aide | g mains | a apparence | e yeux | d distances | b flouter | q/Esc",
            )
        )
    else:
        lines.append("h aide")

    draw_text_lines(frame, tuple(lines))


def save_snapshot(frame: np.ndarray) -> Path:
    snapshot_dir = Path("snapshots")
    snapshot_dir.mkdir(exist_ok=True)

    filename = f"snapshot-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
    path = snapshot_dir / filename
    cv2.imwrite(str(path), frame)
    return path


def download_file(
    urls: str | tuple[str, ...],
    target_path: Path,
    force: bool = False,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        print(f"Deja present: {target_path}")
        return

    url_options = (urls,) if isinstance(urls, str) else urls
    last_error: Exception | None = None

    for url in url_options:
        try:
            print(f"Telechargement: {url}")
            urllib.request.urlretrieve(url, target_path)
            print(f"Fichier pret: {target_path}")
            return
        except (OSError, URLError) as error:
            last_error = error
            if target_path.exists():
                target_path.unlink()
            print(f"Echec du telechargement depuis cette source: {error}")

    raise RuntimeError(f"Telechargement impossible pour {target_path}") from last_error


def download_mobilenet_model(prototxt_path: Path, model_path: Path) -> None:
    download_file(MOBILENET_PROTOTXT_URL, prototxt_path)
    download_file(MOBILENET_MODEL_URL, model_path)


def download_yolo_model(cfg_path: Path, weights_path: Path, names_path: Path) -> None:
    download_file(YOLO_CFG_URL, cfg_path)
    download_file(YOLO_WEIGHTS_URL, weights_path)
    download_file(YOLO_NAMES_URL, names_path)


def download_object_model(args: argparse.Namespace) -> None:
    if args.object_detector == "mobilenet":
        download_mobilenet_model(args.object_prototxt, args.object_model)
        return

    download_yolo_model(args.yolo_cfg, args.yolo_weights, args.yolo_names)


def download_pose_model(prototxt_path: Path, model_path: Path) -> None:
    download_file(POSE_PROTOTXT_URL, prototxt_path)
    download_file(POSE_MODEL_URLS, model_path)


def download_hand_model(prototxt_path: Path, model_path: Path) -> None:
    download_file(HAND_PROTOTXT_URL, prototxt_path)
    download_file(HAND_MODEL_URLS, model_path)


def format_object_summary(
    objects: tuple[DetectedObject, ...],
    show_distances: bool = True,
) -> str:
    if not objects:
        return "aucun"

    return ", ".join(
        format_detected_object(detected, show_distances) for detected in objects[:8]
    )


def format_pose_summary(body_parts: tuple[BodyPart, ...]) -> str:
    if not body_parts:
        return "aucun"

    return ", ".join(part.label for part in body_parts[:8])


def format_hand_summary(hands: tuple[DetectedHand, ...]) -> str:
    if not hands:
        return "aucune"

    return ", ".join(
        f"{hand.handedness}:{hand.finger_count}" for hand in hands[:4]
    )


def run(args: argparse.Namespace) -> int:
    if args.list_windows:
        print_window_list()
        return 0

    if args.download_all_models:
        download_yolo_model(args.yolo_cfg, args.yolo_weights, args.yolo_names)
        download_pose_model(args.pose_prototxt, args.pose_model)
        download_hand_model(args.hand_prototxt, args.hand_model)
        return 0

    if args.download_object_model:
        download_object_model(args)
        return 0

    if args.download_pose_model:
        download_pose_model(args.pose_prototxt, args.pose_model)
        return 0

    if args.download_hand_model:
        download_hand_model(args.hand_prototxt, args.hand_model)
        return 0

    current_mode = args.mode
    face_detector = None
    eye_detector = None
    object_detector = None
    pose_detector = None
    hand_detector = None
    frame_source = open_frame_source(args)
    blur_faces = True
    show_help = True
    show_distances = args.distance_estimates
    show_face_appearance = args.face_appearance
    show_eye_debug = args.eye_debug
    show_hands = args.hands
    processed_frames = 0

    try:
        while True:
            ok, frame = frame_source.read()
            if not ok or frame is None:
                print("Source terminee ou image indisponible.")
                return 0 if frame_source.ends_cleanly else 1

            if mode_uses_faces(current_mode) and face_detector is None:
                face_detector = load_face_detector()

            if (
                mode_uses_faces(current_mode)
                and show_face_appearance
                and eye_detector is None
            ):
                eye_detector = load_eye_detector()

            if mode_uses_objects(current_mode) and object_detector is None:
                object_detector = load_object_detector(args)

            if mode_uses_pose(current_mode) and pose_detector is None:
                pose_detector = PoseDetector(
                    args.pose_prototxt,
                    args.pose_model,
                    max(0.0, min(args.pose_confidence, 1.0)),
                    args.pose_height,
                )

            if (
                show_hands
                and mode_uses_hands(current_mode)
                and hand_detector is None
            ):
                hand_detector = load_hand_detector(args)

            analysis = (
                analyze_frame(
                    frame,
                    face_detector,
                    args.scale,
                    eye_detector=eye_detector,
                    estimate_appearance=show_face_appearance,
                )
                if mode_uses_faces(current_mode) and face_detector is not None
                else None
            )
            objects = (
                object_detector.detect(frame)
                if mode_uses_objects(current_mode) and object_detector is not None
                else ()
            )
            body_parts = (
                pose_detector.detect(frame)
                if mode_uses_pose(current_mode) and pose_detector is not None
                else ()
            )
            hands = (
                hand_detector.detect(frame, body_parts)
                if show_hands
                and mode_uses_hands(current_mode)
                and hand_detector is not None
                else ()
            )
            processed_frames += 1

            if args.no_window:
                face_count = len(analysis.faces) if analysis else 0
                brightness = (
                    f" luminosite={analysis.brightness:.1f}" if analysis else ""
                )
                sharpness = f" nettete={analysis.sharpness:.1f}" if analysis else ""
                print(
                    f"source={frame_source.name} "
                    f"mode={mode_label(current_mode)} "
                    f"visages={face_count} "
                    f"apparence={format_face_appearance_summary(analysis.appearances if analysis else (), show_face_appearance)} "
                    f"objets={format_object_summary(objects, show_distances)} "
                    f"corps={format_pose_summary(body_parts)} "
                    f"mains={format_hand_summary(hands)}"
                    f"{brightness}{sharpness}"
                )
                if args.frames and processed_frames >= args.frames:
                    return 0

                if not args.frames and frame_source.is_static:
                    return 0
                time.sleep(0.5)
                continue

            draw_overlay(
                frame,
                current_mode,
                frame_source.name,
                analysis,
                objects,
                body_parts,
                hands,
                show_hands,
                blur_faces,
                show_help,
                show_distances,
                show_face_appearance,
                show_eye_debug,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(frame_source.wait_ms) & 0xFF

            if key in (ord("q"), 27):
                return 0

            if key == ord("h"):
                show_help = not show_help

            if key == ord("d"):
                show_distances = not show_distances

            if key == ord("a"):
                show_face_appearance = not show_face_appearance

            if key == ord("e"):
                show_eye_debug = not show_eye_debug

            if key == ord("g"):
                show_hands = not show_hands

            if key == ord("m"):
                current_mode = next_mode(current_mode)
                print(f"Mode: {mode_label(current_mode)}")

            selected_mode = mode_from_key(key)
            if selected_mode is not None:
                current_mode = selected_mode
                print(f"Mode: {mode_label(current_mode)}")

            if key == ord("b"):
                blur_faces = not blur_faces

            if key == ord("s"):
                snapshot_path = save_snapshot(frame)
                print(f"Capture sauvegardee: {snapshot_path}")

            if args.frames and processed_frames >= args.frames:
                return 0
    finally:
        if hand_detector is not None:
            hand_detector.close()
        frame_source.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()

    try:
        raise SystemExit(run(args))
    except RuntimeError as error:
        print(f"Erreur: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
