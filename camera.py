import threading
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import (
    AUTO_GESTURE_PREFIX,
    AUTO_MATCH_DISTANCE,
    AUTO_STABILITY_THRESHOLD,
    AUTO_STABLE_FRAMES,
    AUTO_TRAIN_COOLDOWN_SECONDS,
    AUTO_TRAINING_ENABLED,
    CAMERA_INDEX,
    EYE_BLINK_THRESHOLD,
    EYE_MOVEMENT_THRESHOLD,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    RECOGNITION_THRESHOLD,
)
from gestures import GestureStore, normalize_landmarks


class CameraProcessor:
    def __init__(self, gesture_store: GestureStore):
        self.gesture_store = gesture_store

        # ✅ SAFE mediapipe handling
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )

        except AttributeError:
            raise RuntimeError(
                "❌ MediaPipe 'solutions' not available.\n"
                "👉 Fix: Install Python 3.11 and reinstall mediapipe.\n"
                "This code will NOT work properly on Python 3.12."
            )

        # ✅ FIX: better webcam handling for Windows
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Unable to open webcam. Try changing CAMERA_INDEX.")

        self.lock = threading.Lock()
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_landmarks: Optional[np.ndarray] = None

        self.detected_gesture = "none"
        self.match_distance: Optional[float] = None
        self.eye_movement = "center"
        self.eye_phrase = "Looking center"
        self.auto_training_enabled = AUTO_TRAINING_ENABLED
        self.auto_training_status = "Watching for a steady hand pose"
        self._auto_candidate: Optional[np.ndarray] = None
        self._auto_stable_frames = 0
        self._auto_gesture_count = 0
        self._last_auto_train_time = 0.0

    def process_camera_loop(self):
        while True:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.hands.process(rgb)
                face_results = self.face_mesh.process(rgb)
                annotated = frame.copy()

                detected_name = "none"
                detected_landmarks = None
                detected_dist = None
                eye_movement, eye_phrase = self._detect_eye_movement(face_results, annotated)

                if results and results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    detected_landmarks = normalize_landmarks(hand_landmarks)

                    self.mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                    auto_name = self._classify_common_hand_pose(hand_landmarks)
                    match_name, match_dist = self.gesture_store.match(
                        detected_landmarks, RECOGNITION_THRESHOLD
                    )

                    detected_name = auto_name if auto_name != "none" else match_name
                    detected_dist = match_dist
                    self._auto_train_if_stable(detected_landmarks, match_name, match_dist)
                else:
                    self._auto_candidate = None
                    self._auto_stable_frames = 0
                    self.auto_training_status = "Show your hand to auto-learn a gesture"

                self._draw_status_overlay(annotated, detected_name, eye_movement)

                with self.lock:
                    self.latest_annotated_frame = annotated
                    self.latest_landmarks = detected_landmarks
                    self.detected_gesture = detected_name
                    self.match_distance = detected_dist
                    self.eye_movement = eye_movement
                    self.eye_phrase = eye_phrase

            except Exception as e:
                print(f"Camera Loop Error: {e}")

            time.sleep(0.01)

    def get_latest_landmarks(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_landmarks is None else self.latest_landmarks.copy()

    def get_recognition(self) -> Tuple[str, Optional[float]]:
        with self.lock:
            return self.detected_gesture, self.match_distance

    def get_eye_movement(self) -> Tuple[str, str]:
        with self.lock:
            return self.eye_movement, self.eye_phrase

    def get_auto_training_status(self) -> str:
        with self.lock:
            return self.auto_training_status

    def set_auto_training(self, enabled: bool) -> None:
        with self.lock:
            self.auto_training_enabled = enabled
            self.auto_training_status = (
                "Watching for a steady hand pose" if enabled else "Automatic training paused"
            )

    def is_auto_training_enabled(self) -> bool:
        with self.lock:
            return self.auto_training_enabled

    def _auto_train_if_stable(
        self,
        landmarks: np.ndarray,
        match_name: str,
        match_dist: Optional[float],
    ) -> None:
        if not self.auto_training_enabled:
            self.auto_training_status = "Automatic training paused"
            return

        if match_name != "none" and match_dist is not None and match_dist < AUTO_MATCH_DISTANCE:
            self._auto_candidate = landmarks.copy()
            self._auto_stable_frames = 0
            self.auto_training_status = f"Recognizing {match_name}"
            return

        if self._auto_candidate is None:
            self._auto_candidate = landmarks.copy()
            self._auto_stable_frames = 1
            self.auto_training_status = "Hold this gesture steady"
            return

        movement = float(np.mean(np.linalg.norm(landmarks - self._auto_candidate, axis=1)))
        if movement <= AUTO_STABILITY_THRESHOLD:
            self._auto_candidate = (self._auto_candidate + landmarks) / 2.0
            self._auto_stable_frames += 1
        else:
            self._auto_candidate = landmarks.copy()
            self._auto_stable_frames = 1

        if self._auto_stable_frames < AUTO_STABLE_FRAMES:
            self.auto_training_status = f"Learning stable pose {self._auto_stable_frames}/{AUTO_STABLE_FRAMES}"
            return

        now = time.time()
        if now - self._last_auto_train_time < AUTO_TRAIN_COOLDOWN_SECONDS:
            self.auto_training_status = "Gesture learned recently"
            return

        self._auto_gesture_count += 1
        name = f"{AUTO_GESTURE_PREFIX} {self._auto_gesture_count}"
        self.gesture_store.add_gesture(name, self._auto_candidate)
        self._last_auto_train_time = now
        self._auto_stable_frames = 0
        self.auto_training_status = f"Learned {name}"

    def _classify_common_hand_pose(self, hand_landmarks) -> str:
        lm = hand_landmarks.landmark
        fingers = {
            "thumb": lm[4].x < lm[3].x,
            "index": lm[8].y < lm[6].y,
            "middle": lm[12].y < lm[10].y,
            "ring": lm[16].y < lm[14].y,
            "pinky": lm[20].y < lm[18].y,
        }
        raised = sum(1 for is_up in fingers.values() if is_up)

        if raised >= 4:
            return "hello"
        if raised == 0:
            return "need help"
        if fingers["thumb"] and not any(fingers[name] for name in ("index", "middle", "ring", "pinky")):
            return "yes"
        if fingers["index"] and not any(fingers[name] for name in ("middle", "ring", "pinky")):
            return "need water"
        if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            return "thank you"
import numpy as np
import cv2
from typing import Tuple

class Camera:
    # ... existing code ...
    def _detect_eye_movement(self, face_results, frame: np.ndarray) -> Tuple[str, str]:
        try:
            if not face_results or not face_results.multi_face_landmarks:
                return "not detected", "Eyes not detected"
            landmarks = face_results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_corner = landmarks[33]
            right_corner = landmarks[133]
            upper_lid = landmarks[159]
            lower_lid = landmarks[145]
            iris_points = landmarks[468:473] if len(landmarks) > 472 else []
            self._draw_eye_landmarks(frame, landmarks, w, h)
            blink_ratio = self._calculate_blink_ratio(landmarks, w, h)
            if blink_ratio < EYE_BLINK_THRESHOLD:
                return "blink", "Blink detected"
            iris_x, iris_y = self._calculate_iris_position(landmarks, iris_points, w, h)
            cv2.circle(frame, (int(iris_x * w), int(iris_y * h)), 3, (34, 197, 94), -1)
            horizontal, vertical = self._calculate_eye_movement(landmarks, iris_x, iris_y, w, h)
            return self._determine_eye_movement(horizontal, vertical)
        except Exception as e:
            return "error", str(e)
    def _draw_eye_landmarks(self, frame: np.ndarray, landmarks, w: int, h: int) -> None:
        for point in (landmarks[33], landmarks[133], landmarks[159], landmarks[145]):
            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 2, (56, 189, 248), -1)
    def _calculate_blink_ratio(self, landmarks, w: int, h: int) -> float:
        left_corner = landmarks[33]
        right_corner = landmarks[133]
        upper_lid = landmarks[159]
        lower_lid = landmarks[145]
        eye_width = max(abs(right_corner.x - left_corner.x), 1e-6)
        eye_height = max(abs(lower_lid.y - upper_lid.y), 1e-6)
        return eye_height / eye_width
    def _calculate_iris_position(self, landmarks, iris_points, w: int, h: int) -> Tuple[float, float]:
        if iris_points:
            iris_x = float(np.mean([point.x for point in iris_points]))
            iris_y = float(np.mean([point.y for point in iris_points]))
        else:
            iris_x = (landmarks[33].x + landmarks[133].x) / 2.0
            iris_y = (landmarks[159].y + landmarks[145].y) / 2.0
        return iris_x, iris_y
    def _calculate_eye_movement(self, landmarks, iris_x: float, iris_y: float, w: int, h: int) -> Tuple[float, float]:
        left_corner = landmarks[33]
        right_corner = landmarks[133]
        upper_lid = landmarks[159]
        lower_lid = landmarks[145]
        eye_width = max(abs(right_corner.x - left_corner.x), 1e-6)
        eye_height = max(abs(lower_lid.y - upper_lid.y), 1e-6)
        horizontal = ((iris_x - left_corner.x) / eye_width) - 0.5
        vertical = ((iris_y - upper_lid.y) / eye_height) - 0.5
        return horizontal, vertical
    def _determine_eye_movement(self, horizontal: float, vertical: float) -> Tuple[str, str]:
        if horizontal < -EYE_MOVEMENT_THRESHOLD:
            return "look left", "Looking left"
        if horizontal > EYE_MOVEMENT_THRESHOLD:
            return "look right", "Looking right"
        if vertical < -EYE_MOVEMENT_THRESHOLD:
            return "look up", "Looking up"
        if vertical > EYE_MOVEMENT_THRESHOLD:
            return "look down", "Looking down"
        return "center", "Looking center"
    async def _draw_status_overlay(self, frame: np.ndarray, gesture: str, eye_movement: str) -> None:
        # ... existing code ...
        cv2.rectangle(frame, (12, 12), (360, 84), (15, 23, 42), -1)
        cv2.putText(frame, f"Hand: {gesture}", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (229, 231, 235), 2)
        cv2.putText(frame, f"Eyes: {eye_movement}", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (186, 230, 253), 2)

    def generate_mjpeg_stream(self):
        while True:
            with self.lock:
                frame = None if self.latest_annotated_frame is None else self.latest_annotated_frame.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )
