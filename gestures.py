from typing import Dict, List, Optional, Tuple

import numpy as np


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    """Normalize landmarks using wrist-origin + max-distance scaling."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = coords[0]
    centered = coords - wrist

    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist < 1e-6:
        return centered
    return centered / max_dist


def gesture_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance across all 21 landmarks."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


class GestureStore:
    def __init__(self):
        self._gestures: Dict[str, np.ndarray] = {}

    def add_gesture(self, name: str, landmarks: np.ndarray) -> None:
        self._gestures[name] = landmarks.copy()

    def count(self) -> int:
        return len(self._gestures)

    def names(self) -> List[str]:
        return sorted(self._gestures.keys())

    def match(self, live_landmarks: np.ndarray, threshold: float) -> Tuple[str, Optional[float]]:
        if not self._gestures:
            return "none", None

        best_name = "none"
        best_dist = float("inf")
        for name, saved in self._gestures.items():
            dist = gesture_distance(live_landmarks, saved)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist < threshold:
            return best_name, best_dist
        return "none", best_dist
