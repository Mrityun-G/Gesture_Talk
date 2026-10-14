
import base64
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from camera import CameraProcessor
from config import CONTEXTS, DEFAULT_CONTEXT
from gestures import GestureStore

app = Flask(__name__)

store = GestureStore()
camera = CameraProcessor(store)

state_lock = threading.Lock()
frame_lock = threading.Lock()

state = {
    "context": DEFAULT_CONTEXT,
    "gesture": "none",
    "eye_movement": "center",
    "phrase": "Waiting for gesture...",
    "eye_phrase": "Looking center",
}


def phrase_for(gesture_name, context_name):
    gesture_name = gesture_name.lower()

    context_map = {
        "cafe": {"need water": "I would like some water"},
        "hospital": {"need water": "I need water urgently"},
        "college": {"need water": "Can I get some water?"},
        "bank": {"need water": "Can I have some water?"},
    }

    common_map = {
        "hello": "Hello",
        "need help": "I need help",
        "yes": "Yes",
        "thank you": "Thank you",
    }

    if gesture_name.startswith("auto gesture"):
        return f"{gesture_name.title()} detected"

    return context_map.get(context_name, {}).get(
        gesture_name,
        common_map.get(gesture_name, gesture_name),
    )


def get_state_snapshot():
    with state_lock:
        snapshot = dict(state)

    snapshot["trained_gesture_count"] = store.count()
    snapshot["trained_gestures"] = store.names()
    snapshot["auto_training_enabled"] = camera.is_auto_training_enabled()
    snapshot["auto_training_status"] = camera.get_auto_training_status()
    return snapshot


@app.route("/")
def index():
    return render_template("index.html", contexts=CONTEXTS)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "application": "GestureTalk"})


@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")

    if not isinstance(image_data, str) or "," not in image_data:
        return jsonify({"error": "A valid camera frame is required"}), 400

    try:
        raw = base64.b64decode(
            image_data.split(",", 1)[1],
            validate=True,
        )
        buffer = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode camera frame"}), 400

        frame = cv2.flip(frame, 1)

        with frame_lock:
            (
                annotated,
                landmarks,
                gesture_name,
                match_distance,
                eye_movement,
                eye_phrase,
            ) = camera.process_frame(frame)

            with camera.lock:
                camera.latest_annotated_frame = annotated
                camera.latest_landmarks = (
                    None if landmarks is None else landmarks.copy()
                )
                camera.detected_gesture = gesture_name
                camera.match_distance = match_distance
                camera.eye_movement = eye_movement
                camera.eye_phrase = eye_phrase

            with state_lock:
                context_name = state["context"]

                if gesture_name != "none":
                    phrase = phrase_for(gesture_name, context_name)
                elif eye_movement not in (
                    "center", "not detected", "error"
                ):
                    phrase = eye_phrase
                else:
                    phrase = "Waiting for gesture..."

                state.update({
                    "gesture": gesture_name,
                    "eye_movement": eye_movement,
                    "eye_phrase": eye_phrase,
                    "phrase": phrase,
                })

        return jsonify(get_state_snapshot())

    except Exception:
        app.logger.exception("Camera frame processing failed")
        return jsonify({"error": "Frame processing failed"}), 500


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    gesture_name = str(data.get("gesture_name", "")).strip()

    if not gesture_name:
        return jsonify({"error": "Gesture name is required"}), 400

    landmarks = camera.get_latest_landmarks()
    if landmarks is None:
        return jsonify({
            "error": "No hand detected. Show your hand and try again."
        }), 400

    store.add_gesture(gesture_name, landmarks)

    return jsonify({
        "message": "Gesture trained",
        "gesture_name": gesture_name,
    })


@app.route("/auto_training", methods=["POST"])
def auto_training():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))

    camera.set_auto_training(enabled)

    return jsonify({
        "message": "Automatic training updated",
        "enabled": enabled,
    })


@app.route("/recognize")
def recognize():
    return jsonify(get_state_snapshot())


@app.route("/get_state")
def get_state():
    return jsonify(get_state_snapshot())


@app.route("/set_context", methods=["POST"])
def set_context():
    data = request.get_json(silent=True) or {}
    context_name = str(data.get("context", "")).strip().lower()

    if context_name not in CONTEXTS:
        return jsonify({
            "error": f"Invalid context. Choose one of: {CONTEXTS}"
        }), 400

    with state_lock:
        state["context"] = context_name

    return jsonify({
        "message": "Context updated",
        "context": context_name,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
