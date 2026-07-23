import threading
import time
from typing import Dict

from flask import Flask, Response, jsonify, render_template, request

from camera import CameraProcessor
from config import CONTEXTS, CONTEXT_PREFIX, DEFAULT_CONTEXT, SPEECH_COOLDOWN_SECONDS
from gestures import GestureStore
from tts import TTSManager

app = Flask(__name__)

store = GestureStore()
camera = CameraProcessor(store)
tts = TTSManager(cooldown_seconds=SPEECH_COOLDOWN_SECONDS)

state_lock = threading.Lock()
state: Dict[str, str] = {
    "context": DEFAULT_CONTEXT,
    "gesture": "none",
    "eye_movement": "center",
    "phrase": "Waiting for gesture...",
    "eye_phrase": "Looking center",
}


def phrase_for(gesture_name: str, context_name: str) -> str:
    gesture_name = gesture_name.lower()

    context_map = {
        "cafe": {
            "need water": "I would like some water",
        },
        "hospital": {
            "need water": "I need water urgently",
        },
        "college": {
            "need water": "Can I get some water?",
        },
        "bank": {
            "need water": "Can I have some water?",
        },
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

def recognition_state_loop():
    while True:
        try:
            gesture_name, _ = camera.get_recognition()
            eye_movement, eye_phrase = camera.get_eye_movement()

            with state_lock:
                context_name = state["context"]
                if gesture_name == "none":
                    phrase = eye_phrase if eye_movement not in ("center", "not detected") else "Waiting for gesture..."
                else:
                    phrase = phrase_for(gesture_name, context_name)

                state["gesture"] = gesture_name
                state["eye_movement"] = eye_movement
                state["eye_phrase"] = eye_phrase
                state["phrase"] = phrase

            if gesture_name != "none":
                speech_key = f"{context_name}:{gesture_name}"
                tts.try_speak(phrase, speech_key)
            elif eye_movement not in ("center", "not detected"):
                speech_key = f"eye:{eye_movement}"
                tts.try_speak(eye_phrase, speech_key)
        except Exception as e:
            print(f"Recognition loop error: {e}")
        
        time.sleep(0.1)


@app.route("/")
def index():
    return render_template("index.html", contexts=CONTEXTS)


@app.route("/video_feed")
def video_feed():
    return Response(camera.generate_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    gesture_name = str(data.get("gesture_name", "")).strip()

    if not gesture_name:
        return jsonify({"error": "gesture_name is required"}), 400

    landmarks = camera.get_latest_landmarks()
    if landmarks is None:
        return jsonify({"error": "No hand detected. Show your hand and try again."}), 400

    store.add_gesture(gesture_name, landmarks)
    return jsonify({"message": "Gesture trained", "gesture_name": gesture_name})


@app.route("/auto_training", methods=["POST"])
def auto_training():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    camera.set_auto_training(enabled)
    return jsonify({"message": "Automatic training updated", "enabled": enabled})


@app.route("/recognize")
def recognize():
    with state_lock:
        return jsonify(state)


@app.route("/set_context", methods=["POST"])
def set_context():
    data = request.get_json(silent=True) or {}
    context_name = str(data.get("context", "")).strip().lower()

    if context_name not in CONTEXTS:
        return jsonify({"error": f"Invalid context. Choose one of: {CONTEXTS}"}), 400

    with state_lock:
        state["context"] = context_name

    return jsonify({"message": "Context updated", "context": context_name})


@app.route("/get_state")
def get_state():
    with state_lock:
        snapshot = dict(state)
    snapshot["trained_gesture_count"] = store.count()
    snapshot["trained_gestures"] = store.names()
    snapshot["auto_training_enabled"] = camera.is_auto_training_enabled()
    snapshot["auto_training_status"] = camera.get_auto_training_status()
    return jsonify(snapshot)


def start_background_threads():
    threading.Thread(target=camera.process_camera_loop, daemon=True).start()
    threading.Thread(target=recognition_state_loop, daemon=True).start()


if __name__ == "__main__":
    start_background_threads()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
