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
import asyncio
import logging

# Create a logger for error handling
logging.basicConfig(level=logging.ERROR)

async def get_recognition_data(camera):
    # Extract recognition data into a separate async function
    try:
        gesture_name, _ = await camera.get_recognition_async()
        eye_movement, eye_phrase = await camera.get_eye_movement_async()
        return gesture_name, eye_movement, eye_phrase
    except Exception as e:
        logging.error(f"Error getting recognition data: {e}")
        return None, None, None

async def update_state(state, state_lock, gesture_name, eye_movement, eye_phrase, context_name):
    # Update state in a separate async function with proper locking
    async with state_lock:
        if gesture_name == "none":
            phrase = eye_phrase if eye_movement not in ("center", "not detected") else "Waiting for gesture..."
        else:
            phrase = phrase_for(gesture_name, context_name)

        state["gesture"] = gesture_name
        state["eye_movement"] = eye_movement
        state["eye_phrase"] = eye_phrase
        state["phrase"] = phrase

async def speak(tts, phrase, speech_key):
    # Extract speech into a separate async function
    try:
        await tts.try_speak_async(phrase, speech_key)
    except Exception as e:
        logging.error(f"Error speaking: {e}")

async def recognition_state_loop(camera, state, state_lock, tts):
    # Convert the loop to an async function
    while True:
        gesture_name, eye_movement, eye_phrase = await get_recognition_data(camera)
        if gesture_name is None or eye_movement is None or eye_phrase is None:
            continue  # Skip if data is invalid

        context_name = state["context"]
        await update_state(state, state_lock, gesture_name, eye_movement, eye_phrase, context_name)

        if gesture_name != "none":
            speech_key = f"{context_name}:{gesture_name}"
            await speak(tts, state["phrase"], speech_key)
        elif eye_movement not in ("center", "not detected"):
            speech_key = f"eye:{eye_movement}"
            await speak(tts, eye_phrase, speech_key)

        await asyncio.sleep(0.1)  # Use async sleep@app.route("/")
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
