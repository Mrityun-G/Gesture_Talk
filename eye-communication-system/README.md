# Eye Communication System

This is a no-hand communication prototype for people who can use eye movement
but cannot use hand gestures.

## Run

Open this file in a browser:

```text
eye-communication-system/index.html
```

Allow camera permission when the browser asks.

## How It Works

1. The app starts the webcam gaze tracker.
2. Press **Auto calibrate** once.
3. The user looks at each highlighted dot.
4. The system records gaze samples automatically.
5. The user selects letters, words, and commands by looking at a tile.
6. The selected message can be spoken with the browser text-to-speech engine.

## Data

The system stores automatic gaze samples in browser `localStorage`. Use
**Export gaze data** to download the collected samples as:

```text
eye-gaze-training-data.csv
```

This gives you project data without manually creating a separate training
dataset.

## Important Note

This prototype uses WebGazer.js from a CDN, so the first run needs internet
access. For a final offline version, download the WebGazer library into this
folder and update `index.html` to load the local file.
