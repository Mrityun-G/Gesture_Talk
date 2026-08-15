const contextValue = document.getElementById('contextValue');
const gestureValue = document.getElementById('gestureValue');
const eyeValue = document.getElementById('eyeValue');
const phraseValue = document.getElementById('phraseValue');
const learnedCount = document.getElementById('learnedCount');
const trainingStatus = document.getElementById('trainingStatus');
const gestureList = document.getElementById('gestureList');
const learningPill = document.getElementById('learningPill');
const errorBox = document.getElementById('errorBox');
const trainBtn = document.getElementById('trainBtn');
const autoTrainBtn = document.getElementById('autoTrainBtn');
const contextButtons = [...document.querySelectorAll('.btn-context')];

let autoTrainingEnabled = true;

function setError(message = '') {
  errorBox.textContent = message;
}

function setActiveContext(contextName) {
  contextButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.context === contextName);
  });
}

function renderAutoTraining(enabled) {
  autoTrainingEnabled = enabled;
  autoTrainBtn.textContent = enabled ? 'Pause Auto Training' : 'Resume Auto Training';
  autoTrainBtn.classList.toggle('paused', !enabled);
  autoTrainBtn.setAttribute('aria-pressed', String(enabled));
  learningPill.textContent = enabled ? 'Auto learning on' : 'Auto learning paused';
  learningPill.classList.toggle('paused', !enabled);
}

function renderGestureList(names = []) {
  gestureList.innerHTML = '';

  if (!names.length) {
    gestureList.textContent = 'No gestures learned yet';
    return;
  }

  names.forEach((name) => {
    const chip = document.createElement('span');
    chip.className = 'gesture-chip';
    chip.textContent = name;
    gestureList.appendChild(chip);
  });
}

async function refreshState() {
  try {
    const res = await fetch('/get_state');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to fetch state');

    contextValue.textContent = data.context;
    gestureValue.textContent = data.gesture;
    eyeValue.textContent = data.eye_movement;
    phraseValue.textContent = data.phrase;
    learnedCount.textContent = data.trained_gesture_count;
    trainingStatus.textContent = data.auto_training_status;
    renderAutoTraining(Boolean(data.auto_training_enabled));
    renderGestureList(data.trained_gestures || []);
    setActiveContext(data.context);
  } catch (err) {
    setError(err.message);
  }
}

async function setContext(contextName) {
  setError('');
  try {
    const res = await fetch('/set_context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context: contextName }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Context update failed');
    await refreshState();
  } catch (err) {
    setError(err.message);
  }
}

async function setAutoTraining(enabled) {
  setError('');
  renderAutoTraining(enabled);

  try {
    const res = await fetch('/auto_training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Auto training update failed');
    await refreshState();
  } catch (err) {
    setError(err.message);
    renderAutoTraining(!enabled);
  }
# No python code was provided, the given code is in javascript. 
# However, assuming we are rewriting it in python, here's a possible implementation:

import asyncio
import aiohttp
import json

# Define a class to encapsulate the training functionality (SOLID principle)
class GestureTrainer:
    def __init__(self, training_status_element, error_element):
        self.training_status_element = training_status_element
        self.error_element = error_element

    # Extract the prompt logic into a separate method for better readability and reusability
    async def get_gesture_name(self):
        # Since python doesn't have a built-in prompt function like javascript, 
        # we'll use a simple input for demonstration purposes
        gesture_name = input('Name this gesture, for example: Need Water')
        return gesture_name.strip()

    # Define an async method to handle the training logic
    async def train_gesture(self):
        # Clear any previous error messages
        self.error_element.clear()

        # Get the gesture name from the user
        gesture_name = await self.get_gesture_name()
        if not gesture_name:
            return

        try:
            # Use aiohttp for asynchronous http requests
            async with aiohttp.ClientSession() as session:
                async with session.post('/train', 
                                        json={'gesture_name': gesture_name}) as response:
                    # Check if the response was successful
                    if response.status != 200:
                        raise Exception(await response.text())

                    # Parse the response data
                    data = await response.json()
                    # Update the training status
                    self.training_status_element.set_text(f'Saved {data["gesture_name"]}')
                    # Refresh the state (assuming this is an async method)
                    await self.refresh_state()
        except Exception as e:
            # Handle any exceptions and update the error element
            self.error_element.set_text(str(e))

    # Define a method to refresh the state (assuming this is an async method)
    async def refresh_state(self):
        # Implementation of the refresh_state method
        pass

# Usage
trainer = GestureTrainer(training_status_element, error_element)
train_btn.click_handler = trainer.train_gestureautoTrainBtn.addEventListener('click', () => setAutoTraining(!autoTrainingEnabled));
contextButtons.forEach((btn) => {
  btn.addEventListener('click', () => setContext(btn.dataset.context));
});

refreshState();
setInterval(refreshState, 350);
