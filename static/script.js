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
# Note: The provided file is a JavaScript file, but the language specified is Python. 
# The following code is the refactored version in Python.

import asyncio
import aiohttp
import json

class GestureTrainer:
    def __init__(self, training_status_element, error_element, train_button):
        # Initialize the trainer with the necessary elements
        self.training_status_element = training_status_element
        self.error_element = error_element
        self.train_button = train_button
        self.train_button.click_event = self.train_gesture

    async def get_gesture_name(self):
        # Use a separate method to get the gesture name for better readability
        gesture_name = input('Name this gesture, for example: Need Water')
        return gesture_name.strip()

    async def train_gesture(self):
        # Clear any previous error messages
        self.error_element.text = ''
        
        # Get the gesture name from the user
        gesture_name = await self.get_gesture_name()
        if not gesture_name:
            return  # Return early if the gesture name is empty

        try:
            # Use aiohttp for asynchronous HTTP requests
            async with aiohttp.ClientSession() as session:
                async with session.post('/train', 
                                        json={'gesture_name': gesture_name}) as response:
                    # Check if the response was successful
                    if response.status != 200:
                        raise Exception(await response.text())
                    
                    # Parse the response as JSON
                    data = await response.json()
                    # Update the training status
                    self.training_status_element.text = f'Saved {data["gesture_name"]}'
                    # Refresh the state
                    await self.refresh_state()
        except Exception as e:
            # Handle any exceptions that occur during the training process
            self.error_element.text = str(e)

    async def refresh_state(self):
        # This method is not implemented in the original code, so it's left as is
        pass

# Usage
# trainer = GestureTrainer(training_status_element, error_element, train_button)autoTrainBtn.addEventListener('click', () => setAutoTraining(!autoTrainingEnabled));
contextButtons.forEach((btn) => {
  btn.addEventListener('click', () => setContext(btn.dataset.context));
});

refreshState();
setInterval(refreshState, 350);
