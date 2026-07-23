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
}

async function trainGesture() {
  setError('');
  const gestureName = prompt('Name this gesture, for example: Need Water');
  if (!gestureName || !gestureName.trim()) return;

  try {
    const res = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gesture_name: gestureName.trim() }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Training failed');
    trainingStatus.textContent = `Saved ${data.gesture_name}`;
    await refreshState();
  } catch (err) {
    setError(err.message);
  }
}

trainBtn.addEventListener('click', trainGesture);
autoTrainBtn.addEventListener('click', () => setAutoTraining(!autoTrainingEnabled));
contextButtons.forEach((btn) => {
  btn.addEventListener('click', () => setContext(btn.dataset.context));
});

refreshState();
setInterval(refreshState, 350);
