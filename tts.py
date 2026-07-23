import threading
import time
import queue

import pyttsx3


class TTSManager:
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self.last_spoken_time = 0.0
        self.last_spoken_key = ""
        self.state_lock = threading.Lock()
        
        # Use a queue and a dedicated thread for pyttsx3
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Dedicated thread for pyttsx3 to avoid threading issues."""
        try:
            engine = pyttsx3.init()
            while True:
                text = self.queue.get()
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
        except Exception as e:
            print(f"TTS Worker Error: {e}")

    def speak_async(self, text: str):
        """Add text to the speech queue."""
        self.queue.put(text)

    def try_speak(self, text: str, key: str) -> bool:
        """Cooldown-protected speaker trigger."""
        now = time.time()
        should_speak = False

        with self.state_lock:
            if (now - self.last_spoken_time) >= self.cooldown_seconds or key != self.last_spoken_key:
                self.last_spoken_time = now
                self.last_spoken_key = key
                should_speak = True

        if should_speak:
            self.speak_async(text)
        return should_speak
