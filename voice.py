import sounddevice as sd
import numpy as np

class AudioAnalyzer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.energy_threshold = 0.015 
        self.is_speaking = False
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        energy = np.sqrt(np.mean(indata**2))
        self.is_speaking = energy > self.energy_threshold

    def start(self):
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                callback=self._audio_callback
            )
            self.stream.start()
            print("[Audio] Stream started.")
        except Exception as e:
            print(f"[Audio] Warning: Could not start audio stream: {e}")

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                print("[Audio] Stream stopped.")
            except Exception:
                pass

    def get_speaking_status(self):
        return self.is_speaking
