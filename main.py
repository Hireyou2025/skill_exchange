import threading
import time
import cv2
import numpy as np
from collections import deque
from visuals import ScreenRecogniser
from face_pipeline import DetectFace
from blink import BlinkAnalysis
from gaze import GazeTracking
from head_pose import HeadPoseEstimation
from integrity import IntegrityMonitoring
from voice import AudioAnalyzer
from emotion import EmotionInference
from analytics import TimelineAnalytics
from scoring import ScoringEngine
from dashboard import RecruiterDashboard
from report import SessionReporter

class SharedState:
    def __init__(self):
        self.frame = None
        self.landmarks = None
        self.face_count = 0
        self.lock = threading.Lock()

    def update(self, frame, landmarks, face_count):
        with self.lock:
            self.frame = frame
            self.landmarks = landmarks
            self.face_count = face_count

    def get(self):
        with self.lock:
            return self.frame, self.landmarks, self.face_count

class PipelineOrchestrator:
    def __init__(self):
        self.screen_recogniser = ScreenRecogniser()
        self.detect_face = DetectFace()
        
        self.blink_analyzer = BlinkAnalysis()
        self.gaze_tracker = GazeTracking()
        self.head_pose = HeadPoseEstimation()
        self.integrity_monitor = IntegrityMonitoring()
        self.audio_analyzer = AudioAnalyzer()
        self.emotion_inference = EmotionInference()
        self.timeline = TimelineAnalytics()
        self.scoring = ScoringEngine()
        self.dashboard = RecruiterDashboard()
        self.reporter = SessionReporter()

        self.shared_state = SharedState()
        self.running = True

        self.prev_nose_pos = None
        self.jitter_history = deque(maxlen=15)
        self.face_width_history = deque(maxlen=20)

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.analysis_thread.start()
        self.audio_analyzer.start()
        print("[System] Core capture, analysis & audio threads started.")

    def _capture_loop(self):
        while self.running:
            frame = self.screen_recogniser.capture()
            if frame is not None:
                proc_frame, landmarks, face_count = self.detect_face.process(frame)
                self.shared_state.update(proc_frame, landmarks, face_count)
            time.sleep(0.01)

    def _analysis_loop(self):
        while self.running:
            frame, landmarks, face_count = self.shared_state.get()
            if frame is not None:
                disp_frame = frame.copy()
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                lighting_ok = mean_brightness >= 55.0
                
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_ok = laplacian_var >= 90.0
                
                face_visible = face_count > 0 and landmarks is not None
                
                landmarks_stable = True
                if face_visible:
                    h_f, w_f, _ = frame.shape
                    nose_lm = landmarks.landmark[1]
                    nose_pos = (int(nose_lm.x * w_f), int(nose_lm.y * h_f))
                    if self.prev_nose_pos is not None:
                        dist = np.hypot(nose_pos[0] - self.prev_nose_pos[0], nose_pos[1] - self.prev_nose_pos[1])
                        self.jitter_history.append(dist)
                    self.prev_nose_pos = nose_pos
                    
                    if len(self.jitter_history) >= 5:
                        avg_jitter = sum(self.jitter_history) / len(self.jitter_history)
                        if avg_jitter > 6.0:
                            landmarks_stable = False
                else:
                    self.prev_nose_pos = None

                roi_stable = True
                if face_visible:
                    h_f, w_f, _ = frame.shape
                    p1 = landmarks.landmark[234]
                    p2 = landmarks.landmark[454]
                    face_w = np.hypot((p2.x - p1.x) * w_f, (p2.y - p1.y) * h_f)
                    self.face_width_history.append(face_w)
                    
                    if len(self.face_width_history) >= 10:
                        mean_w = sum(self.face_width_history) / len(self.face_width_history)
                        if mean_w > 0:
                            variance = sum((w - mean_w)**2 for w in self.face_width_history) / len(self.face_width_history)
                            std_dev = np.sqrt(variance)
                            if (std_dev / mean_w) > 0.08:
                                roi_stable = False
                
                environmental_metrics = {
                    "lighting_ok": lighting_ok,
                    "blur_ok": blur_ok,
                    "landmarks_stable": landmarks_stable,
                    "face_visible": face_visible,
                    "roi_stable": roi_stable
                }

                integrity_flags = self.integrity_monitor.process(landmarks, face_count)
                is_speaking = self.audio_analyzer.get_speaking_status()
                
                if face_visible:
                    blink_count = self.blink_analyzer.process(landmarks, disp_frame.shape)
                    gaze_dir = self.gaze_tracker.process(landmarks, disp_frame.shape)
                    head_pose = self.head_pose.process(landmarks, disp_frame.shape)

                    realtime_state = self.emotion_inference.process(blink_count, gaze_dir, head_pose, is_speaking, landmarks, disp_frame.shape)
                    self.last_valid_state = realtime_state
                    
                    stats_raw = {
                        "blinks": blink_count,
                        "gaze": gaze_dir,
                        "head_pose": head_pose,
                        "speaking": is_speaking,
                        "integrity_flags": integrity_flags
                    }
                    
                    scores = self.scoring.calculate(realtime_state, integrity_flags, environmental_metrics, stats_raw)
                    self.timeline.process(realtime_state, scores, integrity_flags, environmental_metrics, stats_raw)

                    stats = {
                        "realtime": realtime_state,
                        "scores": scores,
                        "raw": stats_raw,
                        "environmental_metrics": environmental_metrics
                    }
                    
                    canvas = self.dashboard.render(disp_frame, stats)
                else:
                    if not hasattr(self, 'last_valid_state'):
                        self.last_valid_state = {"attention": 1.0, "confidence": 1.0, "focus": 1.0, "stress": 0.0, "speech_energy": 0.0}
                    
                    realtime_state = self.last_valid_state.copy()
                    realtime_state["speech_energy"] = 1.0 if is_speaking else 0.0
                    
                    stats_raw = {
                        "blinks": 0,
                        "gaze": "Unknown",
                        "head_pose": "Unknown",
                        "speaking": is_speaking,
                        "integrity_flags": integrity_flags
                    }
                    
                    scores = self.scoring.calculate(realtime_state, integrity_flags, environmental_metrics, stats_raw)
                    self.timeline.process(realtime_state, scores, integrity_flags, environmental_metrics, stats_raw)
                    
                    stats = {
                        "realtime": realtime_state,
                        "scores": scores,
                        "raw": stats_raw,
                        "environmental_metrics": environmental_metrics
                    }
                    canvas = self.dashboard.render(disp_frame, stats)
                
                cv2.imshow("AI Interview Monitor", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break
            else:
                time.sleep(0.05)

    def stop(self):
        self.running = False
        self.audio_analyzer.stop()
        cv2.destroyAllWindows()
        print("[System] Core pipeline stopped.")

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator()
    orchestrator.start()
    
    try:
        while orchestrator.running:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()
        
    session_dur = time.time() - orchestrator.scoring.explanation_engine.start_time
    orchestrator.scoring.explanation_engine.end_active_violations(session_dur)
    orchestrator.reporter.generate_report(orchestrator.scoring, orchestrator.timeline)
