import time
from collections import deque
from scoring_trace import ScoringTrace
from explanation_engine import ExplanationEngine
from event_reasoner import EventReasoner

class ScoringEngine:
    def __init__(self):
        self.history = deque()
        self.session_averages = {
            "confidence": [],
            "attention": [],
            "focus": [],
            "stress": [],
            "inference_confidence": []
        }
        self.integrity_score = 100.0
        
        self.trace = ScoringTrace()
        self.explanation_engine = ExplanationEngine()

    def calculate(self, realtime_state, integrity_flags, environmental_metrics=None, raw_stats=None):
        current_time = time.time()
        self.history.append((current_time, realtime_state, integrity_flags))
        
        while self.history and current_time - self.history[0][0] > 30:
            self.history.popleft()
            
        def get_rolling_avg(seconds, key):
            vals = [state[key] for ts, state, _ in self.history if current_time - ts <= seconds]
            return sum(vals) / len(vals) if vals else realtime_state.get(key, 0)
            
        self.session_averages["confidence"].append(realtime_state["confidence"])
        self.session_averages["attention"].append(realtime_state["attention"])
        self.session_averages["focus"].append(realtime_state["focus"])
        self.session_averages["stress"].append(realtime_state["stress"])
        
        session_confidence = sum(self.session_averages["confidence"]) / len(self.session_averages["confidence"])
        session_attention = sum(self.session_averages["attention"]) / len(self.session_averages["attention"])
        session_focus = sum(self.session_averages["focus"]) / len(self.session_averages["focus"])
        
        recent_flags = [flags for ts, _, flags in self.history if current_time - ts <= 2.0]
        absent_count = sum(1 for flags in recent_flags if "Face Absent!" in flags)
        multiple_count = sum(1 for flags in recent_flags if "Multiple Faces Detected!" in flags)
        
        if len(recent_flags) > 0:
            if absent_count / len(recent_flags) > 0.5:
                self.integrity_score = max(0.0, self.integrity_score - 1.0)
            if multiple_count / len(recent_flags) > 0.5:
                self.integrity_score = max(0.0, self.integrity_score - 2.0)
            
            if absent_count == 0 and multiple_count == 0:
                self.integrity_score = min(100.0, self.integrity_score + 0.1)

        if environmental_metrics is None:
            environmental_metrics = {
                "lighting_ok": True,
                "blur_ok": True,
                "landmarks_stable": True,
                "face_visible": True,
                "roi_stable": True
            }
        
        if raw_stats is None:
            raw_stats = {
                "gaze": "Center" if realtime_state["attention"] > 0.5 else "Left",
                "head_pose": "Center" if realtime_state["focus"] > 0.5 else "Down",
                "speaking": realtime_state["speech_energy"] > 0.5,
                "integrity_flags": integrity_flags
            }

        session_ts = current_time - self.explanation_engine.start_time
        self.explanation_engine.update_frame_stats(
            realtime_state, raw_stats, environmental_metrics, session_ts
        )
        
        inference_confidence, env_reasons = EventReasoner.calculate_inference_confidence(environmental_metrics)
        self.session_averages["inference_confidence"].append(inference_confidence)
        
        total_frames = self.explanation_engine.total_frames if self.explanation_engine.total_frames > 0 else 1
        gaze_deviation_ratio = (self.explanation_engine.gaze_left_count + self.explanation_engine.gaze_right_count) / total_frames
        
        rolling_confidence = get_rolling_avg(10, "confidence")
        rolling_attention = get_rolling_avg(10, "attention")
        rolling_stress = get_rolling_avg(10, "stress")
        
        session_score, contributors = self.trace.calculate_session_trace(
            session_confidence=session_confidence,
            session_attention=session_attention,
            session_focus=session_focus,
            rolling_stress=rolling_stress,
            integrity_score=self.integrity_score,
            gaze_deviation_ratio=gaze_deviation_ratio
        )
        
        instant_score = realtime_state["confidence"] * 100 - realtime_state["stress"] * 20
        instant_score = max(0.0, min(100.0, instant_score))

        attention_exp = self.explanation_engine.get_attention_explanation(
            round(session_attention * 100, 1), inference_confidence
        )
        confidence_exp = self.explanation_engine.get_confidence_explanation(
            round(session_confidence * 100, 1), inference_confidence
        )
        integrity_exp = self.explanation_engine.get_integrity_explanation(
            round(self.integrity_score, 1), inference_confidence
        )
        
        scores = {
            "instant_score": int(instant_score),
            "rolling_confidence": round(rolling_confidence * 100, 1),
            "rolling_attention": round(rolling_attention * 100, 1),
            "session_score": int(session_score),
            "session_confidence": round(session_confidence * 100, 1),
            "session_attention": round(session_attention * 100, 1),
            "session_focus": round(session_focus * 100, 1),
            "integrity_score": round(self.integrity_score, 1),
            
            "contributors": contributors,
            "inference_confidence": inference_confidence,
            "attention_explanation": attention_exp,
            "confidence_explanation": confidence_exp,
            "integrity_explanation": integrity_exp
        }
        
        return scores
