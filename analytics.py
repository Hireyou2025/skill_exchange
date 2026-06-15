import time
from collections import deque
from event_reasoner import EventReasoner

class TimelineAnalytics:
    def __init__(self):
        self.events = []
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        self.low_attention_start = None
        self.high_stress_start = None
        self.high_comm_start = None
        self.integrity_warning_start = {}

        self.reasoner = EventReasoner()
        self.state_history = deque(maxlen=200)

    def process(self, realtime_state, scores, integrity_flags, environmental_metrics=None, raw_stats=None):
        current_time = time.time()
        
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

        state_snapshot = {
            **realtime_state,
            **raw_stats,
            "face_count": 0 if "Face Absent!" in integrity_flags else (2 if "Multiple Faces Detected!" in integrity_flags else 1)
        }
        self.state_history.append(state_snapshot)
        
        if realtime_state.get("attention", 1.0) < 0.5:
            if self.low_attention_start is None:
                self.low_attention_start = current_time
            elif current_time - self.low_attention_start > 3.0:
                reason_data = self.reasoner.reason_behavioral_event("Attention Drop", list(self.state_history), environmental_metrics)
                self.log_event("Behavior", "Attention Drop", reason_data)
                self.low_attention_start = None
        else:
            self.low_attention_start = None
            
        if realtime_state.get("stress", 0.0) > 0.65:
            if self.high_stress_start is None:
                self.high_stress_start = current_time
            elif current_time - self.high_stress_start > 2.0:
                reason_data = self.reasoner.reason_behavioral_event("Stress Spike", list(self.state_history), environmental_metrics)
                self.log_event("Behavior", "Stress Spike", reason_data)
                self.high_stress_start = None
        else:
            self.high_stress_start = None

        if realtime_state.get("confidence", 0.0) > 0.7 and raw_stats.get("speaking", False):
            if self.high_comm_start is None:
                self.high_comm_start = current_time
            elif current_time - self.high_comm_start > 2.0:
                reason_data = self.reasoner.reason_behavioral_event("Communication Confidence Improved", list(self.state_history), environmental_metrics)
                self.log_event("Behavior", "Communication Confidence Improved", reason_data)
                self.high_comm_start = None
        else:
            self.high_comm_start = None
            
        if integrity_flags:
            for flag in integrity_flags:
                if flag not in self.integrity_warning_start:
                    self.integrity_warning_start[flag] = current_time
                elif current_time - self.integrity_warning_start[flag] > 2.0:
                    reason_data = self.reasoner.reason_behavioral_event("Integrity Warning", list(self.state_history), environmental_metrics)
                    clean_flag = "Absent from ROI" if "Absent" in flag else "Multiple faces detected"
                    reason_data["reason"] = [clean_flag] + reason_data["reason"]
                    self.log_event("Integrity", flag, reason_data)
                    self.integrity_warning_start[flag] = current_time
        else:
            self.integrity_warning_start.clear()
            
    def log_event(self, event_type, event_name, reason_data):
        current_time = time.time()
        
        recent_identical = [
            e for e in self.events 
            if e["name"] == event_name and (current_time - self.start_time) - e["timestamp"] < 5.0
        ]
        if recent_identical:
            return
            
        self.last_log_time = current_time
        timestamp = current_time - self.start_time
        
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        self.events.append({
            "timestamp": timestamp,
            "time_str": time_str,
            "type": event_type,
            "name": event_name,
            "reason_data": reason_data
        })
        
        reasons_str = ", ".join(reason_data.get("reason", []))
        print(f"[Timeline] {time_str} -> {event_type}: {event_name} (Conf: {reason_data.get('confidence', 1.0)}) [Details: {reasons_str}]")
        
    def get_summary(self):
        return f"Total Events Logged: {len(self.events)}"
