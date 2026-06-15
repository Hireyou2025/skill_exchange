import time

class ExplanationEngine:
    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.gaze_center_count = 0
        self.gaze_left_count = 0
        self.gaze_right_count = 0
        
        self.pose_center_count = 0
        self.pose_down_count = 0
        self.pose_left_count = 0
        self.pose_right_count = 0
        self.pose_up_count = 0
        
        self.speaking_count = 0
        
        self.off_screen_start = None
        self.prolonged_glances = []
        
        self.head_down_start = None
        self.head_down_events = []
        
        self.active_violations = {}
        self.completed_violations = []
        
        self.low_lighting_frames = 0
        self.motion_blur_frames = 0
        self.unstable_landmarks_frames = 0
        self.roi_instability_frames = 0
        self.face_absent_frames = 0

    def update_frame_stats(self, realtime_state, raw_stats, env_metrics, timestamp):
        self.total_frames += 1
        
        gaze = raw_stats.get("gaze", "Unknown")
        if gaze == "Center":
            self.gaze_center_count += 1
            if self.off_screen_start is not None:
                duration = timestamp - self.off_screen_start
                if duration >= 1.5:
                    self.prolonged_glances.append((self.off_screen_start, timestamp, duration))
                self.off_screen_start = None
        elif gaze in ["Left", "Right"]:
            if gaze == "Left":
                self.gaze_left_count += 1
            else:
                self.gaze_right_count += 1
            if self.off_screen_start is None:
                self.off_screen_start = timestamp
        
        pose = raw_stats.get("head_pose", "Unknown")
        if pose == "Center":
            self.pose_center_count += 1
            if self.head_down_start is not None:
                duration = timestamp - self.head_down_start
                if duration >= 2.0:
                    self.head_down_events.append((self.head_down_start, timestamp, duration))
                self.head_down_start = None
        elif pose == "Down":
            self.pose_down_count += 1
            if self.head_down_start is None:
                self.head_down_start = timestamp
        elif pose == "Left":
            self.pose_left_count += 1
        elif pose == "Right":
            self.pose_right_count += 1
        elif pose == "Up":
            self.pose_up_count += 1
            
        if raw_stats.get("speaking", False):
            self.speaking_count += 1
            
        if not env_metrics.get("lighting_ok", True):
            self.low_lighting_frames += 1
        if not env_metrics.get("blur_ok", True):
            self.motion_blur_frames += 1
        if not env_metrics.get("landmarks_stable", True):
            self.unstable_landmarks_frames += 1
        if not env_metrics.get("roi_stable", True):
            self.roi_instability_frames += 1
        if not env_metrics.get("face_visible", True):
            self.face_absent_frames += 1

        flags = raw_stats.get("integrity_flags", [])
        
        for flag in ["Face Absent!", "Multiple Faces Detected!"]:
            if flag in flags:
                if flag not in self.active_violations:
                    self.active_violations[flag] = timestamp
            else:
                if flag in self.active_violations:
                    start_t = self.active_violations.pop(flag)
                    duration = timestamp - start_t
                    severity = "High" if duration > 5.0 else "Medium"
                    conf = 0.9 if env_metrics.get("landmarks_stable", True) else 0.6
                    self.completed_violations.append({
                        "type": flag,
                        "start": start_t,
                        "duration": round(duration, 1),
                        "severity": severity,
                        "confidence": conf
                    })

    def end_active_violations(self, total_duration):
        for flag, start_t in list(self.active_violations.items()):
            duration = total_duration - start_t
            severity = "High" if duration > 5.0 else "Medium"
            conf = 0.8
            self.completed_violations.append({
                "type": flag,
                "start": start_t,
                "duration": round(duration, 1),
                "severity": severity,
                "confidence": conf
            })
        self.active_violations.clear()

    def get_attention_explanation(self, score, conf_of_inference):
        total = self.total_frames if self.total_frames > 0 else 1
        gaze_center_pct = int((self.gaze_center_count / total) * 100)
        
        reasons = [
            f"Sustained centered gaze for {gaze_center_pct}% of interview",
            f"{len(self.prolonged_glances)} prolonged off-screen glances detected"
        ]
        
        down_pose_pct = int((self.pose_down_count / total) * 100)
        if down_pose_pct > 10:
            reasons.append(f"Frequent downward head pose detected ({down_pose_pct}% of session)")
            
        if len(self.prolonged_glances) > 3:
            reasons.append("Attention instability observed during periods of off-screen glances")

        return {
            "score": score,
            "explanation": reasons,
            "confidence": conf_of_inference
        }

    def get_confidence_explanation(self, score, conf_of_inference):
        total = self.total_frames if self.total_frames > 0 else 1
        speak_pct = int((self.speaking_count / total) * 100)
        
        reasons = [
            f"Vocal response active for {speak_pct}% of the sessions",
        ]
        
        pose_center_pct = int((self.pose_center_count / total) * 100)
        if pose_center_pct > 70:
            reasons.append(f"Maintained stable centered posture ({pose_center_pct}%)")
        else:
            reasons.append("Frequent movement or head rotation detected")
            
        if self.low_lighting_frames / total > 0.3:
            reasons.append("Behavioral confidence inference adjusted due to low lighting")

        return {
            "score": score,
            "explanation": reasons,
            "confidence": conf_of_inference
        }

    def get_integrity_explanation(self, score, conf_of_inference):
        reasons = []
        
        if not self.completed_violations:
            reasons.append("No sustained integrity anomalies detected during session")
        else:
            for v in self.completed_violations:
                v_name = "Candidate absent from ROI" if "Absent" in v["type"] else "Multiple faces detected"
                reasons.append(
                    f"{v_name} for {v['duration']} seconds (Severity: {v['severity']}, Confidence: {v['confidence']})"
                )
                
        total = self.total_frames if self.total_frames > 0 else 1
        gaze_dev_pct = int(((self.gaze_left_count + self.gaze_right_count) / total) * 100)
        if gaze_dev_pct > 40:
            reasons.append(f"Suspicious gaze deviation frequency exceeded threshold ({gaze_dev_pct}%)")
            
        return {
            "score": score,
            "explanation": reasons,
            "confidence": conf_of_inference
        }

    def generate_recruiter_summary(self, final_score, integrity_score):
        total = self.total_frames if self.total_frames > 0 else 1
        gaze_center_pct = int((self.gaze_center_count / total) * 100)
        speak_pct = int((self.speaking_count / total) * 100)
        
        summary = []
        
        if gaze_center_pct > 75:
            summary.append("Candidate maintained strong eye contact during most responses.")
        elif gaze_center_pct > 50:
            summary.append("Candidate maintained moderate eye contact, with occasional drift.")
        else:
            summary.append("Attention instability observed; candidate frequently looked off-screen.")
            
        down_pose_pct = int((self.pose_down_count / total) * 100)
        if down_pose_pct > 25:
            summary.append("Slight attention drift or head-down posture observed (likely during coding or screen reading).")
            
        if self.unstable_landmarks_frames / total > 0.4:
            summary.append("Behavioral confidence decreased temporarily due to landmark tracking instability.")
        else:
            summary.append("Stress indicators remained within standard baseline limits for most of the interview.")

        if speak_pct > 30:
            summary.append("Communication confidence improved and was well-sustained over the interview duration.")
        else:
            summary.append("Brief vocal responses recorded; communication was mostly passive.")

        if integrity_score > 90:
            summary.append("No major integrity violations detected.")
        else:
            summary.append("Sustained integrity anomalies were recorded; review flags on the session timeline.")
            
        return summary
