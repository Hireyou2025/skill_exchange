class EventReasoner:
    def __init__(self):
        pass

    @staticmethod
    def calculate_inference_confidence(environmental_metrics):
        """
        Calculates the inference confidence (0.0 to 1.0) based on environmental quality.
        Environmental metrics include:
        - lighting_ok (bool)
        - blur_ok (bool)
        - landmarks_stable (bool)
        - face_visible (bool)
        - roi_stable (bool)
        """
        confidence = 1.0
        reasons = []

        if not environmental_metrics.get("lighting_ok", True):
            confidence -= 0.15
            reasons.append("low lighting")
        if not environmental_metrics.get("blur_ok", True):
            confidence -= 0.15
            reasons.append("motion blur")
        if not environmental_metrics.get("landmarks_stable", True):
            confidence -= 0.20
            reasons.append("unstable landmarks")
        if not environmental_metrics.get("face_visible", True):
            confidence -= 0.20
            reasons.append("partial face visibility")
        if not environmental_metrics.get("roi_stable", True):
            confidence -= 0.10
            reasons.append("ROI instability")

        confidence = max(0.1, round(confidence, 2))
        return confidence, reasons

    def reason_behavioral_event(self, event_name, recent_states, environmental_metrics):
        """
        Analyzes the recent state history window to find causal factors for an event
        and returns a structured explanation with inference confidence.
        """
        base_conf, env_reasons = self.calculate_inference_confidence(environmental_metrics)
        reasons = []
        
        if not recent_states:
            return {
                "event": event_name,
                "confidence": base_conf,
                "reason": ["insufficient state history"]
            }

        avg_attention = sum(s.get("attention", 1.0) for s in recent_states) / len(recent_states)
        avg_stress = sum(s.get("stress", 0.0) for s in recent_states) / len(recent_states)
        avg_confidence = sum(s.get("confidence", 1.0) for s in recent_states) / len(recent_states)
        
        gaze_dirs = [s.get("gaze", "Center") for s in recent_states if "gaze" in s]
        head_poses = [s.get("head_pose", "Center") for s in recent_states if "head_pose" in s]
        blink_counts = [s.get("blinks", 0) for s in recent_states if "blinks" in s]
        speaking_statuses = [s.get("speaking", False) for s in recent_states if "speaking" in s]

        if event_name == "Attention Drop":
            if gaze_dirs and gaze_dirs.count("Center") / len(gaze_dirs) < 0.5:
                reasons.append("off-screen gaze")
            if head_poses and head_poses.count("Down") / len(head_poses) > 0.3:
                reasons.append("head-down posture")
            if speaking_statuses and speaking_statuses.count(True) / len(speaking_statuses) < 0.2:
                reasons.append("speech inactivity")
            if not reasons:
                reasons.append("general gaze deviation")
                
        elif event_name == "Stress Spike":
            if len(blink_counts) >= 2:
                blink_increase = blink_counts[-1] - blink_counts[0]
                if blink_increase > 5:
                    reasons.append("increased blink frequency")
            
            if speaking_statuses:
                speak_ratio = speaking_statuses.count(True) / len(speaking_statuses)
                if 0.0 < speak_ratio < 0.3:
                    reasons.append("speech hesitation")
                    
            if gaze_dirs:
                unique_gaze = len(set(gaze_dirs))
                if unique_gaze > 2:
                    reasons.append("gaze instability")
            if not reasons:
                reasons.append("micro-expression variations")
                
        elif event_name == "Communication Confidence Improved":
            if speaking_statuses and speaking_statuses.count(True) / len(speaking_statuses) > 0.5:
                reasons.append("sustained verbal response")
            if avg_attention > 0.7:
                reasons.append("centered gaze alignment")
            if not reasons:
                reasons.append("increased speech activity")

        elif event_name == "Integrity Warning":
            face_counts = [s.get("face_count", 1) for s in recent_states]
            if face_counts and max(face_counts) > 1:
                reasons.append("multiple faces detected")
            elif face_counts and min(face_counts) == 0:
                reasons.append("candidate absent from ROI")
            else:
                reasons.append("suspicious off-screen gaze patterns")
        else:
            reasons.append("unspecified behavioral signal")

        if env_reasons:
            reasons.append(f"inference adjusted due to: {', '.join(env_reasons)}")

        return {
            "event": event_name,
            "confidence": base_conf,
            "reason": reasons
        }
