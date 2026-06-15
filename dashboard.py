import cv2
import numpy as np

class RecruiterDashboard:
    def __init__(self, width=1000, height=600):
        self.width = width
        self.height = height

    def render(self, frame, stats):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        f_h, f_w = frame.shape[:2]
        target_w = int(self.width * 0.55)
        target_h = int(f_h * (target_w / f_w))
        
        if target_h > self.height:
            target_h = self.height
            target_w = int(f_w * (target_h / f_h))
            
        resized_frame = cv2.resize(frame, (target_w, target_h))
        
        y_start = (self.height - target_h) // 2
        canvas[y_start:y_start+target_h, 0:target_w] = resized_frame
        
        dashboard_x = target_w + 20
        y_offset = 35
        
        cv2.putText(canvas, "LIVE BEHAVIORAL INTELLIGENCE", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35
        
        realtime = stats.get("realtime", {})
        scores = stats.get("scores", {})
        env = stats.get("environmental_metrics", {})
        raw = stats.get("raw", {})
        
        cv2.putText(canvas, "SCORING & INFERENCE", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 2)
        y_offset += 25
        
        session_score = scores.get("session_score", 0)
        inf_conf = scores.get("inference_confidence", 1.0)
        
        conf_color = (0, 255, 0) if inf_conf > 0.8 else (0, 200, 255) if inf_conf > 0.5 else (0, 0, 255)
        cv2.putText(canvas, f"Session Score: {session_score}%", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, f"Inference Conf: {int(inf_conf * 100)}%", (dashboard_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 2)
        y_offset += 25
        
        instant_score = scores.get('instant_score', 0)
        score_color = (0, 255, 0) if instant_score > 70 else (0, 255, 255) if instant_score > 40 else (0, 0, 255)
        feeling = realtime.get("feeling", "Neutral")
        cv2.putText(canvas, f"Instant: {instant_score}", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)
        cv2.putText(canvas, f"Feeling: {feeling}", (dashboard_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 35
        
        cv2.putText(canvas, "VISUAL QUALITY ENGINE", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 2)
        y_offset += 25
        
        lit_str = "OK" if env.get("lighting_ok", True) else "Low"
        lit_color = (0, 255, 0) if env.get("lighting_ok", True) else (0, 0, 255)
        
        blur_str = "Clear" if env.get("blur_ok", True) else "Blurry"
        blur_color = (0, 255, 0) if env.get("blur_ok", True) else (0, 0, 255)
        
        jtr_str = "Stable" if env.get("landmarks_stable", True) else "Jitter"
        jtr_color = (0, 255, 0) if env.get("landmarks_stable", True) else (0, 0, 255)
        
        roi_str = "Stable" if env.get("roi_stable", True) else "Unstable"
        roi_color = (0, 255, 0) if env.get("roi_stable", True) else (0, 0, 255)

        cv2.putText(canvas, f"Lighting: ", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, lit_str, (dashboard_x + 80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lit_color, 2)
        
        cv2.putText(canvas, f"Blur: ", (dashboard_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, blur_str, (dashboard_x + 230, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blur_color, 2)
        y_offset += 20
        
        cv2.putText(canvas, f"Landmark: ", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, jtr_str, (dashboard_x + 80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, jtr_color, 2)
        
        cv2.putText(canvas, f"ROI: ", (dashboard_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, roi_str, (dashboard_x + 230, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
        y_offset += 35

        cv2.putText(canvas, "SCORING TRACEBACK CONTRIBUTORS", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 2)
        y_offset += 25
        
        contributors = scores.get("contributors", {})
        for factor, val in contributors.items():
            sign = "+" if val >= 0 else ""
            color = (120, 255, 120) if val >= 0 else (120, 120, 255)
            cv2.putText(canvas, f"- {factor}:", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(canvas, f"{sign}{val}", (dashboard_x + 250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            y_offset += 20
        
        y_offset += 15
        
        cv2.putText(canvas, "LIVE ALERTS & STATE", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 2)
        y_offset += 25
        
        cv2.putText(canvas, f"Gaze: {raw.get('gaze', 'Unknown')}", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"Pose: {raw.get('head_pose', 'Unknown')}", (dashboard_x + 180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        flags = raw.get("integrity_flags", [])
        if len(flags) == 0:
            cv2.putText(canvas, "Security Status: Secure", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            for flag in flags[:2]:
                cv2.putText(canvas, f"- WARNING: {flag}", (dashboard_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 20
                
        cv2.putText(canvas, "Press 'q' to End Session & Generate Report", (dashboard_x, self.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
                
        return canvas
