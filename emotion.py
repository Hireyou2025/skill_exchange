import math

class EmotionInference:
    def __init__(self):
        pass

    def _get_distance(self, p1, p2, width, height):
        x1, y1 = int(p1.x * width), int(p1.y * height)
        x2, y2 = int(p2.x * width), int(p2.y * height)
        return math.hypot(x2 - x1, y2 - y1)

    def process(self, blink_count, gaze_dir, head_pose, is_speaking, landmarks=None, frame_shape=None):
        attention = 0.0
        confidence = 0.0
        focus = 0.0
        stress = 0.0
        speech_energy = 1.0 if is_speaking else 0.0
        feeling = "Neutral"
        
        if gaze_dir == "Center":
            attention += 1.0
            confidence += 0.3
        else:
            attention += 0.2
            
        if head_pose == "Center":
            focus += 1.0
            confidence += 0.3
        elif head_pose == "Down":
            focus += 0.2
        else:
            focus += 0.5
            confidence += 0.1
            
        if is_speaking:
            confidence += 0.2
            attention += 0.1
            
        if blink_count > 20:
            stress += 0.4
            confidence -= 0.1
        elif blink_count > 10:
            stress += 0.1
            
        if landmarks and frame_shape:
            h, w, _ = frame_shape
            
            mouth_open_dist = self._get_distance(landmarks.landmark[13], landmarks.landmark[14], w, h)
            mouth_width_dist = self._get_distance(landmarks.landmark[61], landmarks.landmark[291], w, h)
            brow_dist = self._get_distance(landmarks.landmark[105], landmarks.landmark[159], w, h)
            face_width = self._get_distance(landmarks.landmark[234], landmarks.landmark[454], w, h)
            
            if face_width > 0:
                  mar = mouth_open_dist / face_width
                  mouth_spread = mouth_width_dist / face_width
                  brow_raise = brow_dist / face_width
                  
                  if mar > 0.12 and brow_raise > 0.09:
                      feeling = "Scared / Surprised"
                      stress += 0.2
                      confidence -= 0.1
                  elif mouth_spread > 0.35 and mar < 0.15:
                      feeling = "Happy"
                      confidence += 0.2
                      stress -= 0.1
                  elif brow_raise < 0.04:
                      feeling = "Angry / Stressed"
                      stress += 0.15
                      confidence -= 0.1
                      
        attention = min(max(attention, 0.0), 1.0)
        confidence = min(max(confidence, 0.0), 1.0)
        focus = min(max(focus, 0.0), 1.0)
        stress = min(max(stress, 0.0), 1.0)
        
        return {
            "attention": round(attention, 2),
            "confidence": round(confidence, 2),
            "focus": round(focus, 2),
            "speech_energy": round(speech_energy, 2),
            "stress": round(stress, 2),
            "feeling": feeling
        }
