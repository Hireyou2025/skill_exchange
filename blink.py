import math

class BlinkAnalysis:
    def __init__(self):
        self.previous_closed = False
        self.blink_count = 0
        
    def _eyelid_distance(self, top_lm, bottom_lm, w, h):
        x1, y1 = int(top_lm.x * w), int(top_lm.y * h)
        x2, y2 = int(bottom_lm.x * w), int(bottom_lm.y * h)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    def process(self, landmarks, frame_shape):
        if landmarks is None:
            return self.blink_count

        h, w, _ = frame_shape
        
        left_eye_dist = self._eyelid_distance(landmarks.landmark[159], landmarks.landmark[145], w, h)
        right_eye_dist = self._eyelid_distance(landmarks.landmark[386], landmarks.landmark[374], w, h)

        avg_distance = (left_eye_dist + right_eye_dist) / 2
        closed = avg_distance < 10

        if closed and not self.previous_closed:
            self.blink_count += 1

        self.previous_closed = closed
        return self.blink_count
