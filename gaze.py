class GazeTracking:
    def __init__(self):
        self.previous_direction = None

    def process(self, landmarks, frame_shape):
        if landmarks is None:
            return self.previous_direction

        h, w, _ = frame_shape

        left_iris = landmarks.landmark[468]
        eye_left_corner = landmarks.landmark[33]
        eye_right_corner = landmarks.landmark[133]

        ir_x = int(left_iris.x * w)
        lx = int(eye_left_corner.x * w)
        rx = int(eye_right_corner.x * w)

        if (rx - lx) == 0: 
            return self.previous_direction
        
        ratio = (ir_x - lx) / (rx - lx)
        
        if ratio < 0.35:
            curr_dire = "Left"
          
        elif ratio > 0.50:
            curr_dire = "Right"
        else:
            curr_dire = "Center"

        self.previous_direction = curr_dire
        return curr_dire
