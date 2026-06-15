class IntegrityMonitoring:
    def __init__(self):
        self.absence_frames = 0
        self.absence_threshold = 30 # roughly 1-2s

    def process(self, landmarks, face_count):
        flags = []
        
        if face_count == 0 or landmarks is None:
            self.absence_frames += 1
            if self.absence_frames > self.absence_threshold:
                flags.append("Face Absent!")
        else:
            self.absence_frames = 0
            
        if face_count > 1:
            flags.append("Multiple Faces Detected!")

        return flags
