import cv2
import mediapipe as mp

class DetectFace:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec = self.mp_draw.DrawingSpec(
            color=(255, 0, 0),
            thickness=1,
            circle_radius=1
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face.process(rgb)

        landmarks = None
        face_count = 0
        if result.multi_face_landmarks:
            face_count = len(result.multi_face_landmarks)
            landmarks = result.multi_face_landmarks[0]
            
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                self.draw_spec,
                self.draw_spec
            )

        return frame, landmarks, face_count
