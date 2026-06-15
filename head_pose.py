import cv2
import numpy as np

class HeadPoseEstimation:
    def __init__(self):
        self.previous_pose = "Center"

    def process(self, landmarks, frame_shape):
        if landmarks is None:
            return self.previous_pose

        h, w, c = frame_shape

        face_2d = []
        face_3d = []

        lm_indices = [1, 152, 234, 454]
        for idx in lm_indices:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        if not success:
            return self.previous_pose

        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360

        pose = "Center"
        if y_angle < -10:
            pose = "Left"
        elif y_angle > 10:
            pose = "Right"
        elif x_angle < -10:
            pose = "Down"
        elif x_angle > 10:
            pose = "Up"

        self.previous_pose = pose
        return pose
