import cv2
import mss
import numpy as np

class ScreenRecogniser:
    def __init__(self):
        self.sct = mss.MSS()
        self.monitor = None
        self.select_roi()

    def select_roi(self):
        screen = np.array(self.sct.grab(self.sct.monitors[1]))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        roi = cv2.selectROI(
            "Select Interview Area",
            screen,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyWindow("Select Interview Area")

        x, y, w, h = roi
        self.monitor = {
            "left": x,
            "top": y,
            "width": w,
            "height": h
        }

    def capture(self):
        if not self.monitor or self.monitor["width"] == 0 or self.monitor["height"] == 0:
            return None
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
