import cv2
import numpy as np

class FramePreprocessor:
    def __init__(self, target_width: int = 640, target_height: int = 360, blur_ksize: int = 3):
        self.target_width = target_width
        self.target_height = target_height
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

    def process(self, frame: np.ndarray) -> dict:
        """
        Executes Module 2 pipeline:
        1. Resize
        2. BGR to RGB / Gray conversion
        3. Gaussian blur noise reduction
        4. Frame normalization
        """
        if frame is None:
            return {}

        # 1. Frame Resizing
        resized = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        # 2. Color Conversion
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Noise Reduction via Gaussian Blur
        if self.blur_ksize > 1:
            blurred_bgr = cv2.GaussianBlur(resized, (self.blur_ksize, self.blur_ksize), 0)
            blurred_gray = cv2.GaussianBlur(gray_frame, (self.blur_ksize, self.blur_ksize), 0)
        else:
            blurred_bgr = resized
            blurred_gray = gray_frame

        # 4. Frame Normalization (0.0 - 1.0 float representation for detection confidence)
        normalized = blurred_bgr.astype(np.float32) / 255.0

        return {
            "display_bgr": resized,
            "rgb": rgb_frame,
            "gray": blurred_gray,
            "bgr_blurred": blurred_bgr,
            "normalized": normalized,
            "shape": (self.target_height, self.target_width)
        }