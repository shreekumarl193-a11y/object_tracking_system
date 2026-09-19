import cv2
import os
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

class HybridDetector:
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        # Load YOLOv8 Nano model
        self.yolo = YOLO(yolo_model_path)
        
        # Load OpenCV default Haar Cascades for face & body detection
        face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        body_path = os.path.join(cv2.data.haarcascades, "haarcascade_fullbody.xml")
        self.haar_face = cv2.CascadeClassifier(face_path) if os.path.exists(face_path) else None
        self.haar_body = cv2.CascadeClassifier(body_path) if os.path.exists(body_path) else None

    def detect(self, frame_bgr: np.ndarray, method: str = "yolo", conf_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Detects objects in the input frame using the specified method.
        Returns a list of dicts: [{"bbox": [x, y, w, h], "confidence": float, "class_name": str}]
        """
        detections = []
        h, w = frame_bgr.shape[:2]

        if method.lower() == "yolo":
            results = self.yolo(frame_bgr, conf=conf_threshold, verbose=False)[0]
            for detection_index, box in enumerate(results.boxes, start=1):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                bx = max(0, x1)
                by = max(0, y1)
                bw = min(w - bx, x2 - x1)
                bh = min(h - by, y2 - y1)
                
                cls_id = int(box.cls[0].item())
                label = self.yolo.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                
                detections.append({
                    "object_id": f"{label}-{detection_index}",
                    "bbox": [bx, by, bw, bh],
                    "confidence": round(conf, 3),
                    "class_name": label
                })

        elif method.lower() == "haar":
            if self.haar_face is None and self.haar_body is None:
                return detections

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = self.haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) if self.haar_face is not None else []
            for detection_index, (x, y, bw, bh) in enumerate(faces, start=1):
                detections.append({
                    "object_id": f"face-{detection_index}",
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "confidence": 0.85,
                    "class_name": "face"
                })
            # Detect full bodies if few faces
            if len(detections) == 0:
                bodies = self.haar_body.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 70)) if self.haar_body is not None else []
                for detection_index, (x, y, bw, bh) in enumerate(bodies, start=1):
                    detections.append({
                        "object_id": f"person-{detection_index}",
                        "bbox": [int(x), int(y), int(bw), int(bh)],
                        "confidence": 0.75,
                        "class_name": "person"
                    })

        return detections