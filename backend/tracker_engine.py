import cv2
import time
import math
import io
import csv
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from detector import HybridDetector
from preprocessor import FramePreprocessor

def build_opencv_tracker(tracker_type: str):
    """
    Universal factory for OpenCV trackers.
    Resolves compatibility issues across:
    - opencv-contrib-python 4.5+ (cv2.TrackerKCF.create)
    - opencv legacy module (cv2.legacy.TrackerKCF_create)
    - standard opencv-python builds (TrackerMIL fallback)
    """
    tracker_type = tracker_type.upper()
    
    if tracker_type == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "TrackerKCF") and hasattr(cv2.TrackerKCF, "create"):
            return cv2.TrackerKCF.create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()

    elif tracker_type == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT") and hasattr(cv2.TrackerCSRT, "create"):
            return cv2.TrackerCSRT.create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()

    # Universal fallback if opencv-contrib is not installed
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    if hasattr(cv2, "TrackerMIL") and hasattr(cv2.TrackerMIL, "create"):
        return cv2.TrackerMIL.create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        return cv2.legacy.TrackerMIL_create()

    raise RuntimeError("No OpenCV tracker available. Please run: pip install opencv-contrib-python")


class HybridTrackingEngine:
    """
    Implements Modules 4, 5, 6 & 7:
    - Target Identification & Bounding Box Management
    - Hybrid KCF/CSRT Switching with Stability Evaluation
    - Trajectory Vector Math (From Where to Where, Displacement, Speed, Compass Heading)
    - Module 6: Color Histogram Correlation Re-detection
    - Telemetry Dataset Collection & CSV Export
    """
    def __init__(self, detector: HybridDetector, preprocessor: FramePreprocessor):
        self.detector = detector
        self.preprocessor = preprocessor

        self.is_paused = False
        self.playback_speed = 1.0

        self.is_initialized = False
        self.active_tracker_type = "NONE"
        self.preferred_mode = "HYBRID"
        self.tracker = None

        self.target_id = 1
        self.target_label = "Unselected"
        self.target_class = "Unknown"
        self.target_bbox = None
        self.start_center = None
        self.current_center = None
        self.previous_center = None
        self.trajectory = deque(maxlen=150)

        self.net_displacement = 0.0
        self.cumulative_distance = 0.0
        self.current_speed = 0.0
        self.heading_direction = "Stationary"
        self.heading_angle_deg = 0.0

        self.target_hist = None
        self.consecutive_failures = 0
        self.max_failures_before_recovery = 3
        self.tracking_status = "IDLE"

        self.fps = 0.0
        self.frame_index = 0
        self.last_frame_time = time.time()
        self.telemetry_dataset = []
        self.event_logs = []

    def _log_event(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.event_logs.append(f"[{timestamp}] {message}")
        if len(self.event_logs) > 50:
            self.event_logs.pop(0)

    def _extract_color_histogram(self, frame_bgr: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        bx, by, bw, bh = bbox
        hf, wf = frame_bgr.shape[:2]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(wf, bx + bw), min(hf, by + bh)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Tuple parameters avoid syntax or parsing errors
        channels = (0, 1)
        hist_size = (16, 16)
        ranges = (0, 180, 0, 256)
        hist = cv2.calcHist([hsv], channels, None, hist_size, ranges)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def initialize_target(self, frame_bgr: np.ndarray, bbox: List[int], class_name: str = "Object", mode: str = "HYBRID"):
        try:
            self.preferred_mode = mode
            self.target_bbox = [int(v) for v in bbox]
            self.target_class = class_name
            self.target_label = class_name
            self.target_id += 1

            # Tuple unpacking cleanly avoids index typos
            bx, by, bw, bh = self.target_bbox
            cx = int(bx + bw / 2.0)
            cy = int(by + bh / 2.0)
            self.start_center = (cx, cy)
            self.current_center = (cx, cy)
            self.previous_center = (cx, cy)

            self.trajectory.clear()
            self.trajectory.append((cx, cy, time.time()))
            self.net_displacement = 0.0
            self.cumulative_distance = 0.0
            self.current_speed = 0.0
            self.heading_direction = "Stationary"
            self.heading_angle_deg = 0.0

            init_type = "KCF" if mode == "KCF_ONLY" else "CSRT"
            self.active_tracker_type = init_type
            self.tracker = build_opencv_tracker(init_type)
            self.tracker.init(frame_bgr, tuple(self.target_bbox))
            self.target_hist = self._extract_color_histogram(frame_bgr, self.target_bbox)

            self.is_initialized = True
            self.consecutive_failures = 0
            self.tracking_status = "STABLE"
            self._record_telemetry_row()
            self._log_event(f"Target locked: {self.target_label} at origin ({cx}, {cy}) using {init_type}")
            return True
        except Exception as e:
            self._log_event(f"Error initializing tracker: {str(e)}")
            print("Tracker init error:", e)
            return False

    def _compute_movement_vector(self, now: float):
        if self.start_center is None or self.current_center is None:
            return

        x0, y0 = self.start_center
        xt, yt = self.current_center
        xp, yp = self.previous_center if self.previous_center else (xt, yt)

        # Net straight-line displacement
        self.net_displacement = round(math.hypot(xt - x0, yt - y0), 1)

        # Step distance and cumulative travel
        step = math.hypot(xt - xp, yt - yp)
        self.cumulative_distance = round(self.cumulative_distance + step, 1)

        # Velocity in px/s
        dt = max(0.001, now - self.last_frame_time)
        instant_speed = step / dt
        self.current_speed = round(0.7 * self.current_speed + 0.3 * instant_speed, 1)

        # Heading direction and angle
        dx = xt - xp
        dy = yt - yp
        if step < 1.5:
            self.heading_direction = "Stationary"
            self.heading_angle_deg = 0.0
        else:
            angle = math.degrees(math.atan2(-dy, dx))
            self.heading_angle_deg = round(angle, 1)
            if -22.5 <= angle < 22.5:
                self.heading_direction = "East (→)"
            elif 22.5 <= angle < 67.5:
                self.heading_direction = "North-East (↗)"
            elif 67.5 <= angle < 112.5:
                self.heading_direction = "North (↑)"
            elif 112.5 <= angle < 157.5:
                self.heading_direction = "North-West (↖)"
            elif angle >= 157.5 or angle < -157.5:
                self.heading_direction = "West (←)"
            elif -157.5 <= angle < -112.5:
                self.heading_direction = "South-West (↙)"
            elif -112.5 <= angle < -67.5:
                self.heading_direction = "South (↓)"
            else:
                self.heading_direction = "South-East (↘)"

    def update(self, raw_frame: np.ndarray, detection_algo: str = "yolo") -> Dict[str, Any]:
        t0 = time.time()
        self.frame_index += 1

        prep = self.preprocessor.process(raw_frame)
        display_frame = prep["display_bgr"]

        if not self.is_initialized:
            detections = self.detector.detect(display_frame, method=detection_algo, conf_threshold=0.4)
            self._update_fps(t0)
            return {
                "frame": display_frame,
                "is_tracking": False,
                "detections": detections,
                "telemetry": self.get_telemetry()
            }

        if self.is_paused:
            self._update_fps(t0)
            return {
                "frame": display_frame,
                "is_tracking": True,
                "target_bbox": self.target_bbox,
                "telemetry": self.get_telemetry()
            }

        # Update active tracker
        success, box = self.tracker.update(display_frame)

        if success:
            x, y, w, h = map(int, box)
            hf, wf = display_frame.shape[:2]
            if w > 5 and h > 5 and x < wf and y < hf:
                self.target_bbox = [x, y, w, h]
                self.previous_center = self.current_center
                cx = int(x + w / 2.0)
                cy = int(y + h / 2.0)
                self.current_center = (cx, cy)
                self.trajectory.append((cx, cy, t0))

                self._compute_movement_vector(t0)
                self.consecutive_failures = 0
                self.tracking_status = "STABLE"

                if self.preferred_mode == "HYBRID" and self.active_tracker_type == "CSRT" and self.frame_index % 60 == 0:
                    self._switch_tracker(display_frame, "KCF", "Movement calmed; switched to KCF speed")
            else:
                success = False

        if not success:
            self.consecutive_failures += 1
            self.tracking_status = "DEGRADED"

            if self.preferred_mode == "HYBRID" and self.active_tracker_type == "KCF":
                self._switch_tracker(display_frame, "CSRT", "Rapid motion/drift; switching to CSRT")
            elif self.consecutive_failures >= self.max_failures_before_recovery:
                self._attempt_redetection(display_frame, method=detection_algo)

        # Commit telemetry row every 2 frames
        if self.frame_index % 2 == 0:
            self._record_telemetry_row()

        self.last_frame_time = t0
        self._update_fps(t0)

        return {
            "frame": display_frame,
            "is_tracking": True,
            "target_bbox": self.target_bbox if self.tracking_status != "LOST" else None,
            "telemetry": self.get_telemetry()
        }

    def _switch_tracker(self, frame_bgr: np.ndarray, new_type: str, reason: str):
        if self.target_bbox is None:
            return
        try:
            self._log_event(f"Hybrid switch: {self.active_tracker_type} -> {new_type} ({reason})")
            self.active_tracker_type = new_type
            self.tracker = build_opencv_tracker(new_type)
            self.tracker.init(frame_bgr, tuple(self.target_bbox))
        except Exception as e:
            self._log_event(f"Tracker switch warning: {e}")

    def _attempt_redetection(self, frame_bgr: np.ndarray, method: str) -> bool:
        self.tracking_status = "RECOVERING"
        detections = self.detector.detect(frame_bgr, method=method, conf_threshold=0.35)
        if not detections:
            self.tracking_status = "LOST"
            return False

        best_bbox, best_sim = None, -1.0
        for det in detections:
            cand_hist = self._extract_color_histogram(frame_bgr, det["bbox"])
            if cand_hist is not None and self.target_hist is not None:
                sim = cv2.compareHist(self.target_hist, cand_hist, cv2.HISTCMP_CORREL)
                if sim > best_sim:
                    best_sim = sim
                    best_bbox = det["bbox"]

        if best_bbox is not None and best_sim > 0.45:
            self.target_bbox = best_bbox
            bx, by, bw, bh = best_bbox
            cx = int(bx + bw / 2.0)
            cy = int(by + bh / 2.0)
            self.current_center = (cx, cy)
            self.trajectory.append((cx, cy, time.time()))
            self._log_event(f"Module 6: Re-detected target (Similarity: {best_sim:.2f}). Resuming CSRT.")
            
            self.active_tracker_type = "CSRT"
            self.tracker = build_opencv_tracker("CSRT")
            self.tracker.init(frame_bgr, tuple(self.target_bbox))
            self.consecutive_failures = 0
            self.tracking_status = "STABLE"
            return True

        self.tracking_status = "LOST"
        return False

    def finish_video(self):
        if self.tracking_status != "FINISHED":
            self.tracking_status = "FINISHED"
            self._record_telemetry_row()
            self._log_event(
                f"Video finished at frame {self.frame_index}. "
                f"Net Displacement: {self.net_displacement} px, Path: {self.cumulative_distance} px."
            )

    def _record_telemetry_row(self):
        x0, y0 = self.start_center if self.start_center else (0, 0)
        xt, yt = self.current_center if self.current_center else (0, 0)

        row = {
            "frame_id": self.frame_index,
            "timestamp": time.strftime("%H:%M:%S"),
            "object_name": self.target_label,
            "object_class": self.target_class,
            "start_point": f"({x0}, {y0})",
            "current_point": f"({xt}, {yt})",
            "displacement_px": self.net_displacement,
            "cumulative_dist_px": self.cumulative_distance,
            "speed_px_s": self.current_speed,
            "direction": self.heading_direction,
            "angle_deg": self.heading_angle_deg,
            "active_tracker": self.active_tracker_type,
            "status": self.tracking_status
        }
        self.telemetry_dataset.append(row)
        if len(self.telemetry_dataset) > 500:
            self.telemetry_dataset.pop(0)

    def _update_fps(self, start_time: float):
        elapsed = time.time() - start_time
        instant = 1.0 / elapsed if elapsed > 0 else 30.0
        self.fps = round(0.85 * self.fps + 0.15 * instant, 1)

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "fps": self.fps,
            "is_paused": self.is_paused,
            "playback_speed": self.playback_speed,
            "active_tracker": self.active_tracker_type,
            "mode": self.preferred_mode,
            "status": self.tracking_status,
            "object_name": self.target_label,
            "object_class": self.target_class,
            "target_bbox": self.target_bbox,
            "start_point": list(self.start_center) if self.start_center else None,
            "current_point": list(self.current_center) if self.current_center else None,
            "displacement_px": self.net_displacement,
            "cumulative_dist_px": self.cumulative_distance,
            "speed_px_s": self.current_speed,
            "direction": self.heading_direction,
            "angle_deg": self.heading_angle_deg,
            "trajectory": list(self.trajectory),
            "recent_rows": self.telemetry_dataset[-15:],
            "logs": self.event_logs[-8:]
        }

    def export_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Frame", "Timestamp", "Object_Name", "Class", "Start_Point", "Current_Point", "Displacement_px", "Cumulative_Dist_px", "Speed_px_s", "Direction", "Angle_deg", "Tracker", "Status"])
        for r in self.telemetry_dataset:
            writer.writerow([r["frame_id"], r["timestamp"], r["object_name"], r["object_class"], r["start_point"], r["current_point"], r["displacement_px"], r["cumulative_dist_px"], r["speed_px_s"], r["direction"], r["angle_deg"], r["active_tracker"], r["status"]])
        return output.getvalue()

    def reset(self):
        self.is_initialized = False
        self.tracker = None
        self.target_bbox = None
        self.start_center = None
        self.current_center = None
        self.previous_center = None
        self.trajectory.clear()
        self.net_displacement = 0.0
        self.cumulative_distance = 0.0
        self.current_speed = 0.0
        self.heading_direction = "Stationary"
        self.active_tracker_type = "NONE"
        self.tracking_status = "IDLE"
        self.target_label = "Unselected"
        self.target_class = "Unknown"
        self.frame_index = 0
        self._log_event("Tracker reset: Video rewound to frame 0.")