import cv2
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from detector import HybridDetector
from preprocessor import FramePreprocessor
from tracker_engine import HybridTrackingEngine

app = FastAPI(title="Hybrid Object Tracking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = HybridDetector()
preprocessor = FramePreprocessor(target_width=640, target_height=360)
engine = HybridTrackingEngine(detector, preprocessor)

class VideoSourceManager:
    def __init__(self):
        self.cap = None
        self.source_type = "webcam"
        self.file_path = None
        self.is_paused = False
        self.video_ended = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.last_frame = None

    def open_source(self, source_type="webcam", file_path=None):
        if self.cap is not None:
            self.cap.release()
        
        self.source_type = source_type
        self.file_path = file_path
        self.video_ended = False
        self.is_paused = False
        self.current_frame_idx = 0
        self.last_frame = None

        if source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            self.total_frames = -1
        else:
            self.cap = cv2.VideoCapture(file_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.current_frame_idx = 1
                if self.source_type != "webcam":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0

        return self.cap.isOpened()

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None, True

        if self.video_ended:
            return False, self.last_frame, True

        if self.is_paused:
            return True, self.last_frame, False

        ret, frame = self.cap.read()
        if not ret:
            if self.source_type != "webcam":
                self.video_ended = True
                self.is_paused = True
                return False, self.last_frame, True
            else:
                return False, None, False

        self.last_frame = frame
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.source_type != "webcam" else self.current_frame_idx + 1
        return True, frame, False

    def reset_video(self):
        self.video_ended = False
        self.is_paused = False
        if self.cap is not None and self.source_type != "webcam":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame
                self.current_frame_idx = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True

video_mgr = VideoSourceManager()
video_mgr.open_source("webcam")

@app.post("/api/playback")
async def control_playback(payload: dict):
    action = payload.get("action")
    if action == "play":
        if video_mgr.video_ended:
            video_mgr.reset_video()
            engine.reset()
        video_mgr.is_paused = False
        engine.is_paused = False
    elif action == "pause":
        video_mgr.is_paused = True
        engine.is_paused = True
    elif action == "reset":
        video_mgr.reset_video()
        engine.reset()
    elif action == "set_speed":
        engine.playback_speed = float(payload.get("speed", 1.0))
        
    return {
        "status": "success",
        "is_paused": video_mgr.is_paused,
        "video_ended": video_mgr.video_ended,
        "current_frame": video_mgr.current_frame_idx,
        "total_frames": video_mgr.total_frames
    }

@app.get("/api/export-data")
async def export_data(format: str = "csv"):
    if format.lower() == "json":
        return Response(content=json.dumps(engine.telemetry_dataset, indent=2), media_type="application/json", headers={"Content-Disposition": "attachment; filename=tracking_data.json"})
    return Response(content=engine.export_csv(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=tracking_telemetry_dataset.csv"})

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    temp_path = f"/tmp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    video_mgr.open_source("file", temp_path)
    engine.reset()
    return {"status": "success", "filename": file.filename, "total_frames": video_mgr.total_frames}

@app.post("/api/use-webcam")
async def use_webcam():
    video_mgr.open_source("webcam")
    engine.reset()
    return {"status": "success", "source": "webcam"}

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    detection_algo = "yolo"
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.005)
                msg = json.loads(data)
                cmd = msg.get("action")
                if cmd == "set_detection_algo":
                    detection_algo = msg.get("algo", "yolo")
                elif cmd == "playback_action":
                    sub = msg.get("sub_action")
                    if sub == "play":
                        if video_mgr.video_ended:
                            video_mgr.reset_video()
                            engine.reset()
                        video_mgr.is_paused = False
                        engine.is_paused = False
                    elif sub == "pause":
                        video_mgr.is_paused = True
                        engine.is_paused = True
                    elif sub == "reset":
                        video_mgr.reset_video()
                        engine.reset()
                    elif sub == "set_speed":
                        engine.playback_speed = float(msg.get("speed", 1.0))
                elif cmd == "init_roi":
                    if video_mgr.last_frame is not None:
                        prep = preprocessor.process(video_mgr.last_frame)
                        success = engine.initialize_target(
                            prep["display_bgr"],
                            msg["bbox"],
                            class_name=msg.get("class_name", "Object"),
                            mode=msg.get("mode", "HYBRID")
                        )
                        print(f"Init ROI triggered: Success={success}, BBox={msg['bbox']}")
            except asyncio.TimeoutError:
                pass
            except Exception as cmd_err:
                print("Command handling exception:", cmd_err)

            ret, frame, ended = video_mgr.read_frame()
            
            if ended and video_mgr.source_type != "webcam":
                engine.finish_video()
                frame_to_show = video_mgr.last_frame
            else:
                frame_to_show = frame

            if frame_to_show is None:
                await asyncio.sleep(0.03)
                continue

            if not ended:
                result = engine.update(frame_to_show, detection_algo=detection_algo)
                disp_frame = result["frame"]
                target_bbox = result.get("target_bbox")
                detections = result.get("detections", [])
            else:
                prep = preprocessor.process(frame_to_show)
                disp_frame = prep["display_bgr"]
                target_bbox = engine.target_bbox
                detections = []

            _, buffer = cv2.imencode('.jpg', disp_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "image": f"data:image/jpeg;base64,{jpg_as_text}",
                "telemetry": engine.get_telemetry(),
                "is_tracking": engine.is_initialized,
                "detections": detections,
                "target_bbox": target_bbox,
                "playback_info": {
                    "is_paused": video_mgr.is_paused,
                    "video_ended": video_mgr.video_ended,
                    "current_frame": video_mgr.current_frame_idx,
                    "total_frames": video_mgr.total_frames,
                    "source_type": video_mgr.source_type
                }
            }
            await websocket.send_text(json.dumps(payload))

            base_sleep = 0.025 / max(0.25, engine.playback_speed)
            await asyncio.sleep(base_sleep)

    except WebSocketDisconnect:
        pass
    except Exception as ws_err:
        print("WebSocket loop exception:", ws_err)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)