# Real-Time Single Object Tracking Using Hybrid Detection and Tracking Algorithms

**Major Project Phase – II**  
*Department of Information Science & Engineering, SJCIT, Chickballapur*  
*Group 11 (2026)*

## Overview
A real-time hybrid computer vision tracking system integrating:
- **Object Detection (Module 3)**: YOLOv8 & Haar Cascade fallback
- **Hybrid Tracking (Module 5)**: KCF (High-Speed) & CSRT (High-Accuracy / Occlusion Resistant)
- **Automatic Recovery (Module 6)**: Re-detection via color histogram correlation
- **Telemetry & Movement Graph (Module 7)**: Live velocity ($px/s$), displacement ($px$), 8-point compass bearing, and downloadable CSV/JSON datasets.

## Project Structure
```text
object_tracking_system/
├── backend/
│   ├── app.py             # FastAPI WebSocket & Static Server
│   ├── tracker_engine.py  # 3-Layer Hybrid Tracker & Vector Math
│   ├── detector.py        # YOLOv8 & Haar Cascade
│   ├── preprocessor.py    # Noise reduction & normalization
│   └── requirements.txt   # Dependencies
└── frontend/
    ├── index.html         # Interactive UI Dashboard
    ├── styles.css         # Dark theme styling
    └── app.js             # Canvas overlays & telemetry charts
