# SpeedVision – AI Traffic Speed Detection Web App

Upload any traffic video and get an annotated output video showing real-time speed estimates for every tracked vehicle.

---

## Project Structure

```
speedapp/
├── app.py               # Flask backend (API + processing)
├── templates/
│   └── index.html       # Frontend UI
├── yolov8n.pt           # YOLOv8 model weights (copy here)
├── uploads/             # Temp uploaded videos
└── outputs/             # Processed output videos
```

---

## Setup & Run

### 1. Install Dependencies

```bash
pip install flask ultralytics opencv-python-headless
```

### 2. Place your YOLOv8 model

Copy `yolov8n.pt` into the `speedapp/` folder (same level as `app.py`).

### 3. Start the server

```bash
cd speedapp
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## How It Works

1. **Upload** — User uploads a traffic video (MP4, AVI, MOV, MKV, WEBM).
2. **Detect** — YOLOv8n detects vehicles (cars, trucks, buses, motorcycles) in every frame.
3. **Track** — Persistent IDs are assigned to each vehicle across frames.
4. **Speed** — Pixel displacement per frame is converted to km/h using FPS and a scale factor (0.05 m/pixel).
5. **Annotate** — Bounding boxes, vehicle type, ID, and speed are drawn on each frame. Color indicates speed:
   - 🟢 Green = slow (< 40 km/h)
   - 🟠 Orange = medium (40–80 km/h)
   - 🔴 Red = fast (> 80 km/h)
6. **Download** — Annotated video is available for download along with stats summary.

---

## Calibration Note

The scale factor (`0.05 m/pixel`) is an estimate. For accurate real-world speed readings, calibrate this value based on your camera's height, angle, and focal length.

---

## Tech Stack

- **Backend**: Python, Flask
- **AI Model**: Ultralytics YOLOv8
- **Video Processing**: OpenCV
- **Frontend**: Vanilla HTML/CSS/JS (no framework needed)
