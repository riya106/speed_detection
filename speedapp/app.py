import os
import uuid
import threading
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(job_id, input_path, output_path):
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np

        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['message'] = 'Loading YOLOv8 model...'

        model_path = os.path.join(BASE_DIR, 'yolov8n.pt')
        model = YOLO(model_path)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        prev_positions  = {}
        speed_dict      = {}
        max_speeds      = {}
        frame_count     = 0
        total_vehicles  = set()
        helmet_count    = 0
        no_helmet_count = 0

        # track which rider IDs we've already counted for helmet
        helmet_ids_seen    = set()
        no_helmet_ids_seen = set()

        jobs[job_id]['message'] = 'Detecting vehicles, speeds and helmets...'

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = int((frame_count / max(total_frames, 1)) * 100)
            jobs[job_id]['progress'] = min(progress, 99)

            # ── run tracking (vehicles) ──────────────────────────────────────
            results = model.track(frame, persist=True, verbose=False,
                                  classes=[2, 3, 5, 7])   # car, motorcycle, bus, truck

            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    obj_id  = int(ids[i])
                    cls_id  = int(classes[i])
                    total_vehicles.add(obj_id)

                    label_map = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
                    label = label_map.get(cls_id, 'Vehicle')

                    # speed
                    if obj_id in prev_positions:
                        px, py   = prev_positions[obj_id]
                        distance = np.sqrt((cx-px)**2 + (cy-py)**2)
                        meters   = distance * 0.05
                        speed    = int(meters * fps * 3.6)
                        speed_dict[obj_id] = speed
                        max_speeds[obj_id] = max(max_speeds.get(obj_id, 0), speed)
                    prev_positions[obj_id] = (cx, cy)

                    # box color by speed
                    spd = speed_dict.get(obj_id, 0)
                    if spd > 80:
                        color = (0, 0, 255)
                    elif spd > 40:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} #{obj_id}",
                                (x1, y1 - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    if obj_id in speed_dict:
                        cv2.putText(frame, f"{speed_dict[obj_id]} km/h",
                                    (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # ── helmet check for motorcyclists ───────────────────────
                    if cls_id == 3:   # motorcycle
                        # Crop the upper-body / head region of the rider
                        # (top 40 % of the bounding box)
                        head_y2 = y1 + int((y2 - y1) * 0.40)
                        head_crop = frame[max(0, y1):head_y2, max(0, x1):x2]

                        has_helmet = False
                        if head_crop.size > 0:
                            # Use a simple HSV-based heuristic:
                            # Helmets are often dark/matte objects with low saturation
                            # and relatively uniform color on top of the rider's head.
                            hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                            avg_sat = float(hsv[:, :, 1].mean())
                            avg_val = float(hsv[:, :, 2].mean())

                            # Low saturation + mid-to-dark value → likely a helmet
                            # (hair is usually higher saturation or very dark value)
                            has_helmet = (avg_sat < 60 and 30 < avg_val < 200)

                        if has_helmet:
                            helmet_label = "Helmet ✓"
                            helmet_color = (0, 255, 0)
                            if obj_id not in helmet_ids_seen:
                                helmet_ids_seen.add(obj_id)
                                helmet_count += 1
                        else:
                            helmet_label = "No Helmet ✗"
                            helmet_color = (0, 0, 255)
                            if obj_id not in no_helmet_ids_seen:
                                no_helmet_ids_seen.add(obj_id)
                                no_helmet_count += 1

                        cv2.putText(frame, helmet_label,
                                    (x1, y2 + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    helmet_color, 2)

            # ── HUD overlay ──────────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (310, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"Vehicles : {len(total_vehicles)}",
                        (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Helmet   : {helmet_count}",
                        (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv2.putText(frame, f"No Helmet: {no_helmet_count}",
                        (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

            out.write(frame)

        cap.release()
        out.release()

        speeds    = list(speed_dict.values())
        avg_speed = int(sum(speeds) / len(speeds)) if speeds else 0
        max_speed = max(max_speeds.values()) if max_speeds else 0

        jobs[job_id]['status']      = 'done'
        jobs[job_id]['progress']    = 100
        jobs[job_id]['message']     = 'Processing complete!'
        jobs[job_id]['output_file'] = os.path.basename(output_path)
        jobs[job_id]['stats'] = {
            'total_vehicles':  len(total_vehicles),
            'avg_speed':       avg_speed,
            'max_speed':       max_speed,
            'helmet_count':    helmet_count,
            'no_helmet_count': no_helmet_count,
            'total_frames':    frame_count,
        }

    except Exception as e:
        import traceback
        jobs[job_id]['status']  = 'error'
        jobs[job_id]['message'] = traceback.format_exc()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['video']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    job_id     = str(uuid.uuid4())[:8]
    ext        = file.filename.rsplit('.', 1)[-1].lower()
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.{ext}")
    output_path= os.path.join(app.config['OUTPUT_FOLDER'], f"out_{job_id}.mp4")

    file.save(input_path)
    jobs[job_id] = {'status': 'queued', 'progress': 0,
                    'message': 'Queued...', 'output_file': None, 'stats': None}

    t = threading.Thread(target=process_video, args=(job_id, input_path, output_path))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True,
                     download_name='speed_helmet_detection.mp4')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
