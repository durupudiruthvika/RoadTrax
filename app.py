from flask import Flask, render_template, request, flash, redirect, url_for, session
import os
import cv2
import glob
import base64
import numpy as np
import sqlite3
import requests
from ultralytics import YOLO
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import re

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Optimized for Mobile & Ngrok compatibility
app.config.update(
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=3600
)

# Limit upload size to avoid OOM / worker crashes in cloud runtimes.
# Override via .env: MAX_UPLOAD_MB=120
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '120'))
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
MAX_VIDEO_SECONDS = int(os.getenv('MAX_VIDEO_SECONDS', '60'))

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

# Load environment variables from .env (if present)
load_dotenv()

# Initialize Database for Potholes
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            severity TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL,
            total_potholes INTEGER NOT NULL,
            urgency TEXT NOT NULL,
            address TEXT,
            small INTEGER DEFAULT 0,
            medium INTEGER DEFAULT 0,
            large INTEGER DEFAULT 0,
            result_file TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    # Use ALTER TABLE to add columns if they don't exist
    cols = [
        ('address', 'TEXT'),
        ('small', 'INTEGER DEFAULT 0'),
        ('medium', 'INTEGER DEFAULT 0'),
        ('large', 'INTEGER DEFAULT 0'),
        ('result_file', 'TEXT'),
        ('status', "TEXT DEFAULT 'Reported'"),
        ('detection_type', "TEXT DEFAULT 'Image'")
    ]
    for col_name, col_type in cols:
        try:
            c.execute(f'ALTER TABLE reports ADD COLUMN {col_name} {col_type}')
        except sqlite3.OperationalError:
            pass 
            
    # Create default admin if none exists
    c.execute('SELECT COUNT(*) FROM admins')
    if c.fetchone()[0] == 0:
        # For demo purposes, using a simple password. In production, use hashing.
        c.execute('INSERT INTO admins (username, password) VALUES (?, ?)', ('admin', 'admin1'))
    else:
        # Ensure the admin password is updated if it matches the previous default
        c.execute('UPDATE admins SET password = ? WHERE username = ? AND password = ?', ('admin1', 'admin', 'admin123'))
        
    conn.commit()
    conn.close()

init_db()

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID_HERE').strip()

# ---------------- Model Configuration ----------------
# Allow swapping to a stronger model via env var (e.g. yolov8x.pt, yolo11x.pt,
# or a custom fine-tuned pothole model). Falls back to local best.pt.
MODEL_PATH = os.getenv('MODEL_PATH', 'best.pt')
INFER_IMGSZ = int(os.getenv('INFER_IMGSZ', '640'))  # higher = more accurate, slower; default reduced for cloud speed
VIDEO_FRAME_SKIP = int(os.getenv('VIDEO_FRAME_SKIP', '4'))  # Process 1 in N frames for speed
model = None

# ---------------- Roboflow Workflow Backend ----------------
# Set DETECTION_BACKEND=roboflow in .env to route image / live-frame inference
# through your hosted Roboflow workflow. Video tracking always uses the local
# model because it requires per-object tracking IDs.
DETECTION_BACKEND = os.getenv('DETECTION_BACKEND', 'local').lower()
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '').strip()
ROBOFLOW_API_URL = os.getenv('ROBOFLOW_API_URL', 'https://serverless.roboflow.com').rstrip('/')
ROBOFLOW_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE', '').strip()
ROBOFLOW_WORKFLOW_ID = os.getenv('ROBOFLOW_WORKFLOW_ID', '').strip()
ROBOFLOW_CLASSES = os.getenv('ROBOFLOW_CLASSES', 'Pothole, Sewage/Manhole')

if DETECTION_BACKEND == 'roboflow':
    print(f"[MODEL] Using Roboflow workflow: {ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW_ID}")
else:
    print(f"[MODEL] Using local YOLO backend: {MODEL_PATH} (imgsz={INFER_IMGSZ})")


def ensure_local_model_loaded():
    """Load YOLO model lazily to keep cloud memory usage low."""
    global model
    if model is None:
        print(f"[MODEL] Loading detection model: {MODEL_PATH} (imgsz={INFER_IMGSZ})")
        model = YOLO(MODEL_PATH)
    return model

def _roboflow_predict(img):
    """Call the Roboflow hosted workflow and return normalized detections.

    Returns a list of dicts: {x1, y1, x2, y2, conf, cls}.
    Coordinates are absolute pixel values in the original image.
    """
    if img is None or img.size == 0:
        return []
    if not (ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW_ID):
        print("[WARN] Roboflow backend selected but credentials are missing.")
        return []
    try:
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            return []
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        url = f"{ROBOFLOW_API_URL}/infer/workflows/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_WORKFLOW_ID}"
        payload = {
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
                "image": {"type": "base64", "value": b64},
                "classes": ROBOFLOW_CLASSES,
            },
            "use_cache": True,
        }
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] Roboflow inference failed: {e}")
        return []

    # Roboflow workflow responses are nested. Walk the JSON to find
    # any object that looks like a predictions list.
    detections = []

    def _consume_predictions(preds):
        for p in preds:
            try:
                if 'x' in p and 'y' in p and 'width' in p and 'height' in p:
                    cx, cy, w, h = p['x'], p['y'], p['width'], p['height']
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)
                elif all(k in p for k in ('x1', 'y1', 'x2', 'y2')):
                    x1, y1, x2, y2 = int(p['x1']), int(p['y1']), int(p['x2']), int(p['y2'])
                else:
                    continue
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': float(p.get('confidence', p.get('conf', 0.0))),
                    'cls': str(p.get('class', p.get('class_name', 'object'))),
                })
            except Exception:
                continue

    def _walk(node):
        if isinstance(node, dict):
            preds = node.get('predictions')
            if isinstance(preds, list) and preds and isinstance(preds[0], dict):
                _consume_predictions(preds)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(data)
    return detections


def _local_predict(img, conf=0.15, iou=0.4, augment=True):
    """Run the local YOLO model and return normalized detections."""
    if img is None or img.size == 0:
        return []
    local_model = ensure_local_model_loaded()
    results = local_model.predict(img, conf=conf, iou=iou, augment=augment,
                            imgsz=INFER_IMGSZ, classes=[0], verbose=False)
    out = []
    if not results:
        return out
    for box in (results[0].boxes or []):
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': float(box.conf[0]),
                'cls': 'pothole',
            })
        except Exception:
            continue
    return out


def unified_predict(img, conf=0.15, iou=0.4, augment=True):
    """Backend-agnostic detection. Returns list of detection dicts."""
    infer_img = enhance_for_detection(img)
    if DETECTION_BACKEND == 'roboflow':
        return _roboflow_predict(infer_img)
    return _local_predict(infer_img, conf=conf, iou=iou, augment=augment)


# ---------------- Low-light / Night Enhancement ----------------
def _is_low_light(img, threshold: int = 80) -> bool:
    """Return True when the average luminance is below the threshold."""
    if img is None or img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold

def enhance_for_detection(img):
    """Enhance contrast/brightness so the model detects potholes in low light.

    Uses CLAHE on the L channel of LAB plus a mild gamma boost when the
    scene is dark. Returns the enhanced BGR image. Bright images are
    returned unchanged so daytime accuracy is preserved.
    """
    if img is None or img.size == 0:
        return img
    if not _is_low_light(img):
        return img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        # Gentle gamma correction to lift shadows further
        gamma = 1.4
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(enhanced, table)
    except Exception as e:
        print(f"[WARN] enhance_for_detection failed: {e}")
        return img

# --- Decorators ---
def admin_login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_id'):
            flash('Admin access required.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    if session.get('admin_id'):
        return redirect(url_for('admin_dashboard'))
    return render_template('login.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id FROM admins WHERE username = ? AND password = ?', (username, password))
        admin = c.fetchone()
        conn.close()
        
        if admin:
            session['admin_id'] = username
            session['user_name'] = "Admin"
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials. Access denied.', 'error')
            
    return render_template('admin_login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html', google_client_id=GOOGLE_CLIENT_ID)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('admin_id', None)
    session.pop('user_picture', None)
    return redirect(url_for('index'))

@app.route('/google_login', methods=['POST'])
def google_login():
    token = request.json.get('credential')
    if not token:
        return {"error": "Missing token"}, 400
        
    token = token.strip()
    print(f"DEBUG: Auth attempt. Token length: {len(token)}")
    print(f"DEBUG: Token prefix: {token[:15]}...{token[-15:]}")
    
    try:
        if GOOGLE_CLIENT_ID == "YOUR_GOOGLE_CLIENT_ID_HERE" or not GOOGLE_CLIENT_ID:
            # ONLY for developers who haven't set up a Client ID yet
            import jwt
            unverified_claims = jwt.decode(token, options={"verify_signature": False})
            email = unverified_claims.get('email')
            name = unverified_claims.get('name')
            picture = unverified_claims.get('picture')
            
            if not email:
                return {"error": "No email found in token"}, 401
        else:
            try:
                # Proper production validation with clock skew leeway
                idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID, clock_skew_in_seconds=10)
                email = idinfo.get('email')
                name = idinfo.get('name')
                picture = idinfo.get('picture')
            except ValueError as ve:
                err = str(ve)
                print(f"!!! Google Token Verification Failed: {err}")
                return {"error": f"Invalid token signature: {err}"}, 401
            
        session['user_id'] = email
        session['user_name'] = name if name else email.split('@')[0]
        session['user_picture'] = picture
        return {"status": "success"}, 200
    except Exception as e:
        error_msg = str(e)
        print(f"!!! Google Auth Error on Mobile/Ngrok: {error_msg}")
        return {"error": f"Auth failed: {error_msg}"}, 401

@app.route('/dashboard')
def dashboard():
    if not session.get('user_id'):
        return redirect(url_for('login'))
        
    user_id = session.get('user_id')
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT lat, lng, total_potholes, urgency, address, small, medium, large, result_file, timestamp, status FROM reports WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
    rows = c.fetchall()
    conn.close()
    
    reports = []
    for row in rows:
        reports.append({
            "lat": row[0],
            "lng": row[1],
            "total_potholes": row[2],
            "urgency": row[3],
            "address": row[4] if row[4] else "N/A",
            "small": row[5],
            "medium": row[6],
            "large": row[7],
            "result_file": row[8],
            "timestamp": row[9],
            "status": row[10]
        })
        
    # Aggregate stats for Analytics Dashboard
    stats_summary = {
        "total_potholes": sum(r["total_potholes"] for r in reports),
        "small": sum(r["small"] for r in reports),
        "medium": sum(r["medium"] for r in reports),
        "large": sum(r["large"] for r in reports)
    }
    
    return render_template('dashboard.html', reports=reports, total_reports=len(reports), stats_summary=stats_summary)

@app.route('/map')
def map_view():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('map.html')

@app.route('/live_road_map')
def live_road_map():
    if not session.get('user_id') and not session.get('admin_id'):
        return redirect(url_for('login'))
    return render_template('live_road_map.html')

@app.route('/api/all_reports')
def all_reports():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT lat, lng, total_potholes, urgency, address, timestamp, detection_type, status FROM reports ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    
    reports = []
    for row in rows:
        reports.append({
            "lat": row[0],
            "lng": row[1],
            "total_potholes": row[2],
            "urgency": row[3],
            "address": row[4] if row[4] else "N/A",
            "timestamp": row[5],
            "detection_type": row[6],
            "status": row[7]
        })
    return {"reports": reports}

@app.route('/save_pothole', methods=['POST'])
def save_pothole():
    if not session.get('user_id'):
        return {"error": "Unauthorized"}, 401
    
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    severity = data.get('severity')
    
    if lat and lng and severity:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO potholes (lat, lng, severity) VALUES (?, ?, ?)', (lat, lng, severity))
        conn.commit()
        conn.close()
        return {"status": "success"}, 200
    
    return {"error": "Invalid data"}, 400

@app.route('/submit_report', methods=['POST'])
def submit_report():
    if not session.get('user_id'):
        return {"error": "Unauthorized"}, 401
        
    data = request.json
    lat = data.get('lat')
    lng = data.get('lng')
    total_potholes = data.get('total_potholes', 0)
    urgency = data.get('urgency', 'Unknown')
    address = data.get('address', 'Unknown Location')
    small = data.get('small', 0)
    medium = data.get('medium', 0)
    large = data.get('large', 0)
    result_file = data.get('result_file', '')
    detection_type = data.get('detection_type', 'Image')
    user_id = session.get('user_id')
    
    if lat and lng:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO reports (user_id, lat, lng, total_potholes, urgency, address, small, medium, large, result_file, detection_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                  (user_id, lat, lng, total_potholes, urgency, address, small, medium, large, result_file, detection_type))
        conn.commit()
        conn.close()
        return {"status": "success"}, 200
        
    return {"error": "Invalid coordinates"}, 400

@app.route('/admin-dashboard')
@admin_login_required
def admin_dashboard():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM reports ORDER BY timestamp DESC')
    reports = [dict(zip([column[0] for column in c.description], row)) for row in c.fetchall()]
    
    # System Stats
    c.execute('SELECT COUNT(*) FROM reports')
    total_reports = c.fetchone()[0]
    
    c.execute('SELECT SUM(total_potholes), SUM(small), SUM(medium), SUM(large) FROM reports')
    stats = c.fetchone()
    
    conn.close()
    
    stats_summary = {
        "total_potholes": stats[0] if stats[0] else 0,
        "small": stats[1] if stats[1] else 0,
        "medium": stats[2] if stats[2] else 0,
        "large": stats[3] if stats[3] else 0
    }
    
    return render_template('admin_dashboard.html', reports=reports, total_reports=total_reports, stats_summary=stats_summary)

@app.route('/update_report_status', methods=['POST'])
@admin_login_required
def update_report_status():
    report_id = request.json.get('report_id')
    new_status = request.json.get('status')
    
    if report_id and new_status:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE reports SET status = ? WHERE id = ?', (new_status, report_id))
        conn.commit()
        conn.close()
        return {"status": "success"}, 200
        
    return {"error": "Missing data"}, 400

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file:
            # Sanitize filename
            filename = secure_filename(file.filename)
            # Further clean to be extra safe (remove commas, multiple underscores)
            filename = re.sub(r'[,]+', '_', filename)
            filename = re.sub(r'_+', '_', filename)
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Check extension
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    return process_image(filename, filepath)
                elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    return process_video(filename, filepath)
                else:
                    flash('Unsupported file type', 'error')
                    return redirect(request.url)
            except Exception as e:
                print(f"[ERROR] detect() failed for file '{filename}': {e}")
                return render_template(
                    'detect.html',
                    error_msg=(
                        f"Processing failed: {str(e)}. "
                        "Try a shorter/lower-resolution clip or retry in a few moments."
                    ),
                    uploaded_image=None,
                    uploaded_video=None,
                )

    return render_template('detect.html', uploaded_image=None, uploaded_video=None)

@app.route('/live')
def live_detection():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('detect.html', live_mode=True, uploaded_image=None, uploaded_video=None)

def process_image(filename, filepath):
    img = cv2.imread(filepath)
    if img is None:
        return render_template('detect.html', 
                               error_msg="Invalid input: This is not a valid road image for pothole detection.")

    # Run Inference (backend chosen by DETECTION_BACKEND env: local | roboflow)
    detections = unified_predict(img, conf=0.15, iou=0.4, augment=True)

    potholes = []

    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    # Counters
    small, medium, large = 0, 0, 0
    total_conf = 0

    # Annotated image
    annotated_img = img.copy()

    detected_count = 0

    for det in detections:
        # Bounding box
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']

        w = x2 - x1
        h = y2 - y1
        box_area = w * h
        ratio = (box_area / img_area) * 100

        # Filter out huge objects (Cars) > 20%
        if ratio > 20.0:
            continue
            
        detected_count += 1
        total_conf += conf
        
        # Classification
        if ratio < 1.0:
            severity = "Small"
            color = (0, 255, 0) # Green
            small += 1
        elif 1.0 <= ratio < 4.0:
            severity = "Medium"
            color = (0, 255, 255) # Yellow
            medium += 1
        else:
            severity = "Large"
            color = (0, 0, 255) # Red
            large += 1
            
        # Draw Box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        label = f"Pothole ({severity})"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Top-left overlay
    cv2.putText(annotated_img, f"Total No of Potholes: {detected_count}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
    # Urgency Logic
    if detected_count == 0:
        urgency = "None"
    elif large > 0:
        urgency = "High"
    elif medium > 0:
        urgency = "Medium"
    else:
        urgency = "Low"
        
    # Save Output
    out_filename = 'pred_' + filename
    out_path = os.path.join(app.config['PREDICTION_FOLDER'], out_filename)
    cv2.imwrite(out_path, annotated_img)
    
    # Save a fixed copy for the standard email link as requested
    fixed_out_path = os.path.join(app.config['PREDICTION_FOLDER'], 'detection_result_image.png')
    cv2.imwrite(fixed_out_path, annotated_img)
    
    avg_conf = (total_conf / detected_count) if detected_count > 0 else 0
    
    # Generate Analysis Report Data (Kept strictly on Backend, UI will ignore if needed)
    analysis_report = [
        "--- Analysis Report ---",
        f"Image: {filename}",
        f"Status: {'Pothole-ridden' if detected_count > 0 else 'Clean Road'}",
        f"Total Potholes Detected: {detected_count}",
        f"Average Confidence Score: {round(avg_conf, 2)}",
        "Pothole Severity Distribution:",
        f"  - Small: {small}",
        f"  - Medium: {medium}",
        f"  - Large: {large}",
        f"Suggested Repair Urgency: {urgency}: {'Immediate repair needed' if urgency == 'High' else ('No action needed' if urgency == 'None' else 'Schedule repair')}"
    ]

    return render_template('detect.html',
                           uploaded_image=filename,
                           result_image=out_filename,
                           analysis_report=analysis_report,
                           stats={
                               'total': detected_count,
                               'avg_conf': round(avg_conf, 2),
                               'small': small,
                               'medium': medium,
                               'large': large,
                               'urgency': urgency
                           })

def process_video(filename, filepath):
    # Video vars
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video file")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions")
    if fps <= 0:
        fps = 24

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        duration_sec = total_frames / float(fps)
        if duration_sec > MAX_VIDEO_SECONDS:
            cap.release()
            raise RuntimeError(
                f"Video too long ({int(duration_sec)}s). Max supported length is {MAX_VIDEO_SECONDS}s."
            )
    
    # Optimized Output FPS (Quarter of source for faster processing on limited cloud CPU)
    out_fps = max(1, fps // 4)

    # Try WebM first; fallback to MP4 if codec unavailable in container.
    base_name = os.path.splitext(filename)[0]
    out_filename = 'pred_' + base_name + '.webm'
    out_path = os.path.join(app.config['PREDICTION_FOLDER'], out_filename)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

    if not out.isOpened():
        out_filename = 'pred_' + base_name + '.mp4'
        out_path = os.path.join(app.config['PREDICTION_FOLDER'], out_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError("No supported output codec found for processed video")
    
    # Tracking Variables
    unique_pothole_ids = set()
    total_video_conf = 0
    total_video_boxes = 0
    
    # Severity Tracking per ID (Stores max severity seen for each ID)
    # 0=Small, 1=Medium, 2=Large
    id_severity_map = {} 

    frame_count = 0
    frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Skip frames to reduce inference load (Process 1 in VIDEO_FRAME_SKIP)
        if frame_count % VIDEO_FRAME_SKIP != 0:
            continue
        
        frames_processed += 1
        if frames_processed % 10 == 0:
            print(f"[VIDEO] Processing frame {frame_count}/{total_frames}...")

        # Run Tracking (auto-enhance dark frames so night videos still detect)
        track_frame = enhance_for_detection(frame)
        local_model = ensure_local_model_loaded()
        results = local_model.track(
            track_frame,
            conf=0.2,
            iou=0.5,
            persist=True,
            verbose=False,
            tracker="botsort.yaml",
            classes=[0],
            imgsz=INFER_IMGSZ,
        )
        result = results[0]
        boxes = result.boxes
        
        img_h, img_w = frame.shape[:2]
        img_area = img_h * img_w
        
        if boxes.id is not None:
            # We have IDs
            ids = boxes.id.cpu().numpy().astype(int)
            
            for i, box in enumerate(boxes):
                obj_id = ids[i]
                unique_pothole_ids.add(obj_id)
                
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Accumulate confidence
                total_video_conf += conf
                total_video_boxes += 1
                
                w = x2 - x1
                h = y2 - y1
                ratio = ((w * h) / img_area) * 100

                # Filter out objects that are TOOO big (likely false positives) -> >40% of screen
                if ratio > 40.0:
                    continue
                
                # Determine Severity
                if ratio < 1.0:
                    severity = "Small"
                    sev_score = 0
                    color = (0, 255, 0)
                elif 1.0 <= ratio < 4.0:
                    severity = "Medium"
                    sev_score = 1
                    color = (0, 255, 255)
                else:
                    severity = "Large"
                    sev_score = 2
                    color = (0, 0, 255)
                
                # Update Max Severity for this ID
                if obj_id not in id_severity_map:
                    id_severity_map[obj_id] = sev_score
                else:
                    # Keep the highest severity seen for this pothole
                    if sev_score > id_severity_map[obj_id]:
                        id_severity_map[obj_id] = sev_score

                # Draw Info
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{severity}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Top-left overlay for video
        cv2.putText(frame, f"Total Potholes Found: {len(unique_pothole_ids)}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        out.write(frame)
        
    cap.release()
    out.release()
    
    # --- Final Stats Calculation ---
    total_unique = len(unique_pothole_ids)
    
    # Severity Counts based on unique IDs
    small_count, medium_count, large_count = 0, 0, 0
    
    for uid, score in id_severity_map.items():
        if score == 0:
            small_count += 1
        elif score == 1:
            medium_count += 1
        elif score == 2:
            large_count += 1
            
    # Max Urgency
    if total_unique == 0:
        max_urgency = "None"
    elif large_count > 0:
        max_urgency = "High"
    elif medium_count > 0:
        max_urgency = "Medium"
    else:
        max_urgency = "Low"

    # Calculate Average Confidence
    avg_conf_video = 0
    if total_video_boxes > 0:
        avg_conf_video = round((total_video_conf / total_video_boxes), 2)

    return render_template('detect.html',
                           uploaded_video=filename,
                           result_video=out_filename,
                           video_mode=True,
                           analysis_report=[],
                           stats={
                               'total': total_unique,
                               'avg_conf': avg_conf_video,
                               'small': small_count,
                               'medium': medium_count,
                               'large': large_count,
                               'urgency': max_urgency
                           })


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        'detect.html',
        error_msg=f"Uploaded file is too large. Max allowed size is {MAX_UPLOAD_MB} MB.",
        uploaded_image=None,
        uploaded_video=None,
    ), 413

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    if not session.get('user_id'):
        return {"error": "Unauthorized"}, 401

    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64
        header, encoded = image_data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}, 400

        # Run Inference (selectable backend + low-light boost for night cameras)
        dets = unified_predict(img, conf=0.15, iou=0.5, augment=False)

        detections = []
        for det in dets:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['conf']

            img_h, img_w = img.shape[:2]
            box_area = (x2 - x1) * (y2 - y1)
            ratio = (box_area / (img_h * img_w)) * 100

            if ratio > 40.0: continue
            if ratio < 0.01: continue

            if ratio < 1.0:
                severity = "Small"
                color = "#00ff00"
            elif 1.0 <= ratio < 4.0:
                severity = "Medium"
                color = "#ffff00"
            else:
                severity = "Large"
                color = "#ff0000"

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "severity": severity,
                "color": color
            })

        return {"detections": detections}

    except Exception as e:
        print(f"Error in detect_frame: {e}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '7860'))
    app.run(debug=debug_mode, host=host, port=port)
