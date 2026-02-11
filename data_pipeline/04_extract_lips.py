import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# הגדרת נתיבים לפי המבנה שלך
BASE_DIR = r"C:\VLM-Lip-Reader"
INPUT_DIR = os.path.join(BASE_DIR, "data", "01_raw_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "02_lip_crops")

# יצירת תיקיית פלט אם לא קיימת
os.makedirs(OUTPUT_DIR, exist_ok=True)

# אתחול MediaPipe לפי הקוד שעבד לך בטסט
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_lip_crop(frame, landmarks):
    ih, iw, _ = frame.shape
    # נקודות הציון המקיפות את השפתיים (Outer Lips)
    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    
    x_coords = [int(landmarks.landmark[i].x * iw) for i in lip_indices]
    y_coords = [int(landmarks.landmark[i].y * ih) for i in lip_indices]
    
    if not x_coords or not y_coords: return None
    
    # חישוב Bounding Box עם Padding של 20 פיקסלים
    xmin, xmax = min(x_coords) - 20, max(x_coords) + 20
    ymin, ymax = min(y_coords) - 20, max(y_coords) + 20
    
    # וידוא שלא חרגנו מגבולות הפריים
    xmin, xmax = max(0, xmin), min(iw, xmax)
    ymin, ymax = max(0, ymin), min(ih, ymax)
    
    crop = frame[ymin:ymax, xmin:xmax]
    if crop.size == 0: return None
    return cv2.resize(crop, (112, 112)) # גודל סטנדרטי למודלי Lip-Reading

# רשימת קבצים בתיקייה (מדלג על קבצי temp/part)
video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4') and 'temp' not in f and 'f399' not in f]

if not video_files:
    print(f"⚠️ לא נמצאו סרטוני mp4 תקינים בנתיב: {INPUT_DIR}")
    exit()

print(f"🚀 מתחיל חילוץ שפתיים עבור {len(video_files)} סרטונים...")

for video_file in tqdm(video_files, desc="Processing"):
    cap = cv2.VideoCapture(os.path.join(INPUT_DIR, video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    out_path = os.path.join(OUTPUT_DIR, video_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            lip_frame = get_lip_crop(frame, results.multi_face_landmarks[0])
            if lip_frame is not None:
                if out is None:
                    h, w, _ = lip_frame.shape
                    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                out.write(lip_frame)
                
    cap.release()
    if out: out.release()

print(f"✨ הסתיים! בדוק את התיקייה: {OUTPUT_DIR}")