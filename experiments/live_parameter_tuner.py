import os
import cv2
import numpy as np
import time
import sys
from dotenv import load_dotenv

# --- Imports setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.mediapipe_handler import MediaPipeHandler
from utils.video_processing import (
    extract_face_data, 
    calc_face_size_ratio, 
    calc_yaw_angle,
    calc_movement_metrics,
    get_mouth_roi_params,
    calc_pitch_angle  
)

# Load environment variables
load_dotenv()

def nothing(x):
    pass

def main():
    print("🚀 Starting Live Tuner with matching .env parameters... Press 'q' to quit.")
    
    # 1. Load exact parameters (with defaults if missing)
    ENV_YAW_MIN = float(os.getenv("YAW_THRESHOLD_MIN"))
    ENV_YAW_MAX = float(os.getenv("YAW_THRESHOLD_MAX"))
    
    # New Pitch parameters (Absolute values for looking Up and looking Down)
    ENV_PITCH_UP = float(os.getenv("PITCH_THRESHOLD_UP"))
    ENV_PITCH_DOWN = float(os.getenv("PITCH_THRESHOLD_DOWN"))
    
    ENV_MIN_RATIO = float(os.getenv("MIN_FACE_RATIO"))
    ENV_MAX_MOVE = float(os.getenv("MOVEMENT_THRESHOLD"))
    ENV_MAX_DYAW = float(os.getenv("YAW_THRESHOLD"))
    ENV_MAX_DSIZE = float(os.getenv("SIZE_THRESHOLD"))

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Live Tuner", cv2.WINDOW_NORMAL)
    
    # 2. Create trackbars (OpenCV only accepts integers, so we multiply and divide later)
    cv2.createTrackbar("Yaw Min (x100)", "Live Tuner", int(ENV_YAW_MIN * 100), 100, nothing) 
    cv2.createTrackbar("Yaw Max (x100)", "Live Tuner", int(ENV_YAW_MAX * 100), 1000, nothing) 
    
    # Pitch Trackbars
    cv2.createTrackbar("Max Pitch Up (x100)", "Live Tuner", int(ENV_PITCH_UP * 100), 100, nothing) 
    cv2.createTrackbar("Max Pitch Dn (x100)", "Live Tuner", int(ENV_PITCH_DOWN * 100), 100, nothing) 
    
    cv2.createTrackbar("Min Ratio (x100)", "Live Tuner", int(ENV_MIN_RATIO * 100), 50, nothing) 
    cv2.createTrackbar("Max Move (x1000)", "Live Tuner", int(ENV_MAX_MOVE * 1000), 200, nothing) 
    cv2.createTrackbar("Max dYaw (x100)", "Live Tuner", int(ENV_MAX_DYAW * 100), 200, nothing) 
    cv2.createTrackbar("Max dSize (x100)", "Live Tuner", int(ENV_MAX_DSIZE * 100), 100, nothing) 

    # Initialize MediaPipe handler
    mp_handler = MediaPipeHandler(mode="LIVE", num_faces=2)

    prev_landmarks = None
    prev_yaw = 0.0
    prev_ratio = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # --- Read live trackbar values and convert back to decimals ---
        yaw_min = max(0.01, cv2.getTrackbarPos("Yaw Min (x100)", "Live Tuner") / 100.0)
        yaw_max = cv2.getTrackbarPos("Yaw Max (x100)", "Live Tuner") / 100.0
        
        pitch_up_max = cv2.getTrackbarPos("Max Pitch Up (x100)", "Live Tuner") / 100.0
        pitch_down_max = cv2.getTrackbarPos("Max Pitch Dn (x100)", "Live Tuner") / 100.0
        
        min_ratio = cv2.getTrackbarPos("Min Ratio (x100)", "Live Tuner") / 100.0
        move_thresh = cv2.getTrackbarPos("Max Move (x1000)", "Live Tuner") / 1000.0
        dyaw_thresh = cv2.getTrackbarPos("Max dYaw (x100)", "Live Tuner") / 100.0
        dsize_thresh = cv2.getTrackbarPos("Max dSize (x100)", "Live Tuner") / 100.0

        result = mp_handler.process(frame)

        is_taken = False
        reject_reason = ""
        current_yaw, current_pitch, current_ratio = 0.0, 0.0, 0.0
        dist, yaw_diff, size_diff = 0.0, 0.0, 0.0
        anchors = None

        if not result or not result.face_landmarks:
            reject_reason = "No Face Detected"
            prev_landmarks = None
        elif len(result.face_landmarks) > 1:
            reject_reason = "Multiple Faces Detected"
            prev_landmarks = None
        else:
            landmarks = result.face_landmarks[0]
            
            # Calculate current values
            current_yaw = calc_yaw_angle(landmarks)
            current_pitch = calc_pitch_angle(landmarks) # <-- Pitch calculation
            current_ratio = calc_face_size_ratio(landmarks)
            anchors = extract_face_data(landmarks, w, h)
            
            if prev_landmarks is not None:
                dist, yaw_diff, size_diff = calc_movement_metrics(
                    landmarks, prev_landmarks, current_yaw, prev_yaw, current_ratio, prev_ratio)

            # --- Condition chain (identical to your pipeline logic) ---
            
            # 1. Yaw Angle Check (Looking left/right)
            if not (yaw_min <= current_yaw <= yaw_max):
                reject_reason = f"Bad Yaw Angle ({current_yaw:.2f})"
                prev_landmarks = None
                
            # 2. Pitch Angle Check (Looking up/down)
            # current_pitch is positive for looking UP, negative for looking DOWN
            elif not (-pitch_down_max <= current_pitch <= pitch_up_max):
                reject_reason = f"Bad Pitch Angle ({current_pitch:.2f})"
                prev_landmarks = None
                
            # 3. Face Size Check
            elif current_ratio < min_ratio:
                reject_reason = f"Face Too Small ({current_ratio:.2f})"
                prev_landmarks = None
                
            # 4. Movement Checks (Broken down to see exactly what failed)
            elif prev_landmarks is not None and dist > move_thresh:
                reject_reason = f" Dist ({dist:.3f} > {move_thresh:.3f})"
                prev_landmarks = None
                
            elif prev_landmarks is not None and yaw_diff > dyaw_thresh:
                reject_reason = f" dYaw ({yaw_diff:.2f} > {dyaw_thresh:.2f})"
                prev_landmarks = None
                
            elif prev_landmarks is not None and size_diff > dsize_thresh:
                reject_reason = f" dSize ({size_diff:.2f} > {dsize_thresh:.2f})"
                prev_landmarks = None
                
            else:
                is_taken = True
                prev_landmarks = landmarks
                prev_yaw = current_yaw
                prev_ratio = current_ratio

        # ==========================================
        # 🎨 Visualization (UI)
        # ==========================================
        banner_color = (0, 200, 0) if is_taken else (0, 0, 200)
        cv2.rectangle(frame, (0, 0), (w, h), banner_color, 8)

        if anchors:
            try:
                roi_params = get_mouth_roi_params(anchors, w, h, is_normalized=False)
                box = cv2.boxPoints(roi_params)
                box = np.int32(box)
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            except: pass

        # Top data block (helps you see the exact numbers at any moment)
        cv2.rectangle(frame, (10, 10), (600, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Yaw: {current_yaw:.2f} | Pitch: {current_pitch:.2f} | Ratio: {current_ratio:.3f}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Move: D:{dist:.3f} | dY:{yaw_diff:.2f} | dS:{size_diff:.2f}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Bottom banner
        banner_y_start = h - 80
        cv2.rectangle(frame, (0, banner_y_start), (w, h), banner_color, -1)
        
        status_text = "✅ TAKEN" if is_taken else f"❌ {reject_reason}"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        cv2.putText(frame, status_text, ((w - text_size[0]) // 2, h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.imshow("Live Tuner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_handler.close()

if __name__ == "__main__":
    main()