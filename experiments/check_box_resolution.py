import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request

# --- ייבוא ה-API החדש ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- הגדרות ---
MODEL_PATH = "face_landmarker.task"

# אינדקסים של השפתיים (המעטפת החיצונית)
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

# משתנה גלובלי לשמירת התוצאה האחרונה מה-Callback (בגלל מצב LIVE_STREAM)
latest_result = None

def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading face_landmarker.task model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

# --- פונקציית Callback (רצה בכל פעם שהמודל מסיים חישוב) ---
def result_callback(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def main():

    sum_widths = 0
    sum_heights = 0
    count = 0
    min_width = None
    max_width = None
    min_height = None
    max_height = None
    
    download_model_if_missing()

    # 1. הגדרת המצלמה
    cap = cv2.VideoCapture(0) # 0 = המצלמה הראשית
    
    # ניסיון להגדיר רזולוציה גבוהה (כדי לדמות 1080p או 720p)
    # שים לב: זה תלוי בחומרה של המצלמה שלך
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # בדיקה מה הרזולוציה שהתקבלה בפועל
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Camera active at resolution: {real_w}x{real_h}")

    # 2. הגדרת MediaPipe בשיטת LIVE_STREAM
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM, # <--- מצב לייב אמיתי
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback # חיבור לפונקציה שמתעדכנת
    )

    print("🚀 Starting Live Inspection. Press 'q' to exit.")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # שיקוף (Mirror) לנוחות המשתמש
            frame = cv2.flip(frame, 1)
            
            # חישוב Timestamp במילי-שניות (חובה ב-LIVE_STREAM)
            timestamp_ms = int(time.time() * 1000)
            
            # המרה ל-MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # שליחה לזיהוי (אסינכרוני - לא תוקע את הוידאו)
            landmarker.detect_async(mp_image, timestamp_ms)

            # --- ציור התוצאה האחרונה (אם יש) ---
            if latest_result and latest_result.face_landmarks:
                landmarks = latest_result.face_landmarks[0]
                
                # המרה לפיקסלים
                x_coords = [landmarks[i].x * real_w for i in MOUTH_INDICES]
                y_coords = [landmarks[i].y * real_h for i in MOUTH_INDICES]
                
                min_x, max_x = int(min(x_coords)), int(max(x_coords))
                min_y, max_y = int(min(y_coords)), int(max(y_coords))
                
                width_px = max_x - min_x
                height_px = max_y - min_y
                
                # בחירת צבע לפי האיכות
                if width_px < 60:
                    color = (0, 0, 255) # אדום - קטן מדי
                    status = "BAD (Too Small)"
                elif width_px < 90:
                    color = (0, 255, 255) # צהוב - גבולי
                    status = "OK (Borderline)"
                else:
                    color = (0, 255, 0) # ירוק - מעולה
                    status = "EXCELLENT"

                # ציור ריבוע סביב הפה
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
                
                # כתיבת מידע
                info_text = f"Mouth: {width_px}x{height_px} px | {status}"
                cv2.putText(frame, info_text, (min_x - 20, min_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # כתיבת רזולוציה כללית בצד
                cv2.putText(frame, f"Cam Res: {real_w}x{real_h}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # print the frame rate
                print(f"{width_px}x{height_px}")
                count += 1
                sum_widths += width_px
                sum_heights += height_px
                if min_width is None or width_px < min_width:
                    min_width = width_px
                if max_width is None or width_px > max_width:
                    max_width = width_px
                if min_height is None or height_px < min_height:
                    min_height = height_px
                if max_height is None or height_px > max_height:
                    max_height = height_px

            cv2.imshow('Mouth Resolution Inspector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("Average width: ", sum_widths / count)
    print("Average height: ", sum_heights / count)
    print("Min width: ", min_width)
    print("Max width: ", max_width)
    print("Min height: ", min_height)
    print("Max height: ", max_height)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()