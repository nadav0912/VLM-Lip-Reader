import cv2
import numpy as np
import math
import mediapipe as mp
import os
import sys

# Import Utils (for logging)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.mediapipe_face import MediaPipeHandler

# --- קבועים ---
# אינדקסים לעיניים (לחישוב זווית הראש)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263

# אינדקסים לכל נקודות השפתיים (חיצוני + פנימי) לקבלת מרכז מדויק
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]

# כמה להגדיל את הריבוע ביחס לרוחב הפה
CROP_SCALE = 2.2 

def get_rotated_mouth_roi(landmarks, img_w, img_h):
    """
    מחשב את נתוני הריבוע המסתובב סביב הפה.
    מחזיר:
    - center: (x, y) מרכז הפה
    - size: (width, height) גודל הריבוע (ריבוע שווה צלעות)
    - angle: זווית הסיבוב במעלות
    - eyes: קואורדינטות של העיניים לציור
    """
    
    # 1. חישוב זווית לפי העיניים
    p_eye_l = landmarks[IDX_LEFT_EYE]
    p_eye_r = landmarks[IDX_RIGHT_EYE]
    
    # המרה לפיקסלים
    el_x, el_y = p_eye_l.x * img_w, p_eye_l.y * img_h
    er_x, er_y = p_eye_r.x * img_w, p_eye_r.y * img_h
    
    # חישוב דלתא (הפרשים)
    dy = er_y - el_y
    dx = er_x - el_x
    
    # חישוב זווית (בגלל ש-Y הפוך בתמונה, הזווית חיובית עם השעון)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # 2. חישוב מרכז הפה ורוחב הפה (מכל הנקודות)
    lip_points_x = []
    lip_points_y = []
    
    for idx in LIPS_INDICES:
        p = landmarks[idx]
        lip_points_x.append(p.x * img_w)
        lip_points_y.append(p.y * img_h)
        
    # המרכז הוא הממוצע של כל הנקודות
    center_x = sum(lip_points_x) / len(lip_points_x)
    center_y = sum(lip_points_y) / len(lip_points_y)
    
    # 3. חישוב גודל הריבוע
    # אנחנו רוצים שהריבוע יתבסס על רוחב הפה (המרחק בין המינימום למקסימום ב-X)
    # *הערה*: זה חישוב מקורב לרוחב לפני סיבוב, אבל מספיק טוב לקביעת קנה מידה
    min_x, max_x = min(lip_points_x), max(lip_points_x)
    mouth_width_raw = max_x - min_x
    
    # גודל הריבוע הסופי (רוחב = גובה)
    box_size = mouth_width_raw * CROP_SCALE
    
    return (center_x, center_y), (box_size, box_size), angle_deg, (int(el_x), int(el_y)), (int(er_x), int(er_y))

def main():
    handler = MediaPipeHandler(mode="LIVE", num_faces=1)
    
    cap = cv2.VideoCapture(0)
    # הגדרת רזולוציה
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🎥 Rotating Crop Test Started.")
    print("Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        result = handler.process(frame)

        if result and result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # --- חישוב הגיאומטריה ---
            center, size, angle, eye_l, eye_r = get_rotated_mouth_roi(landmarks, w, h)

            # --- ציור ויזואלי ---

            # 1. קו העיניים והזווית (כדי לוודא שהחישוב נכון)
            color_eyes = (0, 255, 255) # צהוב
            cv2.line(frame, eye_l, eye_r, color_eyes, 1)
            cv2.circle(frame, eye_l, 4, color_eyes, -1)
            cv2.circle(frame, eye_r, 4, color_eyes, -1)
            cv2.putText(frame, f"Angle: {angle:.1f}", (eye_l[0], eye_l[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_eyes, 2)

            # 2. חישוב 4 הפינות של הריבוע המסובב
            # הפונקציה BoxPoints יודעת לקחת (מרכז, גודל, זווית) ולהחזיר פינות
            rect_struct = (center, size, angle) 
            box_points = cv2.boxPoints(rect_struct) 
            box_points = np.int32(box_points) # המרה למספרים שלמים

            # 3. ציור הריבוע המסתובב
            # ירוק כשהזווית סבירה, אדום כשהראש עקום מדי
            color_box = (0, 255, 0)
            if abs(angle) > 15: color_box = (0, 0, 255)
            
            cv2.drawContours(frame, [box_points], 0, color_box, 2)

            # 4. ציור מרכז הפה
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, (255, 0, 0), -1)

        cv2.imshow("Rotated Crop Logic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    handler.close()

if __name__ == "__main__":
    main()