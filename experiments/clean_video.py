import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# --- הגדרות ---
VIDEO_PATH = "videos/7 Principles For Teenagers To Become Millionaires.mp4"       # הסרטון שלך
JSON_PATH = "transcripts/7 Principles For Teenagers To Become Millionaires.json"       # ה-JSON של WhisperX
OUTPUT_PATH = "videos/7 Principles For Teenagers To Become Millionaires_clean.mp4"

# סף רגישות לסיבוב ראש (יחס בין אוזניים לאף)
YAW_THRESHOLD_MIN = 0.25
YAW_THRESHOLD_MAX = 4.0

def get_dominant_speaker_intervals(json_data):
    """
    1. מוצא את הדובר הראשי.
    2. מחזיר רשימה של זמנים (Start, End) שבהם הוא מדבר.
    """
    # זיהוי דובר ראשי
    counts = {}
    for seg in json_data['segments']:
        spk = seg.get('speaker', 'Unknown')
        counts[spk] = counts.get(spk, 0) + len(seg.get('words', []))
    
    dominant_speaker = max(counts, key=counts.get)
    print(f"🎤 Dominant Speaker: {dominant_speaker}")

    # איסוף זמנים של הדובר הראשי בלבד
    intervals = []
    other_speaker_intervals = [] # זמנים שבהם מישהו אחר מדבר (אסור לשמור!)
    
    for seg in json_data['segments']:
        if seg.get('speaker') == dominant_speaker:
            intervals.append((seg['start'], seg['end']))
        else:
            other_speaker_intervals.append((seg['start'], seg['end']))
            
    return intervals, other_speaker_intervals

def is_time_in_intervals(current_time, intervals):
    for start, end in intervals:
        if start <= current_time <= end:
            return True
    return False

def check_face_quality(frame, face_mesh):
    """
    בודק ויזואליה:
    1. יש פרצוף אחד?
    2. הוא מסתכל למצלמה?
    """
    # המרת צבע ל-RGB עבור MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return False, "No Face"
    
    if len(results.multi_face_landmarks) != 1:
        return False, "Multiple Faces"

    # בדיקת זווית (Yaw)
    lm = results.multi_face_landmarks[0].landmark
    nose_x = lm[1].x
    left_ear_x = lm[234].x
    right_ear_x = lm[454].x

    dist_l = abs(nose_x - left_ear_x)
    dist_r = abs(nose_x - right_ear_x)
    ratio = dist_l / (dist_r + 1e-6)

    if ratio < YAW_THRESHOLD_MIN or ratio > YAW_THRESHOLD_MAX:
        return False, "Bad Angle"

    return True, "OK"

def main():
    # 1. טעינת נתונים
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dom_intervals, other_intervals = get_dominant_speaker_intervals(data)

    # 2. הכנת וידאו
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # הכנת Writer לשמירת התוצאה
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # הכנת שכבה אדומה לפילטר
    red_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    red_overlay[:] = (0, 0, 255) # צבע אדום (BGR)

    # אתחול MediaPipe בשיטת VIDEO
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, # מצב וידאו (מהיר יותר)
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    print("🚀 Processing video... (Red = Discarded, Normal = Kept)")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_time = frame_idx / fps
        frame_idx += 1

        # --- לוגיקת ההחלטה (Decision Logic) ---
        
        # 1. בדיקת ויזואלית (תנאי סף עליון)
        is_visual_good, reason = check_face_quality(frame, mp_face_mesh)
        
        # 2. בדיקת אודיו
        is_dominant = is_time_in_intervals(current_time, dom_intervals)
        is_other_speaker = is_time_in_intervals(current_time, other_intervals)

        # 3. ההחלטה הסופית: 1 (שמור) או 0 (זרוק)
        should_keep = False
        
        if is_visual_good:
            if is_dominant:
                should_keep = True  # מדבר + רואים אותו = שמור
            elif not is_other_speaker:
                should_keep = True  # שותק (אף אחד לא מדבר) + רואים אותו = שמור (<SIL>)
            else:
                should_keep = False # מישהו אחר מדבר = זרוק
                reason = "Other Speaker"
        
        # --- צביעת הפריים ---
        if not should_keep:
            # הוספת פילטר אדום (Blending)
            frame = cv2.addWeighted(frame, 0.7, red_overlay, 0.3, 0)
            
            # כתיבת הסיבה על המסך (לדיבוג)
            cv2.putText(frame, f"DISCARD: {reason}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            # פריים תקין - סימון ירוק קטן
            cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)

        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}, Time: {current_time:.2f}s")

    cap.release()
    out.release()
    mp_face_mesh.close()
    print(f"✅ Debug video saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()