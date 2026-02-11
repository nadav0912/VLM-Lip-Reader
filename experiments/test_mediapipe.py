import cv2
import mediapipe as mp
import os

# הגדרות
VIDEO_PATH = r"C:\VLM-Lip-Reader\data\01_raw_videos\Speaker_00_X_XNSeW9jok.mp4"

# אתחול MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils  # אם רוצים לצייר קווים

if not os.path.exists(VIDEO_PATH):
    print(f"❌ הסרטון לא נמצא בנתיב: {VIDEO_PATH}")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("🚀 מפעיל בדיקת זיהוי שפתיים... לחץ על 'q' כדי לצאת.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # המרה ל-RGB עבור MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # נקודות ציון של השפתיים
                for id in [61, 291, 0, 17]:
                    ih, iw, ic = frame.shape
                    x, y = int(face_landmarks.landmark[id].x * iw), int(face_landmarks.landmark[id].y * ih)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow('Lip Detection Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
