import os
import urllib.request
import time
import cv2
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()

class MediaPipeHandler():
    # הגדרת נתיבים קשיחים כברירת מחדל
    MODEL_PATH = "assets/models/face_landmarker.task"

    def __init__(self, mode="VIDEO", num_faces=1, min_confidence=0.5):
        self.mode = mode.upper()
        self.latest_result = None
        self.last_timestamp_ms = 0

        # 1. וודא שהמודל קיים
        self._ensure_model_exists()

        # 2. הגדרות בסיסיות
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        
        if self.mode == "LIVE":
            running_mode = vision.RunningMode.LIVE_STREAM
            callback = self._live_callback
        else:
            running_mode = vision.RunningMode.VIDEO
            callback = None

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces, 
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True, # נדב עשוי להזדקק לזה לניתוח רגשות/דיוק
            result_callback=callback
        )

        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"❌ Error initializing MediaPipe: {e}")
            raise e

    def _ensure_model_exists(self):
        # יצירת התיקייה אם היא לא קיימת
        model_dir = os.path.dirname(self.MODEL_PATH)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading MediaPipe model to {self.MODEL_PATH}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.MODEL_PATH)
            print("✅ Model downloaded successfully.")

    def _live_callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def process(self, frame, timestamp_ms=None):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self.mode == "LIVE":
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            self.landmarker.detect_async(mp_image, timestamp_ms)
            return self.latest_result
        else:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for VIDEO mode")
            return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        if self.landmarker:
            self.landmarker.close()