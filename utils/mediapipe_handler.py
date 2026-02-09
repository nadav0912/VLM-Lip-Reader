import os
import urllib.request
import time
import cv2
from dotenv import load_dotenv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv()

"""
MediaPipe Face Landmarker:
https://developers.google.com/mediapipe/solutions/vision/face_landmarker

This class is a wrapper for the MediaPipe Face Landmarker, 
It handles the model download, initialization, and processing of face landmarks.
It supports both video and live stream processing.
It returns the face landmarks as a DetectionResult object.
"""

class MediaPipeHandler():
    MODEL_PATH = os.getenv("MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH")
    LIPS_INDICES = os.getenv("LIPS_INDICES")

    def __init__(self, mode="VIDEO", num_faces=1, min_confidence=0.5):
        """  
        Args:
            mode (str): "VIDEO" for video processing, "LIVE" for live stream processing.
            num_faces (int): how many faces to detect (1 for LIVE, 2 for VIDEO to detect interruptions).
            min_confidence (float): minimum confidence (0.5 default).
        """
        self.mode = mode.upper()
        self.latest_result = None  # Used to store the latest result in LIVE mode
        self.last_timestamp_ms = 0 # Used to prevent timestamp errors in LIVE mode

        # 1. Check if the model exists
        self._ensure_model_exists()

        # 2. Set the basic options
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        
        # 3. Set the running mode and callback
        if self.mode == "LIVE":
            running_mode = vision.RunningMode.LIVE_STREAM
            callback = self._live_callback
            print(f"MediaPipe configured for LIVE STREAM (Max Faces: {num_faces})")
        else:
            running_mode = vision.RunningMode.VIDEO
            callback = None
            print(f"MediaPipe configured for VIDEO PROCESSING (Max Faces: {num_faces})")

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces, 
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=False, # Not needed for lip reading
            output_facial_transformation_matrixes=False,
            result_callback=callback
        )

        # 4. Create the landmarker
        try:
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"❌ Error initializing MediaPipe: {e}")
            raise e

    def _ensure_model_exists(self):
        #Download the model if it doesn't exist
        if not os.path.exists(os.path.dirname(self.MODEL_PATH)):
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading MediaPipe model to {self.MODEL_PATH}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.MODEL_PATH)
            print("✅ Model downloaded successfully.")

    def _live_callback(self, result, output_image, timestamp_ms):
        # Internal callback function called by MediaPipe when the result is ready (in LIVE mode)
        self.latest_result = result

    def process(self, frame, timestamp_ms=None):
        """
        Main function for face processing.
        
        Args:
            frame: Image in BGR format (like OpenCV returns).
            timestamp_ms: Timestamp of the frame in milliseconds. Required for VIDEO, optional for LIVE.
        
        Returns:
            DetectionResult object or None if no result is available yet.
        """
        # Convert to RGB (required for MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- LIVE MODE ---
        if self.mode == "LIVE":
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000)
            
            # Prevent duplicate or backward timestamps (causes MediaPipe crash)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms

            # Send to asynchronous processing (won't block the program)
            self.landmarker.detect_async(mp_image, timestamp_ms)
            
            # Return the latest known result (to draw on the screen)
            return self.latest_result
        
        # --- VIDEO MODE ---
        else:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for VIDEO mode")
                
            # Synchronous processing (waits for the result)
            return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        # Release resources at the end
        if self.landmarker:
            self.landmarker.close()