import cv2
import mediapipe as mp
import time
import os
import urllib.request

# --- 1. Model Setup ---
MODEL_FILE = 'face_landmarker.task'
if not os.path.exists(MODEL_FILE):
    print("Downloading model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", MODEL_FILE)

# --- 2. Lip Landmark Indices ---
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
ALL_LIP_INDICES = list(set(LIPS_OUTER + LIPS_INNER))

# --- 3. Stats Tracking ---
latency_history = []
fps_on_history = []
fps_off_history = []
frame_timestamps = {}
latest_latency = 0
latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result, latest_latency
    latest_result = result
    if timestamp_ms in frame_timestamps:
        start_time = frame_timestamps.pop(timestamp_ms)
        latency = (time.time() - start_time) * 1000
        latest_latency = latency
        latency_history.append(latency)

# --- 4. Initialize MediaPipe ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    
    num_faces=1, 
    output_face_blendshapes=False, 
    output_facial_transformation_matrixes=False,

    # Confidence parameters returned as requested:
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Try setting to a high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Max Hardware Supported Resolution: {int(actual_w)}x{int(actual_h)}")

p_time = time.time()
detection_active = True

print("!!! CLICK THE VIDEO WINDOW TO MAKE KEYS WORK !!!")
print("Press 'D' to Toggle | Press 'Q' to Quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    # Get current frame dimensions for drawing
    h, w, _ = frame.shape 
    ms_timestamp = int(time.time() * 1000)

    if detection_active:
        frame_timestamps[ms_timestamp] = time.time()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detector.detect_async(mp_image, ms_timestamp)

        # Draw dots
        if latest_result and latest_result.face_landmarks:
            for face_landmarks in latest_result.face_landmarks:
                for idx in ALL_LIP_INDICES:
                    landmark = face_landmarks[idx]
                    # Map normalized coordinates to pixel values
                    x_pixel, y_pixel = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x_pixel, y_pixel), 2, (0, 255, 255), -1) 
    
    # Calculate Frame FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    if detection_active: fps_on_history.append(fps)
    else: fps_off_history.append(fps)

    # Visual Feedback
    color = (0, 255, 0) if detection_active else (0, 0, 255)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    if detection_active:
        cv2.putText(frame, f"AI Lag: {int(latest_latency)}ms", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow('Face Detection Test', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: 
        break
    elif key == ord('d'):
        detection_active = not detection_active
        latest_result = None 
        print(f"Detection Toggled: {detection_active}")

# Cleanup
cap.release()
detector.close()
cv2.destroyAllWindows()

# --- Final Summary ---
print("\n" + "="*40)
print("             FINAL SUMMARY")
print("="*40)
if fps_off_history:
    print(f"Avg FPS (Model OFF): {sum(fps_off_history)/len(fps_off_history):.2f}")
if fps_on_history:
    print(f"Avg FPS (Model ON):  {sum(fps_on_history)/len(fps_on_history):.2f}")
if latency_history:
    print(f"Avg AI Latency (Lag): {sum(latency_history)/len(latency_history):.2f} ms")
print(f"Camera Resolution: {int(actual_w)}x{int(actual_h)}")
print("="*40)