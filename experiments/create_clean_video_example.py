import cv2
import json
import numpy as np
import mediapipe as mp
import os
import urllib.request
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


# --- Settings ---
# Make sure the paths are correct on your computer!
VIDEO_PATH = "data/01_raw_videos/7 Principles For Teenagers To Become Millionaires.mp4"
JSON_PATH = "data/02_raw_transcripts/7 Principles For Teenagers To Become Millionaires.json"
OUTPUT_PATH = "data/03_final_videos/7 Principles For Teenagers To Become Millionaires_clean.mp4"
TEMP_VIDEO_PATH = "temp_video_no_audio.mp4"
MODEL_PATH = "assets/models/face_landmarker.task"  

# Sensitivity for head rotation
YAW_THRESHOLD_MIN = 0.25
YAW_THRESHOLD_MAX = 4.0


def download_model_if_missing():
    # Download the model if it's not in the assets folder
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading face_landmarker.task model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✅ Model downloaded successfully.")

def get_dominant_speaker_intervals(json_data):
    # Get the dominant speaker intervals
    counts = {}
    for seg in json_data['segments']:
        spk = seg.get('speaker', 'Unknown')
        counts[spk] = counts.get(spk, 0) + len(seg.get('words', []))
    
    if not counts: return [], []
    dominant_speaker = max(counts, key=counts.get)
    print(f"🎤 Dominant Speaker: {dominant_speaker}")

    intervals = []
    other_speaker_intervals = []
    
    for seg in json_data['segments']:
        if seg.get('speaker') == dominant_speaker:
            intervals.append((seg['start'], seg['end']))
        else:
            other_speaker_intervals.append((seg['start'], seg['end']))
            
    return intervals, other_speaker_intervals

def is_time_in_intervals(current_time, intervals):
    # Check if the current time is in the intervals
    for start, end in intervals:
        if start <= current_time <= end:
            return True
    return False

def check_face_quality_new(detection_result):
    # Check face quality using the new API (Tasks API)
    # Receive a DetectionResult object
    
    # Check if there is a face
    if not detection_result.face_landmarks:
        return False, "No Face"
    
    if len(detection_result.face_landmarks) != 1:
        return False, "Multiple Faces"

    # Check the angle (Yaw)
    landmarks = detection_result.face_landmarks[0]
    
    nose_x = landmarks[1].x
    left_ear_x = landmarks[234].x
    right_ear_x = landmarks[454].x

    dist_l = abs(nose_x - left_ear_x)
    dist_r = abs(nose_x - right_ear_x)
    
    # Prevent division by zero
    ratio = dist_l / (dist_r + 1e-6) 

    if ratio < YAW_THRESHOLD_MIN or ratio > YAW_THRESHOLD_MAX:
        return False, "Bad Angle"

    return True, "OK"

def draw_text_centered(frame, text, y_pos, font_scale=1.0, color=(255, 255, 255), thickness=2):
    # Draw text with a black outline for maximum readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    
    # Outline (black outline)
    cv2.putText(frame, text, (text_x, y_pos), font, font_scale, (0, 0, 0), thickness + 4)
    # The text itself
    cv2.putText(frame, text, (text_x, y_pos), font, font_scale, color, thickness)

def get_current_subtitles(current_time, json_data):
    # Get the exact text for the current time
    sentence_text = ""
    word_text = ""

    # Search for the sentence (Segment)
    for seg in json_data.get('segments', []):
        if seg['start'] <= current_time <= seg['end']:
            sentence_text = seg['text'].strip()
            break
    
    # Search for the word (Word) - Assume the structure is like you sent
    words_list = json_data.get('word_segments', [])
    
    for w in words_list:
        if w['start'] <= current_time <= w['end']:
            word_text = w['word'].strip()
            break
            
    return sentence_text, word_text

def main():
    # 0. Download the model if it's missing
    download_model_if_missing()

    # 1. Load the JSON file
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dom_intervals, other_intervals = get_dominant_speaker_intervals(data)

    # 2. Initialize MediaPipe 
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        running_mode=vision.RunningMode.VIDEO, # מצב וידאו חובה!
        num_faces=2,
        min_face_detection_confidence=0.5
    )

    # 3. Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video at {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, fps, (width, height))

    # Crate color layers
    red_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    red_overlay[:] = (0, 0, 255) 
    green_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    green_overlay[:] = (0, 255, 0)

    print("🚀 Processing video with New MediaPipe API...")

    # Create the Landmarker inside the context manager
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        with tqdm(total=total_frames, unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # MediaPipe Tasks requires timestamp in milliseconds
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                current_time_sec = timestamp_ms / 1000.0
                
                # Convert to MediaPipe format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # --- Detection ---
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # --- Logic ---
                is_visual_good, reason = check_face_quality_new(detection_result)
                is_dominant = is_time_in_intervals(current_time_sec, dom_intervals)
                is_other_speaker = is_time_in_intervals(current_time_sec, other_intervals)

                should_keep = False
                
                if is_visual_good:
                    if is_dominant:
                        should_keep = True
                    elif not is_other_speaker:
                        should_keep = True # שתיקה לגיטימית
                    else:
                        should_keep = False
                        reason = "Other Speaker"
                
                
                # 1. Filters
                if should_keep:
                    cv2.addWeighted(frame, 0.8, green_overlay, 0.2, 0, frame)
                else:
                    cv2.addWeighted(frame, 0.5, red_overlay, 0.5, 0, frame)
                    cv2.putText(frame, f"DISCARD: {reason}", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 2. Text Overlay 
                sentence, word = get_current_subtitles(current_time_sec, data)
                
                if sentence:
                    draw_text_centered(frame, sentence, height - 120, font_scale=1.1, thickness=2)
                
                if word:
                    draw_text_centered(frame, word, height - 50, font_scale=1.5, color=(0, 255, 255), thickness=3)

                out.write(frame)
                
                pbar.update(1) # Updates the bar by 1 frame

    cap.release()
    out.release()
    print("✅ Visual processing done. Merging audio...")

    # Merge audio 
    try:
        original_clip = VideoFileClip(VIDEO_PATH)
        new_video_clip = VideoFileClip(TEMP_VIDEO_PATH)
        
        # Merge the original audio to the new video
        final_clip = new_video_clip.with_audio(original_clip.audio)        
        
        final_clip.write_videofile(OUTPUT_PATH, codec='libx264', audio_codec='aac', logger=None)
        
        # Clean up
        original_clip.close()
        new_video_clip.close()
        final_clip.close()
        os.remove(TEMP_VIDEO_PATH)
        print(f"🎉 SUCCESS! Clean video saved at:\n{OUTPUT_PATH}")
        
    except Exception as e:
        print(f"❌ Error merging audio: {e}")

if __name__ == "__main__":
    main()