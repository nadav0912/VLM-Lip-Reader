import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

# --- Imports setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.common import setup_logger
from utils.mediapipe_handler import MediaPipeHandler

from utils.video_processing import (
    extract_face_data, 
    calc_face_size_ratio, 
    calc_yaw_angle,
    calc_movement_metrics
)

# --- Configuration ---
load_dotenv()

# Directories
INPUT_VIDEO_DIR = os.getenv("RAW_VIDEOS_DIR", "data/01_raw_videos")
INPUT_TRANSCRIPT_DIR = os.getenv("ROW_TRANSCRIPTS_DIR", "data/02_transcribed")
OUTPUT_ANALYSIS_DIR = os.getenv("ANALYSIS_DIR", "data/03_frame_analysis")
LOG_DIR = os.getenv("LOGS_DIR", "logs")

# Thresholds
MIN_FACE_RATIO = float(os.getenv("MIN_FACE_RATIO", 0.10))
YAW_MIN = float(os.getenv("YAW_THRESHOLD_MIN", 0.25))
YAW_MAX = float(os.getenv("YAW_THRESHOLD_MAX", 4.0))
MOVEMENT_THRESHOLD = float(os.getenv("MOVEMENT_THRESHOLD"))
YAW_THRESHOLD = float(os.getenv("YAW_THRESHOLD"))
SIZE_THRESHOLD = float(os.getenv("SIZE_THRESHOLD"))

os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger('analyzer', os.path.join(LOG_DIR, 'analyze_video.log'))


def precompute_audio_map(words, total_frames, fps):
    """
    Creates a array where:
    2 = Silence
    1 = Valid speech (run MediaPipe)
    0 = other speaker (skip MediaPipe)
    """
    audio_map = np.full(total_frames, 2, dtype=np.int8)

    for w in words:
        start_frame = int(w['start'] * fps)
        end_frame = int(w['end'] * fps)
        
        # Protect against array boundaries
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        # Skip words with length 0 or negative
        if start_frame >= end_frame:
            continue

        if w.get('keep'):
            # Main speaker is speaking -> status 1
            audio_map[start_frame:end_frame] = 1
        else:
            # Other speaker is speaking -> status 0 (reject)
            # We overwrite even if there was silence before
            audio_map[start_frame:end_frame] = 0
            
    return audio_map

def analyze_single_video(video_path, json_path):
    video_name = os.path.basename(video_path)
    file_id = os.path.splitext(video_name)[0]
    output_path = os.path.join(OUTPUT_ANALYSIS_DIR, f"{file_id}_analysis.json")

    if os.path.exists(output_path):
        print(f"⏩ Skipping {video_name} (Exists)")
        return

    # 1. Load Data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = data.get("words", [])
        if not words: return
    except Exception as e:
        logger.error(f"Error {video_name}: {e}")
        return

    # 2. Setup Video    
    mp_handler = MediaPipeHandler(mode="VIDEO", num_faces=2) # We detect 2 faces to check for occlusions
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-calculate duration per frame to avoid cap.get() calls
    frame_duration = 1.0 / fps 

    valid_frames_count = 0
    analyzed_frames = []
    
    # Pre-compute the audio map
    frame_statuses = precompute_audio_map(words, total_frames, fps)

    # Previous frame data
    prev_landmarks = None
    prev_yaw = 0.0
    prev_ratio = 0.0

    print(f"Analyzing {video_name}...")

    # Optimization: Update TQDM every 10 frames to reduce IO overhead
    with tqdm(total=total_frames, unit="fr", miniters=10) as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            current_time_sec = timestamp_ms / 1000.0

            # --- A. Optimized Audio Check ---
            if frame_idx >= len(frame_statuses): break
            audio_status = frame_statuses[frame_idx]            
            
            final_status = 0 # Default: Invalid
            anchors = None
            reject_reason = ""
            
            # --- Logic Flow ---
            if audio_status == 0:
                reject_reason = "Other Speaker"
                prev_landmarks = None
            else:
                # Only run heavy MediaPipe if audio is potentially good
                result = mp_handler.process(frame, timestamp_ms)

                if not result.face_landmarks:
                    reject_reason = "No Face"
                    prev_landmarks = None
                elif len(result.face_landmarks) > 1:
                    reject_reason = "Multiple Faces"
                    prev_landmarks = None
                else:
                    landmarks = result.face_landmarks[0]
                    anchors = extract_face_data(landmarks, width, height)
                    ratio = calc_face_size_ratio(landmarks)
                    yaw = calc_yaw_angle(landmarks)
                    dist, yaw_diff, size_diff = calc_movement_metrics(landmarks, prev_landmarks, yaw, prev_yaw, ratio, prev_ratio)

                    # Yaw Angle Check
                    if not (YAW_MIN <= yaw <= YAW_MAX):
                        prev_landmarks = None
                        reject_reason = f"Bad Angle ({yaw:.2f})"
                    
                    # Face Size Check
                    elif ratio < MIN_FACE_RATIO:
                        prev_landmarks = None
                        reject_reason = f"Face Too Small ({ratio:.2f})"
                    
                    # Movement Check
                    elif dist > MOVEMENT_THRESHOLD or yaw_diff > YAW_THRESHOLD or size_diff > SIZE_THRESHOLD:
                        prev_landmarks = None
                        final_status = -1
                        reject_reason = f"Movement ({dist:.2f}, {yaw_diff:.2f}, {size_diff:.2f})"
                    else:
                        # Success!
                        final_status = int(audio_status)
                        valid_frames_count += 1

                        prev_landmarks = landmarks
                        prev_yaw = yaw
                        prev_ratio = ratio

            # --- Save Data ---
            frame_entry = {
                "i": frame_idx,
                "t": round(current_time_sec, 3),
                "s": final_status
            }
            if anchors:
                frame_entry["a"] = anchors
            if reject_reason != "":
                frame_entry["r"] = reject_reason

            analyzed_frames.append(frame_entry)
            
            frame_idx += 1
            pbar.update(1)

    cap.release()
    mp_handler.close()

    # 3. Save
    output_data = {
        "video_name": video_name,
        "fps": fps,
        "resolution": [width, height],
        "stats": {"valid": valid_frames_count, "total": total_frames},
        "frames": analyzed_frames
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, separators=(',', ':'))

def process_video_wrapper(args):
    # Wrap the function to fit the Pool
    video_path, json_path = args
    analyze_single_video(video_path, json_path)

def main():
    if not os.path.exists(INPUT_VIDEO_DIR): return

    videos = sorted([f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(".mp4")])
    if not videos: return

    # Prepare the list of all tasks to do
    tasks = []
    for video_file in videos:
        json_file = video_file.replace(".mp4", ".json")
        json_path = os.path.join(INPUT_TRANSCRIPT_DIR, json_file)
        if os.path.exists(json_path):
            video_path = os.path.join(INPUT_VIDEO_DIR, video_file)
            tasks.append((video_path, json_path))

    print(f"Starting Parallel Analysis on {len(tasks)} videos using {os.cpu_count()} cores")

    # max_workers = number of cores (can be reduced if the computer is slow)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use tqdm to see overall progress
        list(tqdm(executor.map(process_video_wrapper, tasks), total=len(tasks), unit="vid"))

    print("\nAll videos analyzed successfully.")

if __name__ == "__main__":
    main()