import os

# Set single-threaded execution for numpy/OpenCV to avoid conflicts in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import json
import sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.video_processing import get_mouth_roi_params, fill_missing_landmarks
from utils.common import setup_logger

# --- Config ---
INPUT_VIDEOS_DIR = os.getenv("CLIPS_VIDEOS_DIR")     
INPUT_LABELS_DIR = os.getenv("CLIPS_LABELS_DIR")
OUTPUT_LIPS_DIR = os.getenv("LIPS_VIDEOS_DIR")   
LOG_DIR = os.getenv("LOGS_DIR", "logs")
TARGET_SIZE = int(os.getenv("LIPS_TARGET_SIZE")) 
USE_GRAYSCALE = os.getenv("DATA_GRAYSCALE").lower() == "true"

os.makedirs(OUTPUT_LIPS_DIR, exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger('cut_lips', os.path.join(LOG_DIR, 'cut_lips.log'))


def get_best_video_writer(output_path, fps, size):
    """
    Uses mp4v which is the most stable and reliable on Windows.
    Prevents black frames and distortion.
    """
    # We currently rely on H.264 but the driver is missing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    # If working in grayscale, isColor should be False
    out = cv2.VideoWriter(output_path, fourcc, fps, size, isColor=not USE_GRAYSCALE)  

    if not out.isOpened():
        logger.error(f"CRITICAL ERROR: Could not open video writer for {output_path}")
    
    return out


def process_video_lips(filename):
    # Take a video and a JSON, cut the lips using Affine Transform, and save the new video
    # Input: video_path, json_path, output_path
    # Output: saved_frames_count

    video_path = os.path.join(INPUT_VIDEOS_DIR, filename)
    file_id = os.path.splitext(filename)[0]
    json_path = os.path.join(INPUT_LABELS_DIR, f"{file_id}.json")
    output_path = os.path.join(OUTPUT_LIPS_DIR, filename)

    # If the output already exists, skip processing
    if os.path.exists(output_path):
        logger.info(f"Skipped {filename} (Already exists)")
        return f"Skipped {filename} (Already exists)"

    # 1. Check files
    if not (os.path.exists(video_path) and os.path.exists(json_path)):
        logger.warning(f"⚠️ Skipped {filename} (Missing video or JSON)")
        return f"⚠️ Skipped {filename}"
    
    logger.info(f"⏳ Starting to process {filename}...")

    # 2. Load the data (JSON)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        frames_data = data.get("frames")
    
    # Convert to a dictionary for fast access: index -> landmarks
    landmarks_map = {}
    for f in frames_data:
        if f.get("landmarks"):
            landmarks_map[int(f["index"])] = f["landmarks"]

    # Fill the missing landmarks in the dictionary (happend because the smoothing algorithm in analysis)
    landmarks_map = fill_missing_landmarks(landmarks_map)

    # 3. Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the original resolution (for the help function)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = get_best_video_writer(output_path, fps, (TARGET_SIZE, TARGET_SIZE))

    if not out.isOpened():
        logger.error(f"❌ Failed to create video writer for {filename}")
        return f"❌ Failed to create video writer for {filename}"

    saved_frames_count = 0
    count_no_landmarks = 0
    frames_with_no_landmarks = []
    
    # 4. Loop through the frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: 
            logger.error(f"❌ Error reading frame {i} from {video_path}")
            break
        
        # If there are no landmarks for this frame - skip (don't save black)
        if i not in landmarks_map:
            frames_with_no_landmarks.append(i)
            count_no_landmarks += 1
            logger.warning(f"No landmarks for frame {i} in {json_path}")
            #print(f"❌ No landmarks for frame {i} in {json_path}")
            continue

        lms = landmarks_map[i]

        try:
            # Calculate the square using the help function ---
            rect = get_mouth_roi_params(
                landmarks=lms,
                img_width=orig_w,
                img_height=orig_h,
                is_normalized=False,   
                padding_scale=1.25,
                engale_by_eye=False 
            )

            (center, (roi_size, _), angle) = rect

            # --- Step 2: Create the Affine matrix (crop + rotation + scaling) ---
            # 1. Calculate the scale: how much to shrink/grow to reach TARGET_SIZE
            scale = TARGET_SIZE / float(roi_size)
            
            # 2. Create the rotation matrix around the mouth center
            M = cv2.getRotationMatrix2D(center, angle, scale)
            
            # 3. Move the center: we want the mouth center to be in the middle of the new image (TARGET_SIZE/2)
            M[0, 2] += (TARGET_SIZE / 2.0) - center[0]
            M[1, 2] += (TARGET_SIZE / 2.0) - center[1]
            
            # --- Step 3: Perform the cropping ---
            lips_frame = cv2.warpAffine(frame, M, (TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_CUBIC)            
            
            if USE_GRAYSCALE:
                lips_frame = cv2.cvtColor(lips_frame, cv2.COLOR_BGR2GRAY)

            out.write(lips_frame)
            saved_frames_count += 1
            
        except Exception as e:
            # If the calculation fails due to invalid points, skip
            logger.error(f"❌ Error processing frame {i} in {filename}: {e}")
            continue

    cap.release()
    out.release()

    logger.info(f"✅ {filename} Done: Total frames: {total_frames}, Frames with no landmarks: {frames_with_no_landmarks}, Saved: {saved_frames_count}")
    return f"✅ {filename}: Saved {saved_frames_count} frames"


def main():
    if not os.path.exists(INPUT_VIDEOS_DIR):
        logger.error("❌ No input directory found.")
        print("❌ No input directory found.")
        return

    videos = sorted([f for f in os.listdir(INPUT_VIDEOS_DIR) if f.endswith(".mp4")])
    
    print(f"Cutting lips from {len(videos)} videos...")
    print(f"Target Size: {TARGET_SIZE}x{TARGET_SIZE}")
    
    # Run in parallel
    optimal_workers = max(1, int(os.cpu_count()) // 2) # use 16 cores from 32 of my Intel Core i9-14900HX CPU,
    print(f"Set {optimal_workers} workers for cut lips")

    with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        future_to_video = {executor.submit(process_video_lips, video): video for video in videos}
        
        results = []
        for future in tqdm(as_completed(future_to_video), total=len(videos)):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                video_name = future_to_video[future]
                logger.error(f"Process crashed on video {video_name}: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()