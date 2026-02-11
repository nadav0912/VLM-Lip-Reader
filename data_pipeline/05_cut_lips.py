import cv2
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.video_processing import get_mouth_roi_params, fill_missing_landmarks

# --- Config ---
INPUT_VIDEOS_DIR = os.getenv("FINAL_VIDEOS_DIR")     
INPUT_LABELS_DIR = os.getenv("FINAL_LABELS_DIR")
OUTPUT_LIPS_DIR = os.getenv("LIPS_VIDEOS_DIR")   

TARGET_SIZE = int(os.getenv("LIPS_TARGET_SIZE")) 

os.makedirs(OUTPUT_LIPS_DIR, exist_ok=True)

def get_best_video_writer(output_path, fps, size):
    """
    Uses mp4v which is the most stable and reliable on Windows.
    Prevents black frames and distortion.
    """
    # We currently rely on H.264 but the driver is missing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    # Critical check - did we manage to open the file for writing?
    if not out.isOpened():
        print(f"❌ CRITICAL ERROR: Could not open video writer for {output_path}")
        # Last try with MJPG (works everywhere but larger files)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path.replace(".mp4", ".avi"), fourcc, fps, size)
    
    return out

def process_video_lips(filename):
    # Take a video and a JSON, cut the lips using Affine Transform, and save the new video
    # Input: video_path, json_path, output_path
    # Output: saved_frames_count

    video_path = os.path.join(INPUT_VIDEOS_DIR, filename)
    file_id = os.path.splitext(filename)[0]
    json_path = os.path.join(INPUT_LABELS_DIR, f"{file_id}.json")
    output_path = os.path.join(OUTPUT_LIPS_DIR, filename)

    # 1. Check files
    if not (os.path.exists(video_path) and os.path.exists(json_path)):
        return f"⚠️ Skipped {filename}"
    
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
    #out = get_best_video_writer(output_path, fps, (orig_w, orig_h))


    if not out.isOpened():
        return f"❌ Failed to create video writer for {filename}"

    saved_frames_count = 0
    count_no_landmarks = 0
    frames_with_no_landmarks = []
    
    # 4. Loop through the frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: 
            #print(f"❌ Error reading frame {i} from {video_path}")
            break
        
        # If there are no landmarks for this frame - skip (don't save black)
        if i not in landmarks_map:
            frames_with_no_landmarks.append(i)
            count_no_landmarks += 1
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
            
            out.write(lips_frame)
            saved_frames_count += 1
            
        except Exception as e:
            # If the calculation fails due to invalid points, skip
            print(f"❌ Error processing frame {i} in {json_path}: {e}")
            continue

    cap.release()
    out.release()
    print(f"\nvideo: {filename}, Total frames: {total_frames}, Frames with no landmarks: {frames_with_no_landmarks}")
    return f"✅ {filename}: Saved {saved_frames_count} frames"


def main():
    if not os.path.exists(INPUT_VIDEOS_DIR):
        print("❌ No input directory found.")
        return

    videos = sorted([f for f in os.listdir(INPUT_VIDEOS_DIR) if f.endswith(".mp4")])
    
    print(f"Cutting lips from {len(videos)} videos...")
    print(f"Target Size: {TARGET_SIZE}x{TARGET_SIZE}")
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_video_lips, videos), total=len(videos)))

    print("\nDone.")

if __name__ == "__main__":
    main()