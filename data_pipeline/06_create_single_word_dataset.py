import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import json
import sys
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.common import setup_logger

# --- Config ---
VID_DIR = os.getenv("LIPS_VIDEOS_DIR")
LBL_DIR = os.getenv("CLIPS_LABELS_DIR")
OUT_DIR = os.getenv("SINGLE_WORD_CLIPS_DIR")
LOG_DIR = os.getenv("LOGS_DIR", "logs")
PAD = int(os.getenv("PADDING_FRAMES"))
MASTER_JSON = os.path.join(OUT_DIR, "labels.json")

os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger('extract_words', os.path.join(LOG_DIR, 'extract_words.log'))

os.makedirs(OUT_DIR, exist_ok=True)

# Initialize the master JSON file if it doesn't exist
if not os.path.exists(MASTER_JSON):
    with open(MASTER_JSON, 'w', encoding='utf-8') as f:
        json.dump({}, f)

def clean_word_for_filename(word):
    return re.sub(r'[\\/*?:"<>|]', "", word).strip()

def update_master_json(new_entries):
    # Update the master JSON file with the new entries
    if not new_entries: return
    try:
        with open(MASTER_JSON, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            data.update(new_entries)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate() # Delete leftovers if the file is too short
    except Exception as e:
        print(f"⚠️ Error updating JSON: {e}")

def process_video(filename):
    # Cut words from the video and return a dictionary of {filename: word}
    vid_path = os.path.join(VID_DIR, filename)
    json_path = os.path.join(LBL_DIR, filename.replace('.mp4', '.json'))
    
    if not (os.path.exists(vid_path) and os.path.exists(json_path)):
        logger.warning(f"File not found: {vid_path} or {json_path}")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        text_data = data.get("text", {}) 
        words = text_data.get("words", [])
        source_metadata = data.get("metadata", {})
        speaker = source_metadata.get("speaker", "Unknown")
        gender = source_metadata.get("gender", "Unknown")


    if not words:
        logger.info(f"No words found in {filename}")
        return {}

    local_results = {}

    try:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Test the color/grayscale of the first frame
        ret, frame = cap.read()
        if not ret: 
            cap.release()
            return {}

        is_color = False
        if len(frame.shape) == 3:
            b, g, r = cv2.split(frame)
            is_color = not (np.array_equal(b, g) and np.array_equal(g, r))
            
        height, width = frame.shape[:2]
    
        # Reset the video to the beginning after the test
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        file_id = os.path.splitext(filename)[0]
        
        # 1. Prepare the list of active clips with their start/end frames and names
        active_clips = []
        for i, w in enumerate(words):
            start_f = max(0, int(w["start"] * fps) - PAD)
            end_f = min(total_frames, int(w["end"] * fps) + PAD)
            
            if end_f - start_f < 3: continue 

            safe_word = clean_word_for_filename(w['word'])
            if not safe_word: continue
                
            clip_name = f"{file_id}_{i:03d}_{safe_word}.mp4"
            active_clips.append({
                "start": start_f, 
                "end": end_f, 
                "name": clip_name, 
                "word": w['word'],
                "original_start_sec": w["start"],
                "original_end_sec": w["end"],    
                "writer": None, 
                "out_path": os.path.join(OUT_DIR, clip_name)
            })

        # 2. Process the video frame by frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            if not is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for clip in active_clips:
                # If the current frame is within the clip's time range, write it to the clip's video
                if clip["start"] <= frame_idx < clip["end"]:
                    if clip["writer"] is None:
                        clip["writer"] = cv2.VideoWriter(clip["out_path"], fourcc, fps, (width, height), isColor=is_color)
                    clip["writer"].write(frame)
                    
                # If we just passed the end of the clip, release the writer and save the result
                elif frame_idx == clip["end"] and clip["writer"] is not None:
                    clip["writer"].release()
                    clip["writer"] = None
                    local_results[clip["name"]] = {
                        "word": clip["word"],
                        "metadata": {
                            "source_video": filename,
                            "speaker": speaker,
                            "gender": gender,
                            "original_start_sec": clip["original_start_sec"],
                            "original_end_sec": clip["original_end_sec"],
                            "source_start_frame": clip["start"],
                            "source_end_frame": clip["end"],
                            "duration_frames": clip["end"] - clip["start"],
                            "fps": fps,
                            "is_color": is_color
                        }
                    }
                    
            frame_idx += 1

        # Final cleanup for additional safety
        for clip in active_clips:
            if clip["writer"] is not None:
                clip["writer"].release()
                local_results[clip["name"]] = {
                        "word": clip["word"],
                        "metadata": {
                            "source_video": filename,
                            "speaker": speaker,
                            "gender": gender,
                            "original_start_sec": clip["original_start_sec"],
                            "original_end_sec": clip["original_end_sec"],
                            "source_start_frame": clip["start"],
                            "source_end_frame": clip["end"],
                            "duration_frames": clip["end"] - clip["start"],
                            "fps": fps,
                            "resolution": [width, height],
                            "is_color": is_color
                        }
                    }

        logger.info(f"✅ {filename}: Saved {len(local_results)} words")
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        print(f"❌ Error processing video {filename}: {e}")
        
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    
    logger.info(f"Finished processing {filename}. Extracted {len(local_results)} words.")
    return local_results

def main():
    videos = [f for f in os.listdir(VID_DIR) if f.endswith(".mp4")]
    print(f"Extracting words from {len(videos)} videos to {OUT_DIR}...")

    all_results = {}

    optimal_workers = max(1, int(os.cpu_count() * 0.8))
    with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        futures = {executor.submit(process_video, vid): vid for vid in videos}
        
        # as_completed returns a result as soon as the video is done processing
        for future in tqdm(as_completed(futures), total=len(videos)):
            try:
                result_dict = future.result()
                if result_dict:
                    all_results.update(result_dict)
            except Exception as e:
                logger.error(f"Process crashed: {e}")
                print(f"❌ Process crashed: {e}")

    # Save all results to the master JSON file at once
    if all_results:
        logger.info(f"Saving {len(all_results)} labels to JSON...")
        print("💾 Saving all labels to JSON...")
        update_master_json(all_results)

    print("✅ Done.")

if __name__ == "__main__":
    main()