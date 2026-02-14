import cv2
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
VID_DIR = os.getenv("LIPS_VIDEOS_DIR")
LBL_DIR = os.getenv("FINAL_LABELS_DIR")
OUT_DIR = os.getenv("SINGLE_WORD_CLIPS_DIR")
PAD = int(os.getenv("PADDING_FRAMES", 2))
MASTER_JSON = os.path.join(OUT_DIR, "labels.json")

os.makedirs(OUT_DIR, exist_ok=True)

# Initialize the master JSON file if it doesn't exist
if not os.path.exists(MASTER_JSON):
    with open(MASTER_JSON, 'w', encoding='utf-8') as f:
        json.dump({}, f)

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
        print(f"File not found: {vid_path} or {json_path}")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f).get("text", {}) 
        words = text_data.get("words", [])

    if not words:
        print(f"No words found in {filename}")
        return {}

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Test the color/grayscale of the first frame
    ret, frame = cap.read()
    if not ret: return {}
    is_color = len(frame.shape) == 3
    height, width = frame.shape[:2]
    
    local_results = {}
    count_short_noise = 0
    
    for i, w in enumerate(words):
        start_f = max(0, int(w["start"] * fps) - PAD)
        end_f = min(total_frames, int(w["end"] * fps) + PAD)
        
        if end_f - start_f < 3: 
            count_short_noise += 1
            continue # Filter out short noise

        # File name: ID_WordIndex_Word.mp4
        file_id = os.path.splitext(filename)[0]
        clip_name = f"{file_id}_{i:03d}_{w['word']}.mp4"
        out_path = os.path.join(OUT_DIR, clip_name)
        
        # Save the clip
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=is_color)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(end_f - start_f):
            ret, frame = cap.read()
            if ret: out.write(frame)
            else: break
        out.release()
        
        local_results[clip_name] = w['word']

    cap.release()
    return local_results

def main():
    videos = [f for f in os.listdir(VID_DIR) if f.endswith(".mp4")]
    print(f"Extracting words from {len(videos)} videos to {OUT_DIR}...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_video, vid): vid for vid in videos}
        
        # as_completed returns a result as soon as the video is done processing
        for future in tqdm(as_completed(futures), total=len(videos)):
            result_dict = future.result()
            # Update the master JSON file in real time for each video that is done processing
            update_master_json(result_dict)

    print("✅ Done.")

if __name__ == "__main__":
    main()