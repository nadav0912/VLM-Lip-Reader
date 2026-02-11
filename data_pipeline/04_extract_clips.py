import json
import os
import numpy as np
import subprocess
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.visual_data import create_debug_video

# Paths
INPUT_VIDEO_DIR = os.getenv("RAW_VIDEOS_DIR")
INPUT_TRANSCRIPT_DIR = os.getenv("ROW_TRANSCRIPTS_DIR")
INPUT_ANALYSIS_DIR = os.getenv("ANALYSIS_DIR")
OUTPUT_DATASET_DIR = os.getenv("FINAL_DATASET_DIR")
VIDEOS_OUT_DIR = os.getenv("FINAL_VIDEOS_DIR")
LABELS_OUT_DIR = os.getenv("FINAL_LABELS_DIR")

# Create the directories
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
os.makedirs(VIDEOS_OUT_DIR, exist_ok=True)
os.makedirs(LABELS_OUT_DIR, exist_ok=True)

# Settings to extract the clips
SMOOTHING_TOLERANCE = float(os.getenv("SMOOTHING_TOLERANCE")) # How many bad frames in a row we tolerate (skip)
MAX_INTERNAL_SILENCE_SEC = float(os.getenv("MAX_INTERNAL_SILENCE_SEC")) # Maximum silence duration in a single sequence. Over this - we split.
PADDING_SEC = float(os.getenv("PADDING_SEC")) # How much silence to add before and after the speech. If it's more - we add padding.
MIN_CLIP_DURATION_SEC = float(os.getenv("MIN_CLIP_DURATION_SEC")) # Minimum duration of the final clip. Shorter than this - we discard.

# ==========================================
# PART 1: RECOGNIZE THE CLIPS
# ==========================================
def load_analysis_data(analysis_path):
    if not os.path.exists(analysis_path):
        return None
    with open(analysis_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_status_array(frames_data, total_frames):
    # Convert the list of frames to a numpy array for fast access
    # 0=Invalid, 1=Speaking, 2=Silence
    arr = np.zeros(total_frames, dtype=np.int8) 
    for f in frames_data:
        idx = f["i"]
        if idx < total_frames:
            arr[idx] = f["s"]
    return arr


def smooth_status_array(status_arr):
    # Fix small gaps of invalid frames (0) within a valid sequence.
    # Not change -1 (movement) values.
    arr = status_arr.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 0:
            j = i + 1
            while j < n and arr[j] == 0:
                j += 1
            
            bad_run_length = j - i
            
            # If the bad run is short, and it's surrounded by valid frames
            if bad_run_length <= SMOOTHING_TOLERANCE:
                left_val = arr[i-1] if i > 0 else 0
                right_val = arr[j] if j < n else 0
                
                if left_val > 0 and right_val > 0:
                    # Convert the bad run to 1 (speaking) to maintain the sequence
                    arr[i:j] = 1 
            i = j
        else:
            i += 1
    return arr


def find_speech_islands(status_arr, fps):
    # Find the speech islands (1) and merge them if they are close.
    n = len(status_arr)
    max_silence_frames = int(MAX_INTERNAL_SILENCE_SEC * fps)
    
    merged_clips = []
    
    current_start = -1
    current_end = -1
    
    for i in range(n):
        s = status_arr[i]
        
        if s == 1: # דיבור
            if current_start == -1:
                current_start = i
            current_end = i # Extend the end
        
        elif s == 2: # Silence
            # If we are inside a sequence, check if the silence is too long
            if current_start != -1:
                silence_gap = i - current_end
                if silence_gap > max_silence_frames:
                    # Silence too long -> Close the clip
                    merged_clips.append((current_start, current_end))
                    current_start = -1
                    current_end = -1
        
        elif s <= 0: # Invalid (0) OR Scene Cut (-1)
            # Break the sequence -> Close the clip immediately
            if current_start != -1:
                merged_clips.append((current_start, current_end))
                current_start = -1
                current_end = -1

    # Close any remaining open clip
    if current_start != -1:
        merged_clips.append((current_start, current_end))
        
    return merged_clips


def add_padding_and_filter(clips, status_arr, fps):
    # Add padding (silence) on both sides *only if it exists and is valid*.
    final_manifest = []
    n = len(status_arr)
    padding_frames = int(PADDING_SEC * fps)
    min_len_frames = int(MIN_CLIP_DURATION_SEC * fps)
    
    for start, end in clips:
        # --- Padding Start (Left) ---
        new_start = start
        # Try to go back until the boundary (padding_frames)
        # But stop if we encounter something that is not silence (like 0 or end of video)
        for k in range(1, padding_frames + 1):
            idx = start - k
            if idx < 0 or status_arr[idx] != 2: 
                break 
            new_start = idx
            
        # --- Padding End (Right) ---
        new_end = end
        # Try to go forward until the boundary
        for k in range(1, padding_frames + 1):
            idx = end + k
            if idx >= n or status_arr[idx] != 2:
                break
            new_end = idx
            
        # --- Check the final duration ---
        duration_frames = new_end - new_start + 1
        if duration_frames >= min_len_frames:
            final_manifest.append({
                "start_frame": int(new_start),
                "end_frame": int(new_end),
            })
            
    return final_manifest


def create_clips_manifest(analysis_path):
    data = load_analysis_data(analysis_path)
    if not data: return None
    
    fps = data.get("fps")
    total_frames = data["stats"]["total"]
    
    # 1. Get the data
    raw_arr = get_status_array(data["frames"], total_frames)
    
    # 2. Smoothing
    smooth_arr = smooth_status_array(raw_arr)
    
    # 3. Find the speech islands (Islands)
    raw_clips = find_speech_islands(smooth_arr, fps)
    
    # 4. Add padding and filter
    final_clips = add_padding_and_filter(raw_clips, smooth_arr, fps)
        
    return {
        "video_name": data.get("video_name"),
        "total_clips": len(final_clips),
        "segments": final_clips
    }


# ==========================================
# PART 2: CUT CLIPS & GENERATE LABELS
# ==========================================

def cut_video_clip_ffmpeg(video_path, start_time, duration, output_path):
    #Cut the video physically by exact times.
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.6f}",    # Start time
        "-i", video_path,              # Input video
        "-t", f"{duration:.6f}",       # Duration
        "-an",                         # No audio
        "-c:v", "libx264",             # We use the regular and good encoder
        "-crf", "17",                  # Almost perfect quality (Lossless-like)
        "-preset", "fast",             # Speed
        
        # The flag that saves the state:
        # keyint=1: every frame is a keyframe (like a standalone image)
        # scenecut=0: turns off smart optimizations that might move frames
        "-x264-params", "keyint=1:scenecut=0", 
        "-loglevel", "error",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e}")
        return False


def create_label_json(segment, frames_map, full_words_list, fps, resolution, source_name, output_path):
    #Create the final JSON.
    #It receives the full words list, filters only what belongs to the clip,
    #and converts all times (words and frames) to relative time (starting from 0.0).
 
    width, height = resolution
    s_frame = segment["start_frame"]
    e_frame = segment["end_frame"]
    
    # 1. Get the exact start time of the clip
    start_frame_data = frames_map.get(s_frame)
    if start_frame_data:
        clip_start_time = start_frame_data["t"]
    else:
        # Assume the frames are missing in the map, calculate based on FPS
        clip_start_time = s_frame / fps

    # Calculate the end time for filtering the words
    end_frame_data = frames_map.get(e_frame)
    if end_frame_data:
        clip_end_time = end_frame_data["t"]
    else:
        clip_end_time = e_frame / fps

    # 2. Filter and match the words (Words Processing)
    relative_words = []

    # Find start index to look for words with binary search
    start_idx = bisect_left(full_words_list, clip_start_time, key=lambda x: x["end"])
    start_idx = max(0, start_idx - 3)

    for i in range(start_idx, len(full_words_list)):
        w = full_words_list[i]
        
        # If wrod start after clip end we can stop the loop
        if w["start"] > clip_end_time:
            break
            
        w_center = (w["start"] + w["end"]) / 2
        
        if clip_start_time <= w_center <= clip_end_time:
            start = round(max(0.0, w["start"] - clip_start_time), 3)
            end = round(max(0.0, w["end"] - clip_start_time), 3)
            relative_words.append({"start": start, "end": end, "word": w["word"]}) # Remove speaker and keep from word

    # 3. Process the frames (Frames Processing)
    clip_frames_data = []
    
    for i in range(s_frame, e_frame + 1):
        original_data = frames_map.get(i)
        
        entry = {
            "index": i - s_frame,       # Index in the clip (0, 1, 2...)
            "original_index": i,        # Original index (for fallback)
            "timestamp": 0.0,           # Will be calculated later
            "landmarks": None,
            "landmarks_normalized": None
        }
        
        if original_data:
            # Calculate relative time for the frame
            rel_time = original_data["t"] - clip_start_time
            entry["timestamp"] = round(max(0.0, rel_time), 4)

            # Normalize the landmarks (if exists)
            if "a" in original_data:
                anchors = original_data["a"]
                entry["landmarks"] = anchors # Save the pixels for the landmark
                
                # Create normalized coordinates (0.0 to 1.0)
                entry["landmarks_normalized"] = {
                    k: [round(p[0]/width, 5), round(p[1]/height, 5)] 
                    for k, p in anchors.items()
                }
        
        clip_frames_data.append(entry)

    # 4. Build the full text of the clip
    clip_text = " ".join([w["word"] for w in relative_words])

    # 5. Save the final object
    label_data = {
        "metadata": {
            "source_video": source_name,
            "fps": fps,
            "resolution": resolution,
            "duration": segment.get("duration", clip_end_time - clip_start_time),
            "total_frames": len(clip_frames_data),
            "clip_word_count": len(relative_words)
        },
        "text": {
            "sentence": clip_text,
            "words": relative_words  # המילים עם הזמנים החדשים
        },
        "frames": clip_frames_data   # הפריימים עם הזמנים החדשים והנקודות
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, separators=(',', ':'))

# ==========================================
# PART 3: WORKER & MAIN
# ==========================================
def process_video_task(args):
    #The function that runs in parallel on each video
    video_idx, filename = args
    
    file_id = os.path.splitext(filename)[0]
    video_path = os.path.join(INPUT_VIDEO_DIR, filename)
    analysis_path = os.path.join(INPUT_ANALYSIS_DIR, f"{file_id}_analysis.json")
    transcript_path = os.path.join(INPUT_TRANSCRIPT_DIR, f"{file_id}.json")
    
    if not (os.path.exists(analysis_path) and os.path.exists(transcript_path)):
        return f"⚠️ Skipped {filename} (Data missing)"

    # Find the good clips from the video
    manifest = create_clips_manifest(analysis_path)
    if not manifest or not manifest["segments"]:
        return f"No clips in {filename}"

    segments = manifest["segments"]
    
    # Map for fast access to the exact times
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
        fps = analysis_data.get("fps")
        resolution = analysis_data.get("resolution")
        frames_list = analysis_data.get("frames", [])

    frames_map = {frames_list[i]["i"]: frames_list[i] for i in range(len(frames_list))}

    # Load the total word count for statistics
    with open(transcript_path, 'r', encoding='utf-8') as f:
        words = json.load(f).get("words", [])
        
    created_count = 0
    
    # 2. Cut and save the clips
    for clip_idx, seg in enumerate(segments):
        # Anonymous name: v001_c001
        base_name = f"v{video_idx:03d}_c{clip_idx:03d}"
        mp4_out = os.path.join(VIDEOS_OUT_DIR, f"{base_name}.mp4")
        json_out = os.path.join(LABELS_OUT_DIR, f"{base_name}.json")
        
        # Calculate the exact time based on the original frames
        s_idx, e_idx = seg["start_frame"], seg["end_frame"]
    
        real_start = frames_map[s_idx]["t"]
        real_end = frames_map[e_idx]["t"]
        # +1 frame duration buffer
        exact_duration = (real_end - real_start) + (2.0 / fps)

        # 1. Cut the video
        if cut_video_clip_ffmpeg(video_path, real_start, exact_duration, mp4_out):
            # 2. Create the JSON label
            create_label_json(seg, frames_map, words, fps, resolution, filename, json_out)
            created_count += 1
            
    return f"Video {video_idx:03d}: Created {created_count} clips"

def main():
    videos = sorted([f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(".mp4")])
    if not videos:
        print("❌ No videos found.")
        return

    print(f"🚀 Processing {len(videos)} videos -> {OUTPUT_DATASET_DIR}")
    
    # Create task list
    tasks = [(i+1, v) for i, v in enumerate(videos)]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_video_task, tasks), total=len(tasks)))

    print("\n🏁 DONE:")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()