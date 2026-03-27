import os
import json
import yt_dlp
import re
import sys
import glob
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add the project root to the path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common import setup_logger, sanitize_filename

load_dotenv()

# Paths
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "data/01_raw_videos")
SOURCES_FILE = os.getenv("SOURCES_FILE", "assets/configs/source_urls.json")
COOKIES_FILE = os.getenv("COOKIES_FILE", "assets/config/youtube_cookies.txt")
LOG_FOLDER = os.getenv("LOGS_DIR", "logs")
METADATA_FILE =  os.path.join(DOWNLOAD_DIR, f"metadata.json")

# Quality settings
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", 1080))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", 720)) 
TARGET_FPS = int(os.getenv("TARGET_FPS", 25))
MIN_FPS = int(os.getenv("MIN_FPS_TO_DOWNLOAD", 20))
MAX_WORKERS = int(os.getenv("DOWNLOWDER_MAX_WORKERS", 3))

# Setup loggers
error_logger = setup_logger('error_logger', os.path.join(LOG_FOLDER, 'download_errors.log'))
success_logger = setup_logger('success_logger', os.path.join(LOG_FOLDER, 'download_success.log'))

# Global metadata dictionary and lock for thread safety
metadata_lock = threading.Lock()
global_metadata = {}

def filter_video_quality(info, *, incomplete):
    # Filter to check the metadata before downloading
    # Return an error string if the video doesn't meet the standards
    video_fps = info.get('fps')
    video_height = info.get('height')
    
    # Check FPS
    if video_fps is not None and video_fps < MIN_FPS:
        return f"FPS {video_fps} is too low (Min: {MIN_FPS})"

    # Check resolution
    if video_height is not None and video_height < MIN_HEIGHT:
        return f"Height {video_height}p is too low (Min: {MIN_HEIGHT}p)"
        
    return None

def download_single_video(entry):
    # Function to download a single video
    # Return a status message (string) at the end
    url = entry.get('url')
    
    if not url: return "❌ No URL provided"

    # Extract video ID for filename (YouTube specific)
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match: return f"❌ Invalid URL format: {url}"
    
    final_filename = video_id_match.group(1)
    output_path = os.path.join(DOWNLOAD_DIR, f"{final_filename}.mp4")

    video_meta = {'orig_fps': 'N/A', 'duration': 0, 'height': 'N/A'}

    # yt-dlp settings
    ydl_opts = {
        # 1. Quality: The best possible, up to the threshold we set
        # 'format': f'bestvideo[height<={TARGET_HEIGHT}][ext=mp4]+bestaudio[ext=m4a]/best[height<={TARGET_HEIGHT}][ext=mp4]',
        'format': f'bestvideo[height<={TARGET_HEIGHT}]+bestaudio/best[height<={TARGET_HEIGHT}]',

        # 2. Save path
        'outtmpl': os.path.join(DOWNLOAD_DIR, f'{final_filename}.%(ext)s'),
        
        # 3. Filter (Matcher)
        'match_filter': filter_video_quality,
        
        # 4. Cookies (Critical for preventing bans)
        'cookiefile': COOKIES_FILE if os.path.exists(COOKIES_FILE) else None,
        
        # 5. Retries
        'retries': 10,
        'fragment_retries': 10,
        
        # 6. Conversion and Processing (FFmpeg)
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': {
            'video_convertor': [
                '-r', str(TARGET_FPS),                 # Set target FPS (will drop frames if original is higher, or duplicate if lower)
                '-vf', f'scale=-2:{TARGET_HEIGHT}',    # Scale height to target and adjust width to maintain aspect ratio.
                '-c:v', 'h264_nvenc',                  # Use NVIDIA hardware acceleration (if available)
                '-pix_fmt', 'yuv420p',                 # Ensure compatibility with most players and models (some models require this pixel format)
                '-preset', 'p4',                       # Set a faster preset for encoding (p1 is the fastest, p7 is the slowest but best quality)
                '-cq', '23',                           # Use CQ (Constant Quality) instead of CRF for NVIDIA GPUs
                '-c:a', 'aac',                         # Use AAC for audio (widely supported and efficient)
                '-b:a', '128k',                        # Set audio bitrate to 128 kbps (good balance for speech/audio content)
                '-ac', '1',                            # Convert to mono (single channel - ideal for speech recognition)
                '-ar', '16000'                         # Change sample rate to 16kHz (standard for Whisper and models)
            ]
        },
        
        # Quiet (for tqdm)
        'quiet': True,
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 1. Extract & Download in ONE step
            info = ydl.extract_info(url, download=True)
            video_meta['orig_fps'] = info.get('fps', 'Unknown')
            video_meta['height'] = info.get('height', 'Unknown')
            video_meta['duration'] = info.get('duration', 0)

           # 2. Verify & Log
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:
                duration_min = round(video_meta['duration'] / 60, 2)

                # Save metadata in the global dictionary (thread-safe)
                full_meta = entry.copy() 
                full_meta.update(video_meta) 
                full_meta['video_id'] = final_filename 
                
                with metadata_lock:
                    global_metadata[final_filename] = full_meta
                
                log_msg = (
                    f"VIDEO: {final_filename} | "
                    f"DURATION: {duration_min} min | "
                    f"RES: {video_meta['height']}p -> {TARGET_HEIGHT}p | "
                    f"FPS: {video_meta['orig_fps']} -> {TARGET_FPS}"
                )
                success_logger.info(log_msg)
                return f"✅ DONE: {final_filename}"
            else:
                return f"❌ FAILED (Empty/Missing): {final_filename}"
            


    except yt_dlp.utils.DownloadError as e:
        # Clean all video related files if had downlowd erorr
        for temp_file in glob.glob(os.path.join(DOWNLOAD_DIR, f"{final_filename}*")):
            try: os.remove(temp_file)
            except: pass
        
        error_msg = str(e)
        if "Skipping" in error_msg:
            reason = error_msg.split('Skipping: ')[-1]
            error_logger.warning(f"Filter blocked {final_filename}: {reason}")
            return f"⛔ FILTERED: {final_filename} ({reason})"
        else:
            error_logger.error(f"Download Error on {final_filename}: {e}")
            return f"❌ ERROR: {final_filename}"
            
    except Exception as e:
        # Clean general download files in case of unexpected erorr 
        for temp_file in glob.glob(os.path.join(DOWNLOAD_DIR, f"{final_filename}*")):
            try: os.remove(temp_file)
            except: pass
        
        error_logger.error(f"Unexpected Error on {final_filename}: {e}")
        return f"❌ CRITICAL: {final_filename}"

def main():
    global global_metadata

    # Configuration file check
    if not os.path.exists(SOURCES_FILE):
        print(f"❌ Config file missing: {SOURCES_FILE}")
        return
    
    if os.path.exists(COOKIES_FILE):
        print(f"🍪 Cookies found at: {COOKIES_FILE}")
    else:
        print(f"⚠️  Cookies NOT found at: {COOKIES_FILE} (YouTube might block you)")



    # Create target directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Load the list of videos
    with open(SOURCES_FILE, 'r', encoding='utf-8') as f:
        all_videos = json.load(f)

    # Load prev metadata and 
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            global_metadata = json.load(f)

    videos_to_download = []
    for video in all_videos:
        url = video.get('url', '')
        match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
        if match:
            video_id = match.group(1)
            if video_id in global_metadata:
                continue # Skip videos lredy downloaded
        videos_to_download.append(video)

    if not videos_to_download:
        print(f"All {len(all_videos)} videos are already downloaded and fully documented in metadata!")
        return

    print(f"\nStarting Pipeline Step 1: Downloading {len(videos_to_download)} videos...")
    print(f"Settings: Max {TARGET_HEIGHT}p | {TARGET_FPS} fps | Min Quality: {MIN_HEIGHT}p/{MIN_FPS}fps")
    print(f"Output: {DOWNLOAD_DIR}\n")

    # Manage downloads in parallel with progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_video = {executor.submit(download_single_video, video): video for video in videos_to_download}
        
        # Tqdm wraps as_completed
        with tqdm(total=len(videos_to_download), unit="video", desc="Processing") as pbar:
            for future in as_completed(future_to_video):
                result = future.result()
                
                # Write logical results to tqdm (to avoid breaking the progress bar)
                if "❌" in result or "⛔" in result:
                    # Errors and filters are printed to the screen
                    pbar.write(result)
                elif "✅" in result:
                    pass
                pbar.update(1)


    # Save the global metadata to a JSON file at the end
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\nFinished. Check '{DOWNLOAD_DIR}' for videos and 'download_errors.log' for issues.")

if __name__ == "__main__":
    main()