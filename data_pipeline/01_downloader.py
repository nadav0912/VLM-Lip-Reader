import os
import json
import yt_dlp
import logging
import re
import sys
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

# Quality settings
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", 1080))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", 720)) 
TARGET_FPS = int(os.getenv("TARGET_FPS", 25))
MIN_FPS = int(os.getenv("MIN_FPS_TO_DOWNLOAD", 20))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 3))

error_logger = setup_logger('error_logger', os.path.join(LOG_FOLDER, 'download_errors.log'))
success_logger = setup_logger('success_logger', os.path.join(LOG_FOLDER, 'download_success.log'))

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
    raw_title = entry.get('title', 'Unknown_Video')
    speaker = entry.get('speaker', 'Unknown_Speaker')
    
    if not url: return "❌ No URL provided"

    # Create a unique and clean file name
    safe_title = sanitize_filename(raw_title)
    safe_speaker = sanitize_filename(speaker)
    final_filename = f"{safe_speaker}_{safe_title}"
    output_path = os.path.join(DOWNLOAD_DIR, f"{final_filename}.mp4")
    
    # --- Sanity Check ---
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        # If the file exists and is larger than 1MB (it's probably valid)
        if file_size > 1024 * 1024:
            return f"⏭️  SKIP (Exists): {final_filename}"
        else:
            # The file exists but it's "garbage" (too small/empty) - delete and re-download
            try:
                os.remove(output_path)
                error_logger.info(f"Deleted corrupted file: {final_filename}")
            except OSError:
                return f"❌ Error deleting corrupted file: {final_filename}"

    video_meta = {'orig_fps': 'N/A', 'duration': 0, 'height': 'N/A'}

    # yt-dlp settings
    ydl_opts = {
        # 1. Quality: The best possible, up to the threshold we set
        'format': f'bestvideo[height<={TARGET_HEIGHT}][ext=mp4]+bestaudio[ext=m4a]/best[height<={TARGET_HEIGHT}][ext=mp4]',
        
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
        
        'postprocessor_args': [
            '-r', str(TARGET_FPS),                 # Fixed FPS
            '-vf', f'scale=-2:{TARGET_HEIGHT}',    # Fixed Resolution
            '-c:v', 'libx264',                     # Video encoding
            '-preset', 'fast',                     # Encoding speed
            '-crf', '23',                          # Quality
            '-c:a', 'aac',                         # Audio encoding
            '-b:a', '128k'
        ],
        
        # Quiet (for tqdm)
        'quiet': True,
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 1. Extract Metadata
            info = ydl.extract_info(url, download=False)
            video_meta['orig_fps'] = info.get('fps', 'Unknown')
            video_meta['height'] = info.get('height', 'Unknown')
            video_meta['duration'] = info.get('duration', 0)
            
            # 2. Download
            ydl.download([url])
            
           # 3. Verify & Log
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:
                duration_min = round(video_meta['duration'] / 60, 2)
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
        error_msg = str(e)
        if "Skipping" in error_msg:
            reason = error_msg.split('Skipping: ')[-1]
            error_logger.warning(f"Filter blocked {final_filename}: {reason}")
            return f"⛔ FILTERED: {final_filename} ({reason})"
        else:
            error_logger.error(f"Download Error on {final_filename}: {e}")
            return f"❌ ERROR: {final_filename}"
            
    except Exception as e:
        error_logger.error(f"Unexpected Error on {final_filename}: {e}")
        return f"❌ CRITICAL: {final_filename}"

def main():
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
        videos = json.load(f)

    print(f"\nStarting Pipeline Step 1: Downloading {len(videos)} videos...")
    print(f"Settings: Max {TARGET_HEIGHT}p | {TARGET_FPS} fps | Min Quality: {MIN_HEIGHT}p/{MIN_FPS}fps")
    print(f"Output: {DOWNLOAD_DIR}\n")

    # Manage downloads in parallel with progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_video = {executor.submit(download_single_video, video): video for video in videos}
        
        # Tqdm wraps as_completed
        with tqdm(total=len(videos), unit="video", desc="Processing") as pbar:
            for future in as_completed(future_to_video):
                result = future.result()
                
                # Write logical results to tqdm (to avoid breaking the progress bar)
                if "❌" in result or "⛔" in result:
                    # Errors and filters are printed to the screen
                    pbar.write(result)
                elif "✅" in result:
                    pass
                pbar.update(1)

    print(f"\nFinished. Check '{DOWNLOAD_DIR}' for videos and 'download_errors.log' for issues.")

if __name__ == "__main__":
    main()