import os
import json
import yt_dlp
import logging
import re
import sys
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# הוספת שורש הפרויקט לנתיב
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common import setup_logger, sanitize_filename

load_dotenv()

# --- הגדרת נתיבים ---
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "data/01_raw_videos")
SOURCES_FILE = os.getenv("SOURCES_FILE", "assets/configs/source_urls.json")
LOG_FOLDER = os.getenv("LOGS_DIR", "logs")

# הגדרות איכות
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", 1080))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", 720)) 
TARGET_FPS = int(os.getenv("TARGET_FPS", 25))
MIN_FPS = int(os.getenv("MIN_FPS_TO_DOWNLOAD", 20))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 2)) 

error_logger = setup_logger('error_logger', os.path.join(LOG_FOLDER, 'download_errors.log'))
success_logger = setup_logger('success_logger', os.path.join(LOG_FOLDER, 'download_success.log'))

def filter_video_quality(info, *, incomplete):
    video_fps = info.get('fps')
    video_height = info.get('height')
    
    if video_fps is not None and video_fps < MIN_FPS:
        return f"FPS {video_fps} is too low (Min: {MIN_FPS})"
    if video_height is not None and video_height < MIN_HEIGHT:
        return f"Height {video_height}p is too low (Min: {MIN_HEIGHT}p)"
    return None

def download_single_video(entry):
    url = entry.get('url')
    speaker = entry.get('speaker_id', 'Unknown_Speaker')
    video_id = url.split("v=")[-1] if "v=" in url else "video"
    
    if not url: return "❌ No URL provided"

    safe_speaker = sanitize_filename(speaker)
    final_filename = f"{safe_speaker}_{video_id}"
    output_path = os.path.join(DOWNLOAD_DIR, f"{final_filename}.mp4")
    
    if os.path.exists(output_path):
        if os.path.getsize(output_path) > 1024 * 1024:
            return f"⏭️  SKIP: {final_filename}"

    # --- תיקון מבנה ה-ydl_opts ---
    ydl_opts = {
        'format': f'bestvideo[height<={TARGET_HEIGHT}][ext=mp4]+bestaudio[ext=m4a]/best[height<={TARGET_HEIGHT}][ext=mp4]',
        'outtmpl': os.path.join(DOWNLOAD_DIR, f'{final_filename}.%(ext)s'),
        'match_filter': filter_video_quality,
        
        # שימוש בקובץ הקוקיז מהמיקום שציינת
        'cookiefile': 'assets/configs/youtube_cookies.txt', 
        
        'retries': 10,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': [
            '-r', str(TARGET_FPS),
            '-vf', f'scale=-2:{TARGET_HEIGHT}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23'
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            return f"✅ DONE: {final_filename}"
    except Exception as e:
        error_logger.error(f"Error on {final_filename}: {e}")
        return f"❌ ERROR: {final_filename}"

def main():
    if not os.path.exists(SOURCES_FILE):
        print(f"❌ Config file missing: {SOURCES_FILE}")
        return
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    with open(SOURCES_FILE, 'r', encoding='utf-8') as f:
        videos = json.load(f)

    print(f"\n🚀 Starting Pipeline Step 1 (Using Cookie File)")
    print(f"Downloading {len(videos)} videos to {DOWNLOAD_DIR}...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {executor.submit(download_single_video, video): video for video in videos}
        with tqdm(total=len(videos), unit="video", desc="Downloading") as pbar:
            for future in as_completed(future_to_video):
                result = future.result()
                if "❌" in result:
                    pbar.write(result)
                pbar.update(1)

if __name__ == "__main__":
    main()