import os
import json
import yt_dlp
import logging
import sys
import time
import random
from dotenv import load_dotenv

# Add project root
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.common import setup_logger, sanitize_filename

load_dotenv()

# Paths
# תיקון: וידוא ש-DOWNLOAD_DIR מוגדר היטב לשימוש בכל הקוד
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", r"C:\VLM-Lip-Reader\data\01_raw_videos")
SOURCES_FILE = os.getenv("SOURCES_FILE", "assets/configs/source_urls.json")
COOKIES_FILE = os.getenv("COOKIES_FILE", "assets/config/youtube_cookies.txt")
LOG_FOLDER = os.getenv("LOGS_DIR", "logs")

# Quality
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", 1080))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", 720))
TARGET_FPS = int(os.getenv("TARGET_FPS", 25))
MIN_FPS = int(os.getenv("MIN_FPS_TO_DOWNLOAD", 20))

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

error_logger = setup_logger('error_logger', os.path.join(LOG_FOLDER, 'download_errors.log'))
success_logger = setup_logger('success_logger', os.path.join(LOG_FOLDER, 'download_success.log'))

# ---------------------------

def filter_video_quality(info, *, incomplete):
    fps = info.get('fps')
    height = info.get('height')

    if fps is not None and fps < MIN_FPS:
        return f"FPS {fps} < {MIN_FPS}"

    if height is not None and height < MIN_HEIGHT:
        return f"Height {height} < {MIN_HEIGHT}"

    return None

# ---------------------------

def human_sleep():
    t = random.uniform(20, 45)
    print(f"😴 Sleeping {round(t,1)}s to avoid bot detection...")
    time.sleep(t)

# ---------------------------

def download_single_video(entry):
    url = entry.get("url")
    if not url:
        return " Missing URL"

    # שומרים על ה-speaker_id ללוגים
    speaker = entry.get("speaker_id", "unknown")
    safe_speaker = sanitize_filename(speaker)
    
    # --- הגדרות ההורדה עם תיקון השמות ---
    ydl_opts = {
        # תיקון: שימוש ב-DOWNLOAD_DIR במקום INPUT_DIR שלא היה קיים
        # תבנית השם: כותרת הסרטון (כדי שיהיה קל לזהות)
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        
        # הגבלה לתווי וינדוס תקינים ומניעת תקלות בשמות
        'restrictfilenames': True,
        
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'match_filter': filter_video_quality,
        'cookiefile': COOKIES_FILE if os.path.exists(COOKIES_FILE) else None,
        'nopart': True, 
        'retries': 10,
        'rate_limit': '1M',
        'quiet': True,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        },
        'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
        'postprocessor_args': [
            '-r', str(TARGET_FPS),
            '-vf', f'scale=-2:{TARGET_HEIGHT}',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k'
        ]
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # חילוץ המידע כדי לקבל את השם הסופי
            info = ydl.extract_info(url, download=True)
            final_title = info.get('title', 'video')
            return f"✅ DONE: {final_title}"
    except Exception as e:
        error_logger.error(f"{url}: {e}")
        return f" ERROR: {url}"

# ---------------------------

def main():
    if not os.path.exists(SOURCES_FILE):
        print(f" Missing config: {SOURCES_FILE}")
        return

    with open(SOURCES_FILE, encoding="utf-8") as f:
        videos = json.load(f)

    print(f"🎬 Downloading {len(videos)} videos (SAFE MODE)\n")

    for idx, video in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}]")
        result = download_single_video(video)
        print(result)
        
        # שינה בין הורדות רק אם יש יותר מסרטון אחד
        if idx < len(videos):
            human_sleep()

    print("\n Finished safely.")

if __name__ == "__main__":
    main()