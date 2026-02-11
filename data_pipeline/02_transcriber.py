import os
import gc
import json
import torch
import whisperx
import faster_whisper
from glob import glob
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import string
import re
from collections import Counter

# ========================================================
# --- תיקון DLL וסביבת עבודה לכרטיסי אינטל / CPU ---
# ========================================================
# הוספת נתיבי הספריות שהורדו כדי למנוע שגיאות DLL חסרים
os.environ["PATH"] += os.pathsep + r'C:\Users\liory\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cudnn\bin'
os.environ["PATH"] += os.pathsep + r'C:\Users\liory\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin'

# תיקון נתיבים לייבוא תיקיית utils מתיקיית השורש
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.common import setup_logger

# ========================================================
# --- Monkey Patch לתיקון שגיאת multilingual ב-faster_whisper ---
# ========================================================
def apply_faster_whisper_patch():
    try:
        target = None
        if hasattr(faster_whisper, "transcription") and hasattr(faster_whisper.transcription, "TranscriptionOptions"):
            target = faster_whisper.transcription.TranscriptionOptions
        elif hasattr(faster_whisper, "TranscriptionOptions"):
            target = faster_whisper.TranscriptionOptions
        
        if target:
            original_init = target.__init__
            def patched_init(self, *args, **kwargs):
                kwargs.pop("multilingual", None)
                return original_init(self, *args, **kwargs)
            target.__init__ = patched_init
    except Exception:
        pass

apply_faster_whisper_patch()

# תיקון טעינת מודלים ב-PyTorch 2.4+
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# ========================================================
# --- הגדרות נתיבים ומודל ---
# ========================================================
load_dotenv()

INPUT_DIR = r"C:\VLM-Lip-Reader\data\01_raw_videos"
OUTPUT_DIR = r"C:\VLM-Lip-Reader\data\02_transcribed"
LOG_DIR = r"C:\VLM-Lip-Reader\logs"
HF_TOKEN = "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"

# הגדרות הרצה (מכיוון שאין NVIDIA GPU, נשתמש ב-CPU)
MODEL_SIZE = "base"  # שיניתי ל-base כדי שזה ירוץ מהר על המעבד שלך
DEVICE = "cpu"       
COMPUTE_TYPE = "int8"
BATCH_SIZE = 1       # על CPU עדיף batch קטן כדי לא לחנוק את ה-RAM

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger = setup_logger("transcriber", os.path.join(LOG_DIR, "transcribe_pipeline.log"))

def release_memory(*args):
    for obj in args:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clean_text(text):
    if not text: return ""
    return re.sub(r"[^\w\s]", "", text.lower().strip())

# ==========================================
# STAGE 1: Transcription
# ==========================================
def stage_1_transcribe(videos):
    print(f"\nSTAGE 1: Transcribing {len(videos)} videos on {DEVICE}...")
    try:
            # גרסה פשוטה יותר של טעינה שעובדת על כל גרסאות WhisperX
            model = whisperx.load_model(
                MODEL_SIZE, 
                DEVICE, 
                compute_type=COMPUTE_TYPE, 
                language="en"
            )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for video_path in tqdm(videos, desc="Transcribing"):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
        
        if os.path.exists(output_path):
            continue

        try:
            audio = whisperx.load_audio(video_path)
            result = model.transcribe(audio, batch_size=BATCH_SIZE)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            release_memory(audio, result)
        except Exception as e:
            logger.error(f"Stage 1 Error on {file_id}: {e}")

    release_memory(model)

# ==========================================
# STAGE 2: Alignment
# ==========================================
def stage_2_align(videos):
    print(f"\nSTAGE 2: Aligning words to timestamps...")
    try:
        align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    except Exception as e:
        print(f"❌ Failed to load align model: {e}")
        return

    for video_path in tqdm(videos, desc="Aligning"):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
        
        if not os.path.exists(output_path): continue

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            
            if "segments" in result and len(result["segments"]) > 0 and "words" in result["segments"][0]:
                continue

            audio = whisperx.load_audio(video_path)
            result = whisperx.align(result.get("segments", []), align_model, align_metadata, audio, DEVICE)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            release_memory(audio)
        except Exception as e:
            logger.error(f"Stage 2 Error on {file_id}: {e}")

    release_memory(align_model)

# ==========================================
# STAGE 4: Cleaning & Normalization
# ==========================================
def stage_4_processing(videos):
    print(f"\nSTAGE 4: Normalizing JSON files...")
    for video_path in tqdm(videos, desc="Processing"):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
        
        if not os.path.exists(output_path): continue

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            all_words = []
            speaker_counter = Counter()

            for segment in data.get("segments", []):
                for word_obj in segment.get("words", []):
                    cleaned = clean_text(word_obj.get("word", ""))
                    if not cleaned: continue
                    
                    speaker = word_obj.get("speaker", "SPEAKER_00")
                    speaker_counter[speaker] += 1
                    
                    if word_obj.get("start") is not None:
                        all_words.append(word_obj)

            main_speaker = speaker_counter.most_common(1)[0][0] if speaker_counter else None
            
            # שמירת המילים בפורמט שטוח לאימון קל יותר
            data["words"] = all_words
            data["main_speaker"] = main_speaker
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Stage 4 Error on {file_id}: {e}")

def main():
    videos = sorted(glob(os.path.join(INPUT_DIR, "*.mp4")))
    if not videos:
        print("⚠️ No videos found in 01_raw_videos")
        return

    print(f"🚀 Starting Pipeline on {len(videos)} videos (CPU Mode)")
    
    stage_1_transcribe(videos)
    stage_2_align(videos)
    # הערה: ויתרנו על Stage 3 (Diarization) כי הוא הכי כבד ל-CPU
    stage_4_processing(videos)

    print(f"\n✅ Finished! Transcripts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()