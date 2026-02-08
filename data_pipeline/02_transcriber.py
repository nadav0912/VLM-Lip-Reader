import os
import gc
import json
import torch
import whisperx
from glob import glob
from dotenv import load_dotenv
from tqdm import tqdm
import sys

# Import Utils (for logging)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.common import setup_logger

# Fix for loading models in newer versions of PyTorch
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Optimization for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Settings
load_dotenv()

INPUT_DIR = os.getenv("RAW_VIDEOS_DIR", "data/01_raw_videos")
OUTPUT_DIR = os.getenv("ROW_TRANSCRIPTS_DIR", "data/02_transcribed")
LOG_DIR = os.getenv("LOGS_DIR", "logs")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Model settings
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", 8)) 
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8") 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create logger
os.makedirs(LOG_DIR, exist_ok=True)
logger = setup_logger('transcriber', os.path.join(LOG_DIR, 'transcribe_pipeline.log'))

def release_memory():
    # Clean up the GPU memory
    gc.collect()
    torch.cuda.empty_cache()

def get_video_files():
    return sorted(glob(os.path.join(INPUT_DIR, "*.mp4")))

# ==========================================
# Stage 1: Transcription
# ==========================================
def stage_1_transcribe(videos):
    print(f"\nSTAGE 1: Transcribing {len(videos)} videos...")
    
    # Load the model once
    print(f"Loading Whisper Model ({MODEL_SIZE})...")
    try:
        model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    for video_path in tqdm(videos, desc="Stage 1 - Transcribing"):
        filename = os.path.basename(video_path)
        file_id = os.path.splitext(filename)[0]
        json_output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

        # Skip if already exists
        if os.path.exists(json_output_path):
            continue

        try:
            audio = whisperx.load_audio(video_path)
            result = model.transcribe(audio, batch_size=BATCH_SIZE)
            
            # Save intermediate result
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            # Release local memory
            del audio
            del result
            
        except Exception as e:
            logger.error(f"Stage 1 Error on {filename}: {e}")
            print(f"❌ Error: {e}")

    # Delete the model from memory at the end of the stage
    del model
    release_memory()
    print("✅ Stage 1 Complete. Model unloaded.")


# ==========================================
# Stage 2: Alignment
# ==========================================
def stage_2_align(videos):
    print(f"\nSTAGE 2: Aligning text...")

    # Load the Align model (assuming English by default for execution)
    print("Loading Align Model...")
    try:
        # Here we use en, if you have other languages you need to load dynamically
        align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    except Exception as e:
        print(f"❌ Failed to load align model: {e}")
        return

    for video_path in tqdm(videos, desc="Stage 2 - Aligning"):
        filename = os.path.basename(video_path)
        file_id = os.path.splitext(filename)[0]
        json_output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

        if not os.path.exists(json_output_path):
            continue

        try:
            # Load the JSON from stage 1
            with open(json_output_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # Check if Align was already performed (if there are words?)
            if "segments" in result and len(result["segments"]) > 0 and "words" in result["segments"][0]:
                continue 

            audio = whisperx.load_audio(video_path)
            
            # Perform the alignment
            result = whisperx.align(
                result["segments"], align_model, align_metadata, audio, DEVICE, return_char_alignments=False
            )

            # Save the updated result
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            del audio
            del result

        except Exception as e:
            logger.error(f"Stage 2 Error on {filename}: {e}")

    # Delete the models from memory
    del align_model
    del align_metadata
    release_memory()
    print("✅ Stage 2 Complete. Model unloaded.")


# ==========================================
# Stage 3: Diarization
# ==========================================
def stage_3_diarize(videos):
    print(f"\nSTAGE 3: Identifying Speakers...")
    
    # Import here to avoid dependency issues
    from whisperx.diarize import DiarizationPipeline

    print("Loading Diarization Pipeline...")
    try:
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    except Exception as e:
        print(f"❌ Failed to load Diarization model: {e}")
        return

    for video_path in tqdm(videos, desc="Stage 3 - Diarizing"):
        filename = os.path.basename(video_path)
        file_id = os.path.splitext(filename)[0]
        json_output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

        if not os.path.exists(json_output_path):
            continue

        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # Check if speakers were already identified
            if "segments" in result and len(result["segments"]) > 0 and "speaker" in result["segments"][0]:
                continue

            audio = whisperx.load_audio(video_path)
            
            # Perform the diarization
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=2)
            
            # Assign the speakers to the words
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Save the final result
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            del audio
            del result
            del diarize_segments

        except Exception as e:
            logger.error(f"Stage 3 Error on {filename}: {e}")

    # Delete the model from memory
    del diarize_model
    release_memory()
    print("✅ Stage 3 Complete. Pipeline Finished.")


def main():
    if not HF_TOKEN:
        print("❌ Error: HUGGINGFACE_TOKEN missing in .env")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    videos = get_video_files()
    
    if not videos:
        print("⚠️ No videos found.")
        return

    print(f"🚀 Starting Pipeline on {len(videos)} videos")
    print(f"Status: CUDA Available? {torch.cuda.is_available()}")

    # Run the stages one after the other to avoid OOM
    stage_1_transcribe(videos)
    stage_2_align(videos)
    stage_3_diarize(videos)
    
    print(f"\nDone! All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()