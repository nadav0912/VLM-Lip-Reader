import os
import gc
import json
import torch
import whisperx
from glob import glob
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import string
from collections import Counter

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

def clean_text(text):
    if not text: return ""
    text = text.lower().strip()
    # remove punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~) from the ends
    return text.strip(string.punctuation) 


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

# ==========================================
# Stage 4: Normalize, Clean & Filter 
# ==========================================
def stage_4_processing(videos):
    print(f"\nSTAGE 4: Normalizing, Cleaning & Filtering...")

    for video_path in tqdm(videos, desc="Stage 4 - Processing"):
        filename = os.path.basename(video_path)
        file_id = os.path.splitext(filename)[0]
        json_output_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")

        if not os.path.exists(json_output_path):
            continue

        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if the words are already processed
            if "words" in data and isinstance(data["words"], list):
                continue 

            # Flattening Words
            all_words = []
            speaker_counter = Counter()

            if "segments" in data:
                for segment in data["segments"]:
                    for word_obj in segment["words"]:
                        # Clean the word
                        raw_word = word_obj.get("word", "")
                        cleaned_word = clean_text(raw_word)
                
                        if not cleaned_word:
                            continue

                        # Count the speaker
                        speaker = word_obj.get("speaker")
                        speaker_counter[speaker] += 1
                            
                        # Add the word if the start and end are not None
                        if word_obj["start"] is not None and word_obj["end"] is not None:
                            all_words.append(word_obj)
                        else:
                            logger.warning(f"Word '{cleaned_word}' has no start or end time in {filename}")

            # Identify the main speaker
            main_speaker = None
            if speaker_counter:
                main_speaker = speaker_counter.most_common(1)[0][0]
            
            # Filter and tag the words
            count_kept_words = 0
            for w in all_words:
                # Keep the word if it is the main speaker
                if main_speaker and w["speaker"] == main_speaker:
                    w["keep"] = True
                    count_kept_words += 1
                else:
                    w["keep"] = False

            # Sort by start time
            #all_words.sort(key=lambda x: x["start"])
            
            # Check for time anomalies
            for i in range(1, len(all_words)):
                if all_words[i]["start"] < all_words[i-1]["start"]:
                    logger.warning(f"Time anomaly in {filename}: Word '{all_words[i]['word']}' starts before the previous word starts.")

            # Delete the old words list if it exists
            if "word_segments" in data:
                del data["word_segments"]

            # Delete the old segments list if it exists
            if "segments" in data:
                del data["segments"]
            
            # Save the new words list
            data["words"] = all_words
            data["main_speaker"] = main_speaker
            data["stats"] = {
                "total_words": len(all_words),
                "kept_words": count_kept_words
            }

            # Save the final result
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Stage 4 Error on {filename}: {e}")
            print(f"❌ Error processing {filename}: {e}")

    print("✅ Stage 4 Complete. JSON files normalized and cleaned.")


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
    stage_4_processing(videos)

    print(f"\nDone! All results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()