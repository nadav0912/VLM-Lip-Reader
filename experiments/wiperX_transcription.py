import torch
import whisperx
from whisperx.utils import get_writer
import json
import gc
import torch
import os
from dotenv import load_dotenv
import torch.serialization
from whisperx.diarize import DiarizationPipeline

# Patch torch.load to allow loading of non-weights only models
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
torch.backends.cuda.matmul.allow_tf32 = True

load_dotenv()

print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Torch CUDA version: {torch.version.cuda}")

# --- הגדרות ---
device = "cuda" 
audio_file = "7 Principles For Teenagers To Become Millionaires.mp4"
output_dir = "output_results"
hf_token = os.getenv("HUGGINGFACE_TOKEN") # שים כאן את הטוקן שלך
batch_size = 4
compute_type = "int8"

def print_gpu_utilization():

    if torch.cuda.is_available():
        # זיכרון שמוקצה כרגע לאובייקטים של PyTorch
        allocated = torch.cuda.memory_allocated() / 1024**3
        # זיכרון שה-Driver של NVIDIA שמר עבור PyTorch (כולל זיכרון פנוי בתוך ה-Cache)
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"[GPU Memory] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        print("CUDA is not available.")
print("Status before loading model:")
print_gpu_utilization()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. תמלול ראשוני (Transcription)
print("--- Step 1: Transcribing ---")
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="en")

print("Status after transcription:")
print_gpu_utilization()


# 2. יישור זמנים (Alignment)
print("--- Step 2: Aligning words ---")
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# ניקוי זיכרון לפני ה-Diarization
gc.collect()
torch.cuda.empty_cache()

# 3. זיהוי דוברים (Diarization)
print("--- Step 3: Identifying Speakers ---")
diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

# הרצת הדיאריזציה על האודיו
diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=2)

# הצמדת הדוברים לתוצאות התמלול הקיים
result = whisperx.assign_word_speakers(diarize_segments, result)


# --- Step 4: Saving files ---
print("--- Step 4: Saving files ---")
# וידוא ששדה השפה קיים (פותר את ה-KeyError: 'language')
if "language" not in result:
    result["language"] = "en" 

# הגדרות כתיבה
writer_options = {
    "max_line_width": 1000,
    "max_line_count": 2,
    "highlight_words": False
}

# שמירת SRT
srt_writer = get_writer("srt", output_dir)
srt_writer(result, audio_file, writer_options)

# שמירת JSON מפורט עבור ה-Lip Reader
json_path = os.path.join(output_dir, f"{os.path.basename(audio_file)}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"Success! Final data saved to: {output_dir}")

del model
gc.collect()
torch.cuda.empty_cache()

print("Status after clearing cache:")
print_gpu_utilization()