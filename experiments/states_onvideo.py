import json
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

import os

input_video_dir = os.getenv("RAW_VIDEOS_DIR", "")
input_transcript_dir = os.getenv("ROW_TRANSCRIPTS_DIR", "") 
print(input_video_dir)
print(input_transcript_dir)

VIDEO_PATH = os.path.join(input_video_dir, "Iman_Gadzhi_7_Principles_For_Teenagers_To_Become_Millionaires.mp4")
JSON_PATH = os.path.join(input_transcript_dir, "Iman_Gadzhi_7_Principles_For_Teenagers_To_Become_Millionaires.json")


def analyze_dataset(video_path, json_path):
    video_path = Path(video_path)
    json_path = Path(json_path)
    # 1. בדיקת קיום קבצים
    if not video_path.exists() or not json_path.exists():
        print("Error: One or more files not found.")
        return

    # 2. טעינת נתוני הוידאו
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    # 3. טעינת נתוני ה-JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # נשתמש ב-word_segments כי זה הנתון המדויק ביותר
    words = data.get('words', [])
    
    if not words:
        print("Error: No 'word_segments' found in JSON.")
        return

    # 4. עיבוד וחישוב סטטיסטיקות
    word_durations = []  # משך זמן בשניות
    word_frames = []     # משך זמן בפריימים
    gaps = []            # שתיקות בין מילים
    scores = []          # רמת ביטחון של המודל
    speaker_stats = {}   # סטטיסטיקה לכל דובר

    prev_end = 0.0

    for w in words:
        start = w['start']
        end = w['end']
        duration = end - start
        
        # חישוב פריימים (מעגלים למספר השלם הקרוב)
        frames = (duration * fps)
        
        word_durations.append(duration)
        word_frames.append(frames)
        scores.append(w.get('score', 0))

        # חישוב הפער מהמילה הקודמת (שתיקה)
        if prev_end > 0:
            gap = start - prev_end
            if gap > 0: # מתעלמים מחפיפות קטנות אם יש
                gaps.append(gap)
        prev_end = end

        # סטטיסטיקת דוברים
        spk = w.get('speaker', 'Unknown')
        speaker_stats[spk] = speaker_stats.get(spk, 0) + 1

    # המרה ל-numpy arrays לחישובים מהירים
    np_durations = np.array(word_durations)
    np_frames = np.array(word_frames)
    np_scores = np.array(scores)
    np_gaps = np.array(gaps)

    # --- הדפסת הדוח ---
    print("="*60)
    print(f"🎬 VIDEO ANALYSIS REPORT: {os.path.basename(video_path)}")
    print("="*60)
    
    print(f"Video Stats:")
    print(f"  • Resolution:   {width}x{height}")
    print(f"  • FPS:          {fps:.2f}")
    print(f"  • Total Frames: {total_frames}")
    print(f"  • Duration:     {timedelta(seconds=int(duration_sec))}")

    print("-" * 60)
    print(f"📝 TEXT/WORD STATS (Total Words: {len(words)})")
    
    # חישוב סטטיסטיקות זמן (שניות)
    print(f"\nTime per Word (Seconds):")
    print(f"  • Mean (Average):   {np.mean(np_durations):.4f} sec")
    print(f"  • Std Dev (Sigma):  {np.std(np_durations):.4f} sec")
    print(f"  • Median:           {np.median(np_durations):.4f} sec")
    print(f"  • Min:              {np.min(np_durations):.4f} sec")
    print(f"  • Max:              {np.max(np_durations):.4f} sec")

    # חישוב סטטיסטיקות פריימים
    print(f"\nFrames per Word (at {fps} FPS):")
    print(f"  • Mean (Average):   {np.mean(np_frames):.2f} frames")
    print(f"  • Std Dev (Sigma):  {np.std(np_frames):.2f} frames")
    print(f"  • Median:           {np.median(np_frames):.1f} frames")
    print(f"  • Min:              {np.min(np_frames)} frames")
    print(f"  • Max:              {np.max(np_frames)} frames")

    print(f"\nConfidence Scores (Model Certainty):")
    print(f"  • Average Score:    {np.mean(np_scores):.2%}")
    print(f"  • Lowest Score:     {np.min(np_scores):.2%}")

    if len(gaps) > 0:
        print(f"\nSilence/Gaps between words:")
        print(f"  • Average Gap:      {np.mean(np_gaps):.4f} sec")
        print(f"  • Max Gap:          {np.max(np_gaps):.4f} sec")

    print("-" * 60)
    print("🗣️  SPEAKER DISTRIBUTION")
    for spk, count in speaker_stats.items():
        percentage = (count / len(words)) * 100
        print(f"  • {spk}: {count} words ({percentage:.1f}%)")

    print("-" * 60)
    print("⚠️  OUTLIERS & WARNINGS")
    
    # בדיקת מילים קצרות מדי (פחות מ-3 פריימים זה בעייתי למודל)
    short_words_idx = np.where(np_frames < 3)[0]
    print(f"  • Extremely short words (<3 frames): {len(short_words_idx)}")
    if len(short_words_idx) > 0:
        print(f"    Examples: {[words[i]['word'] for i in short_words_idx[:5]]}")

    # בדיקת מילים ארוכות מדי (אולי שגיאת סנכרון)
    long_words_idx = np.where(np_durations > 1.5)[0] # מילה מעל 1.5 שניות
    print(f"  • Very long words (>1.5 sec): {len(long_words_idx)}")
    if len(long_words_idx) > 0:
        print(f"    Examples: {[words[i]['word'] for i in long_words_idx[:5]]}")

    print("="*60)

if __name__ == "__main__":
    analyze_dataset(VIDEO_PATH, JSON_PATH)