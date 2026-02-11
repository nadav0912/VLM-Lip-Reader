import cv2
import json
import os
import numpy as np
import subprocess
from tqdm import tqdm
import math

def create_debug_video(video_path, analysis_path, transcript_path, segments, output_path):
    """
    Generates a visual debug video with:
    - Green overlay for KEEP segments / Red for DISCARD
    - Text overlay with status and subtitles
    - Rotated bounding box around the mouth
    - Original audio merged
    """
    
    # 1. Load External Data
    if not os.path.exists(analysis_path) or not os.path.exists(transcript_path):
        print("❌ Visualizer: Missing analysis or transcript files.")
        return

    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
        
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)

    # 2. Setup Video Processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Visualizer: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary output (video only, no audio)
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # 3. Create Fast Lookup Maps
    
    # A. Keep Mask
    keep_mask = np.zeros(total_frames, dtype=bool)
    for seg in segments:
        s, e = seg['start_frame'], seg['end_frame']
        s = max(0, s)
        e = min(total_frames, e)
        keep_mask[s:e+1] = True

    # B. Frame Analysis Map
    frame_info_map = {f["i"]: f for f in analysis_data["frames"]}

    # C. Words List
    words = transcript_data.get("words", [])
    word_idx = 0

    print(f"🎨 Generating Debug Video: {os.path.basename(output_path)}...")

    # 4. Render Loop
    for i in tqdm(range(total_frames), unit="fr", leave=False):
        ret, frame = cap.read()
        if not ret: break

        # --- Color Overlay ---
        overlay = frame.copy()
        is_kept = keep_mask[i]
        
        if is_kept:
            color = (0, 255, 0) # Green
            status_text = "KEEP"
            box_color = (255, 0, 0) # Blue box for valid frames
        else:
            color = (0, 0, 255) # Red
            status_text = "DISCARD"
            box_color = (0, 0, 255) # Red box for invalid frames
            
        cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # --- Draw Mouth Bounding Box ---
        f_info = frame_info_map.get(i, {})
        anchors = f_info.get("a") # Get landmarks
        
        if anchors:
            try:
                # Extract points
                ml = anchors["mouth_l"]
                mr = anchors["mouth_r"]
                mt = anchors["mouth_t"]
                mb = anchors["mouth_b"]
                
                # 1. Calculate Center
                center_x = (ml[0] + mr[0]) // 2
                center_y = (mt[1] + mb[1]) // 2
                
                # 2. Calculate Width and Height (with some padding)
                w = int(math.hypot(mr[0] - ml[0], mr[1] - ml[1]) * 1.5) # 1.5x padding
                h = int(math.hypot(mb[0] - mt[0], mb[1] - mt[1]) * 1.8) # 1.8x padding (mouth opens vertically)
                
                # 3. Calculate Rotation Angle
                # ArcTangent of the slope between left and right mouth corners
                dy = mr[1] - ml[1]
                dx = mr[0] - ml[0]
                angle = math.degrees(math.atan2(dy, dx))
                
                # 4. Create Rotated Rectangle
                rect = ((center_x, center_y), (w, h), angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box) # Convert to integer
                
                # Draw the box
                cv2.drawContours(frame, [box], 0, box_color, 2)
                
            except Exception as e:
                pass # Skip drawing if calculation fails

        # --- Top Info (Status & Reason) ---
        raw_status = f_info.get("s", "?")
        reject_reason = f_info.get("r", "")
        
        info_str = f"Frame: {i} | {status_text} | Raw: {raw_status}"
        if reject_reason:
            info_str += f" | {reject_reason}"
            
        cv2.putText(frame, info_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
        cv2.putText(frame, info_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # --- Bottom Info (Subtitles) ---
        current_time = i / fps
        subtitle = ""
        
        while word_idx < len(words):
            w = words[word_idx]
            if w["end"] < current_time:
                word_idx += 1
                continue
            if w["start"] <= current_time <= w["end"]:
                subtitle = w["word"]
                break
            if w["start"] > current_time:
                break 

        if subtitle:
            text_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 60
            
            cv2.putText(frame, subtitle, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)
            cv2.putText(frame, subtitle, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

        out.write(frame)

    cap.release()
    out.release()

    # 5. Merge Audio
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_output) 
    except Exception as e:
        print(f"⚠️ Audio merge failed: {e}")
        if os.path.exists(temp_output):
            os.replace(temp_output, output_path)

    print("✅ Debug video ready.")


def create_captioned_video(video_path, json_path, output_path):
    """
    Overlay subtitles dynamically.
    Adapts font size and position to video resolution.
    Debugs timing issues.
    """
    
    # 1. בדיקת קבצים
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        return

    # 2. טעינת נתונים
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # תמיכה בשני הפורמטים (אם זה JSON של סרטון מלא או של קליפ)
        words = data.get("words", []) 
        if not words and "text" in data: # פורמט של Dataset
             words = data["text"].get("words", [])

    if not words:
        print("❌ No words found in JSON!")
        return

    # 3. הגדרות וידאו
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 Video Info: {width}x{height} @ {fps}fps")
    print(f"📝 First word starts at: {words[0]['start']}s")
    print(f"📝 Last word ends at: {words[-1]['end']}s")

    # קובץ זמני
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    # --- חישוב דינמי של גודל פונט ---
    # נכוון לכך שהטקסט יתפוס בערך 80% מהרוחב אם הוא ארוך, או גודל סביר
    # פקטור בסיס: רוחב 1920 נותן סקייל 2.0
    font_scale = max(0.5, width / 1000.0) 
    thickness = max(1, int(font_scale * 2))
    
    # מיקום Y: תמיד 10% מלמטה
    text_y_pos = int(height * 0.90)

    word_idx = 0
    words_drawn_count = 0

    print("🚀 Rendering frames...")
    
    for i in tqdm(range(total_frames), unit="fr"):
        ret, frame = cap.read()
        if not ret: break

        current_time = i / fps 

        # לוגיקה למציאת המילה (יעילה)
        while word_idx < len(words) and words[word_idx]["end"] < current_time:
            word_idx += 1
        
        active_word = None
        if word_idx < len(words):
            w = words[word_idx]
            if w["start"] <= current_time <= w["end"]:
                active_word = w["word"]

        # --- ציור הטקסט ---
        if active_word:
            words_drawn_count += 1
            text = active_word.upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # חישוב גודל כדי למרכז
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x = (width - text_w) // 2
            
            # ציור מסגרת שחורה (Outline) חזקה לקונטרסט
            cv2.putText(frame, text, (x, text_y_pos), font, font_scale, (0,0,0), thickness * 3, cv2.LINE_AA)
            # ציור הטקסט הלבן
            cv2.putText(frame, text, (x, text_y_pos), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    if words_drawn_count == 0:
        print("\n⚠️ WARNING: Video created but NO words were drawn.")
        print("Possible reason: Timestamp mismatch.")
        print(f"Video duration: {total_frames/fps:.2f}s")
        print(f"First word start: {words[0]['start']:.2f}s")
        if words[0]['start'] > (total_frames/fps):
            print("👉 The words start AFTER the video ends! Did you forget to normalize times to 0.0?")
    else:
        print(f"\n✅ Drawn content on {words_drawn_count} frames.")

    # 5. איחוד אודיו
    print("🔊 Merging audio...")
    command = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-i", video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-loglevel", "error",
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"🎉 Final video ready: {output_path}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
    except Exception as e:
        print(f"❌ Error merging audio: {e}")


if __name__ == "__main__":
    v_path = "data/04_dataset/videos/v002_c000.mp4"
    j_path = "data/04_dataset/labels/v002_c000.json"
    o_path = "test_subtitle_result.mp4"
    
    create_captioned_video(v_path, j_path, o_path)