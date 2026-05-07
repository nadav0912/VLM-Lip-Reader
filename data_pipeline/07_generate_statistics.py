import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import os
import sys
import json
import shutil
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.video_processing import get_mouth_roi_params

# ================================================================================== #
# GLOBAL CONFIGURATION & PATHS
# ================================================================================== #
# 1. Single Words Paths
WORDS_SRC_DIR = os.getenv("SINGLE_WORD_CLIPS_DIR")
SINGLE_WORDS_JSON = os.path.join(WORDS_SRC_DIR, "labels.json") if WORDS_SRC_DIR else ""

# 2. Sentences Paths
SENTENCES_LABELS_DIR = os.getenv("CLIPS_LABELS_DIR")
SENTENCES_SRC_DIR = os.getenv("CLIPS_DATASET_DIR")
SENTENCES_VIDEOS_DIR = os.getenv("CLIPS_VIDEOS_DIR")
SENTENCES_LIPS_DIR = os.getenv("LIPS_VIDEOS_DIR")

# 3. Output Release Paths (Build everything in 07_statistics first)
STATISTICS_DIR = os.getenv("STATISTICS_DIR")
STATS_IMG_DIR = os.path.join(STATISTICS_DIR, "visualizations")
STATS_JSON_FILE = os.path.join(STATISTICS_DIR, "dataset_summary.json")
EXAMPLES_DIR = os.path.join(STATISTICS_DIR, "examples")

# Final destination for packaging
KAGGLE_RELEASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "data/Kaggle_Release_Build"))

# Set font for subtitles in examples
FONT_PATH = "arial.ttf"


# ================================================================================== #
# CREATE DATAFRAMES 
# ================================================================================== #
def generate_single_words_dataFrame(json_path):
    """
    Reads a JSON file containing metadata about single word clips, 
    processes the data, and returns a DataFrame with relevant information.
    """
    if not os.path.exists(json_path):
        print(f"❌ Error: Could not find {json_path}")
        return None, {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for clip_name, details in data.items():
        word = details.get("word", "").strip().lower() 
        meta = details.get("metadata", {})
        
        records.append({
            "clip_name": clip_name,
            "word": word,
            "speaker": meta.get("speaker", "Unknown"),
            "source_video": meta.get("source_video", "Unknown"),
            "gender": meta.get("gender", "Unknown"),
            "duration_frames": meta.get("duration_frames", 0),
            "fps": meta.get("fps", 25.0) 
        })

    df = pd.DataFrame(records)

    return df


def process_sentences_data(labels_dir):
    """
    Reads all JSON files in the specified directory, extracts relevant information about sentence clips,
    and generates a DataFrame with the processed data.
    """
    if not os.path.exists(labels_dir):
        print(f"❌ Error: Could not find directory {labels_dir}")
        return pd.DataFrame(), {}

    records = []
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(labels_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not decode {filename}")
                continue
                
            meta = data.get("metadata", {})
            text_data = data.get("text", {})
            words_list = text_data.get("words", [])
            
            records.append({
                "clip_name": filename.replace(".json", ""), 
                "source_video": meta.get("source_video", "Unknown"),
                "sentence": text_data.get("sentence", "").strip(),
                "word_count": len(words_list),
                "duration_frames": meta.get("total_frames", meta.get("duration_frames", 0)),
                "duration_sec": meta.get("duration", 0.0),
                "speaker": meta.get("speaker", "Unknown"),
                "gender": meta.get("gender", "Unknown"),
                "fps": meta.get("fps", 25.0)
            })

    df = pd.DataFrame(records)
    
    return df


# ================================================================================== #
# PLOTTING FUNCTIONS
# ================================================================================== #
def plot_top_words_bar(df, output_dir, top_n=75):
    # Bar plot of the most frequent words in the dataset
    if df.empty: return
    
    plt.figure(figsize=(12, 15)) 
    
    top_words = df['word'].value_counts().head(top_n)
    sns.barplot(x=top_words.values, y=top_words.index, palette="viridis")
    
    plt.title(f"Top {top_n} Most Frequent Words", fontsize=16)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    
    plt.yticks(fontsize=8)
    
    plt.savefig(os.path.join(output_dir, f"words_top_{top_n}_frequencies.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_words_per_speaker(df, output_dir):
    # Bar plot showing the number of word clips per speaker
    if df.empty: return
    plt.figure(figsize=(10, 6))
    speaker_counts = df['speaker'].value_counts()
    
    sns.barplot(x=speaker_counts.index, y=speaker_counts.values, palette="magma")
    plt.title("Number of Word Clips per Speaker")
    plt.xlabel("Speaker")
    plt.ylabel("Number of Clips")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "words_per_speaker.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_word_duration_distribution(df, output_dir):
    # Distribution of word durations in frames
    if df.empty: return
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_frames'], bins=100, kde=True, color='lightgreen')
    
    plt.title("Distribution of Single Word videos Durations in Frames (with Padding)")
    plt.xlabel("Duration (Frames)")
    plt.ylabel("Number of Clips")
    plt.savefig(os.path.join(output_dir, "words_duration_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_word_cloud(df, output_dir):
    # Word cloud of the most common words in the dataset
    if df.empty: return
    
    text_for_cloud = " ".join(df['word'].astype(str))
    
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_cloud)
        
        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "words_cloud.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate word cloud: {e}")


def plot_sentence_word_count_distribution(df, output_dir):
    # Distribution of the number of words per sentence in the dataset
    if df.empty: return
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=100, kde=True, color='skyblue')
    
    plt.title("Distribution of Words per Sentence")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Clips")
    plt.savefig(os.path.join(output_dir, "sentences_word_count_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentence_duration_distribution(df, output_dir):
    # Distribution of sentence durations in frames
    if df.empty: return
    plt.figure(figsize=(10, 6))
    sns.histplot(df['duration_frames'], bins=100, kde=True, color='salmon')
    
    plt.title("Distribution of Sentence Durations (Frames)")
    plt.xlabel("Duration (Frames)")
    plt.ylabel("Number of Clips")
    plt.savefig(os.path.join(output_dir, "sentences_duration_dist.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ================================================================================== #
# STATISTICS CALCULATION FOR summary_stats.json FILE
# ================================================================================== #
def update_stats_json(json_path, category_name, stats_dict):
    """ פונקציית עזר שמעדכנת קובץ JSON קיים בלי לדרוס נתונים אחרים """
    data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
                
    data[category_name] = stats_dict
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def calculate_and_save_single_words_stats(df, json_path):
    if df.empty: return
    
    word_counts = df['word'].value_counts()
    
    stats = {
        "Total Clips": len(df),
        "Unique Words (Vocabulary)": len(word_counts),
        "Rare Words (Appeared Only Once)": int((word_counts == 1).sum()),
        "Top 10 Frequent Words": word_counts.head(10).to_dict(),
        "Total Duration (Frames)": int(df['duration_frames'].sum()),
        "Average Frames per Word": float(round(df['duration_frames'].mean(), 2)),
        "Speakers Count": int(df['speaker'].nunique()),
        "Gender Distribution": df['gender'].value_counts().to_dict()
    }
    
    update_stats_json(json_path, "Single_Words", stats)
    print("✅ Single Words statistics saved to JSON.")


def calculate_and_save_sentences_stats(df, json_path):
    if df.empty: return
    
    total_seconds = df['duration_sec'].sum()
    total_minutes = total_seconds / 60.0
    total_hours = total_minutes / 60.0
    
    stats = {
        "Total Clips": len(df),
        "Unique Speakers": int(df['speaker'].nunique()),
        "Total Duration (Frames)": int(df['duration_frames'].sum()),
        "Total Duration (Time)": {
            "Seconds": float(round(total_seconds, 2)),
            "Minutes": float(round(total_minutes, 2)),
            "Hours": float(round(total_hours, 2))
        },
        "Duration per Clip (Seconds)": {
            "Average": float(round(df['duration_sec'].mean(), 2)),
            "Median": float(round(df['duration_sec'].median(), 2)),
            "Minimum": float(round(df['duration_sec'].min(), 2)),
            "Maximum": float(round(df['duration_sec'].max(), 2))
        },
        "Gender Distribution": df['gender'].value_counts().to_dict(),
        "Word Count Stats": {
            "Average": float(round(df['word_count'].mean(), 2)),
            "Median": int(df['word_count'].median()),
            "Minimum": int(df['word_count'].min()),
            "Maximum": int(df['word_count'].max())
        }
    }
    
    update_stats_json(json_path, "Sentences", stats)
    print(f"✅ Sentences statistics saved. (Total Video Time: {round(total_minutes, 2)} minutes)")


# ================================================================================== #
# DATASET PACKAGING (FOR MANUAL UPLOAD)
# ================================================================================== #
def smart_copy(src, dst):
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            smart_copy(os.path.join(src, item), os.path.join(dst, item))
    else:
        if not os.path.exists(dst): shutil.copy2(src, dst)


def package_dataset_for_kaggle():
    print("\n📦 Packaging dataset for manual Kaggle upload (Safe Copy)...")
    sentences_dest = os.path.join(KAGGLE_RELEASE_DIR, "sentences")
    words_dest = os.path.join(KAGGLE_RELEASE_DIR, "single_words")
    
    os.makedirs(os.path.join(sentences_dest, "full_face_videos"), exist_ok=True)
    os.makedirs(os.path.join(sentences_dest, "cropped_lips_videos"), exist_ok=True)
    os.makedirs(os.path.join(sentences_dest, "raw_labels"), exist_ok=True)
    os.makedirs(os.path.join(words_dest, "cropped_lips_videos"), exist_ok=True)

    if os.path.exists(SENTENCES_VIDEOS_DIR): smart_copy(SENTENCES_VIDEOS_DIR, os.path.join(sentences_dest, "full_face_videos"))
    if os.path.exists(SENTENCES_LIPS_DIR): smart_copy(SENTENCES_LIPS_DIR, os.path.join(sentences_dest, "cropped_lips_videos"))
    if os.path.exists(SENTENCES_LABELS_DIR): smart_copy(SENTENCES_LABELS_DIR, os.path.join(sentences_dest, "raw_labels"))

    if os.path.exists(WORDS_SRC_DIR):
        for item in os.listdir(WORDS_SRC_DIR):
            src_item = os.path.join(WORDS_SRC_DIR, item)
            if item.endswith(".mp4"):
                smart_copy(src_item, os.path.join(words_dest, "cropped_lips_videos", item))
            elif item == "labels.json":
                smart_copy(src_item, os.path.join(words_dest, "raw_labels.json"))

    if os.path.exists(STATISTICS_DIR):
        smart_copy(STATISTICS_DIR, KAGGLE_RELEASE_DIR)


# ================================================================================== #
# VISUAL EXAMPLES GENERATOR (OPENCV & PILLOW)
# ================================================================================== #
def draw_text_with_outline(img_pil, text, position, font_size=16):
    """ מצייר טקסט על התמונה עם קו מתאר שחור לקריאות מקסימלית """
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()
        
    x, y = position
    for adj in range(-2, 3):
        for opp in range(-2, 3):
            if adj != 0 or opp != 0:
                draw.text((x+adj, y+opp), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return img_pil


def get_active_word(words_list, current_time):
    for w in words_list:
        if w["start"] <= current_time <= w["end"]:
            return w["word"]
    return ""


def process_single_video_example(input_video_path, json_data, output_path, draw_landmarks=False):
    """ מעבד סרטון בודד (פנים או שפתיים) ומוסיף לו כתוביות """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    words_list = json_data["text"]["words"]
    frames_data = {f["index"]: f for f in json_data.get("frames", [])}
    
    # --- תנאי גודל פונט ---
    # אם זה סרטון שפתיים (רוחב קטן), פונט קטן. אחרת, פונט גדול לסרטון פנים מלא.
    is_lips_video = w < 500  # סרטוני שפתיים הם לרוב סביב 100-200 פיקסלים ברוחב
    font_size = 14 if is_lips_video else 100
    y_offset = h - 35 if is_lips_video else h - 120
    char_width_approx = 3 if is_lips_video else 18
    # ----------------------

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        current_time = frame_idx / fps
        active_word = get_active_word(words_list, current_time)
        
        if draw_landmarks and frame_idx in frames_data:
            lm = frames_data[frame_idx]["landmarks"]
            if lm:
                rect = get_mouth_roi_params(lm, w, h, is_normalized=False)
                box = cv2.boxPoints(rect)
                box = np.int32(box) 
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        if active_word:
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text_x = (w // 2) - (len(active_word) * char_width_approx) 
            draw_text_with_outline(img_pil, active_word, (text_x, y_offset), font_size=font_size)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def process_side_by_side_example(full_vid_path, lips_vid_path, json_data, output_path):
    """ מחבר את סרטון הפנים וסרטון השפתיים אחד ליד השני עם כתוביות למטה """
    cap_f = cv2.VideoCapture(full_vid_path)
    cap_l = cv2.VideoCapture(lips_vid_path)
    
    if not cap_f.isOpened() or not cap_l.isOpened(): return

    fps = cap_f.get(cv2.CAP_PROP_FPS)
    w_f, h_f = int(cap_f.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_l, h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # הגדלת אזור השפתיים שיתאים לגובה הפנים המלאות (כדי שאפשר יהיה לחבר יפה)
    scale = h_f / h_l if h_l > 0 else 1
    new_w_l = int(w_l * scale)
    combined_w = w_f + new_w_l

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (combined_w, h_f))
    words_list = json_data["text"]["words"]
    frames_data = {f["index"]: f for f in json_data.get("frames", [])}
    
    frame_idx = 0
    while True:
        ret_f, frame_f = cap_f.read()
        ret_l, frame_l = cap_l.read()
        
        if not ret_f or not ret_l: break
        
        # ציור מלבן על הפנים
        if frame_idx in frames_data:
            lm = frames_data[frame_idx]["landmarks"]
            if lm:
                rect = get_mouth_roi_params(lm, w_f, h_f, is_normalized=False)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(frame_f, [box], 0, (0, 255, 0), 2)
            
        # שינוי גודל וחיבור הצדדים
        frame_l_resized = cv2.resize(frame_l, (new_w_l, h_f))
        combined_frame = cv2.hconcat([frame_f, frame_l_resized])
        
        # הוספת כתוביות באמצע הסרטון המשולב
        active_word = get_active_word(words_list, frame_idx / fps)
        if active_word:
            img_pil = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            text_x = (combined_w // 2) - (len(active_word) * 20)
            draw_text_with_outline(img_pil, active_word, (text_x, h_f - 150), font_size=100)
            combined_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        out.write(combined_frame)
        frame_idx += 1

    cap_f.release()
    cap_l.release()
    out.release()


def process_slow_mo_word_example(video_filename, word_data, output_path):
    """ מאט סרטון מילה בודדת לרבע מהירות (0.25x), ממרכז טקסט מוקטן ומוסיף תווית """
    input_path = os.path.join(WORDS_SRC_DIR, video_filename)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return

    # שומרים על ה-FPS המקורי וניצור את ההאטה על ידי הכפלת פריימים (4x)
    original_fps = cap.get(cv2.CAP_PROP_FPS) 
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (w, h))

    word_text = word_data["word"]
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 1. תווית הילוך איטי (פונט קטנטן למעלה)
        draw_text_with_outline(img_pil, "0.25x", (5, 5), font_size=8)
        
        # 2. כתובית המילה (פונט מוקטן וממורכז)
        text_x = (w // 2) - (len(word_text) * 4)
        draw_text_with_outline(img_pil, word_text, (text_x, h - 30), font_size=14)
        
        frame_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # כתיבת הפריים 4 פעמים = האטה של 0.25x
        out.write(frame_with_text)
        out.write(frame_with_text)
        out.write(frame_with_text)
        out.write(frame_with_text)

    cap.release()
    out.release()


def generate_all_visual_examples():
    """ בוחרת דגימות באופן אקראי ויעיל ומייצרת את כל סוגי הדוגמאות """
    print("\n🎬 Generating Visual Examples...")
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    # --- 1. משפטים: פנים, שפתיים, וצד-לצד ---
    if os.path.exists(SENTENCES_LABELS_DIR):
        all_json_files = [f for f in os.listdir(SENTENCES_LABELS_DIR) if f.endswith('.json')]
        
        # מערבבים את רשימת שמות הקבצים כדי להתחיל מנקודה אקראית
        random.shuffle(all_json_files)
        
        selected_clips = []
        
        # עוברים על הקבצים רק עד שאנחנו מוצאים 3 שעונים על התנאי
        for jf in all_json_files:
            with open(os.path.join(SENTENCES_LABELS_DIR, jf), 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if data.get("metadata", {}).get("clip_word_count", 0) > 10:
                        selected_clips.append((jf, data))
                        
                        # ברגע שמצאנו 3, מפסיקים את הלולאה (אין צורך לקרוא עוד קבצים)
                        if len(selected_clips) == 3:
                            break
                except json.JSONDecodeError:
                    continue

        if selected_clips:
            example_types = ["fullface", "lips", "side_by_side"]
            
            for i, ex_type in enumerate(example_types):
                # אם מצאנו פחות מ-3, נשתמש בראשון שוב כשנצטרך
                sample_data = selected_clips[i] if i < len(selected_clips) else selected_clips[0]
                clip_id = sample_data[1]["metadata"]["clip_id"]
                
                f_vid = os.path.join(SENTENCES_VIDEOS_DIR, f"{clip_id}.mp4")
                l_vid = os.path.join(SENTENCES_LIPS_DIR, f"{clip_id}.mp4")
                
                if ex_type == "fullface" and os.path.exists(f_vid):
                    print(f"   -> Creating Full Face example for {clip_id}...")
                    process_single_video_example(f_vid, sample_data[1], os.path.join(EXAMPLES_DIR, f"example_fullface.mp4"), draw_landmarks=True)
                
                elif ex_type == "lips" and os.path.exists(l_vid):
                    print(f"   -> Creating Lips Only example for {clip_id}...")
                    process_single_video_example(l_vid, sample_data[1], os.path.join(EXAMPLES_DIR, f"example_lips.mp4"))
                    
                elif ex_type == "side_by_side" and os.path.exists(f_vid) and os.path.exists(l_vid):
                    print(f"   -> Creating Side-by-Side example for {clip_id}...")
                    process_side_by_side_example(f_vid, l_vid, sample_data[1], os.path.join(EXAMPLES_DIR, f"example_side_by_side.mp4"))

    # --- 2. מילים בודדות: Slow Motion ---
    if os.path.exists(SINGLE_WORDS_JSON):
        with open(SINGLE_WORDS_JSON, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
            
        sample_keys = random.sample(list(words_data.keys()), min(3, len(words_data)))
        for i, key in enumerate(sample_keys):
            print(f"   -> Creating Slow-Mo Word example {i+1}...")
            process_slow_mo_word_example(key, words_data[key], os.path.join(EXAMPLES_DIR, f"example_word_{i+1}.mp4"))

    print("✅ All visual examples created successfully in Kaggle_Release_Build/examples!")


# ================================================================================== #
# MAIN EXECUTION
# ================================================================================== #
def main():
    print("🚀 Starting Data Processing, Statistics Generation, and Packaging...")

    if os.path.exists(KAGGLE_RELEASE_DIR):
        shutil.rmtree(KAGGLE_RELEASE_DIR)
    os.makedirs(KAGGLE_RELEASE_DIR, exist_ok=True)
    os.makedirs(STATS_IMG_DIR, exist_ok=True)
    
    with open(STATS_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)

    # 3. PROCESS SINGLE WORDS
    print("\n⏳ Processing Single Words Dataset...")
    df_words = generate_single_words_dataFrame(SINGLE_WORDS_JSON)
    if df_words is not None and not df_words.empty:
        words_csv_path = os.path.join(STATISTICS_DIR, "single_words_metadata.csv")
        df_words.to_csv(words_csv_path, index=False, encoding='utf-8-sig')
        
        print("   -> Generating Single Words visualizations...")
        plot_top_words_bar(df_words, STATS_IMG_DIR)
        plot_words_per_speaker(df_words, STATS_IMG_DIR)
        plot_word_duration_distribution(df_words, STATS_IMG_DIR)
        generate_word_cloud(df_words, STATS_IMG_DIR)
        
        calculate_and_save_single_words_stats(df_words, STATS_JSON_FILE)

    # 4. PROCESS SENTENCES
    print("\n⏳ Processing Sentences Dataset...")
    df_sentences = process_sentences_data(SENTENCES_LABELS_DIR)
    if df_sentences is not None and not df_sentences.empty:
        sentences_csv_path = os.path.join(STATISTICS_DIR, "sentences_metadata.csv")
        df_sentences.to_csv(sentences_csv_path, index=False, encoding='utf-8-sig')
        
        print("   -> Generating Sentences visualizations...")
        plot_sentence_word_count_distribution(df_sentences, STATS_IMG_DIR)
        plot_sentence_duration_distribution(df_sentences, STATS_IMG_DIR)
        
        calculate_and_save_sentences_stats(df_sentences, STATS_JSON_FILE)
    
    # 5. CREATE EXAMPLES
    generate_all_visual_examples()

    # 6. PACKAGE
    package_dataset_for_kaggle()

    # 7. FINISH
    print("\n🎉 All done!")
    print(f"📁 Your dataset is perfectly organized and waiting in: {KAGGLE_RELEASE_DIR}")
    print("\nNext steps on your Linux server:")
    print("1. Run this command to zip the folder:")
    print(f"   zip -r vlm_lip_reading_dataset.zip Kaggle_Release_Build/")
    print("2. Download the .zip file to your PC using WinSCP.")
    print("3. Upload manually via Kaggle.com -> Create -> New Dataset.")

if __name__ == "__main__":
    main()