import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import os
import json
from dotenv import load_dotenv
load_dotenv()


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
# MAIN EXECUTION
# ================================================================================== #
def main():
    print("🚀 Starting Data Processing and Statistics Generation...")

    # 1. Define paths (Adjust these env variables to match your actual .env keys if needed)
    SINGLE_WORDS_JSON =  os.getenv("SINGLE_WORD_CLIPS_DIR") + "/labels.json"
    SENTENCES_LABELS_DIR = os.getenv("CLIPS_LABELS_DIR")
    
    # Kaggle Release Structure
    OUT_DIR = os.getenv("STATISTICS_DIR")
    STATS_IMG_DIR = os.path.join(OUT_DIR, "stats")
    STATS_JSON_FILE = os.path.join(OUT_DIR, "summary_stats.json")

    # 2. Create necessary directories
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_IMG_DIR, exist_ok=True)
    
    # Initialize/Clear the summary_stats.json file at the start of the run
    with open(STATS_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)

    # ==========================================
    # 3. PROCESS SINGLE WORDS
    # ==========================================
    print("\n⏳ Processing Single Words Dataset...")
    df_words = generate_single_words_dataFrame(SINGLE_WORDS_JSON)
    
    if df_words is not None and not df_words.empty:
        # Save CSV for Kaggle users
        words_csv_path = os.path.join(OUT_DIR, "single_words_metadata.csv")
        df_words.to_csv(words_csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved {words_csv_path}")
        
        # Generate Graphs
        print("📊 Generating Single Words Graphs...")
        plot_top_words_bar(df_words, STATS_IMG_DIR)
        plot_words_per_speaker(df_words, STATS_IMG_DIR)
        plot_word_duration_distribution(df_words, STATS_IMG_DIR)
        generate_word_cloud(df_words, STATS_IMG_DIR)
        
        # Calculate and Save Stats
        calculate_and_save_single_words_stats(df_words, STATS_JSON_FILE)
    else:
        print("⚠️ No single words data processed.")

    # ==========================================
    # 4. PROCESS SENTENCES
    # ==========================================
    print("\n⏳ Processing Sentences Dataset...")
    df_sentences = process_sentences_data(SENTENCES_LABELS_DIR)
    
    if df_sentences is not None and not df_sentences.empty:
        # Save CSV for Kaggle users
        sentences_csv_path = os.path.join(OUT_DIR, "sentences_metadata.csv")
        df_sentences.to_csv(sentences_csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 Saved {sentences_csv_path}")
        
        # Generate Graphs
        print("📊 Generating Sentences Graphs...")
        plot_sentence_word_count_distribution(df_sentences, STATS_IMG_DIR)
        plot_sentence_duration_distribution(df_sentences, STATS_IMG_DIR)
        
        # Calculate and Save Stats
        calculate_and_save_sentences_stats(df_sentences, STATS_JSON_FILE)
    else:
        print("⚠️ No sentences data processed.")

    print(f"\n🎉 All done! Everything is packed and ready in the '{OUT_DIR}' directory.")



if __name__ == "__main__":
    main()