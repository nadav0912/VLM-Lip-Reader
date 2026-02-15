import torch
import json
import os
from string import ascii_lowercase
import statistics
import re
from collections import Counter

def pad_collate(batch):
    """
    Function that takes a list of samples (Tuples) from the Dataset,
    finds the longest video in the Batch, and pads the others with zeros.
    """
    batch = [item for item in batch if item is not None]
    
    # If in case all the Batch is empty (very rare), return nothing
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
    # batch looks like this: [(video1, label1), (video2, label2), ...]
    # 1. Separate videos and labels
    videos = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 2. Find the longest video in this batch
    # shape is (C, Time, H, W) -> We check the Time dimension (index 1)
    max_len = max([v.size(1) for v in videos])
    
    # Get the rest of the dimensions from the first video
    C, _, H, W = videos[0].size()
    
    # 3. Create an empty tensor of the maximum size (Batch_Size, C, Max_Len, H, W)
    batch_size = len(videos)
    padded_videos = torch.zeros(batch_size, C, max_len, H, W)
    
    # Save the original lengths (useful for advanced models that need Masking)
    lengths = []

    # 4. Fill the tensor
    for i, vid in enumerate(videos):
        current_len = vid.size(1)
        lengths.append(current_len)
        
        # Copy the current video into the larger tensor
        # [:current_len] ensures we fill from the start, and the rest remains 0
        padded_videos[i, :, :current_len, :, :] = vid

    # Convert the labels to a tensor
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    return padded_videos, labels, lengths


def print_dataset_stats(labels_path):
    # Load the labels from the JSON file
    if not os.path.exists(labels_path):
        print(f"❌ Error: File not found at {labels_path}")
        return

    print(f"📂 Loading labels from: {labels_path} ...")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the list of words (the Values in the JSON)
    all_words = list(data.values())
    total_samples = len(all_words)
    
    # Count the frequency for each word
    counter = Counter(all_words)
    unique_words = list(counter.keys())
    vocab_size = len(unique_words)
    frequencies = list(counter.values())

    # --- Statistical calculations ---
    if frequencies:
        avg_freq = statistics.mean(frequencies)
        median_freq = statistics.median(frequencies)
        try:
            stdev = statistics.stdev(frequencies)
        except:
            stdev = 0
        
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        
        # Count "Singletons" (words that appear only once)
        singletons = len([f for f in frequencies if f == 1])
        singleton_percentage = (singletons / vocab_size) * 100
    else:
        avg_freq, median_freq, stdev, max_freq, min_freq, singletons, singleton_percentage = 0,0,0,0,0,0,0

    # --- Print the report ---
    print("\n" + "="*50)
    print(f"📊 DATASET STATISTICS REPORT")
    print("="*50)
    
    print(f"🔹 Total Video Clips:      {total_samples}")
    print(f"🔹 Vocabulary Size:        {vocab_size} unique words")
    print("-" * 50)
    
    print(f"📈 Frequency Analysis:")
    print(f"   • Mean (Average):       {avg_freq:.2f} clips per word")
    print(f"   • Median:               {median_freq:.2f} clips per word")
    print(f"   • Std Deviation:        {stdev:.2f}")
    print(f"   • Min / Max:            {min_freq} / {max_freq}")
    print("-" * 50)
    
    print(f"⚠️ Imbalance Check:")
    print(f"   • Singletons (1 sample): {singletons} words ({singleton_percentage:.1f}%)")
    if singleton_percentage > 20:
        print("     (Warning: High number of singletons might confuse the model during validation)")
    
    print("-" * 50)
    print(f"🏆 Top 10 Most Common Words:")
    for i, (word, count) in enumerate(counter.most_common(10), 1):
        print(f"   {i}. {word:<15} ({count} clips)")
        
    print("-" * 50)
    print(f"👻 Bottom 5 Least Common Words:")
    for word, count in counter.most_common()[:-6:-1]:
        print(f"   • {word:<15} ({count} clips)")
        
    print("="*50 + "\n")


def analyze_word_lengths(all_words):
    """
    Analyzes the length of words (number of characters).
    Prints ALL single-character words to help identify data cleaning issues.
    """
    unique_words = list(set(all_words))
    
    # מיפוי אורכים: מפתח = אורך, ערך = רשימת מילים
    length_map = {}
    for word in unique_words:
        w_len = len(word)
        if w_len not in length_map:
            length_map[w_len] = []
        length_map[w_len].append(word)
        
    sorted_lengths = sorted(length_map.keys())
    
    print("\n" + "="*50)
    print(f"📏 WORD LENGTH ANALYSIS")
    print("="*50)
    
    # --- 1. מילים קצרות במיוחד (אות אחת) - השינוי כאן ---
    one_letter_words = length_map.get(1, [])
    print(f"⚠️ Single-Character Words ({len(one_letter_words)} found):")
    
    if one_letter_words:
        # ממיין ומדפיס את הרשימה המלאה בצורה ברורה
        print(f"   👉 LIST: {sorted(one_letter_words)}")
        print("      (Tip: 'a' and 'i' are valid. Others like 's', 't' are likely noise.)")
    else:
        print("   ✅ None found.")
        
    print("-" * 50)

    # --- 2. מילים ארוכות במיוחד (Top 5 הכי ארוכות) ---
    if sorted_lengths:
        max_len = sorted_lengths[-1]
        longest_words = length_map[max_len]
        print(f"🦒 Longest Words ({max_len} chars):")
        for w in longest_words[:5]: # מציג רק 5 ראשונים
            print(f"   👉 {w}")
    
    print("-" * 50)
    
    # --- 3. התפלגות (היסטוגרמה טקסטואלית) ---
    print(f"📊 Length Distribution (Unique Words):")
    total_unique = len(unique_words)
    for length in sorted_lengths:
        # מציג רק אורכים שיש בהם תוכן (עד אורך 20 למשל כדי לא להציף)
        if length <= 20 or len(length_map[length]) > 0:
            count = len(length_map[length])
            percentage = (count / total_unique) * 100
            bar = "█" * int(percentage / 2) # כל קו מייצג 2%
            print(f"   {length:2d} chars: {count:4d} words ({percentage:5.1f}%) | {bar}")
            
    print("="*50 + "\n")


def analyze_character_coverage(all_words):
    """
    Checks if the dataset contains all letters of the alphabet.
    Important to ensure the model learns visual features for all phonemes/visemes.
    """
    # concatenate all words into a single huge string
    huge_string = "".join(all_words).lower()
    char_counts = Counter(huge_string)
    
    alphabet = ascii_lowercase # a, b, c, d, e, ...
    missing_chars = [char for char in alphabet if char not in char_counts]
    
    print("\n" + "="*50)
    print(f"🔤 CHARACTER COVERAGE REPORT")
    print("="*50)
    
    # missing letters
    if missing_chars:
        print(f"❌ MISSING CHARACTERS (Model won't learn these!):")
        print(f"   👉 {', '.join(missing_chars).upper()}")
    else:
        print(f"✅ Full Alphabet Coverage (A-Z present)")
        
    print("-" * 50)
    
    # rarest letters (less than 100 occurrences in the entire dataset)
    rare_threshold = 100
    rare_chars = [char for char, count in char_counts.items() if count < rare_threshold and char in alphabet]
    
    if rare_chars:
        print(f"⚠️ Rare Characters (<{rare_threshold} instances):")
        for char in rare_chars:
            print(f"   • '{char.upper()}': {char_counts[char]}")
    else:
        print("✅ All characters define robustly represented.")
        
    print("="*50 + "\n")


def analyze_data_cleanliness(all_words):
    """
    Performs a 'Sanity Check' on the text labels to ensure quality.
    Checks for:
    1. Uppercase letters (normalization issues).
    2. Non-alphabetic characters (numbers, punctuation).
    3. Leading/Trailing whitespace.
    """
    print("\n" + "="*50)
    print(f"🧹 DATA CLEANLINESS REPORT")
    print("="*50)
    
    unique_words = list(set(all_words))
    problems_found = False

    # --- 1. Uppercase Check ---
    # בודק אם יש מילים שלא הומרו לאותיות קטנות
    uppercase_words = [w for w in unique_words if any(c.isupper() for c in w)]
    
    if uppercase_words:
        problems_found = True
        print(f"❌ UPPERCASE ISSUES DETECTED!")
        print(f"   Found {len(uppercase_words)} words with uppercase letters.")
        print(f"   Examples: {uppercase_words[:5]} ...")
        print("   👉 Fix: Apply .lower() on your dataset generation script.")
    else:
        print(f"✅ Case Normalization: OK (All lowercase)")
        
    print("-" * 50)

    # --- 2. Punctuation & Numbers Check ---
    # בודק אם יש מילים שמכילות משהו שהוא לא אותיות (a-z)
    # Regex: מחפש כל דבר שאינו a-z ואינו רווח (למקרה של ביטויים בני 2 מילים)
    dirty_pattern = re.compile(r'[^a-z ]')
    dirty_words = [w for w in unique_words if dirty_pattern.search(w)]
    
    if dirty_words:
        problems_found = True
        print(f"⚠️ DIRTY CHARACTERS DETECTED (Punctuation/Numbers)!")
        print(f"   Found {len(dirty_words)} words with non-alphabetic chars.")
        print(f"   Examples: {dirty_words[:5]} ...")
        print("   👉 Fix: Use regex to remove punctuation (e.g., 'hello!' -> 'hello').")
    else:
        print(f"✅ Clean Characters: OK (Only a-z present)")

    print("-" * 50)

    # --- 3. Whitespace Check ---
    # בודק אם יש רווחים "שקופים" בהתחלה או בסוף
    whitespace_issues = [w for w in unique_words if w.strip() != w]
    
    if whitespace_issues:
        problems_found = True
        print(f"❌ WHITESPACE ISSUES DETECTED!")
        print(f"   Found {len(whitespace_issues)} words with leading/trailing spaces.")
        print(f"   Examples: {list(map(repr, whitespace_issues[:5]))}") # repr מראה את הרווחים
        print("   👉 Fix: Apply .strip() on your labels.")
    else:
        print(f"✅ Whitespace: OK (No hidden spaces)")

    print("="*50 + "\n")
    return problems_found

# --- example of running (if you run the file directly) ---
if __name__ == "__main__":
    # assume this is your path
    path = "data/06_single_words_dataset/labels.json"
    print_dataset_stats(path)