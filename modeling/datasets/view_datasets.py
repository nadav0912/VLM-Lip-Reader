import os
import json
import sys
from dotenv import load_dotenv
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.torch_dataset import print_dataset_stats, analyze_word_lengths, analyze_character_coverage, analyze_data_cleanliness

def view_single_words_dataset(labels_file_path):
    # load the data once for all functions
    if os.path.exists(labels_file_path):
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_words_list = list(data.values())

        # 2. run the existing functions (frequencies)
        print_dataset_stats(labels_file_path) # your original function
        
        # 3. run the new functions
        analyze_word_lengths(all_words_list)      # word lengths
        analyze_character_coverage(all_words_list) # character coverage
        analyze_data_cleanliness(all_words_list) # data cleanliness
        
    else:
        print(f"File not found: {labels_file_path}")


if __name__ == "__main__":
    labels_file_path = "data/06_single_words_dataset/labels.json"
    view_single_words_dataset(labels_file_path)
