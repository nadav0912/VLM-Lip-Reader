import json
import os

# 1. Path Definitions
BASE_DIR = r"C:\VLM-Lip-Reader"
LINKS_FILE = os.path.join(BASE_DIR, "links.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "assets", "configs", "source_urls.json")

print(f"🔍 Searching for file at: {LINKS_FILE}")

# 2. Check if the source file exists
if not os.path.exists(LINKS_FILE):
    print(f"Error: The file links.txt.txt was not found in {BASE_DIR}!")
    print("Please ensure the file from Word is saved there with the exact name.")
else:
    # 3. Read the file lines
    with open(LINKS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_sources = []
    current_speaker = "unknown"

    # 4. Process Speakers and Links
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if "Speaker" in line:
            # Converts "Speaker 01" to "Speaker_01"
            current_speaker = line.split(":")[0].replace(" ", "_")
        elif "youtube.com" in line or "youtu.be" in line:
            new_sources.append({
                "url": line,
                "speaker_id": current_speaker
            })

    # 5. Create config directory and save JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_sources, f, indent=4, ensure_ascii=False)

    print(f"Success! {len(new_sources)} links have been added to source_urls.json")