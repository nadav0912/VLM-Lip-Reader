import json
import os

# 1. הגדרות נתיבים
BASE_DIR = r"C:\VLM-Lip-Reader"
LINKS_FILE = os.path.join(BASE_DIR, "links.txt")
# תיקייה חדשה שבה יישמרו הקבצים המפוצלים
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "configs", "speakers")

print(f"🔍 מחפש את הקובץ בנתיב: {LINKS_FILE}")

# 2. בדיקה אם קובץ המקור קיים
if not os.path.exists(LINKS_FILE):
    print(f"Error: הקובץ links.txt לא נמצא ב-{BASE_DIR}!")
else:
    # 3. קריאת השורות
    with open(LINKS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # מילון שיחזיק רשימת סרטונים לכל דובר
    speakers_data = {}
    current_speaker = "unknown"

    # 4. עיבוד הדוברים והלינקים
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if "Speaker" in line:
            # הופך "Speaker 01" ל-"Speaker_01"
            current_speaker = line.split(":")[0].replace(" ", "_")
            if current_speaker not in speakers_data:
                speakers_data[current_speaker] = []
        elif "youtube.com" in line or "youtu.be" in line:
            speakers_data[current_speaker].append({
                "url": line,
                "speaker_id": current_speaker
            })

    # 5. יצירת התיקייה ושמירת קובץ נפרד לכל דובר
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for speaker, videos in speakers_data.items():
        file_name = f"{speaker.lower()}.json"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(videos, f, indent=4, ensure_ascii=False)
        
        print(f"✅ נוצר קובץ עבור {speaker}: {len(videos)} סרטונים.")

    print(f"\n✨ הצלחה! כל הדוברים פוצלו לקבצים נפרדים בתיקייה: {OUTPUT_DIR}")