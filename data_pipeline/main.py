import subprocess
import sys
import time
from datetime import timedelta

# --- הגדרת הקבצים לפי הסדר ---
SCRIPTS = [
    "data_pipeline/01_downloader.py",
    "data_pipeline/02_transcriber.py",
    "data_pipeline/03_analyze_video.py",
    "data_pipeline/04_extract_clips.py",
    "data_pipeline/05_cut_lips.py"
]

def format_time(seconds):
    #Converts seconds to a readable format: 00:00:00
    return str(timedelta(seconds=int(seconds)))

def main():
    print(f"Starting Pipeline with {len(SCRIPTS)} steps...\n")
    
    total_start_time = time.time()
    stats = [] 

    # --- לולאת ההרצה ---
    for script_name in SCRIPTS:
        print(f"\nRunning: {script_name}...")
        step_start = time.time()

        try:
            # sys.executable ensures we use the same Python that runs this script
            # check=True causes an immediate stop if there is an error
            subprocess.run([sys.executable, script_name], check=True)
            
            step_duration = time.time() - step_start
            stats.append((script_name, step_duration))
            print(f"✅ Done. Taken: {format_time(step_duration)}\n")
            
        except subprocess.CalledProcessError:
            print(f"\nError: Pipeline stopped at '{script_name}'")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nPipeline stopped by user.")
            sys.exit(0)

    # --- Summary and statistics ---
    total_time = time.time() - total_start_time
    
    print("="*60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total Time: {format_time(total_time)}")
    print("="*60)
    print(f"{'Script Name':<30} | {'Time':<10} | {'% of Total'}")
    print("-" * 60)
    
    for name, duration in stats:
        percent = (duration / total_time) * 100
        print(f"{name:<30} | {format_time(duration):<10} | {percent:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()