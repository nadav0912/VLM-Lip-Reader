import yt_dlp
import subprocess
import json
import os


def print_real_file_specs(file_path):
    """
    Reads the actual file from your disk to verify quality.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Use ffprobe to get the truth from the file
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_streams', '-show_format', file_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        # Look for the video stream (usually index 0)
        video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
        
        width = video_stream.get('width')
        height = video_stream.get('height')
        # Calculate FPS from the fractional string (e.g., "30/1" or "24000/1001")
        fps_fraction = video_stream.get('r_frame_rate', '0/0')
        fps = eval(fps_fraction) if '/' in fps_fraction else 0
        
        # Bitrate can be in the stream or the general format container
        bitrate_raw = video_stream.get('bit_rate') or data['format'].get('bit_rate', 0)
        bitrate_kbps = int(bitrate_raw) // 1000
        
        print("\n" + "="*40)
        print(f"✅ DOWNLOAD VERIFIED")
        print(f"File: {os.path.basename(file_path)}")
        print(f"Resolution: {width}x{height} ({( 'HD' if height >= 720 else 'SD' )})")
        print(f"Frame Rate: {fps:.2f} FPS")
        print(f"Bitrate:    {bitrate_kbps} kbps")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"⚠️ Could not read file specs: {e}")


def download_and_verify(urls):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                # 1. Download the video
                info = ydl.extract_info(url, download=True)
                
                # 2. Get the actual path of the final file
                # yt-dlp might change the extension during merging, so we check the info dict
                final_filename = ydl.prepare_filename(info)
                
                # Ensure we point to the .mp4 if it was merged
                if not os.path.exists(final_filename):
                    name, _ = os.path.splitext(final_filename)
                    final_filename = name + ".mp4"

                # 3. Inspect the physical file
                print_real_file_specs(final_filename)
                
            except Exception as e:
                print(f"❌ Failed to process {url}: {e}")

# Array of target URLs
video_urls = [
    'https://www.youtube.com/watch?v=1-izXBhkiHw'
]

if __name__ == "__main__":
    download_and_verify(video_urls)