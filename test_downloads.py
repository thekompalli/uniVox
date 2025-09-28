import shutil
from pathlib import Path
from datetime import datetime

def find_and_download_latest():
    """Find and download the latest processed audio file"""
    audio_dir = Path("data/audio")
    
    if not audio_dir.exists():
        print("Audio directory not found")
        return
    
    # Find all processed files
    processed_files = []
    for job_dir in audio_dir.iterdir():
        if job_dir.is_dir():
            processed_file = job_dir / f"{job_dir.name}_processed.wav"
            if processed_file.exists():
                processed_files.append(processed_file)
    
    if not processed_files:
        print("No processed files found")
        return
    
    # Get latest file
    latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
    
    # Create downloads directory
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    # Copy file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = downloads_dir / f"latest_processed_{timestamp}.wav"
    shutil.copy2(latest_file, output_file)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"Downloaded: {output_file}")
    print(f"Size: {file_size:.1f} MB")
    print(f"From job: {latest_file.parent.name}")

if __name__ == "__main__":
    find_and_download_latest()