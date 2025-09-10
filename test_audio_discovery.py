#!/usr/bin/env python3
"""Test audio file discovery logic"""

from pathlib import Path

def test_audio_discovery(job_id):
    """Test the audio file discovery for a specific job"""
    try:
        print(f"Testing audio file discovery for job: {job_id}")
        
        # Find audio file in job directory (same logic as Celery task)
        audio_dir = Path("data/audio") / job_id
        print(f"Looking in directory: {audio_dir}")
        
        if not audio_dir.exists():
            print(f"ERROR: Audio directory not found: {audio_dir}")
            return None
        
        # List all files in directory
        all_files = list(audio_dir.glob("*"))
        print(f"All files in directory: {all_files}")
        
        # Find the audio file (should start with "original_")
        audio_files = list(audio_dir.glob("original_*"))
        print(f"Audio files found: {audio_files}")
        
        if not audio_files:
            print(f"ERROR: No audio file found in directory")
            return None
        
        audio_file_path = str(audio_files[0])  # Use first match
        print(f"SUCCESS: Found audio file: {audio_file_path}")
        
        # Check if file actually exists
        if Path(audio_file_path).exists():
            file_size = Path(audio_file_path).stat().st_size
            print(f"File exists and is {file_size} bytes")
        else:
            print(f"ERROR: File path exists in glob but file not accessible")
        
        return audio_file_path
        
    except Exception as e:
        print(f"ERROR: Exception during discovery: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with the current job
    test_job_id = "2987b7cb-232c-4e20-bca0-66f3fb77d660"
    test_audio_discovery(test_job_id)