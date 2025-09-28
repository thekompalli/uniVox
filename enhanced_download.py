import shutil
import numpy as np
import librosa
import json
from pathlib import Path
from datetime import datetime

def calculate_snr(audio: np.ndarray, frame_length: int = 2048) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        audio: Input audio signal
        frame_length: Frame length for analysis
        
    Returns:
        SNR in dB
    """
    try:
        # Simple SNR estimation
        # Find the loudest frames as signal
        frames = []
        hop_length = frame_length // 2
        
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            frames.append(frame)
        
        if not frames:
            return 10.0
        
        frames = np.array(frames)
        frame_powers = np.mean(frames**2, axis=1)
        
        # Signal: top 10% of frames
        signal_threshold = np.percentile(frame_powers, 90)
        signal_frames = frames[frame_powers >= signal_threshold]
        signal_power = np.mean(signal_frames**2) if len(signal_frames) > 0 else 0
        
        # Noise: bottom 20% of frames
        noise_threshold = np.percentile(frame_powers, 20)
        noise_frames = frames[frame_powers <= noise_threshold]
        noise_power = np.mean(noise_frames**2) if len(noise_frames) > 0 else 1e-8
        
        # Calculate SNR
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        return float(snr_db)
        
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return 10.0  # Default reasonable SNR

def get_audio_metrics(original_audio, processed_audio, sample_rate):
    """Calculate comprehensive audio enhancement metrics"""
    
    # Calculate SNRs
    original_snr = calculate_snr(original_audio)
    processed_snr = calculate_snr(processed_audio)
    snr_improvement = processed_snr - original_snr
    
    # Calculate RMS levels
    original_rms = np.sqrt(np.mean(original_audio**2))
    processed_rms = np.sqrt(np.mean(processed_audio**2))
    
    # Calculate dynamic range
    original_dynamic_range = np.max(original_audio) - np.min(original_audio)
    processed_dynamic_range = np.max(processed_audio) - np.min(processed_audio)
    
    # Calculate spectral characteristics
    try:
        original_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=original_audio, sr=sample_rate)[0])
        processed_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=processed_audio, sr=sample_rate)[0])
    except:
        original_spectral_centroid = 0
        processed_spectral_centroid = 0
    
    return {
        'original_snr': original_snr,
        'processed_snr': processed_snr,
        'snr_improvement': snr_improvement,
        'original_rms': original_rms,
        'processed_rms': processed_rms,
        'rms_change_percent': ((processed_rms - original_rms) / original_rms * 100) if original_rms > 0 else 0,
        'original_dynamic_range': original_dynamic_range,
        'processed_dynamic_range': processed_dynamic_range,
        'original_spectral_centroid': original_spectral_centroid,
        'processed_spectral_centroid': processed_spectral_centroid,
        'duration_seconds': len(processed_audio) / sample_rate
    }

def find_and_download_latest():
    """Find and download the latest processed audio file with enhancement metrics"""
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
    job_id = latest_file.parent.name
    
    print(f"Processing latest file from job: {job_id}")
    print(f"File path: {latest_file}")
    
    # Find original file
    job_dir = latest_file.parent
    original_files = list(job_dir.glob("original_*"))
    
    if not original_files:
        print("Warning: Original file not found, cannot calculate enhancement metrics")
        original_file = None
    else:
        original_file = original_files[0]
        print(f"Original file: {original_file.name}")
    
    # Create downloads directory
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(exist_ok=True)
    
    # Copy processed file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = downloads_dir / f"latest_processed_{timestamp}.wav"
    shutil.copy2(latest_file, output_file)
    
    # Get basic file info
    file_size = output_file.stat().st_size / (1024 * 1024)
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Downloaded: {output_file}")
    print(f"Size: {file_size:.1f} MB")
    print(f"From job: {job_id}")
    
    # Calculate enhancement metrics if original file exists
    if original_file:
        try:
            print("\nCalculating enhancement metrics...")
            
            # Load both audio files
            print("Loading original audio...")
            original_audio, orig_sr = librosa.load(str(original_file), sr=None)
            
            print("Loading processed audio...")
            processed_audio, proc_sr = librosa.load(str(latest_file), sr=None)
            
            # Ensure same sample rate for comparison
            if orig_sr != proc_sr:
                print(f"Resampling original audio from {orig_sr}Hz to {proc_sr}Hz")
                original_audio = librosa.resample(original_audio, orig_sr=orig_sr, target_sr=proc_sr)
            
            # Calculate metrics
            metrics = get_audio_metrics(original_audio, processed_audio, proc_sr)
            
            print("\n" + "="*60)
            print("ENHANCEMENT METRICS")
            print("="*60)
            print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
            print(f"Sample Rate: {proc_sr} Hz")
            
            print(f"\nSIGNAL-TO-NOISE RATIO:")
            print(f"  Original SNR:    {metrics['original_snr']:>8.2f} dB")
            print(f"  Processed SNR:   {metrics['processed_snr']:>8.2f} dB")
            print(f"  SNR Improvement: {metrics['snr_improvement']:>+8.2f} dB")
            
            print(f"\nAUDIO LEVELS:")
            print(f"  Original RMS:    {metrics['original_rms']:>8.4f}")
            print(f"  Processed RMS:   {metrics['processed_rms']:>8.4f}")
            print(f"  RMS Change:      {metrics['rms_change_percent']:>+8.1f}%")
            
            print(f"\nDYNAMIC RANGE:")
            print(f"  Original:        {metrics['original_dynamic_range']:>8.4f}")
            print(f"  Processed:       {metrics['processed_dynamic_range']:>8.4f}")
            
            if metrics['original_spectral_centroid'] > 0:
                print(f"\nSPECTRAL CHARACTERISTICS:")
                print(f"  Original Centroid:  {metrics['original_spectral_centroid']:>8.1f} Hz")
                print(f"  Processed Centroid: {metrics['processed_spectral_centroid']:>8.1f} Hz")
            
            # Quality assessment
            print(f"\nQUALITY ASSESSMENT:")
            if metrics['snr_improvement'] > 3:
                quality = "EXCELLENT"
            elif metrics['snr_improvement'] > 1:
                quality = "GOOD"
            elif metrics['snr_improvement'] > -1:
                quality = "MODERATE"
            else:
                quality = "POOR"
            
            print(f"  Enhancement Quality: {quality}")
            
            if metrics['snr_improvement'] > 0:
                print(f"  Status: ✓ Audio quality improved")
            else:
                print(f"  Status: ⚠ No significant improvement detected")
            
            # Save metrics to file
            metrics_file = downloads_dir / f"metrics_{timestamp}.json"
            metrics_data = {
                'job_id': job_id,
                'timestamp': timestamp,
                'original_file': str(original_file),
                'processed_file': str(latest_file),
                'downloaded_file': str(output_file),
                'metrics': metrics
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            print(f"\nMetrics saved to: {metrics_file}")
            
        except Exception as e:
            print(f"\nError calculating enhancement metrics: {e}")
            print("File downloaded successfully, but metrics calculation failed")
    
    print("="*60)

if __name__ == "__main__":
    print("AUDIO ENHANCEMENT DOWNLOADER")
    print("="*60)
    find_and_download_latest()