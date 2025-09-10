"""
Audio Processing Service
Core audio preprocessing and feature extraction
"""
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from src.config.app_config import app_config
from src.utils.audio_utils import AudioUtils
from src.models.vad_inference import VADInference
from src.repositories.audio_repository import AudioRepository

logger = logging.getLogger(__name__)


class AudioProcessingService:
    """Audio preprocessing and feature extraction service"""
    
    def __init__(self):
        self.audio_utils = AudioUtils()
        self.vad_model = VADInference()
        self.audio_repo = AudioRepository()
        
    async def process_audio(self, job_id: str, audio_file_path: str) -> Dict[str, Any]:
        """
        Main audio processing pipeline
        
        Args:
            job_id: Job identifier
            audio_file_path: Path to audio file
            
        Returns:
            Processing results with segments and features
        """
        try:
            logger.info(f"Starting audio processing for job {job_id}")
            
            # Load and normalize audio
            audio_data, sample_rate = await self.load_and_normalize_audio(audio_file_path)
            
            # Voice Activity Detection
            speech_segments = await self.voice_activity_detection(audio_data, sample_rate)
            
            # Audio quality enhancement
            enhanced_audio = await self.enhance_audio_quality(audio_data, sample_rate)
            
            # Feature extraction
            features = await self.extract_features(enhanced_audio, sample_rate)
            
            # Save processed audio and features
            processed_path = await self.save_processed_audio(
                job_id, enhanced_audio, sample_rate
            )
            
            result = {
                "job_id": job_id,
                "original_path": audio_file_path,
                "processed_path": processed_path,
                "duration": float(len(enhanced_audio) / sample_rate),  # Ensure regular float
                "sample_rate": int(sample_rate),  # Ensure regular int
                "speech_segments": speech_segments,
                "features": features,
                "quality_metrics": await self.calculate_quality_metrics(
                    audio_data, enhanced_audio, sample_rate
                )
            }
            
            logger.info(f"Audio processing completed for job {job_id}")
            return result
            
        except Exception as e:
            logger.exception(f"Error in audio processing for job {job_id}: {e}")
            raise
    
    async def load_and_normalize_audio(
        self, 
        audio_file_path: str
    ) -> Tuple[np.ndarray, int]:
        """
        Load and normalize audio file
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Normalized audio data and sample rate
        """
        try:
            # Load audio file
            audio_data, original_sr = librosa.load(
                audio_file_path, 
                sr=None,  # Keep original sample rate initially
                mono=False
            )
            
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # Resample to target sample rate
            if original_sr != app_config.target_sample_rate:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=original_sr,
                    target_sr=app_config.target_sample_rate
                )
            
            # Normalize audio amplitude
            audio_data = self.audio_utils.normalize_audio(audio_data)
            
            # Remove silence from beginning and end
            audio_data = self.audio_utils.trim_silence(audio_data)
            
            logger.info(f"Loaded and normalized audio: {len(audio_data)} samples, "
                       f"{app_config.target_sample_rate}Hz")
            
            return audio_data, app_config.target_sample_rate
            
        except Exception as e:
            logger.exception(f"Error loading audio file {audio_file_path}: {e}")
            raise
    
    async def voice_activity_detection(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> List[Dict[str, float]]:
        """
        Detect speech segments using VAD
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            
        Returns:
            List of speech segments with start/end times
        """
        try:
            logger.info("Running voice activity detection")
            
            # Use Silero VAD for initial detection
            vad_segments = await self.vad_model.detect_speech(audio_data, sample_rate)
            
            # Post-process segments
            filtered_segments = []
            for segment in vad_segments:
                duration = segment['end'] - segment['start']
                
                # Filter out very short segments
                if duration >= app_config.min_segment_duration:
                    filtered_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'confidence': segment.get('confidence', 1.0)
                    })
            
            # Merge nearby segments
            merged_segments = self._merge_nearby_segments(filtered_segments)
            
            logger.info(f"VAD detected {len(merged_segments)} speech segments")
            return merged_segments
            
        except Exception as e:
            logger.exception(f"Error in voice activity detection: {e}")
            raise
    
    async def enhance_audio_quality(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """
        Enhance audio quality for better processing
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Enhanced audio signal
        """
        try:
            logger.info("Enhancing audio quality")
            
            enhanced_audio = audio_data.copy()
            
            # Calculate SNR
            snr = self.audio_utils.calculate_snr(enhanced_audio)
            
            # Apply noise reduction if SNR is low
            if snr < 10:  # 10 dB threshold
                logger.info(f"Low SNR detected ({snr:.2f} dB), applying noise reduction")
                enhanced_audio = self.audio_utils.reduce_noise(
                    enhanced_audio, sample_rate
                )
            
            # Apply pre-emphasis filter
            enhanced_audio = self.audio_utils.apply_preemphasis(enhanced_audio)
            
            # Dynamic range compression
            enhanced_audio = self.audio_utils.dynamic_range_compression(enhanced_audio)
            
            # Final normalization
            enhanced_audio = self.audio_utils.normalize_audio(enhanced_audio)
            
            logger.info("Audio quality enhancement completed")
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in audio enhancement: {e}")
            # Return original audio if enhancement fails
            return audio_data
    
    async def extract_features(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio features for downstream processing
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        try:
            logger.info("Extracting audio features")
            
            features = {}
            
            # MFCC features (for general audio analysis) - convert to list for JSON serialization
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=13,
                hop_length=512,
                win_length=1024
            )
            features['mfcc_shape'] = mfcc.shape  # Store shape info
            features['mfcc_mean'] = float(np.mean(mfcc))  # Store statistical summary
            features['mfcc_std'] = float(np.std(mfcc))
            
            # Mel spectrogram (for deep learning models) - store summary stats only
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=80,
                hop_length=512,
                win_length=1024
            )
            features['mel_spectrogram_shape'] = mel_spec.shape
            features['mel_spectrogram_mean'] = float(np.mean(mel_spec))
            features['mel_spectrogram_std'] = float(np.std(mel_spec))
            
            # Spectral features - convert to lists
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            features['spectral_centroids_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroids_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['zero_crossing_rate_std'] = float(np.std(zcr))
            
            # Chroma features (for music/speech discrimination) - store summary
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=sample_rate
            )
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # Tempo and beat features - ensure JSON serializable
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=sample_rate
            )
            features['tempo'] = float(tempo)
            features['beats_count'] = len(beats)
            features['beats_mean_interval'] = float(np.mean(np.diff(beats))) if len(beats) > 1 else 0.0
            
            logger.info("Feature extraction completed")
            return features
            
        except Exception as e:
            logger.exception(f"Error in feature extraction: {e}")
            raise
    
    async def save_processed_audio(
        self, 
        job_id: str, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> str:
        """
        Save processed audio to storage
        
        Args:
            job_id: Job identifier
            audio_data: Processed audio signal
            sample_rate: Sample rate
            
        Returns:
            Path to saved audio file
        """
        try:
            # Create output filename
            filename = f"{job_id}_processed.wav"
            
            # Save using audio repository
            file_path = await self.audio_repo.save_processed_audio(
                job_id, audio_data, sample_rate, filename
            )
            
            logger.info(f"Saved processed audio to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.exception(f"Error saving processed audio: {e}")
            raise
    
    async def calculate_quality_metrics(
        self, 
        original_audio: np.ndarray,
        processed_audio: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, float]:
        """
        Calculate audio quality metrics
        
        Args:
            original_audio: Original audio signal
            processed_audio: Processed audio signal
            sample_rate: Sample rate
            
        Returns:
            Quality metrics dictionary
        """
        try:
            metrics = {}
            
            # Signal-to-Noise Ratio
            metrics['snr_original'] = float(self.audio_utils.calculate_snr(original_audio))
            metrics['snr_processed'] = float(self.audio_utils.calculate_snr(processed_audio))
            
            # Dynamic range
            metrics['dynamic_range_original'] = float(np.max(original_audio) - np.min(original_audio))
            metrics['dynamic_range_processed'] = float(np.max(processed_audio) - np.min(processed_audio))
            
            # RMS energy
            metrics['rms_original'] = float(np.sqrt(np.mean(original_audio**2)))
            metrics['rms_processed'] = float(np.sqrt(np.mean(processed_audio**2)))
            
            # Spectral centroid (measure of brightness)
            metrics['spectral_centroid'] = float(np.mean(
                librosa.feature.spectral_centroid(y=processed_audio, sr=sample_rate)
            ))
            
            # Zero crossing rate (measure of noisiness)
            metrics['zero_crossing_rate'] = float(np.mean(
                librosa.feature.zero_crossing_rate(processed_audio)
            ))
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating quality metrics: {e}")
            return {}
    
    def _merge_nearby_segments(
        self, 
        segments: List[Dict[str, float]], 
        max_gap: float = 0.3
    ) -> List[Dict[str, float]]:
        """
        Merge speech segments that are close together
        
        Args:
            segments: List of speech segments
            max_gap: Maximum gap to merge (seconds)
            
        Returns:
            Merged segments
        """
        if not segments:
            return segments
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x['start'])
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last_merged = merged[-1]
            
            # Check if segments should be merged
            gap = current['start'] - last_merged['end']
            if gap <= max_gap:
                # Merge segments
                merged[-1] = {
                    'start': last_merged['start'],
                    'end': current['end'],
                    'confidence': (last_merged['confidence'] + current['confidence']) / 2
                }
            else:
                merged.append(current)
        
        return merged
    
    async def get_audio_chunks(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        chunk_duration: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Split audio into processing chunks
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            chunk_duration: Chunk duration in seconds
            
        Returns:
            List of audio chunks with metadata
        """
        try:
            chunk_samples = chunk_duration * sample_rate
            total_duration = len(audio_data) / sample_rate
            
            chunks = []
            for i in range(0, len(audio_data), chunk_samples):
                chunk_start = i / sample_rate
                chunk_end = min((i + chunk_samples) / sample_rate, total_duration)
                chunk_data = audio_data[i:i + chunk_samples]
                
                if len(chunk_data) > 0:
                    chunks.append({
                        'data': chunk_data,
                        'start_time': chunk_start,
                        'end_time': chunk_end,
                        'duration': len(chunk_data) / sample_rate,
                        'sample_rate': sample_rate
                    })
            
            logger.info(f"Created {len(chunks)} audio chunks")
            return chunks
            
        except Exception as e:
            logger.exception(f"Error creating audio chunks: {e}")
            raise
    
    def process_audio_sync(self, job_id: str, audio_file_path: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_audio method for Celery tasks
        
        Args:
            job_id: Job identifier
            audio_file_path: Path to audio file
            
        Returns:
            Processing results with segments and features
        """
        import asyncio
        
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async process_audio method
            result = loop.run_until_complete(
                self.process_audio(job_id, audio_file_path)
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in sync audio processing for job {job_id}: {e}")
            raise
        finally:
            # Clean up the event loop
            try:
                loop.close()
            except:
                pass