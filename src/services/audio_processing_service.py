"""
Audio Processing Service
Core audio preprocessing and feature extraction
"""
import logging
import numpy as np
import librosa
import time
import soundfile as sf
import scipy.signal  # Add this import for signal processing
import scipy.ndimage  # Add this import for image processing functions used in audio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import shutil
from datetime import datetime
import json
from typing import Dict, Any, List, Tuple, Optional

from src.config.app_config import app_config
from src.utils.audio_utils import AudioUtils
from src.models.vad_inference import VADInference
from src.repositories.audio_repository import AudioRepository

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time  # Add this import for timing

from src.config.app_config import app_config
from src.utils.audio_utils import AudioUtils
from src.models.vad_inference import VADInference
from src.repositories.audio_repository import AudioRepository

import shutil
from datetime import datetime
import json

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

# Add these complete method implementations to your src/services/audio_processing_service.py file
# Place them after the voice_activity_detection method

    async def enhance_audio_quality(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        use_advanced_noise_removal: bool = None,
        noise_removal_method: str = None,
        noise_removal_strength: float = None
    ) -> np.ndarray:
        """
        Enhance audio quality for better processing
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate
            use_advanced_noise_removal: Whether to use advanced noise removal (None for config default)
            noise_removal_method: Method for noise removal (None for config default)
            noise_removal_strength: Strength of noise removal (None for config default)
            
        Returns:
            Enhanced audio signal
        """
        try:
            logger.info("Enhancing audio quality")
            
            # Get configuration values - with fallback if config doesn't exist yet
            if use_advanced_noise_removal is None:
                use_advanced_noise_removal = getattr(app_config, 'enable_advanced_noise_removal', True)
            if noise_removal_method is None:
                noise_removal_method = getattr(app_config, 'default_noise_removal_method', 'spectral_subtraction')
                if hasattr(noise_removal_method, 'value'):
                    noise_removal_method = noise_removal_method.value
            if noise_removal_strength is None:
                noise_removal_strength = getattr(app_config, 'noise_removal_strength', 0.8)
            
            enhanced_audio = audio_data.copy()
            processing_steps = []
            
            # Step 1: Initial audio analysis
            initial_snr = self.audio_utils.calculate_snr(enhanced_audio)
            logger.info(f"Initial audio SNR: {initial_snr:.2f} dB")
            processing_steps.append(f"Initial SNR: {initial_snr:.2f} dB")
            
            # Step 2: Apply noise reduction if SNR is low
            snr_threshold_advanced = getattr(app_config, 'noise_removal_snr_threshold_advanced', 15.0)
            snr_threshold_basic = getattr(app_config, 'noise_removal_snr_threshold_basic', 10.0)
            
            if use_advanced_noise_removal and initial_snr < snr_threshold_advanced:
                logger.info(f"Low SNR detected ({initial_snr:.2f} dB), applying advanced noise reduction")
                
                # Use the enhanced noise reduction from audio_utils
                enhanced_audio = self.audio_utils.reduce_noise(
                    enhanced_audio, 
                    sample_rate, 
                    method=noise_removal_method
                )
                
                # Calculate improvement
                final_snr = self.audio_utils.calculate_snr(enhanced_audio)
                snr_improvement = final_snr - initial_snr
                processing_steps.append(f"Noise reduction ({noise_removal_method}): +{snr_improvement:.1f} dB SNR")
                
                logger.info(f"Noise reduction completed. Method: {noise_removal_method}, "
                        f"SNR improvement: +{snr_improvement:.1f} dB")
            
            elif initial_snr < snr_threshold_basic:
                logger.info(f"Very low SNR detected ({initial_snr:.2f} dB), applying basic noise reduction")
                enhanced_audio = self.audio_utils.reduce_noise(enhanced_audio, sample_rate)
                processing_steps.append("Basic noise reduction applied")
            
            # Step 3: Apply pre-emphasis filter (for speech processing)
            if self._contains_speech(enhanced_audio, sample_rate):
                enhanced_audio = self.audio_utils.apply_preemphasis(enhanced_audio)
                processing_steps.append("Pre-emphasis applied")
                logger.debug("Pre-emphasis filter applied")
            
            # Step 4: Dynamic range processing
            dynamic_range = np.max(enhanced_audio) - np.min(enhanced_audio)
            if dynamic_range > 1.5:  # If very wide dynamic range
                enhanced_audio = self.audio_utils.dynamic_range_compression(enhanced_audio)
                processing_steps.append("Dynamic range compression applied")
                logger.debug("Dynamic range compression applied")
            
            # Step 5: Final normalization and safety checks
            enhanced_audio = self.audio_utils.normalize_audio(enhanced_audio)
            
            # Ensure no clipping or artifacts
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
            
            # Step 6: Quality verification
            final_snr = self.audio_utils.calculate_snr(enhanced_audio)
            total_improvement = final_snr - initial_snr
            
            processing_steps.append(f"Final SNR: {final_snr:.2f} dB (+{total_improvement:.1f} dB total)")
            
            logger.info(f"Audio quality enhancement completed. "
                    f"SNR improved from {initial_snr:.2f} dB to {final_snr:.2f} dB "
                    f"(+{total_improvement:.1f} dB improvement)")
            
            # Store processing information for debugging/monitoring
            self.last_processing_steps = processing_steps
            
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in audio quality enhancement: {e}")
            # Return original audio if processing fails
            return audio_data

    def _contains_speech(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Check if audio contains speech content"""
        try:
            # Simple speech detection based on spectral characteristics
            stft = librosa.stft(audio_data, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            
            # Check energy in speech frequency range (300-3000 Hz)
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
            speech_mask = (freq_bins >= 300) & (freq_bins <= 3000)
            
            speech_energy = np.mean(magnitude[speech_mask, :])
            total_energy = np.mean(magnitude)
            
            speech_ratio = speech_energy / (total_energy + 1e-8)
            
            # If speech frequencies contain more than 40% of energy, likely contains speech
            return speech_ratio > 0.4
            
        except Exception as e:
            logger.exception(f"Error detecting speech content: {e}")
            return True  # Default to assuming speech is present

    # Also add these additional helper methods that may be missing:

    async def remove_background_noise(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        method: str = 'spectral_subtraction',
        strength: float = 0.8,
        preserve_speech: bool = True,
        auto_detect_method: bool = True
    ) -> Dict[str, Any]:
        """
        Remove background noise from audio with advanced techniques
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate
            method: Noise reduction method
            strength: Noise reduction strength (0.0 to 1.0)
            preserve_speech: Whether to prioritize speech preservation
            auto_detect_method: Automatically select best method based on audio characteristics
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        try:
            logger.info(f"Starting background noise removal for audio of {len(audio_data)} samples")
            
            import time
            start_time = time.time()
            
            # Apply noise reduction using audio_utils
            denoised_audio = self.audio_utils.reduce_noise(
                audio_data, sample_rate, method=method
            )
            
            processing_time = time.time() - start_time
            
            # Calculate basic metrics
            original_snr = self.audio_utils.calculate_snr(audio_data)
            final_snr = self.audio_utils.calculate_snr(denoised_audio)
            snr_improvement = final_snr - original_snr
            
            result = {
                "denoised_audio": denoised_audio,
                "original_audio": audio_data,
                "sample_rate": sample_rate,
                "method_used": method,
                "processing_time": processing_time,
                "noise_reduction_strength": strength,
                "preserve_speech": preserve_speech,
                "quality_improvement": {
                    "snr_before": original_snr,
                    "snr_after": final_snr,
                    "snr_improvement": snr_improvement,
                    "noise_reduction_db": max(0, snr_improvement)
                },
                "noise_reduction_metrics": {
                    "original_snr": float(original_snr),
                    "final_snr": float(final_snr),
                    "snr_improvement": float(snr_improvement),
                    "noise_reduction_quality": 'excellent' if snr_improvement > 3 else 'good' if snr_improvement > 1 else 'moderate'
                }
            }
            
            logger.info(f"Background noise removal completed in {processing_time:.2f}s. "
                    f"SNR improved from {original_snr:.1f}dB to {final_snr:.1f}dB")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in background noise removal: {e}")
            raise        
    
    # async def enhance_audio_quality(
    #     self, 
    #     audio_data: np.ndarray, 
    #     sample_rate: int
    # ) -> np.ndarray:
    #     """
    #     Enhance audio quality for better processing
        
    #     Args:
    #         audio_data: Input audio signal
    #         sample_rate: Sample rate
            
    #     Returns:
    #         Enhanced audio signal
    #     """
    #     try:
    #         logger.info("Enhancing audio quality")
            
    #         enhanced_audio = audio_data.copy()
            
    #         # Calculate SNR
    #         snr = self.audio_utils.calculate_snr(enhanced_audio)
            
    #         # Apply noise reduction if SNR is low
    #         if snr < 10:  # 10 dB threshold
    #             logger.info(f"Low SNR detected ({snr:.2f} dB), applying noise reduction")
    #             enhanced_audio = self.audio_utils.reduce_noise(
    #                 enhanced_audio, sample_rate
    #             )
            
    #         # Apply pre-emphasis filter
    #         enhanced_audio = self.audio_utils.apply_preemphasis(enhanced_audio)
            
    #         # Dynamic range compression
    #         enhanced_audio = self.audio_utils.dynamic_range_compression(enhanced_audio)
            
    #         # Final normalization
    #         enhanced_audio = self.audio_utils.normalize_audio(enhanced_audio)
            
    #         logger.info("Audio quality enhancement completed")
    #         return enhanced_audio
            
    #     except Exception as e:
    #         logger.exception(f"Error in audio enhancement: {e}")
    #         # Return original audio if enhancement fails
    #         return audio_data
    
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

    # Add this method to your AudioProcessingService class in src/services/audio_processing_service.py

    async def remove_background_noise(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        method: str = 'advanced_spectral',
        strength: float = 0.8,
        preserve_speech: bool = True,
        auto_detect_method: bool = True
    ) -> Dict[str, Any]:
        """
        Remove background noise from audio with advanced techniques
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate
            method: Noise reduction method ('advanced_spectral', 'adaptive_wiener', 'multi_band', 'auto')
            strength: Noise reduction strength (0.0 to 1.0)
            preserve_speech: Whether to prioritize speech preservation
            auto_detect_method: Automatically select best method based on audio characteristics
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        try:
            logger.info(f"Starting background noise removal for audio of {len(audio_data)} samples")
            
            # Step 1: Analyze audio characteristics for method selection
            audio_analysis = await self._analyze_audio_for_noise_removal(audio_data, sample_rate)
            
            # Step 2: Auto-select method if requested
            if auto_detect_method or method == 'auto':
                selected_method = self._select_optimal_noise_method(audio_analysis)
                logger.info(f"Auto-selected noise reduction method: {selected_method}")
            else:
                selected_method = method
            
            # Step 3: Pre-processing for noise reduction
            preprocessed_audio = await self._preprocess_for_noise_removal(audio_data, sample_rate)
            
            # Step 4: Apply noise reduction
            start_time = time.time()
            
            if selected_method == 'advanced_spectral':
                denoised_audio = self.audio_utils.remove_background_noise(
                    preprocessed_audio, sample_rate, 'advanced_spectral', strength, preserve_speech
                )
            elif selected_method == 'adaptive_wiener':
                denoised_audio = self.audio_utils.remove_background_noise(
                    preprocessed_audio, sample_rate, 'adaptive_wiener', strength, preserve_speech
                )
            elif selected_method == 'multi_band':
                denoised_audio = self.audio_utils.remove_background_noise(
                    preprocessed_audio, sample_rate, 'multi_band', strength, preserve_speech
                )
            elif selected_method == 'hybrid':
                # Apply multiple methods in sequence for maximum noise reduction
                denoised_audio = await self._apply_hybrid_noise_reduction(
                    preprocessed_audio, sample_rate, strength, preserve_speech
                )
            else:
                # Fallback to existing method
                denoised_audio = self.audio_utils.reduce_noise(preprocessed_audio, sample_rate)
            
            processing_time = time.time() - start_time
            
            # Step 5: Post-processing and quality assessment
            final_audio = await self._postprocess_denoised_audio(denoised_audio, sample_rate)
            
            # Step 6: Calculate noise reduction metrics
            metrics = await self._calculate_noise_reduction_metrics(
                audio_data, final_audio, sample_rate
            )
            
            # Step 7: Prepare result
            result = {
                "denoised_audio": final_audio,
                "original_audio": audio_data,
                "sample_rate": sample_rate,
                "method_used": selected_method,
                "processing_time": processing_time,
                "noise_reduction_strength": strength,
                "preserve_speech": preserve_speech,
                "audio_analysis": audio_analysis,
                "noise_reduction_metrics": metrics,
                "quality_improvement": {
                    "snr_before": audio_analysis.get('snr_db', 0),
                    "snr_after": metrics.get('final_snr', 0),
                    "snr_improvement": metrics.get('snr_improvement', 0),
                    "noise_reduction_db": metrics.get('noise_reduction_db', 0)
                }
            }
            
            logger.info(f"Background noise removal completed in {processing_time:.2f}s. "
                    f"SNR improved from {audio_analysis.get('snr_db', 0):.1f}dB to {metrics.get('final_snr', 0):.1f}dB")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in background noise removal: {e}")
            raise

    async def _analyze_audio_for_noise_removal(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Analyze audio characteristics to determine optimal noise removal approach"""
        try:
            analysis = {}
            
            # Calculate SNR
            analysis['snr_db'] = self.audio_utils.calculate_snr(audio_data)
            
            # Analyze frequency content
            stft = librosa.stft(audio_data, n_fft=1024, hop_length=256)
            magnitude = np.abs(stft)
            
            # Frequency distribution analysis
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
            
            # Low frequency energy (0-300 Hz) - often noise
            low_freq_mask = freq_bins <= 300
            low_freq_energy = np.mean(magnitude[low_freq_mask, :])
            
            # Speech frequency energy (300-3000 Hz)
            speech_freq_mask = (freq_bins >= 300) & (freq_bins <= 3000)
            speech_freq_energy = np.mean(magnitude[speech_freq_mask, :])
            
            # High frequency energy (3000+ Hz)
            high_freq_mask = freq_bins >= 3000
            high_freq_energy = np.mean(magnitude[high_freq_mask, :])
            
            total_energy = np.mean(magnitude)
            
            analysis.update({
                'low_freq_ratio': low_freq_energy / total_energy,
                'speech_freq_ratio': speech_freq_energy / total_energy,
                'high_freq_ratio': high_freq_energy / total_energy,
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(S=magnitude)[0]),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(S=magnitude)[0]),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            })
            
            # Estimate noise type
            if analysis['low_freq_ratio'] > 0.3:
                analysis['dominant_noise_type'] = 'low_frequency'  # Hum, rumble
            elif analysis['high_freq_ratio'] > 0.4:
                analysis['dominant_noise_type'] = 'high_frequency'  # Hiss, fan noise
            else:
                analysis['dominant_noise_type'] = 'broadband'  # General background noise
            
            # Estimate stationarity of noise
            # Calculate frame-wise energy variance
            frame_energy = np.mean(magnitude, axis=0)
            energy_variance = np.var(frame_energy)
            analysis['noise_stationarity'] = 'stationary' if energy_variance < np.mean(frame_energy) * 0.1 else 'non_stationary'
            
            # Detect speech presence
            speech_segments = await self.voice_activity_detection(audio_data, sample_rate)
            analysis['speech_ratio'] = sum((seg['end'] - seg['start']) for seg in speech_segments) / (len(audio_data) / sample_rate)
            
            return analysis
            
        except Exception as e:
            logger.exception(f"Error analyzing audio for noise removal: {e}")
            return {'snr_db': 10.0, 'dominant_noise_type': 'broadband', 'speech_ratio': 0.5}

    def _select_optimal_noise_method(self, audio_analysis: Dict[str, Any]) -> str:
        """Select optimal noise reduction method based on audio characteristics"""
        try:
            snr = audio_analysis.get('snr_db', 10.0)
            noise_type = audio_analysis.get('dominant_noise_type', 'broadband')
            speech_ratio = audio_analysis.get('speech_ratio', 0.5)
            low_freq_ratio = audio_analysis.get('low_freq_ratio', 0.2)
            
            # Decision logic for method selection
            if snr < 5:
                # Very noisy audio - use aggressive multi-band approach
                return 'multi_band'
            elif snr < 10 and noise_type == 'low_frequency':
                # Low SNR with low-frequency noise - spectral subtraction works well
                return 'advanced_spectral'
            elif speech_ratio > 0.7:
                # Speech-heavy audio - use adaptive Wiener for speech preservation
                return 'adaptive_wiener'
            elif noise_type == 'broadband' and snr >= 10:
                # Moderate SNR with broadband noise - adaptive Wiener
                return 'adaptive_wiener'
            elif low_freq_ratio > 0.4:
                # Lot of low-frequency content - multi-band processing
                return 'multi_band'
            else:
                # Default to advanced spectral subtraction
                return 'advanced_spectral'
                
        except Exception as e:
            logger.exception(f"Error selecting optimal noise method: {e}")
            return 'advanced_spectral'

    async def _preprocess_for_noise_removal(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Preprocess audio before noise removal"""
        try:
            preprocessed = audio_data.copy()
            
            # Step 1: Remove DC offset
            preprocessed = preprocessed - np.mean(preprocessed)
            
            # Step 2: Apply gentle high-pass filter to remove very low frequencies
            from scipy import signal
            sos = signal.butter(2, 80, btype='high', fs=sample_rate, output='sos')
            preprocessed = signal.sosfilt(sos, preprocessed)
            
            # Step 3: Normalize to prevent clipping during processing
            max_val = np.max(np.abs(preprocessed))
            if max_val > 0:
                preprocessed = preprocessed / max_val * 0.95
            
            return preprocessed
            
        except Exception as e:
            logger.exception(f"Error in preprocessing for noise removal: {e}")
            return audio_data

    async def _apply_hybrid_noise_reduction(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        strength: float,
        preserve_speech: bool
    ) -> np.ndarray:
        """Apply multiple noise reduction methods in sequence"""
        try:
            logger.info("Applying hybrid noise reduction approach")
            
            # Step 1: Start with gentle spectral subtraction
            result = self.audio_utils.remove_background_noise(
                audio_data, sample_rate, 'advanced_spectral', strength * 0.6, preserve_speech
            )
            
            # Step 2: Apply adaptive Wiener filtering
            result = self.audio_utils.remove_background_noise(
                result, sample_rate, 'adaptive_wiener', strength * 0.8, preserve_speech
            )
            
            # Step 3: Final multi-band processing for fine-tuning
            result = self.audio_utils.remove_background_noise(
                result, sample_rate, 'multi_band', strength * 0.4, preserve_speech
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in hybrid noise reduction: {e}")
            return audio_data

    async def _postprocess_denoised_audio(
        self,
        denoised_audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Post-process denoised audio to improve quality"""
        try:
            processed = denoised_audio.copy()
            
            # Step 1: Apply gentle de-essing if needed (reduce harsh sibilants)
            processed = self._apply_gentle_deessing(processed, sample_rate)
            
            # Step 2: Restore some high-frequency content that might have been lost
            processed = self._restore_high_frequencies(processed, sample_rate)
            
            # Step 3: Final normalization
            processed = self.audio_utils.normalize_audio(processed, method='peak')
            
            # Step 4: Apply soft limiting to prevent clipping
            processed = self._apply_soft_limiting(processed)
            
            return processed
            
        except Exception as e:
            logger.exception(f"Error in post-processing denoised audio: {e}")
            return denoised_audio

    def _apply_gentle_deessing(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle de-essing to reduce harsh sibilants"""
        try:
            from scipy import signal
            
            # Target sibilant frequency range (4-8 kHz)
            nyquist = sample_rate / 2
            low_freq = 4000 / nyquist
            high_freq = min(8000 / nyquist, 0.95)
            
            # Design bandpass filter for sibilant detection
            sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
            sibilant_signal = signal.sosfilt(sos, audio)
            
            # Calculate envelope of sibilant content
            envelope = np.abs(signal.hilbert(sibilant_signal))
            
            # Apply gentle compression when sibilants are detected
            threshold = np.percentile(envelope, 80)
            compression_ratio = 2.0
            
            gain_reduction = np.ones_like(audio)
            sibilant_frames = envelope > threshold
            
            if np.any(sibilant_frames):
                excess = envelope[sibilant_frames] / threshold
                gain_reduction[sibilant_frames] = 1.0 / (1.0 + (excess - 1.0) / compression_ratio)
            
            # Smooth gain changes
            from scipy.ndimage import gaussian_filter1d
            gain_reduction = gaussian_filter1d(gain_reduction, sigma=sample_rate * 0.001)  # 1ms smoothing
            
            return audio * gain_reduction
            
        except Exception as e:
            logger.exception(f"Error in gentle de-essing: {e}")
            return audio

    def _restore_high_frequencies(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Restore high-frequency content that might have been over-attenuated"""
        try:
            # Apply gentle high-frequency boost above 4kHz
            from scipy import signal
            
            nyquist = sample_rate / 2
            freq = 4000 / nyquist
            
            if freq < 0.95:  # Only if 4kHz is below Nyquist
                # Design gentle high-frequency emphasis filter
                b, a = signal.butter(2, freq, btype='high')
                
                # Apply with low gain to avoid artifacts
                emphasis_gain = 0.1  # Very gentle boost
                emphasized = signal.filtfilt(b, a, audio)
                
                # Mix with original
                result = audio + emphasis_gain * emphasized
                
                # Ensure no clipping
                max_val = np.max(np.abs(result))
                if max_val > 1.0:
                    result = result / max_val
                
                return result
            
            return audio
            
        except Exception as e:
            logger.exception(f"Error restoring high frequencies: {e}")
            return audio

    def _apply_soft_limiting(self, audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """Apply soft limiting to prevent clipping"""
        try:
            # Soft limiting using tanh function
            # Only apply to samples above threshold
            mask = np.abs(audio) > threshold
            
            if np.any(mask):
                # Apply soft limiting
                limited = audio.copy()
                over_threshold = audio[mask]
                
                # Soft limit using tanh
                sign = np.sign(over_threshold)
                magnitude = np.abs(over_threshold)
                
                # Map [threshold, inf) to [threshold, 1.0) using tanh
                normalized_excess = (magnitude - threshold) / (1.0 - threshold)
                soft_limited_excess = np.tanh(normalized_excess) * (1.0 - threshold)
                soft_limited_magnitude = threshold + soft_limited_excess
                
                limited[mask] = sign * soft_limited_magnitude
                
                return limited
            
            return audio
            
        except Exception as e:
            logger.exception(f"Error in soft limiting: {e}")
            return audio

    async def _calculate_noise_reduction_metrics(
        self,
        original_audio: np.ndarray,
        denoised_audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, float]:
        """Calculate metrics to assess noise reduction effectiveness"""
        try:
            metrics = {}
            
            # Calculate SNR before and after
            original_snr = self.audio_utils.calculate_snr(original_audio)
            final_snr = self.audio_utils.calculate_snr(denoised_audio)
            
            metrics['original_snr'] = float(original_snr)
            metrics['final_snr'] = float(final_snr)
            metrics['snr_improvement'] = float(final_snr - original_snr)
            
            # Calculate RMS levels
            original_rms = np.sqrt(np.mean(original_audio**2))
            denoised_rms = np.sqrt(np.mean(denoised_audio**2))
            
            metrics['original_rms'] = float(original_rms)
            metrics['denoised_rms'] = float(denoised_rms)
            
            # Estimate noise reduction in dB
            if original_rms > 0:
                rms_reduction_db = 20 * np.log10(denoised_rms / original_rms)
                metrics['rms_change_db'] = float(rms_reduction_db)
            else:
                metrics['rms_change_db'] = 0.0
            
            # Calculate spectral distance metrics
            original_spectrum = np.abs(librosa.stft(original_audio))
            denoised_spectrum = np.abs(librosa.stft(denoised_audio))
            
            # Spectral distortion
            min_frames = min(original_spectrum.shape[1], denoised_spectrum.shape[1])
            orig_spec_db = 20 * np.log10(original_spectrum[:, :min_frames] + 1e-8)
            den_spec_db = 20 * np.log10(denoised_spectrum[:, :min_frames] + 1e-8)
            
            spectral_distortion = np.mean(np.abs(orig_spec_db - den_spec_db))
            metrics['spectral_distortion_db'] = float(spectral_distortion)
            
            # Estimate noise reduction effectiveness
            if metrics['snr_improvement'] > 3:
                metrics['noise_reduction_quality'] = 'excellent'
            elif metrics['snr_improvement'] > 1:
                metrics['noise_reduction_quality'] = 'good'
            elif metrics['snr_improvement'] > -1:
                metrics['noise_reduction_quality'] = 'moderate'
            else:
                metrics['noise_reduction_quality'] = 'poor'
            
            # Estimate noise reduction amount in dB
            # This is a rough estimate based on SNR improvement and RMS change
            estimated_noise_reduction = max(0, metrics['snr_improvement'])
            metrics['noise_reduction_db'] = float(estimated_noise_reduction)
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating noise reduction metrics: {e}")
            return {
                'original_snr': 10.0,
                'final_snr': 10.0,
                'snr_improvement': 0.0,
                'noise_reduction_quality': 'unknown'
            }
    # Replace your existing enhance_audio_quality method in src/services/audio_processing_service.py

# Update these methods in your src/services/audio_processing_service.py

# Update the enhance_audio_quality method:
# Add these complete method implementations to your src/services/audio_processing_service.py file
# Place them after the voice_activity_detection method

# Update the process_audio_with_advanced_noise_removal method:
async def process_audio_with_advanced_noise_removal(
    self, 
    job_id: str, 
    audio_file_path: str,
    noise_removal_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced version of process_audio with advanced noise removal options
    
    Args:
        job_id: Job identifier
        audio_file_path: Path to audio file
        noise_removal_config: Configuration for noise removal
        
    Returns:
        Processing results with enhanced noise removal
    """
    try:
        logger.info(f"Starting enhanced audio processing with noise removal for job {job_id}")
        
        # Get default configuration from app_config
        default_config = app_config.noise_removal_config
        
        # Merge with provided config
        if noise_removal_config:
            config = {**default_config, **noise_removal_config}
        else:
            config = default_config
        
        # Load and normalize audio
        audio_data, sample_rate = await self.load_and_normalize_audio(audio_file_path)
        
        # Voice Activity Detection
        speech_segments = await self.voice_activity_detection(audio_data, sample_rate)
        
        # Enhanced audio quality with advanced noise removal
        enhanced_audio = await self.enhance_audio_quality(
            audio_data, 
            sample_rate,
            use_advanced_noise_removal=config['enabled'],
            noise_removal_method=config['method'],
            noise_removal_strength=config['strength']
        )
        
        # Feature extraction
        features = await self.extract_features(enhanced_audio, sample_rate)
        
        # Save processed audio
        processed_path = await self.save_processed_audio(job_id, enhanced_audio, sample_rate)
        
        # Calculate comprehensive quality metrics
        quality_metrics = await self.calculate_quality_metrics(
            audio_data, enhanced_audio, sample_rate
        )
        
        # Add noise removal specific metrics if available
        if hasattr(self, 'last_processing_steps'):
            quality_metrics['processing_steps'] = self.last_processing_steps
        
        result = {
            "job_id": job_id,
            "original_path": audio_file_path,
            "processed_path": processed_path,
            "duration": float(len(enhanced_audio) / sample_rate),
            "sample_rate": int(sample_rate),
            "speech_segments": speech_segments,
            "features": features,
            "quality_metrics": quality_metrics,
            "noise_removal_config": config,
            "enhancement_applied": True
        }
        
        logger.info(f"Enhanced audio processing completed for job {job_id}")
        return result
        
    except Exception as e:
        logger.exception(f"Error in enhanced audio processing for job {job_id}: {e}")
        raise

# Update any method selection logic to use the configuration:
def _select_optimal_noise_method(self, audio_analysis: Dict[str, Any]) -> str:
    """Select optimal noise reduction method based on audio characteristics"""
    try:
        snr = audio_analysis.get('snr_db', 10.0)
        noise_type = audio_analysis.get('dominant_noise_type', 'broadband')
        speech_ratio = audio_analysis.get('speech_ratio', 0.5)
        low_freq_ratio = audio_analysis.get('low_freq_ratio', 0.2)
        
        # Use configuration values for thresholds
        very_noisy_threshold = 5.0  # Could make this configurable too
        speech_heavy_threshold = app_config.noise_removal_preserve_speech
        
        # Decision logic for method selection
        if snr < very_noisy_threshold:
            # Very noisy audio - use aggressive multi-band approach
            return 'multi_band'
        elif snr < app_config.noise_removal_snr_threshold_basic and noise_type == 'low_frequency':
            # Low SNR with low-frequency noise - spectral subtraction works well
            return 'advanced_spectral'
        elif speech_ratio > 0.7 and speech_heavy_threshold:
            # Speech-heavy audio - use adaptive Wiener for speech preservation
            return 'adaptive_wiener'
        elif noise_type == 'broadband' and snr >= app_config.noise_removal_snr_threshold_basic:
            # Moderate SNR with broadband noise - adaptive Wiener
            return 'adaptive_wiener'
        elif low_freq_ratio > 0.4:
            # Lot of low-frequency content - multi-band processing
            return 'multi_band'
        else:
            # Default to advanced spectral subtraction
            return 'advanced_spectral'
            
    except Exception as e:
        logger.exception(f"Error selecting optimal noise method: {e}")
        return 'advanced_spectral'
    
def _contains_speech(self, audio_data: np.ndarray, sample_rate: int) -> bool:
    """Check if audio contains speech content"""
    try:
        # Simple speech detection based on spectral characteristics
        stft = librosa.stft(audio_data, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        
        # Check energy in speech frequency range (300-3000 Hz)
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        speech_mask = (freq_bins >= 300) & (freq_bins <= 3000)
        
        speech_energy = np.mean(magnitude[speech_mask, :])
        total_energy = np.mean(magnitude)
        
        speech_ratio = speech_energy / (total_energy + 1e-8)
        
        # If speech frequencies contain more than 40% of energy, likely contains speech
        return speech_ratio > 0.4
        
    except Exception as e:
        logger.exception(f"Error detecting speech content: {e}")
        return True  # Default to assuming speech is present

async def _optimize_frequency_response(
    self, 
    audio_data: np.ndarray, 
    sample_rate: int
) -> np.ndarray:
    """Optimize frequency response for better clarity"""
    try:
        from scipy import signal
        
        # Analyze current frequency response
        f, psd = signal.welch(audio_data, fs=sample_rate, nperseg=1024)
        
        # Design subtle EQ adjustments
        optimized_audio = audio_data.copy()
        
        # Gentle high-frequency enhancement for clarity (above 2kHz)
        nyquist = sample_rate / 2
        if 2000 < nyquist:
            high_freq = 2000 / nyquist
            
            # Design gentle high-shelf filter
            sos = signal.butter(2, high_freq, btype='high', output='sos')
            
            # Apply with minimal gain
            high_enhanced = signal.sosfilt(sos, audio_data)
            optimized_audio = audio_data + 0.1 * high_enhanced  # Very subtle enhancement
        
        # Gentle low-frequency cleanup (below 100Hz)
        if 100 < nyquist:
            low_freq = 100 / nyquist
            
            # High-pass filter to remove rumble
            sos = signal.butter(1, low_freq, btype='high', output='sos')
            optimized_audio = signal.sosfilt(sos, optimized_audio)
        
        # Ensure no clipping after EQ
        max_val = np.max(np.abs(optimized_audio))
        if max_val > 1.0:
            optimized_audio = optimized_audio / max_val
        
        return optimized_audio
        
    except Exception as e:
        logger.exception(f"Error optimizing frequency response: {e}")
        return audio_data

    # Add this method to also integrate with your main processing pipeline
async def process_audio_with_advanced_noise_removal(
        self, 
        job_id: str, 
        audio_file_path: str,
        noise_removal_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced version of process_audio with advanced noise removal options
        
        Args:
            job_id: Job identifier
            audio_file_path: Path to audio file
            noise_removal_config: Configuration for noise removal
            
        Returns:
            Processing results with enhanced noise removal
        """
        try:
            logger.info(f"Starting enhanced audio processing with noise removal for job {job_id}")
            
            # Default noise removal configuration
            default_config = {
                'enabled': True,
                'method': 'auto',
                'strength': 0.8,
                'preserve_speech': True,
                'auto_detect_method': True
            }
            
            # Merge with provided config
            if noise_removal_config:
                config = {**default_config, **noise_removal_config}
            else:
                config = default_config
            
            # Load and normalize audio
            audio_data, sample_rate = await self.load_and_normalize_audio(audio_file_path)
            
            # Voice Activity Detection
            speech_segments = await self.voice_activity_detection(audio_data, sample_rate)
            
            # Enhanced audio quality with advanced noise removal
            enhanced_audio = await self.enhance_audio_quality(
                audio_data, 
                sample_rate,
                use_advanced_noise_removal=config['enabled'],
                noise_removal_method=config['method'],
                noise_removal_strength=config['strength']
            )
            
            # Feature extraction
            features = await self.extract_features(enhanced_audio, sample_rate)
            
            # Save processed audio
            processed_path = await self.save_processed_audio(job_id, enhanced_audio, sample_rate)
            
            # Calculate comprehensive quality metrics
            quality_metrics = await self.calculate_quality_metrics(
                audio_data, enhanced_audio, sample_rate
            )
            
            # Add noise removal specific metrics if available
            if hasattr(self, 'last_processing_steps'):
                quality_metrics['processing_steps'] = self.last_processing_steps
            
            result = {
                "job_id": job_id,
                "original_path": audio_file_path,
                "processed_path": processed_path,
                "duration": float(len(enhanced_audio) / sample_rate),
                "sample_rate": int(sample_rate),
                "speech_segments": speech_segments,
                "features": features,
                "quality_metrics": quality_metrics,
                "noise_removal_config": config,
                "enhancement_applied": True
            }
            
            logger.info(f"Enhanced audio processing completed for job {job_id}")
            return result
            
        except Exception as e:
            logger.exception(f"Error in enhanced audio processing for job {job_id}: {e}")
            raise    
# Add these methods to your AudioProcessingService class in src/services/audio_processing_service.py



def save_denoised_audio(
    self, 
    job_id: str, 
    original_audio: np.ndarray,
    denoised_audio: np.ndarray, 
    sample_rate: int,
    output_dir: str = "downloads"
) -> str:
    """
    Save denoised audio file with comparison to original
    
    Args:
        job_id: Job identifier
        original_audio: Original audio signal
        denoised_audio: Processed/denoised audio signal
        sample_rate: Sample rate
        output_dir: Output directory for saved files
        
    Returns:
        Path to saved denoised audio file
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_short = job_id[:8] if len(job_id) > 8 else job_id
        
        # Save denoised audio
        denoised_filename = f"denoised_{job_short}_{timestamp}.wav"
        denoised_path = output_path / denoised_filename
        sf.write(denoised_path, denoised_audio, sample_rate)
        
        # Calculate and log improvement metrics
        original_snr = self.audio_utils.calculate_snr(original_audio)
        denoised_snr = self.audio_utils.calculate_snr(denoised_audio)
        improvement = denoised_snr - original_snr
        
        # Save metadata
        metadata = {
            "job_id": job_id,
            "timestamp": timestamp,
            "original_snr_db": float(original_snr),
            "denoised_snr_db": float(denoised_snr),
            "snr_improvement_db": float(improvement),
            "sample_rate": int(sample_rate),
            "duration_seconds": float(len(denoised_audio) / sample_rate),
            "original_file_size": len(original_audio),
            "denoised_file_size": len(denoised_audio),
            "denoised_file_path": str(denoised_path.absolute())
        }
        
        # Save metadata file
        metadata_path = output_path / f"metadata_{job_short}_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Denoised audio saved: {denoised_path}")
        logger.info(f"SNR improvement: {improvement:.2f} dB ({original_snr:.2f} -> {denoised_snr:.2f})")
        
        return str(denoised_path.absolute())
        
    except Exception as e:
        logger.exception(f"Error saving denoised audio: {e}")
        raise

def save_comparison_audio(
    self, 
    job_id: str, 
    original_audio: np.ndarray,
    denoised_audio: np.ndarray, 
    sample_rate: int,
    output_dir: str = "downloads"
) -> Dict[str, str]:
    """
    Save both original and denoised audio for comparison
    
    Args:
        job_id: Job identifier
        original_audio: Original audio signal
        denoised_audio: Processed/denoised audio signal
        sample_rate: Sample rate
        output_dir: Output directory for saved files
        
    Returns:
        Dictionary with paths to saved files
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_short = job_id[:8] if len(job_id) > 8 else job_id
        
        # Save original audio
        original_filename = f"original_{job_short}_{timestamp}.wav"
        original_path = output_path / original_filename
        sf.write(original_path, original_audio, sample_rate)
        
        # Save denoised audio
        denoised_filename = f"denoised_{job_short}_{timestamp}.wav"
        denoised_path = output_path / denoised_filename
        sf.write(denoised_path, denoised_audio, sample_rate)
        
        # Create a combined comparison file (original first, then denoised)
        silence_gap = np.zeros(int(sample_rate * 0.5))  # 0.5 second gap
        combined_audio = np.concatenate([original_audio, silence_gap, denoised_audio])
        combined_filename = f"comparison_{job_short}_{timestamp}.wav"
        combined_path = output_path / combined_filename
        sf.write(combined_path, combined_audio, sample_rate)
        
        # Calculate metrics
        original_snr = self.audio_utils.calculate_snr(original_audio)
        denoised_snr = self.audio_utils.calculate_snr(denoised_audio)
        improvement = denoised_snr - original_snr
        
        result = {
            "original_path": str(original_path.absolute()),
            "denoised_path": str(denoised_path.absolute()),
            "comparison_path": str(combined_path.absolute()),
            "snr_improvement": improvement,
            "original_snr": original_snr,
            "denoised_snr": denoised_snr
        }
        
        logger.info(f"Comparison files saved:")
        logger.info(f"  Original: {original_path}")
        logger.info(f"  Denoised: {denoised_path}")
        logger.info(f"  Comparison: {combined_path}")
        logger.info(f"  SNR improvement: {improvement:.2f} dB")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error saving comparison audio: {e}")
        raise

def download_processed_audio_from_job(self, job_id: str, output_dir: str = "downloads") -> str:
    """
    Download the processed audio file from a completed job
    
    Args:
        job_id: Job identifier
        output_dir: Output directory for downloaded file
        
    Returns:
        Path to downloaded file
    """
    try:
        # Find the processed audio file
        job_audio_dir = Path("data") / "audio" / job_id
        processed_file = job_audio_dir / f"{job_id}_processed.wav"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed audio file not found for job {job_id}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy to downloads
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_short = job_id[:8]
        output_filename = f"downloaded_{job_short}_{timestamp}.wav"
        output_file = output_path / output_filename
        
        shutil.copy2(processed_file, output_file)
        
        # Get file info
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"Downloaded processed audio: {output_file}")
        logger.info(f"File size: {file_size:.1f} MB")
        
        return str(output_file.absolute())
        
    except Exception as e:
        logger.exception(f"Error downloading processed audio for job {job_id}: {e}")
        raise

def download_latest_processed_audio(self, output_dir: str = "downloads") -> str:
    """
    Download the most recently processed audio file
    
    Args:
        output_dir: Output directory for downloaded file
        
    Returns:
        Path to downloaded file
    """
    try:
        audio_base_dir = Path("data") / "audio"
        
        if not audio_base_dir.exists():
            raise FileNotFoundError("Audio directory not found")
        
        # Find all processed files
        processed_files = []
        for job_dir in audio_base_dir.iterdir():
            if job_dir.is_dir():
                processed_file = job_dir / f"{job_dir.name}_processed.wav"
                if processed_file.exists():
                    processed_files.append(processed_file)
        
        if not processed_files:
            raise FileNotFoundError("No processed audio files found")
        
        # Get the most recent file
        latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
        job_id = latest_file.parent.name
        
        logger.info(f"Found latest processed file from job: {job_id}")
        
        return self.download_processed_audio_from_job(job_id, output_dir)
        
    except Exception as e:
        logger.exception(f"Error downloading latest processed audio: {e}")
        raise

def list_processed_audio_files(self) -> List[Dict[str, Any]]:
    """
    List all available processed audio files
    
    Returns:
        List of processed audio file information
    """
    try:
        audio_base_dir = Path("data") / "audio"
        
        if not audio_base_dir.exists():
            return []
        
        processed_files = []
        
        for job_dir in audio_base_dir.iterdir():
            if job_dir.is_dir():
                job_id = job_dir.name
                processed_file = job_dir / f"{job_id}_processed.wav"
                
                if processed_file.exists():
                    file_stat = processed_file.stat()
                    
                    # Get original filename if available
                    original_files = list(job_dir.glob("original_*"))
                    original_name = original_files[0].name if original_files else "Unknown"
                    
                    processed_files.append({
                        "job_id": job_id,
                        "processed_file_path": str(processed_file.absolute()),
                        "original_filename": original_name,
                        "size_mb": file_stat.st_size / (1024 * 1024),
                        "modified_time": datetime.fromtimestamp(file_stat.st_mtime),
                        "file_exists": True
                    })
        
        # Sort by modification time (newest first)
        processed_files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        logger.info(f"Found {len(processed_files)} processed audio files")
        
        return processed_files
        
    except Exception as e:
        logger.exception(f"Error listing processed audio files: {e}")
        return []

def test_noise_removal_on_file(
    self, 
    input_file_path: str, 
    output_dir: str = "downloads"
) -> Dict[str, Any]:
    """
    Test noise removal on a specific audio file and save results
    
    Args:
        input_file_path: Path to input audio file
        output_dir: Output directory for results
        
    Returns:
        Test results with file paths and metrics
    """
    try:
        logger.info(f"Testing noise removal on: {input_file_path}")
        
        # Load audio file
        audio_data, sample_rate = librosa.load(input_file_path, sr=None)
        
        # Apply noise removal
        enhanced_audio = self.enhance_audio_quality(audio_data, sample_rate)
        
        # Generate test job ID
        test_job_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save comparison files
        comparison_result = self.save_comparison_audio(
            test_job_id, audio_data, enhanced_audio, sample_rate, output_dir
        )
        
        # Calculate detailed metrics
        original_snr = self.audio_utils.calculate_snr(audio_data)
        enhanced_snr = self.audio_utils.calculate_snr(enhanced_audio)
        
        test_result = {
            "test_job_id": test_job_id,
            "input_file": input_file_path,
            "output_directory": output_dir,
            "sample_rate": sample_rate,
            "duration_seconds": len(audio_data) / sample_rate,
            "original_snr_db": float(original_snr),
            "enhanced_snr_db": float(enhanced_snr),
            "snr_improvement_db": float(enhanced_snr - original_snr),
            "files": comparison_result,
            "test_timestamp": datetime.now().isoformat()
        }
        
        # Save test results
        results_file = Path(output_dir) / f"test_results_{test_job_id}.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(test_result, f, indent=2)
        
        logger.info("Noise removal test completed")
        logger.info(f"SNR improvement: {test_result['snr_improvement_db']:.2f} dB")
        logger.info(f"Results saved to: {results_file}")
        
        return test_result
        
    except Exception as e:
        logger.exception(f"Error testing noise removal: {e}")
        raise
# Add these methods to your AudioProcessingService class (at the end of the class)

def list_processed_audio_files(self) -> List[Dict[str, Any]]:
    """
    List all available processed audio files
    
    Returns:
        List of processed audio file information
    """
    try:
        audio_base_dir = Path("data") / "audio"
        
        if not audio_base_dir.exists():
            logger.warning("Audio directory not found")
            return []
        
        processed_files = []
        
        for job_dir in audio_base_dir.iterdir():
            if job_dir.is_dir():
                job_id = job_dir.name
                processed_file = job_dir / f"{job_id}_processed.wav"
                
                if processed_file.exists():
                    file_stat = processed_file.stat()
                    
                    # Get original filename if available
                    original_files = list(job_dir.glob("original_*"))
                    original_name = original_files[0].name if original_files else "Unknown"
                    
                    processed_files.append({
                        "job_id": job_id,
                        "processed_file_path": str(processed_file.absolute()),
                        "original_filename": original_name,
                        "size_mb": file_stat.st_size / (1024 * 1024),
                        "modified_time": datetime.fromtimestamp(file_stat.st_mtime),
                        "file_exists": True
                    })
        
        # Sort by modification time (newest first)
        processed_files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        logger.info(f"Found {len(processed_files)} processed audio files")
        
        return processed_files
        
    except Exception as e:
        logger.exception(f"Error listing processed audio files: {e}")
        return []

def download_latest_processed_audio(self, output_dir: str = "downloads") -> str:
    """
    Download the most recently processed audio file
    
    Args:
        output_dir: Output directory for downloaded file
        
    Returns:
        Path to downloaded file
    """
    try:
        audio_base_dir = Path("data") / "audio"
        
        if not audio_base_dir.exists():
            raise FileNotFoundError("Audio directory not found")
        
        # Find all processed files
        processed_files = []
        for job_dir in audio_base_dir.iterdir():
            if job_dir.is_dir():
                processed_file = job_dir / f"{job_dir.name}_processed.wav"
                if processed_file.exists():
                    processed_files.append(processed_file)
        
        if not processed_files:
            raise FileNotFoundError("No processed audio files found")
        
        # Get the most recent file
        latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
        job_id = latest_file.parent.name
        
        logger.info(f"Found latest processed file from job: {job_id}")
        
        return self.download_processed_audio_from_job(job_id, output_dir)
        
    except Exception as e:
        logger.exception(f"Error downloading latest processed audio: {e}")
        raise

def download_processed_audio_from_job(self, job_id: str, output_dir: str = "downloads") -> str:
    """
    Download the processed audio file from a completed job
    
    Args:
        job_id: Job identifier
        output_dir: Output directory for downloaded file
        
    Returns:
        Path to downloaded file
    """
    try:
        # Find the processed audio file
        job_audio_dir = Path("data") / "audio" / job_id
        processed_file = job_audio_dir / f"{job_id}_processed.wav"
        
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed audio file not found for job {job_id}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy to downloads
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_short = job_id[:8]
        output_filename = f"downloaded_{job_short}_{timestamp}.wav"
        output_file = output_path / output_filename
        
        shutil.copy2(processed_file, output_file)
        
        # Get file info
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"Downloaded processed audio: {output_file}")
        logger.info(f"File size: {file_size:.1f} MB")
        
        return str(output_file.absolute())
        
    except Exception as e:
        logger.exception(f"Error downloading processed audio for job {job_id}: {e}")
        raise    