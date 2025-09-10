"""
Audio Utilities
Core audio processing functions and utilities
"""
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import scipy.signal
from scipy.io import wavfile

from src.config.app_config import app_config
from src.api.schemas.process_schemas import AudioSpecs

logger = logging.getLogger(__name__)


class AudioUtils:
    """Utility class for audio processing operations"""
    
    def __init__(self):
        self.target_sr = app_config.target_sample_rate
        self.target_channels = app_config.target_channels
    
    async def get_audio_specs(self, file_path: Path) -> AudioSpecs:
        """
        Extract audio file specifications
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio specifications
        """
        try:
            # Use librosa to get basic info
            info = sf.info(str(file_path))
            
            return AudioSpecs(
                sample_rate=info.samplerate,
                channels=info.channels,
                duration=info.frames / info.samplerate,
                format=info.format.lower(),
                bit_depth=self._get_bit_depth(info.subtype),
                file_size=file_path.stat().st_size
            )
            
        except Exception as e:
            logger.exception(f"Error getting audio specs for {file_path}: {e}")
            raise
    
    def _get_bit_depth(self, subtype: str) -> int:
        """Extract bit depth from audio subtype"""
        bit_depth_mapping = {
            'PCM_16': 16,
            'PCM_24': 24,
            'PCM_32': 32,
            'FLOAT': 32,
            'DOUBLE': 64
        }
        return bit_depth_mapping.get(subtype, 16)
    
    def load_audio(
        self, 
        file_path: str, 
        target_sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with preprocessing
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (uses config default if None)
            mono: Convert to mono if True
            
        Returns:
            Audio data and sample rate
        """
        try:
            target_sr = target_sr or self.target_sr
            
            # Load audio
            audio_data, sample_rate = librosa.load(
                file_path,
                sr=target_sr,
                mono=mono
            )
            
            logger.debug(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.exception(f"Error loading audio file {file_path}: {e}")
            raise
    
    def normalize_audio(self, audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """
        Normalize audio amplitude
        
        Args:
            audio: Input audio signal
            method: Normalization method ('peak', 'rms', 'lufs')
            
        Returns:
            Normalized audio signal
        """
        try:
            if method == 'peak':
                # Peak normalization
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                    
            elif method == 'rms':
                # RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_rms = 0.2  # Target RMS level
                    audio = audio * (target_rms / rms)
                    
            elif method == 'lufs':
                # LUFS-based normalization (simplified)
                # In practice, you'd use pyloudnorm or similar
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_lufs = -23  # Standard broadcast level
                    current_lufs = 20 * np.log10(rms + 1e-8)
                    gain_db = target_lufs - current_lufs
                    gain_linear = 10**(gain_db / 20)
                    audio = audio * gain_linear
            
            # Ensure values are within [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            
            return audio
            
        except Exception as e:
            logger.exception(f"Error normalizing audio: {e}")
            return audio
    
    def trim_silence(
        self, 
        audio: np.ndarray, 
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Remove silence from beginning and end of audio
        
        Args:
            audio: Input audio signal
            top_db: Threshold in dB below peak for silence
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Trimmed audio signal
        """
        try:
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            logger.debug(f"Trimmed audio from {len(audio)} to {len(trimmed)} samples")
            return trimmed
            
        except Exception as e:
            logger.warning(f"Error trimming silence: {e}")
            return audio
    
    def calculate_snr(self, audio: np.ndarray, frame_length: int = 2048) -> float:
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
            # Find the loudest frame as signal
            frames = self._frame_audio(audio, frame_length, frame_length // 2)
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
            logger.warning(f"Error calculating SNR: {e}")
            return 10.0  # Default reasonable SNR
    
    def reduce_noise(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        method: str = 'spectral_subtraction'
    ) -> np.ndarray:
        """
        Apply noise reduction to audio
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            method: Noise reduction method
            
        Returns:
            Noise-reduced audio signal
        """
        try:
            if method == 'spectral_subtraction':
                return self._spectral_subtraction_nr(audio, sample_rate)
            elif method == 'wiener_filter':
                return self._wiener_filter_nr(audio)
            else:
                logger.warning(f"Unknown noise reduction method: {method}")
                return audio
                
        except Exception as e:
            logger.exception(f"Error in noise reduction: {e}")
            return audio
    
    def _spectral_subtraction_nr(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """Spectral subtraction noise reduction"""
        try:
            # STFT
            hop_length = 256
            n_fft = 1024
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames (assuming initial silence)
            noise_frames = min(10, magnitude.shape[1] // 4)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            subtracted_magnitude = magnitude - alpha * noise_spectrum
            
            # Ensure magnitude doesn't go negative
            subtracted_magnitude = np.maximum(
                subtracted_magnitude, 
                0.1 * magnitude
            )
            
            # Reconstruct signal
            enhanced_stft = subtracted_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in spectral subtraction: {e}")
            return audio
    
    def _wiener_filter_nr(self, audio: np.ndarray) -> np.ndarray:
        """Wiener filter noise reduction (simplified)"""
        try:
            # Simple Wiener filter implementation
            # Estimate signal power and noise power
            frame_length = 1024
            frames = self._frame_audio(audio, frame_length, frame_length // 2)
            
            # Signal power estimation (top 50% of frames)
            frame_powers = np.mean(frames**2, axis=1)
            signal_power = np.percentile(frame_powers, 75)
            
            # Noise power estimation (bottom 25% of frames)
            noise_power = np.percentile(frame_powers, 25)
            
            # Wiener gain
            wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
            
            # Apply gain
            enhanced_audio = audio * wiener_gain
            
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in Wiener filtering: {e}")
            return audio
    
    def apply_preemphasis(
        self, 
        audio: np.ndarray, 
        coeff: float = 0.97
    ) -> np.ndarray:
        """
        Apply pre-emphasis filter
        
        Args:
            audio: Input audio signal
            coeff: Pre-emphasis coefficient
            
        Returns:
            Pre-emphasized audio signal
        """
        try:
            # Pre-emphasis: y[n] = x[n] - coeff * x[n-1]
            emphasized = np.append(audio[0], audio[1:] - coeff * audio[:-1])
            return emphasized
            
        except Exception as e:
            logger.warning(f"Error applying pre-emphasis: {e}")
            return audio
    
    def dynamic_range_compression(
        self, 
        audio: np.ndarray,
        threshold: float = -20,  # dB
        ratio: float = 4.0,
        attack_time: float = 0.01,  # seconds
        release_time: float = 0.1   # seconds
    ) -> np.ndarray:
        """
        Apply dynamic range compression
        
        Args:
            audio: Input audio signal
            threshold: Compression threshold in dB
            ratio: Compression ratio
            attack_time: Attack time in seconds
            release_time: Release time in seconds
            
        Returns:
            Compressed audio signal
        """
        try:
            # Simple compressor implementation
            # Convert to dB
            audio_db = 20 * np.log10(np.abs(audio) + 1e-8)
            
            # Find samples above threshold
            above_threshold = audio_db > threshold
            
            # Calculate gain reduction
            excess_db = audio_db - threshold
            gain_reduction_db = np.zeros_like(audio_db)
            gain_reduction_db[above_threshold] = excess_db[above_threshold] * (1 - 1/ratio)
            
            # Apply attack/release smoothing (simplified)
            # In practice, you'd use proper envelope following
            smoothed_gain = scipy.signal.lfilter([1-0.1], [1, -0.1], gain_reduction_db)
            
            # Convert back to linear and apply
            gain_reduction_linear = 10**(-smoothed_gain / 20)
            compressed_audio = audio * gain_reduction_linear
            
            return compressed_audio
            
        except Exception as e:
            logger.warning(f"Error in dynamic range compression: {e}")
            return audio
    
    def _frame_audio(
        self, 
        audio: np.ndarray, 
        frame_length: int, 
        hop_length: int
    ) -> np.ndarray:
        """Create overlapping frames from audio"""
        try:
            num_frames = (len(audio) - frame_length) // hop_length + 1
            frames = np.zeros((num_frames, frame_length))
            
            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frames[i] = audio[start:end]
            
            return frames
            
        except Exception as e:
            logger.exception(f"Error framing audio: {e}")
            return np.array([audio])
    
    def resample_audio(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Input audio signal
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio signal
        """
        try:
            if orig_sr == target_sr:
                return audio
            
            resampled = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
            
            logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz")
            return resampled
            
        except Exception as e:
            logger.exception(f"Error resampling audio: {e}")
            return audio
    
    def save_audio(
        self, 
        audio: np.ndarray, 
        file_path: str, 
        sample_rate: int,
        format: str = 'wav'
    ):
        """
        Save audio to file
        
        Args:
            audio: Audio signal to save
            file_path: Output file path
            sample_rate: Sample rate
            format: Audio format
        """
        try:
            # Ensure audio is in valid range
            audio = np.clip(audio, -1.0, 1.0)
            
            if format.lower() == 'wav':
                sf.write(file_path, audio, sample_rate)
            else:
                # Use librosa for other formats
                sf.write(file_path, audio, sample_rate, format=format.upper())
            
            logger.debug(f"Saved audio to {file_path}")
            
        except Exception as e:
            logger.exception(f"Error saving audio to {file_path}: {e}")
            raise
    
    def convert_format(
        self, 
        input_path: str, 
        output_path: str, 
        target_format: str = 'wav',
        target_sr: Optional[int] = None
    ):
        """
        Convert audio file format
        
        Args:
            input_path: Input file path
            output_path: Output file path
            target_format: Target format
            target_sr: Target sample rate (optional)
        """
        try:
            # Load audio
            audio, sr = self.load_audio(input_path, target_sr)
            
            # Save in target format
            self.save_audio(audio, output_path, sr, target_format)
            
            logger.info(f"Converted {input_path} to {output_path} ({target_format})")
            
        except Exception as e:
            logger.exception(f"Error converting audio format: {e}")
            raise
    
    def extract_segment(
        self, 
        audio: np.ndarray, 
        start_time: float, 
        end_time: float, 
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract audio segment by time
        
        Args:
            audio: Full audio signal
            start_time: Start time in seconds
            end_time: End time in seconds
            sample_rate: Sample rate
            
        Returns:
            Extracted audio segment
        """
        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if start_sample >= end_sample:
                logger.warning(f"Invalid segment times: {start_time}-{end_time}s")
                return np.array([])
            
            segment = audio[start_sample:end_sample]
            
            return segment
            
        except Exception as e:
            logger.exception(f"Error extracting segment: {e}")
            return np.array([])
    
    def calculate_audio_features(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Calculate various audio features for analysis
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        try:
            features = {}
            
            # Basic statistics
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
            features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))
            features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)))
            
            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features['tempo'] = float(tempo)
            
            # Dynamic range
            features['dynamic_range'] = float(np.max(audio) - np.min(audio))
            
            # SNR
            features['snr_db'] = self.calculate_snr(audio)
            
            return features
            
        except Exception as e:
            logger.exception(f"Error calculating audio features: {e}")
            return {}