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
    # Add these methods to your src/utils/audio_utils.py file
# Add these methods to your src/utils/audio_utils.py file

    def remove_background_noise(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        method: str = 'advanced_spectral',
        noise_reduction_strength: float = 0.8,
        preserve_speech: bool = True
    ) -> np.ndarray:
        """
        Advanced background noise removal with multiple techniques
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            method: Noise reduction method ('advanced_spectral', 'adaptive_wiener', 'rnnoise', 'multi_band')
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
            preserve_speech: Whether to prioritize speech preservation
            
        Returns:
            Noise-reduced audio signal
        """
        try:
            logger.info(f"Applying background noise removal with method: {method}")
            
            if method == 'advanced_spectral':
                return self._advanced_spectral_subtraction(audio, sample_rate, noise_reduction_strength)
            elif method == 'adaptive_wiener':
                return self._adaptive_wiener_filter(audio, sample_rate, preserve_speech)
            elif method == 'multi_band':
                return self._multi_band_noise_reduction(audio, sample_rate, noise_reduction_strength)
            elif method == 'rnnoise':
                return self._rnn_noise_reduction(audio, sample_rate)
            else:
                logger.warning(f"Unknown noise reduction method: {method}, using default")
                return self.reduce_noise(audio, sample_rate)
                
        except Exception as e:
            logger.exception(f"Error in background noise removal: {e}")
            return audio

    def _advanced_spectral_subtraction(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        strength: float = 0.8
    ) -> np.ndarray:
        """Advanced spectral subtraction with improved speech preservation"""
        try:
            # Parameters
            hop_length = 256
            n_fft = 1024
            
            # STFT
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Improved noise estimation using multiple methods
            noise_estimate = self._estimate_noise_spectrum(magnitude, method='multi_frame')
            
            # Adaptive spectral subtraction with voice activity detection
            vad_mask = self._simple_vad_mask(magnitude, noise_estimate)
            
            # Apply spectral subtraction with adaptive parameters
            alpha = 2.0 * strength  # Over-subtraction factor
            beta = 0.01  # Spectral floor factor
            
            # Calculate subtraction with VAD consideration
            subtracted_magnitude = np.zeros_like(magnitude)
            for freq_bin in range(magnitude.shape[0]):
                for frame in range(magnitude.shape[1]):
                    if vad_mask[freq_bin, frame]:  # Speech present
                        # Less aggressive subtraction for speech frames
                        subtraction_factor = alpha * 0.5
                    else:  # Noise only
                        # More aggressive subtraction for noise frames
                        subtraction_factor = alpha
                    
                    noise_level = noise_estimate[freq_bin, 0] if noise_estimate.shape[1] == 1 else noise_estimate[freq_bin, frame]
                    subtracted_magnitude[freq_bin, frame] = magnitude[freq_bin, frame] - subtraction_factor * noise_level
                    
                    # Apply spectral floor
                    min_magnitude = beta * magnitude[freq_bin, frame]
                    subtracted_magnitude[freq_bin, frame] = max(subtracted_magnitude[freq_bin, frame], min_magnitude)
            
            # Reconstruct signal
            enhanced_stft = subtracted_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            # Post-processing: smooth transitions
            enhanced_audio = self._smooth_audio_transitions(enhanced_audio, sample_rate)
            
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in advanced spectral subtraction: {e}")
            return audio

    def _adaptive_wiener_filter(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        preserve_speech: bool = True
    ) -> np.ndarray:
        """Adaptive Wiener filter with speech detection"""
        try:
            # STFT parameters
            hop_length = 256
            n_fft = 1024
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            power_spectrum = magnitude ** 2
            
            # Estimate noise power spectrum
            noise_power = self._estimate_noise_power_spectrum(power_spectrum)
            
            # Calculate Wiener gain for each time-frequency bin
            wiener_gain = np.zeros_like(power_spectrum)
            
            for freq_bin in range(power_spectrum.shape[0]):
                for frame in range(power_spectrum.shape[1]):
                    signal_power = power_spectrum[freq_bin, frame]
                    noise_power_estimate = noise_power[freq_bin, 0] if noise_power.shape[1] == 1 else noise_power[freq_bin, frame]
                    
                    # Adaptive gain calculation
                    if preserve_speech:
                        # Use speech-aware gain calculation
                        speech_probability = self._estimate_speech_probability(
                            signal_power, noise_power_estimate, freq_bin, sample_rate
                        )
                        min_gain = 0.1 + 0.4 * speech_probability  # Higher floor for speech
                    else:
                        min_gain = 0.1
                    
                    gain = signal_power / (signal_power + noise_power_estimate + 1e-8)
                    wiener_gain[freq_bin, frame] = max(gain, min_gain)
            
            # Apply Wiener filter
            filtered_magnitude = magnitude * wiener_gain
            enhanced_stft = filtered_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in adaptive Wiener filter: {e}")
            return audio

    def _multi_band_noise_reduction(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        strength: float = 0.8
    ) -> np.ndarray:
        """Multi-band noise reduction for better preservation of speech characteristics"""
        try:
            # Define frequency bands (Hz)
            bands = [
                (0, 300),      # Low frequencies
                (300, 1000),   # Low-mid frequencies  
                (1000, 3000),  # Mid frequencies (speech critical)
                (3000, 6000),  # High-mid frequencies
                (6000, sample_rate//2)  # High frequencies
            ]
            
            # STFT
            hop_length = 256
            n_fft = 1024
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Frequency bins
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            
            enhanced_magnitude = magnitude.copy()
            
            # Process each frequency band separately
            for i, (low_freq, high_freq) in enumerate(bands):
                # Find frequency bin range for this band
                band_mask = (freqs >= low_freq) & (freqs < high_freq)
                band_indices = np.where(band_mask)[0]
                
                if len(band_indices) == 0:
                    continue
                
                # Extract band magnitude
                band_magnitude = magnitude[band_indices, :]
                
                # Band-specific noise reduction parameters
                if low_freq >= 1000 and high_freq <= 3000:
                    # Speech critical band - less aggressive
                    band_strength = strength * 0.6
                    min_gain = 0.3
                elif low_freq < 300:
                    # Low frequency band - more aggressive (often contains noise)
                    band_strength = strength * 1.2
                    min_gain = 0.1
                else:
                    # Other bands - standard processing
                    band_strength = strength
                    min_gain = 0.2
                
                # Apply noise reduction to this band
                noise_estimate = np.mean(band_magnitude[:, :5], axis=1, keepdims=True)  # First 5 frames
                
                for j, freq_idx in enumerate(band_indices):
                    for frame in range(magnitude.shape[1]):
                        original_mag = magnitude[freq_idx, frame]
                        noise_level = noise_estimate[j, 0]
                        
                        # Calculate reduction
                        reduction_factor = band_strength * (noise_level / (original_mag + 1e-8))
                        reduction_factor = min(reduction_factor, 0.9)  # Limit max reduction
                        
                        # Apply reduction with minimum gain
                        gain = max(1.0 - reduction_factor, min_gain)
                        enhanced_magnitude[freq_idx, frame] = original_mag * gain
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.exception(f"Error in multi-band noise reduction: {e}")
            return audio

    def _estimate_noise_spectrum(
        self, 
        magnitude: np.ndarray, 
        method: str = 'multi_frame'
    ) -> np.ndarray:
        """Improved noise spectrum estimation"""
        try:
            if method == 'multi_frame':
                # Use multiple frames for better noise estimation
                num_noise_frames = min(10, magnitude.shape[1] // 4)
                
                # Use both beginning and end frames (assuming they contain noise)
                start_frames = magnitude[:, :num_noise_frames//2]
                end_frames = magnitude[:, -num_noise_frames//2:] if magnitude.shape[1] > num_noise_frames else magnitude[:, :1]
                
                # Combine estimates
                noise_spectrum = np.minimum(
                    np.mean(start_frames, axis=1, keepdims=True),
                    np.mean(end_frames, axis=1, keepdims=True)
                )
                
            elif method == 'minimum_tracking':
                # Track minimum values across time for each frequency bin
                window_size = min(20, magnitude.shape[1] // 2)
                noise_spectrum = np.zeros((magnitude.shape[0], 1))
                
                for freq_bin in range(magnitude.shape[0]):
                    # Use minimum tracking for noise estimation
                    windowed_mins = []
                    for i in range(0, magnitude.shape[1] - window_size + 1, window_size // 2):
                        window_min = np.min(magnitude[freq_bin, i:i+window_size])
                        windowed_mins.append(window_min)
                    
                    noise_spectrum[freq_bin, 0] = np.median(windowed_mins) if windowed_mins else np.min(magnitude[freq_bin, :])
            
            else:
                # Default: use first few frames
                num_frames = min(5, magnitude.shape[1] // 4)
                noise_spectrum = np.mean(magnitude[:, :num_frames], axis=1, keepdims=True)
            
            return noise_spectrum
            
        except Exception as e:
            logger.exception(f"Error estimating noise spectrum: {e}")
            # Fallback to simple estimation
            return np.mean(magnitude[:, :5], axis=1, keepdims=True)

    def _simple_vad_mask(
        self, 
        magnitude: np.ndarray, 
        noise_estimate: np.ndarray
    ) -> np.ndarray:
        """Simple voice activity detection mask"""
        try:
            # Calculate SNR for each time-frequency bin
            snr_threshold = 3.0  # dB
            snr_linear_threshold = 10**(snr_threshold / 10)
            
            vad_mask = np.zeros_like(magnitude, dtype=bool)
            
            for freq_bin in range(magnitude.shape[0]):
                noise_level = noise_estimate[freq_bin, 0] if noise_estimate.shape[1] == 1 else noise_estimate[freq_bin, :]
                
                # Calculate local SNR
                local_snr = (magnitude[freq_bin, :] ** 2) / (noise_level ** 2 + 1e-8)
                
                # Mark as speech if SNR is above threshold
                vad_mask[freq_bin, :] = local_snr > snr_linear_threshold
            
            # Apply temporal smoothing to reduce false positives
            from scipy import ndimage
            vad_mask = ndimage.binary_opening(vad_mask, structure=np.ones((3, 3)))
            
            return vad_mask
            
        except Exception as e:
            logger.exception(f"Error creating VAD mask: {e}")
            return np.ones_like(magnitude, dtype=bool)  # Default to all speech

    def _estimate_noise_power_spectrum(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Estimate noise power spectrum using minimum statistics"""
        try:
            # Use minimum tracking across time for each frequency bin
            window_length = min(20, power_spectrum.shape[1] // 3)
            noise_power = np.zeros((power_spectrum.shape[0], 1))
            
            for freq_bin in range(power_spectrum.shape[0]):
                # Track minimum power in sliding windows
                min_values = []
                for i in range(0, power_spectrum.shape[1] - window_length + 1, window_length // 2):
                    window_min = np.min(power_spectrum[freq_bin, i:i+window_length])
                    min_values.append(window_min)
                
                # Use median of minimums as noise estimate
                noise_power[freq_bin, 0] = np.median(min_values) if min_values else np.min(power_spectrum[freq_bin, :])
            
            return noise_power
            
        except Exception as e:
            logger.exception(f"Error estimating noise power spectrum: {e}")
            return np.mean(power_spectrum[:, :5], axis=1, keepdims=True)

    def _estimate_speech_probability(
        self, 
        signal_power: float, 
        noise_power: float, 
        freq_bin: int, 
        sample_rate: int
    ) -> float:
        """Estimate probability that current frame contains speech"""
        try:
            # Calculate SNR
            snr = signal_power / (noise_power + 1e-8)
            snr_db = 10 * np.log10(snr)
            
            # Base probability from SNR
            if snr_db > 10:
                snr_prob = 0.9
            elif snr_db > 5:
                snr_prob = 0.7
            elif snr_db > 0:
                snr_prob = 0.5
            else:
                snr_prob = 0.2
            
            # Frequency-based probability (speech is more likely in certain frequency ranges)
            freq_hz = freq_bin * sample_rate / 2048  # Assuming n_fft=1024
            
            if 300 <= freq_hz <= 3000:  # Primary speech frequencies
                freq_prob = 0.8
            elif 100 <= freq_hz <= 6000:  # Extended speech range
                freq_prob = 0.6
            else:
                freq_prob = 0.3
            
            # Combine probabilities
            speech_probability = (snr_prob + freq_prob) / 2
            return min(speech_probability, 1.0)
            
        except Exception as e:
            logger.exception(f"Error estimating speech probability: {e}")
            return 0.5  # Default moderate probability

    def _smooth_audio_transitions(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply smooth transitions to reduce artifacts"""
        try:
            # Simple high-frequency smoothing to reduce artifacts
            from scipy import signal
            
            # Design a gentle low-pass filter to remove artifacts above speech range
            nyquist = sample_rate / 2
            cutoff = min(8000, nyquist * 0.9)  # Cutoff at 8kHz or 90% of Nyquist
            
            # Butterworth filter
            sos = signal.butter(4, cutoff, btype='low', fs=sample_rate, output='sos')
            smoothed_audio = signal.sosfilt(sos, audio)
            
            return smoothed_audio
            
        except Exception as e:
            logger.exception(f"Error smoothing audio transitions: {e}")
            return audio

    def _rnn_noise_reduction(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        RNN-based noise reduction (placeholder for future implementation)
        This would require a trained RNN model for noise reduction
        """
        try:
            logger.info("RNN noise reduction not implemented, falling back to spectral subtraction")
            return self._advanced_spectral_subtraction(audio, sample_rate, 0.8)
            
        except Exception as e:
            logger.exception(f"Error in RNN noise reduction: {e}")
            return audio  