"""
Audio Utilities
Helper functions for audio processing and analysis
"""

import numpy as np
import librosa
import io
from typing import Tuple, Optional, Dict, Any
import logging
import wave

logger = logging.getLogger(__name__)

class AudioUtils:
    """Utility functions for audio processing"""

    @staticmethod
    def analyze_audio_buffer(audio_buffer: io.BytesIO) -> Optional[Dict[str, Any]]:
        """
        Analyze audio from buffer

        Args:
            audio_buffer: Audio data buffer

        Returns:
            Audio analysis results or None if failed
        """
        try:
            # Load audio with librosa
            audio, sample_rate = librosa.load(audio_buffer, sr=None, mono=False)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)

            # Basic analysis
            duration = len(audio) / sample_rate
            rms_energy = float(np.sqrt(np.mean(audio**2)))

            # Spectral analysis
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

            # Tempo and rhythm
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)

            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': 1,
                'rms_energy': rms_energy,
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'tempo': float(tempo),
                'num_beats': len(beat_frames)
            }

        except Exception as e:
            logger.exception(f"Error analyzing audio buffer: {e}")
            return None

    @staticmethod
    def estimate_snr(audio: np.ndarray, frame_length: int = 2048) -> float:
        """
        Estimate Signal-to-Noise Ratio

        Args:
            audio: Audio signal
            frame_length: Frame length for analysis

        Returns:
            Estimated SNR in dB
        """
        try:
            # Use STFT for frequency domain analysis
            stft = librosa.stft(audio, n_fft=frame_length)
            magnitude = np.abs(stft)

            # Simple SNR estimation
            # Signal: high-energy frames, Noise: low-energy frames
            frame_energy = np.mean(magnitude, axis=0)

            # Use percentiles to estimate signal and noise
            noise_threshold = np.percentile(frame_energy, 25)  # Bottom 25% as noise
            signal_threshold = np.percentile(frame_energy, 75)  # Top 25% as signal

            noise_energy = np.mean(frame_energy[frame_energy <= noise_threshold])
            signal_energy = np.mean(frame_energy[frame_energy >= signal_threshold])

            if noise_energy > 0:
                snr = 20 * np.log10(signal_energy / noise_energy)
                return max(0, min(50, snr))  # Clamp between 0 and 50 dB
            else:
                return 30.0  # Default high SNR

        except Exception as e:
            logger.exception(f"Error estimating SNR: {e}")
            return 15.0  # Default moderate SNR

    @staticmethod
    def detect_voice_activity(audio: np.ndarray, sample_rate: int,
                            frame_length: float = 0.025, hop_length: float = 0.010) -> Dict[str, Any]:
        """
        Detect voice activity in audio

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            frame_length: Frame length in seconds
            hop_length: Hop length in seconds

        Returns:
            Voice activity detection results
        """
        try:
            frame_samples = int(frame_length * sample_rate)
            hop_samples = int(hop_length * sample_rate)

            # Frame the audio
            frames = librosa.util.frame(audio, frame_length=frame_samples, hop_length=hop_samples)

            # Calculate features for each frame
            frame_energy = np.sum(frames**2, axis=0)
            frame_zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=0)), axis=0)

            # Normalize features
            energy_mean = np.mean(frame_energy)
            energy_std = np.std(frame_energy)
            zcr_mean = np.mean(frame_zcr)

            # Voice activity detection thresholds
            energy_threshold = energy_mean + 0.5 * energy_std
            zcr_threshold = zcr_mean * 0.5

            # Classify frames
            voice_frames = (frame_energy > energy_threshold) & (frame_zcr > zcr_threshold)

            # Calculate statistics
            total_frames = len(voice_frames)
            voice_frame_count = np.sum(voice_frames)
            voice_ratio = voice_frame_count / total_frames if total_frames > 0 else 0

            # Find voice segments
            voice_segments = []
            in_voice = False
            start_frame = 0

            for i, is_voice in enumerate(voice_frames):
                if is_voice and not in_voice:
                    start_frame = i
                    in_voice = True
                elif not is_voice and in_voice:
                    end_frame = i
                    start_time = start_frame * hop_length
                    end_time = end_frame * hop_length
                    voice_segments.append({'start': start_time, 'end': end_time})
                    in_voice = False

            # Handle case where voice continues to end
            if in_voice:
                end_time = len(voice_frames) * hop_length
                start_time = start_frame * hop_length
                voice_segments.append({'start': start_time, 'end': end_time})

            return {
                'voice_ratio': voice_ratio,
                'voice_frame_count': voice_frame_count,
                'total_frames': total_frames,
                'voice_segments': voice_segments,
                'has_voice': voice_ratio > 0.1  # At least 10% voice activity
            }

        except Exception as e:
            logger.exception(f"Error in voice activity detection: {e}")
            return {
                'voice_ratio': 0.5,
                'voice_frame_count': 0,
                'total_frames': 0,
                'voice_segments': [],
                'has_voice': True  # Assume voice if detection fails
            }

    @staticmethod
    def analyze_spectral_features(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract spectral features from audio

        Args:
            audio: Audio signal
            sample_rate: Sample rate

        Returns:
            Dictionary of spectral features
        """
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)

            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'zero_crossing_rate_std': float(np.std(zero_crossing_rate)),
                'mfcc_mean': [float(x) for x in np.mean(mfccs, axis=1)],
                'mfcc_std': [float(x) for x in np.std(mfccs, axis=1)],
                'chroma_mean': [float(x) for x in np.mean(chroma, axis=1)],
                'contrast_mean': [float(x) for x in np.mean(contrast, axis=1)]
            }

        except Exception as e:
            logger.exception(f"Error extracting spectral features: {e}")
            return {}

    @staticmethod
    def detect_audio_quality_issues(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect potential audio quality issues

        Args:
            audio: Audio signal
            sample_rate: Sample rate

        Returns:
            Dictionary of quality issues and recommendations
        """
        issues = []
        recommendations = []
        quality_score = 100

        try:
            # Check for clipping
            clipped_samples = np.sum(np.abs(audio) >= 0.99)
            clipping_ratio = clipped_samples / len(audio)

            if clipping_ratio > 0.01:  # More than 1% clipped
                issues.append(f"Audio clipping detected ({clipping_ratio:.2%} of samples)")
                recommendations.append("Reduce input gain to avoid clipping")
                quality_score -= 20

            # Check dynamic range
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))

            if dynamic_range < 10:
                issues.append(f"Low dynamic range ({dynamic_range:.1f} dB)")
                recommendations.append("Increase recording level or reduce compression")
                quality_score -= 15

            # Check for silence
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio) < silence_threshold)
            silence_ratio = silent_samples / len(audio)

            if silence_ratio > 0.5:
                issues.append(f"High silence ratio ({silence_ratio:.2%})")
                recommendations.append("Check microphone connection and recording levels")
                quality_score -= 10

            # Check frequency content
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)

            # Check for sufficient high-frequency content (above 1kHz)
            high_freq_energy = np.sum(magnitude[np.abs(freqs) > 1000])
            total_energy = np.sum(magnitude)
            high_freq_ratio = high_freq_energy / total_energy

            if high_freq_ratio < 0.1:
                issues.append("Limited high-frequency content")
                recommendations.append("Check for low-pass filtering or poor microphone response")
                quality_score -= 10

            # SNR estimation
            snr = AudioUtils.estimate_snr(audio)

            if snr < 10:
                issues.append(f"Low signal-to-noise ratio ({snr:.1f} dB)")
                recommendations.append("Record in a quieter environment or use noise reduction")
                quality_score -= 15
            elif snr < 15:
                issues.append(f"Moderate signal-to-noise ratio ({snr:.1f} dB)")
                recommendations.append("Consider noise reduction for better results")
                quality_score -= 5

            return {
                'issues': issues,
                'recommendations': recommendations,
                'quality_score': max(0, quality_score),
                'metrics': {
                    'clipping_ratio': clipping_ratio,
                    'dynamic_range': dynamic_range,
                    'silence_ratio': silence_ratio,
                    'high_freq_ratio': high_freq_ratio,
                    'snr': snr
                }
            }

        except Exception as e:
            logger.exception(f"Error detecting quality issues: {e}")
            return {
                'issues': ["Unable to analyze audio quality"],
                'recommendations': ["Try uploading the file again"],
                'quality_score': 50,
                'metrics': {}
            }

    @staticmethod
    def convert_audio_format(audio_data: bytes, target_format: str = 'wav') -> Optional[bytes]:
        """
        Convert audio to target format

        Args:
            audio_data: Original audio data
            target_format: Target format ('wav', 'mp3', etc.)

        Returns:
            Converted audio data or None if failed
        """
        try:
            # Load audio with librosa
            audio_buffer = io.BytesIO(audio_data)
            audio, sample_rate = librosa.load(audio_buffer, sr=None)

            # Convert to target format
            if target_format.lower() == 'wav':
                # Convert to WAV
                output_buffer = io.BytesIO()

                # Normalize to 16-bit range
                audio_int = (audio * 32767).astype(np.int16)

                # Write WAV file
                with wave.open(output_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int.tobytes())

                output_buffer.seek(0)
                return output_buffer.getvalue()

            else:
                logger.warning(f"Unsupported target format: {target_format}")
                return None

        except Exception as e:
            logger.exception(f"Error converting audio format: {e}")
            return None

    @staticmethod
    def generate_audio_summary(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Generate comprehensive audio summary

        Args:
            audio: Audio signal
            sample_rate: Sample rate

        Returns:
            Comprehensive audio summary
        """
        try:
            # Basic properties
            duration = len(audio) / sample_rate
            rms_energy = float(np.sqrt(np.mean(audio**2)))

            # Voice activity detection
            vad_results = AudioUtils.detect_voice_activity(audio, sample_rate)

            # Spectral features
            spectral_features = AudioUtils.analyze_spectral_features(audio, sample_rate)

            # Quality analysis
            quality_analysis = AudioUtils.detect_audio_quality_issues(audio, sample_rate)

            # SNR estimation
            snr = AudioUtils.estimate_snr(audio)

            # Tempo analysis
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
            except:
                tempo = 0
                beat_frames = []

            return {
                'basic_properties': {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': 1,
                    'rms_energy': rms_energy,
                    'snr_estimate': snr
                },
                'voice_activity': vad_results,
                'spectral_features': spectral_features,
                'quality_analysis': quality_analysis,
                'rhythm': {
                    'tempo': float(tempo),
                    'num_beats': len(beat_frames)
                }
            }

        except Exception as e:
            logger.exception(f"Error generating audio summary: {e}")
            return {
                'basic_properties': {
                    'duration': 0,
                    'sample_rate': sample_rate,
                    'channels': 1,
                    'rms_energy': 0,
                    'snr_estimate': 0
                },
                'voice_activity': {'has_voice': True, 'voice_ratio': 0.5},
                'spectral_features': {},
                'quality_analysis': {
                    'issues': ['Analysis failed'],
                    'recommendations': ['Try re-uploading the file'],
                    'quality_score': 50
                },
                'rhythm': {'tempo': 0, 'num_beats': 0}
            }