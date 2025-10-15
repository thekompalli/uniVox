"""
Audio Uploader Component
Streamlit component for uploading and validating audio files
"""

import streamlit as st
import numpy as np
import librosa
import io
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioUploader:
    """Audio file upload and validation component"""

    def __init__(self):
        """Initialize audio uploader"""
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
        self.max_file_size = 500 * 1024 * 1024  # 500MB
        self.min_duration = 1.0  # 1 second
        self.max_duration = 3600.0  # 1 hour

    def render(self) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        """
        Render audio upload component

        Returns:
            Tuple of (audio_file_bytes, file_info) or (None, None) if no file
        """
        st.subheader("📁 Upload Audio File")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=self.supported_formats,
            help=f"""
            **Supported formats:** {', '.join(self.supported_formats)}

            **Requirements:**
            - Maximum file size: 500MB
            - Duration: 1 second to 1 hour
            - Sample rate: 8kHz to 48kHz
            - Channels: Mono or Stereo (will be converted to mono)
            """
        )

        if uploaded_file is None:
            # Show upload instructions
            st.info("""
            👆 **Upload an audio file to get started**

            **Supported formats:** WAV, MP3, FLAC, OGG, M4A

            **Tips for best results:**
            - Use high-quality recordings (SNR > 5dB)
            - Ensure clear speech without excessive background noise
            - Multi-speaker files are supported
            - Mixed language content is automatically detected
            """)
            return None, None

        # Validate and process uploaded file
        return self._process_uploaded_file(uploaded_file)

    def _process_uploaded_file(self, uploaded_file) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        """
        Process and validate uploaded audio file

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Tuple of (audio_bytes, file_info) or (None, None) if invalid
        """
        try:
            # Get file info
            file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size / (1024 * 1024),  # MB
                'type': uploaded_file.type or self._guess_file_type(uploaded_file.name)
            }

            # Validate file size
            if uploaded_file.size > self.max_file_size:
                st.error(f"❌ File too large: {file_info['size']:.1f}MB (max: 500MB)")
                return None, None

            # Read file bytes
            file_bytes = uploaded_file.getvalue()

            # Validate audio content
            audio_info = self._validate_audio_content(file_bytes, uploaded_file.name)

            if audio_info is None:
                return None, None

            # Merge file info with audio info
            file_info.update(audio_info)

            # Show validation results
            self._show_validation_results(file_info)

            return file_bytes, file_info

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.exception(f"Error processing uploaded file: {e}")
            return None, None

    def _validate_audio_content(self, file_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
        """
        Validate audio content using librosa

        Args:
            file_bytes: Raw file bytes
            filename: Original filename

        Returns:
            Audio information dict or None if invalid
        """
        try:
            # Create file-like object from bytes
            audio_buffer = io.BytesIO(file_bytes)

            # Load audio with librosa
            with st.spinner("Analyzing audio file..."):
                try:
                    # Try to load audio
                    audio, sample_rate = librosa.load(audio_buffer, sr=None, mono=False)

                    # Convert to mono if stereo
                    if audio.ndim > 1:
                        audio = librosa.to_mono(audio)

                    duration = len(audio) / sample_rate

                    # Validate duration
                    if duration < self.min_duration:
                        st.error(f"❌ Audio too short: {duration:.1f}s (minimum: {self.min_duration}s)")
                        return None

                    if duration > self.max_duration:
                        st.error(f"❌ Audio too long: {duration:.1f}s (maximum: {self.max_duration}s)")
                        return None

                    # Calculate additional metrics
                    rms_energy = np.sqrt(np.mean(audio**2))
                    snr_estimate = self._estimate_snr(audio)

                    audio_info = {
                        'duration': f"{duration:.1f}s",
                        'duration_seconds': duration,
                        'sample_rate': sample_rate,
                        'channels': 1,  # Always mono after conversion
                        'rms_energy': rms_energy,
                        'snr_estimate': snr_estimate,
                        'has_speech': self._detect_speech_activity(audio, sample_rate)
                    }

                    return audio_info

                except Exception as e:
                    st.error(f"❌ Invalid audio file: {str(e)}")
                    return None

        except Exception as e:
            st.error(f"❌ Error analyzing audio: {str(e)}")
            return None

    def _estimate_snr(self, audio: np.ndarray, frame_length: int = 2048) -> float:
        """
        Estimate Signal-to-Noise Ratio

        Args:
            audio: Audio signal
            frame_length: Frame length for analysis

        Returns:
            Estimated SNR in dB
        """
        try:
            # Simple SNR estimation using energy in different frequency bands
            stft = librosa.stft(audio, n_fft=frame_length)
            magnitude = np.abs(stft)

            # Estimate signal and noise energy
            signal_energy = np.mean(np.max(magnitude, axis=0))  # Max energy per frame
            noise_energy = np.mean(np.min(magnitude, axis=0))   # Min energy per frame

            if noise_energy > 0:
                snr = 20 * np.log10(signal_energy / noise_energy)
                return max(0, min(50, snr))  # Clamp between 0 and 50 dB
            else:
                return 30.0  # Default if cannot estimate

        except Exception:
            return 15.0  # Default SNR estimate

    def _detect_speech_activity(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Detect if audio contains speech activity

        Args:
            audio: Audio signal
            sample_rate: Sample rate

        Returns:
            True if speech detected, False otherwise
        """
        try:
            # Use simple energy-based voice activity detection
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop

            # Calculate frame energy
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            frame_energy = np.sum(frames**2, axis=0)

            # Threshold based on energy statistics
            energy_threshold = np.percentile(frame_energy, 30)  # 30th percentile as noise floor
            speech_frames = frame_energy > (energy_threshold * 5)  # 5x above noise floor

            # Require at least 10% of frames to have speech
            speech_ratio = np.sum(speech_frames) / len(speech_frames)
            return speech_ratio > 0.1

        except Exception:
            return True  # Assume speech if detection fails

    def _guess_file_type(self, filename: str) -> str:
        """
        Guess MIME type from file extension

        Args:
            filename: File name

        Returns:
            Guessed MIME type
        """
        extension = Path(filename).suffix.lower()
        mime_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4'
        }
        return mime_types.get(extension, 'audio/wav')

    def _show_validation_results(self, file_info: Dict[str, Any]):
        """
        Show audio validation results

        Args:
            file_info: File information dictionary
        """
        st.subheader("✅ File Validation Results")

        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Duration",
                file_info['duration'],
                help="Total audio duration"
            )

        with col2:
            st.metric(
                "Sample Rate",
                f"{file_info['sample_rate']} Hz",
                help="Audio sample rate"
            )

        with col3:
            # SNR with color coding
            snr = file_info['snr_estimate']
            snr_delta = None
            if snr >= 15:
                snr_delta = "Good"
            elif snr >= 10:
                snr_delta = "Fair"
            else:
                snr_delta = "Poor"

            st.metric(
                "Est. SNR",
                f"{snr:.1f} dB",
                delta=snr_delta,
                help="Estimated Signal-to-Noise Ratio"
            )

        with col4:
            speech_status = "✅ Detected" if file_info['has_speech'] else "⚠️ Not detected"
            st.metric(
                "Speech Activity",
                speech_status,
                help="Whether speech activity was detected"
            )

        # Show warnings or recommendations
        warnings = []
        recommendations = []

        if file_info['snr_estimate'] < 10:
            warnings.append("Low SNR detected - audio quality may affect results")
            recommendations.append("Consider using audio with less background noise")

        if not file_info['has_speech']:
            warnings.append("No clear speech activity detected")
            recommendations.append("Ensure the audio contains spoken content")

        if file_info['duration_seconds'] < 5:
            recommendations.append("Very short audio - results may be limited")

        if file_info['sample_rate'] < 16000:
            recommendations.append("Low sample rate - consider higher quality audio for best results")

        # Display warnings
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")

        # Display recommendations
        if recommendations:
            with st.expander("💡 Recommendations"):
                for rec in recommendations:
                    st.info(f"💡 {rec}")

        # Audio quality summary
        quality_score = self._calculate_quality_score(file_info)
        quality_color = "green" if quality_score >= 80 else "orange" if quality_score >= 60 else "red"

        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; border-left: 4px solid {quality_color};">
            <strong>Audio Quality Score: {quality_score}/100</strong><br>
            {'Excellent quality for processing' if quality_score >= 80 else
             'Good quality, should work well' if quality_score >= 60 else
             'Lower quality, results may vary'}
        </div>
        """, unsafe_allow_html=True)

    def _calculate_quality_score(self, file_info: Dict[str, Any]) -> int:
        """
        Calculate overall audio quality score

        Args:
            file_info: File information

        Returns:
            Quality score (0-100)
        """
        score = 100

        # SNR contribution (40 points)
        snr = file_info['snr_estimate']
        if snr >= 20:
            snr_score = 40
        elif snr >= 15:
            snr_score = 35
        elif snr >= 10:
            snr_score = 25
        elif snr >= 5:
            snr_score = 15
        else:
            snr_score = 5

        # Speech detection (30 points)
        speech_score = 30 if file_info['has_speech'] else 5

        # Sample rate (20 points)
        sr = file_info['sample_rate']
        if sr >= 44100:
            sr_score = 20
        elif sr >= 22050:
            sr_score = 18
        elif sr >= 16000:
            sr_score = 15
        elif sr >= 8000:
            sr_score = 10
        else:
            sr_score = 5

        # Duration (10 points)
        duration = file_info['duration_seconds']
        if duration >= 30:
            duration_score = 10
        elif duration >= 10:
            duration_score = 8
        elif duration >= 5:
            duration_score = 6
        else:
            duration_score = 3

        total_score = min(100, snr_score + speech_score + sr_score + duration_score)
        return max(0, total_score)