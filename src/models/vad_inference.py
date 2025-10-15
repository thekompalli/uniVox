"""
Voice Activity Detection Inference
Silero VAD model wrapper for speech detection
"""
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config.app_config import app_config, model_config

logger = logging.getLogger(__name__)


class VADInference:
    """Silero VAD model inference wrapper"""
    
    def __init__(self):
        self.model = None
        self.utils = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000  # Silero VAD requires 16kHz
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Silero VAD model"""
        try:
            logger.info("Initializing Silero VAD model")
            
            # Load pre-trained Silero VAD model
            # torch.hub expects the GitHub repo in the form "owner/repo"
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                onnx=False
            )
            
            # Move to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            # Extract utilities
            (self.get_speech_timestamps, 
             self.save_audio, 
             self.read_audio, 
             self.VADIterator, 
             self.collect_chunks) = self.utils
            
            logger.info(f"Silero VAD model initialized on {self.device}")
            
        except Exception as e:
            logger.exception(f"Error initializing Silero VAD model: {e}")
            raise
    
    async def detect_speech(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: int = 30,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Detect speech segments in audio
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Original sample rate
            min_speech_duration_ms: Minimum speech duration in ms
            max_speech_duration_s: Maximum speech duration in seconds
            min_silence_duration_ms: Minimum silence duration in ms
            window_size_samples: VAD window size
            speech_pad_ms: Padding around speech segments in ms
            
        Returns:
            List of speech segments with start/end times and confidence
        """
        try:
            # Run VAD in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._detect_speech_sync,
                audio_data,
                sample_rate,
                min_speech_duration_ms,
                max_speech_duration_s,
                min_silence_duration_ms,
                window_size_samples,
                speech_pad_ms
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in VAD detection: {e}")
            raise
    
    def _detect_speech_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        min_speech_duration_ms: int,
        max_speech_duration_s: int,
        min_silence_duration_ms: int,
        window_size_samples: int,
        speech_pad_ms: int
    ) -> List[Dict[str, Any]]:
        """Synchronous speech detection"""
        try:
            # Resample to 16kHz if needed
            if sample_rate != self.sample_rate:
                audio_tensor = self._resample_audio(
                    audio_data, sample_rate, self.sample_rate
                )
            else:
                audio_tensor = torch.from_numpy(audio_data).float()
            
            # Ensure audio is on correct device
            audio_tensor = audio_tensor.to(self.device)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                max_speech_duration_s=max_speech_duration_s,
                min_silence_duration_ms=min_silence_duration_ms,
                window_size_samples=window_size_samples,
                speech_pad_ms=speech_pad_ms,
                return_seconds=True
            )
            
            # Convert to our format
            segments = []
            for i, timestamp in enumerate(speech_timestamps):
                segment = {
                    'start': float(timestamp['start']),
                    'end': float(timestamp['end']),
                    'confidence': self._calculate_segment_confidence(
                        audio_tensor, timestamp, self.sample_rate
                    ),
                    'segment_id': i
                }
                segments.append(segment)
            
            logger.debug(f"VAD detected {len(segments)} speech segments")
            return segments
            
        except Exception as e:
            logger.exception(f"Error in synchronous VAD detection: {e}")
            raise
    
    async def detect_speech_streaming(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int,
        chunk_duration_ms: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Streaming VAD for real-time processing
        
        Args:
            audio_chunks: List of audio chunks
            sample_rate: Sample rate
            chunk_duration_ms: Chunk duration in milliseconds
            
        Returns:
            List of speech detection results per chunk
        """
        try:
            # Initialize VAD iterator
            vad_iterator = self.VADIterator(
                self.model,
                sampling_rate=self.sample_rate
            )
            
            results = []
            current_time = 0.0
            chunk_size_samples = int(sample_rate * chunk_duration_ms / 1000)
            
            for chunk in audio_chunks:
                # Resample if needed
                if sample_rate != self.sample_rate:
                    chunk_tensor = self._resample_audio(
                        chunk, sample_rate, self.sample_rate
                    )
                else:
                    chunk_tensor = torch.from_numpy(chunk).float()
                
                chunk_tensor = chunk_tensor.to(self.device)
                
                # Process chunk
                speech_dict = vad_iterator(chunk_tensor, return_seconds=True)
                
                result = {
                    'chunk_start': current_time,
                    'chunk_end': current_time + chunk_duration_ms / 1000,
                    'speech_prob': float(speech_dict.get('speech_prob', 0.0)),
                    'is_speech': bool(speech_dict.get('is_speech', False))
                }
                
                results.append(result)
                current_time += chunk_duration_ms / 1000
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in streaming VAD: {e}")
            raise
    
    async def get_speech_confidence_scores(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_size_ms: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frame-level speech confidence scores
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            window_size_ms: Window size in milliseconds
            
        Returns:
            Tuple of (timestamps, confidence_scores)
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._get_confidence_scores_sync,
                audio_data,
                sample_rate,
                window_size_ms
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error getting confidence scores: {e}")
            raise
    
    def _get_confidence_scores_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_size_ms: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous confidence score computation"""
        try:
            # Resample if needed
            if sample_rate != self.sample_rate:
                audio_tensor = self._resample_audio(
                    audio_data, sample_rate, self.sample_rate
                )
            else:
                audio_tensor = torch.from_numpy(audio_data).float()
            
            audio_tensor = audio_tensor.to(self.device)
            
            # Silero VAD forward requires fixed frame length: 512 @16k, 256 @8k
            frame_size = 512 if self.sample_rate == 16000 else 256
            # Use requested window size as hop (stride) in samples
            hop_samples = max(1, int(self.sample_rate * window_size_ms / 1000))
            
            timestamps = []
            confidence_scores = []
            
            # Process in sliding windows of fixed size required by the model
            for start in range(0, len(audio_tensor) - frame_size + 1, hop_samples):
                end = start + frame_size
                window_audio = audio_tensor[start:end]
                # Get speech probability for this fixed-size window
                with torch.no_grad():
                    speech_prob = self.model(window_audio.unsqueeze(0), self.sample_rate)
                    speech_prob = float(speech_prob.squeeze())
                
                # Convert to time
                time_start = start / self.sample_rate
                timestamps.append(time_start)
                confidence_scores.append(speech_prob)
            
            return np.array(timestamps), np.array(confidence_scores)
            
        except Exception as e:
            logger.exception(f"Error in sync confidence scores: {e}")
            raise
    
    def _calculate_segment_confidence(
        self,
        audio_tensor: torch.Tensor,
        timestamp: Dict[str, float],
        sample_rate: int
    ) -> float:
        """Calculate confidence for a speech segment"""
        try:
            start_sample = int(timestamp['start'] * sample_rate)
            end_sample = int(timestamp['end'] * sample_rate)
            
            # Extract segment
            segment = audio_tensor[start_sample:end_sample]
            
            if len(segment) == 0:
                return 0.0
            
            # Get average speech probability for segment using fixed-size frames
            frame_size = 512 if sample_rate == 16000 else 256
            hop_size = frame_size // 4

            probs = []
            if len(segment) < frame_size:
                pad = torch.zeros(frame_size - len(segment), device=segment.device, dtype=segment.dtype)
                window = torch.cat([segment, pad], dim=0)
                with torch.no_grad():
                    prob = self.model(window.unsqueeze(0), sample_rate)
                    probs.append(float(prob.squeeze()))
            else:
                for start in range(0, len(segment) - frame_size + 1, hop_size):
                    end = start + frame_size
                    window = segment[start:end]
                    with torch.no_grad():
                        prob = self.model(window.unsqueeze(0), sample_rate)
                        probs.append(float(prob.squeeze()))
            
            return np.mean(probs) if probs else 0.5
            
        except Exception as e:
            logger.exception(f"Error calculating segment confidence: {e}")
            return 0.5
    
    def _resample_audio(
        self,
        audio_data: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> torch.Tensor:
        """Resample audio using torchaudio"""
        try:
            import torchaudio.transforms as T
            
            audio_tensor = torch.from_numpy(audio_data).float()
            
            if orig_sr != target_sr:
                resampler = T.Resample(orig_sr, target_sr)
                audio_tensor = resampler(audio_tensor)
            
            return audio_tensor
            
        except ImportError:
            # Fallback to librosa if torchaudio not available
            import librosa
            resampled = librosa.resample(
                audio_data,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
            return torch.from_numpy(resampled).float()
    
    async def filter_non_speech(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speech_threshold: float = 0.5
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Remove non-speech regions from audio
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate
            speech_threshold: Speech probability threshold
            
        Returns:
            Tuple of (filtered_audio, speech_segments)
        """
        try:
            # Get speech segments
            speech_segments = await self.detect_speech(audio_data, sample_rate)
            
            if not speech_segments:
                return np.array([]), []
            
            # Extract and concatenate speech regions
            speech_audio_parts = []
            filtered_segments = []
            current_time = 0.0
            
            for segment in speech_segments:
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Extract speech segment
                speech_part = audio_data[start_sample:end_sample]
                speech_audio_parts.append(speech_part)
                
                # Update segment timing for filtered audio
                duration = len(speech_part) / sample_rate
                filtered_segment = {
                    'original_start': segment['start'],
                    'original_end': segment['end'],
                    'filtered_start': current_time,
                    'filtered_end': current_time + duration,
                    'confidence': segment['confidence']
                }
                filtered_segments.append(filtered_segment)
                
                current_time += duration
            
            # Concatenate all speech parts
            filtered_audio = np.concatenate(speech_audio_parts) if speech_audio_parts else np.array([])
            
            logger.info(f"Filtered audio: {len(filtered_audio)/sample_rate:.2f}s from "
                       f"{len(audio_data)/sample_rate:.2f}s original")
            
            return filtered_audio, filtered_segments
            
        except Exception as e:
            logger.exception(f"Error filtering non-speech: {e}")
            return audio_data, []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get VAD model information"""
        return {
            'model_name': 'Silero VAD',
            'sample_rate': self.sample_rate,
            'device': str(self.device),
            'model_version': 'v3.1',
            'supported_languages': ['multilingual'],
            'window_size_samples': 512,
            'supported_sample_rates': [8000, 16000]
        }
