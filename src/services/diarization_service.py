"""
Speaker Diarization Service
pyannote.audio integration for speaker diarization
"""
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
import soundfile as sf

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import torchaudio

from src.config.app_config import app_config, model_config
from src.models.diarization_inference import DiarizationInference
from src.repositories.audio_repository import AudioRepository
from src.utils.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class DiarizationService:
    """Speaker diarization service using pyannote.audio"""
    
    def __init__(self):
        self.diarization_model = None
        self.audio_repo = AudioRepository()
        self.audio_utils = AudioUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize diarization models"""
        try:
            logger.info("Initializing diarization models")
            # Direct HF cache to local models directory to reuse downloads
            import os
            cache_dir = os.getenv("HF_CACHE_DIR") or os.getenv("TRANSFORMERS_CACHE") or str(app_config.models_dir / "huggingface")
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)

            # Initialize pyannote pipeline with HuggingFace token
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            self.diarization_model = Pipeline.from_pretrained(
                model_config.pyannote_model,
                use_auth_token=hf_token if hf_token else True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.diarization_model.to(torch.device("cuda"))
            
            logger.info("Diarization models initialized successfully")
            
        except Exception as e:
            logger.exception(f"Error initializing diarization models: {e}")
            raise
    
    async def diarize_audio(
        self, 
        job_id: str,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio
        
        Args:
            job_id: Job identifier
            audio_data: Audio signal
            sample_rate: Sample rate
            num_speakers: Optional number of speakers hint
            
        Returns:
            Diarization results with segments and embeddings
        """
        try:
            logger.info(f"Starting speaker diarization for job {job_id}")
            
            # Run diarization in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._diarize_sync,
                audio_data,
                sample_rate,
                num_speakers
            )
            
            # Post-process results
            processed_result = await self._post_process_diarization(
                job_id, result, audio_data, sample_rate
            )
            
            logger.info(f"Diarization completed for job {job_id}: "
                       f"{len(processed_result['segments'])} segments, "
                       f"{processed_result['num_speakers']} speakers")
            
            return processed_result
            
        except Exception as e:
            logger.exception(f"Error in speaker diarization for job {job_id}: {e}")
            raise
    
    def _diarize_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synchronous diarization implementation"""
        try:
            # Convert numpy array to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Create temporary audio file for pyannote
            temp_path = app_config.temp_dir / "temp_diarization.wav"
            torchaudio.save(temp_path, audio_tensor, sample_rate)
            
            # Configure pipeline parameters
            if num_speakers:
                self.diarization_model.instantiate({
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": max(2, num_speakers - 1),
                        "max_cluster_size": num_speakers + 2
                    }
                })
            
            # Run diarization
            diarization_result = self.diarization_model(str(temp_path))
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            # Extract segments and labels
            segments = []
            labels = set()
            
            for turn, _, label in diarization_result.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': str(label),
                    'duration': turn.duration,
                    'confidence': 1.0  # pyannote doesn't provide confidence by default
                })
                labels.add(str(label))
            
            # Return only JSON-serializable content; keep raw objects out of results
            return {
                'segments': segments,
                'num_speakers': len(labels),
                'speaker_labels': list(labels)
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous diarization: {e}")
            raise
    
    async def _post_process_diarization(
        self,
        job_id: str,
        diarization_result: Dict[str, Any],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Post-process diarization results"""
        try:
            segments = diarization_result['segments']
            
            # Filter short segments
            filtered_segments = []
            for segment in segments:
                if segment['duration'] >= app_config.min_segment_duration:
                    filtered_segments.append(segment)
                else:
                    logger.debug(f"Filtering short segment: {segment['duration']:.3f}s")
            
            # Merge overlapping segments from same speaker
            merged_segments = self._merge_speaker_segments(filtered_segments)
            
            # Extract speaker embeddings for each segment
            embeddings = await self._extract_segment_embeddings(
                audio_data, sample_rate, merged_segments
            )
            
            # Calculate confidence scores based on segment characteristics
            confidence_segments = self._calculate_segment_confidence(
                merged_segments, audio_data, sample_rate
            )
            
            # Save diarization results
            await self._save_diarization_results(job_id, {
                'segments': confidence_segments,
                'embeddings': embeddings,
                'num_speakers': diarization_result['num_speakers'],
                'speaker_labels': diarization_result['speaker_labels']
            })
            
            return {
                'segments': confidence_segments,
                'embeddings': embeddings,
                'num_speakers': len(set(s['speaker'] for s in confidence_segments)),
                'speaker_labels': list(set(s['speaker'] for s in confidence_segments)),
                'quality_metrics': self._calculate_diarization_quality(confidence_segments)
            }
            
        except Exception as e:
            logger.exception(f"Error in diarization post-processing: {e}")
            raise
    
    def _merge_speaker_segments(
        self, 
        segments: List[Dict[str, Any]], 
        gap_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Merge segments from same speaker that are close together"""
        if not segments:
            return segments
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        merged = []
        
        for segment in sorted_segments:
            if not merged:
                merged.append(segment)
                continue
            
            last_segment = merged[-1]
            
            # Check if same speaker and segments are close
            if (segment['speaker'] == last_segment['speaker'] and 
                segment['start'] - last_segment['end'] <= gap_threshold):
                
                # Merge segments
                merged[-1] = {
                    'start': last_segment['start'],
                    'end': segment['end'],
                    'speaker': segment['speaker'],
                    'duration': segment['end'] - last_segment['start'],
                    'confidence': (last_segment['confidence'] + segment['confidence']) / 2
                }
            else:
                merged.append(segment)
        
        return merged
    
    async def _extract_segment_embeddings(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract speaker embeddings for each segment"""
        try:
            embeddings = []
            
            for segment in segments:
                # Extract audio segment
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                
                if len(segment_audio) < sample_rate * 0.1:  # Skip very short segments
                    # Use zero embedding for very short segments
                    embeddings.append(np.zeros(256))  # ECAPA embedding dim
                    continue
                
                # Extract embedding (simplified - would use actual embedding model)
                embedding = await self._extract_embedding(segment_audio, sample_rate)
                embeddings.append(embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.exception(f"Error extracting segment embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(segments), 256))
    
    async def _extract_embedding(
        self, 
        audio_segment: np.ndarray, 
        sample_rate: int
    ) -> np.ndarray:
        """Extract speaker embedding from audio segment"""
        try:
            # This would use a proper speaker embedding model (ECAPA-TDNN)
            # For now, return a simple feature-based representation
            
            # Basic spectral features as embedding
            import librosa
            
            # Ensure minimum length
            if len(audio_segment) < sample_rate * 0.1:
                audio_segment = np.pad(audio_segment, (0, int(sample_rate * 0.1)))
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_segment,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=512,
                hop_length=256
            )
            
            # Statistical pooling
            embedding = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1)
            ])
            
            # Pad or truncate to 256 dimensions
            if len(embedding) < 256:
                embedding = np.pad(embedding, (0, 256 - len(embedding)))
            else:
                embedding = embedding[:256]
            
            # L2 normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error extracting embedding: {e}")
            return np.zeros(256)
    
    def _calculate_segment_confidence(
        self,
        segments: List[Dict[str, Any]],
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Calculate confidence scores for segments"""
        try:
            confidence_segments = []
            
            for segment in segments:
                # Extract segment audio
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                
                # Calculate confidence based on multiple factors
                duration_factor = min(1.0, segment['duration'] / 2.0)  # Longer = more confident
                energy_factor = min(1.0, np.mean(np.abs(segment_audio)) * 10)  # Higher energy = more confident
                
                # SNR-based confidence
                snr = self.audio_utils.calculate_snr(segment_audio)
                snr_factor = min(1.0, max(0.1, snr / 20.0))  # Normalize SNR to 0-1
                
                # Combined confidence
                confidence = (duration_factor + energy_factor + snr_factor) / 3.0
                
                confidence_segment = segment.copy()
                confidence_segment['confidence'] = confidence
                confidence_segments.append(confidence_segment)
            
            return confidence_segments
            
        except Exception as e:
            logger.exception(f"Error calculating segment confidence: {e}")
            # Return original segments with default confidence
            for segment in segments:
                segment['confidence'] = 0.8
            return segments
    
    def _calculate_diarization_quality(
        self, 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate quality metrics for diarization results"""
        try:
            if not segments:
                return {}
            
            # Calculate metrics
            total_duration = sum(s['duration'] for s in segments)
            avg_segment_duration = np.mean([s['duration'] for s in segments])
            avg_confidence = np.mean([s['confidence'] for s in segments])
            num_speakers = len(set(s['speaker'] for s in segments))
            
            # Speaker balance (how evenly distributed the speakers are)
            speaker_durations = {}
            for segment in segments:
                speaker = segment['speaker']
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + segment['duration']
            
            if num_speakers > 1:
                speaker_balance = 1.0 - np.std(list(speaker_durations.values())) / np.mean(list(speaker_durations.values()))
            else:
                speaker_balance = 1.0
            
            return {
                'total_speech_duration': total_duration,
                'average_segment_duration': avg_segment_duration,
                'average_confidence': avg_confidence,
                'num_speakers_detected': num_speakers,
                'speaker_balance': max(0.0, speaker_balance),
                'num_segments': len(segments)
            }
            
        except Exception as e:
            logger.exception(f"Error calculating diarization quality: {e}")
            return {}
    
    async def _save_diarization_results(
        self, 
        job_id: str, 
        results: Dict[str, Any]
    ):
        """Save diarization results"""
        try:
            await self.audio_repo.save_processing_results(
                job_id, "diarization", results
            )
            logger.info(f"Saved diarization results for job {job_id}")
            
        except Exception as e:
            logger.exception(f"Error saving diarization results: {e}")
    
    def load_processed_data_sync(self, job_id: str) -> Dict[str, Any]:
        """Load processed audio data synchronously (for Celery tasks).

        Reads preprocessing_results.json written during preprocessing and
        loads the processed audio file into memory for downstream tasks.
        """
        try:
            job_dir = app_config.data_dir / "audio" / job_id
            meta_path = job_dir / "preprocessing_results.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing preprocessing results for job {job_id}: {meta_path}")

            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            processed_path = meta.get('processed_path')
            if not processed_path:
                raise ValueError("processed_path missing in preprocessing results")

            # Load processed audio
            audio_data, sample_rate = sf.read(processed_path)
            if audio_data.ndim > 1:
                # Convert to mono if stereo
                audio_data = audio_data.mean(axis=1)

            return {
                'audio_data': audio_data.astype(np.float32),
                'sample_rate': int(sample_rate),
                'speech_segments': meta.get('speech_segments', [])
            }

        except Exception as e:
            logger.exception(f"Error loading processed data: {e}")
            # Return empty but well-formed structure on failure
            return {'audio_data': np.array([]), 'sample_rate': 16000, 'speech_segments': []}
    
    def diarize_audio_sync(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int
    ) -> Dict[str, Any]:
        """Synchronous diarization for Celery tasks"""
        try:
            return self._diarize_sync(audio_data, sample_rate)
            
        except Exception as e:
            logger.exception(f"Error in synchronous diarization: {e}")
            raise
