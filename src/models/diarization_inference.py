"""
Diarization Inference
pyannote.audio integration for speaker diarization
"""
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import torchaudio

from src.config.app_config import app_config, model_config
from src.models.triton_client import TritonClient, TritonModelWrapper

logger = logging.getLogger(__name__)


class DiarizationInference:
    """pyannote.audio diarization model wrapper"""
    
    def __init__(self, triton_client: Optional[TritonClient] = None):
        self.pipeline = None
        self.embedding_model = None
        self.triton_client = triton_client
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.temp_dir = Path(app_config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize diarization models"""
        try:
            logger.info("Initializing pyannote.audio diarization pipeline")
            
            # Load pre-trained diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                model_config.pyannote_model,
                use_auth_token=getattr(model_config, 'huggingface_token', None)
            )
            
            # Move pipeline to appropriate device
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
            
            logger.info(f"Diarization pipeline initialized on {self.device}")
            
            # Initialize speaker embedding model if available
            try:
                from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
                self.embedding_model = PretrainedSpeakerEmbedding(
                    model_config.speaker_embedding_model,
                    device=torch.device(self.device)
                )
                logger.info("Speaker embedding model initialized")
                
            except Exception as e:
                logger.warning(f"Could not initialize speaker embedding model: {e}")
            
        except Exception as e:
            logger.exception(f"Error initializing diarization models: {e}")
            raise
    
    async def diarize(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            num_speakers: Known number of speakers (optional)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            Diarization results with segments and embeddings
        """
        try:
            # Run diarization in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._diarize_sync,
                audio_data,
                sample_rate,
                num_speakers,
                min_speakers,
                max_speakers
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in diarization: {e}")
            raise
    
    def _diarize_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict[str, Any]:
        """Synchronous diarization implementation"""
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(
                suffix='.wav', 
                dir=self.temp_dir, 
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Convert to tensor and save
                audio_tensor = torch.from_numpy(audio_data).float()
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                torchaudio.save(temp_path, audio_tensor, sample_rate)
                
                # Configure pipeline parameters
                if num_speakers:
                    # Set exact number of speakers (pyannote 3.x API)
                    self.pipeline.instantiate({
                        "num_speakers": int(num_speakers)
                    })
                else:
                    # Provide a reasonable range
                    self.pipeline.instantiate({
                        "min_speakers": int(min_speakers),
                        "max_speakers": int(max_speakers)
                    })
                
                # Run diarization
                diarization_result = self.pipeline(str(temp_path))
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
            
            # Extract results
            segments = []
            speaker_labels = set()
            
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                segment = {
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'duration': float(turn.duration),
                    'speaker': str(speaker),
                    'confidence': 1.0  # pyannote doesn't provide confidence by default
                }
                segments.append(segment)
                speaker_labels.add(str(speaker))
            
            # Calculate embeddings if model available
            embeddings = []
            if self.embedding_model:
                embeddings = self._extract_embeddings_sync(
                    audio_data, sample_rate, segments
                )
            
            result = {
                'segments': segments,
                'num_speakers': len(speaker_labels),
                'speaker_labels': list(speaker_labels),
                'embeddings': embeddings,
                'raw_annotation': diarization_result
            }
            
            logger.info(f"Diarization completed: {len(segments)} segments, "
                       f"{len(speaker_labels)} speakers")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in synchronous diarization: {e}")
            raise
    
    def _extract_embeddings_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Extract speaker embeddings for segments"""
        try:
            if not self.embedding_model:
                return []
            
            embeddings = []
            
            for segment in segments:
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]
                
                if len(segment_audio) < sample_rate * 0.1:  # Skip very short segments
                    embeddings.append(np.zeros(512))  # Default embedding dimension
                    continue
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(segment_audio).float()
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Create segment object for embedding extraction
                segment_obj = Segment(segment['start'], segment['end'])
                
                # Extract embedding
                with tempfile.NamedTemporaryFile(
                    suffix='.wav', 
                    dir=self.temp_dir, 
                    delete=False
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    
                    torchaudio.save(temp_path, audio_tensor, sample_rate)
                    
                    # Get embedding
                    embedding = self.embedding_model.crop(
                        str(temp_path), 
                        segment_obj
                    )
                    
                    embeddings.append(embedding.data.cpu().numpy().flatten())
                    
                    # Clean up
                    temp_path.unlink(missing_ok=True)
            
            return embeddings
            
        except Exception as e:
            logger.exception(f"Error extracting embeddings: {e}")
            # Return zero embeddings as fallback
            return [np.zeros(512) for _ in segments]
    
    async def rttm_diarize(
        self,
        audio_file_path: str,
        rttm_output_path: str,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform diarization and save in RTTM format
        
        Args:
            audio_file_path: Path to audio file
            rttm_output_path: Path to save RTTM output
            num_speakers: Number of speakers (optional)
            
        Returns:
            Diarization results
        """
        try:
            # Run diarization directly on file
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._rttm_diarize_sync,
                audio_file_path,
                rttm_output_path,
                num_speakers
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in RTTM diarization: {e}")
            raise
    
    def _rttm_diarize_sync(
        self,
        audio_file_path: str,
        rttm_output_path: str,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synchronous RTTM diarization"""
        try:
            # Configure pipeline
            if num_speakers:
                self.pipeline.instantiate({
                    "num_speakers": int(num_speakers)
                })
            
            # Run diarization
            diarization = self.pipeline(audio_file_path)
            
            # Save to RTTM format
            with open(rttm_output_path, 'w') as rttm_file:
                diarization.write_rttm(rttm_file)
            
            # Extract segments
            segments = []
            speaker_labels = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'duration': float(turn.duration),
                    'speaker': str(speaker)
                }
                segments.append(segment)
                speaker_labels.add(str(speaker))
            
            return {
                'segments': segments,
                'num_speakers': len(speaker_labels),
                'speaker_labels': list(speaker_labels),
                'rttm_path': rttm_output_path
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous RTTM diarization: {e}")
            raise
    
    async def segment_audio_by_speaker(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Segment audio by speaker
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            segments: Diarization segments
            
        Returns:
            Dictionary mapping speakers to their audio segments
        """
        try:
            speaker_audio = {}
            
            for segment in segments:
                speaker = segment['speaker']
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]
                
                if speaker not in speaker_audio:
                    speaker_audio[speaker] = []
                
                speaker_audio[speaker].append(segment_audio)
            
            return speaker_audio
            
        except Exception as e:
            logger.exception(f"Error segmenting audio by speaker: {e}")
            return {}
    
    async def merge_speaker_segments(
        self,
        segments: List[Dict[str, Any]],
        max_gap: float = 0.5,
        same_speaker_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Merge nearby segments from same speaker
        
        Args:
            segments: Input segments
            max_gap: Maximum gap to merge (seconds)
            same_speaker_only: Only merge same speaker segments
            
        Returns:
            Merged segments
        """
        try:
            if not segments:
                return segments
            
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x['start'])
            merged_segments = [sorted_segments[0]]
            
            for current in sorted_segments[1:]:
                last_merged = merged_segments[-1]
                
                # Check merge conditions
                gap = current['start'] - last_merged['end']
                same_speaker = current['speaker'] == last_merged['speaker']
                
                if gap <= max_gap and (not same_speaker_only or same_speaker):
                    # Merge segments
                    merged_segments[-1] = {
                        'start': last_merged['start'],
                        'end': current['end'],
                        'duration': current['end'] - last_merged['start'],
                        'speaker': current['speaker'] if same_speaker else 'mixed',
                        'confidence': (last_merged.get('confidence', 1.0) + 
                                     current.get('confidence', 1.0)) / 2
                    }
                else:
                    merged_segments.append(current)
            
            logger.info(f"Merged {len(segments)} segments to {len(merged_segments)}")
            return merged_segments
            
        except Exception as e:
            logger.exception(f"Error merging segments: {e}")
            return segments
    
    async def post_process_diarization(
        self,
        segments: List[Dict[str, Any]],
        min_segment_duration: float = 0.5,
        min_speaker_time: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Post-process diarization results
        
        Args:
            segments: Input segments
            min_segment_duration: Minimum segment duration
            min_speaker_time: Minimum total time per speaker
            
        Returns:
            Post-processed segments
        """
        try:
            # Filter by minimum segment duration
            filtered_segments = [
                s for s in segments 
                if s['duration'] >= min_segment_duration
            ]
            
            # Calculate total time per speaker
            speaker_times = {}
            for segment in filtered_segments:
                speaker = segment['speaker']
                speaker_times[speaker] = speaker_times.get(speaker, 0) + segment['duration']
            
            # Filter speakers with insufficient time
            valid_speakers = {
                speaker for speaker, total_time in speaker_times.items()
                if total_time >= min_speaker_time
            }
            
            # Keep only segments from valid speakers
            final_segments = [
                s for s in filtered_segments
                if s['speaker'] in valid_speakers
            ]
            
            logger.info(f"Post-processed: {len(segments)} -> {len(final_segments)} segments, "
                       f"{len(valid_speakers)} speakers")
            
            return final_segments
            
        except Exception as e:
            logger.exception(f"Error in post-processing: {e}")
            return segments
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get diarization model information"""
        return {
            'model_name': 'pyannote.audio',
            'model_version': model_config.pyannote_model,
            'device': str(self.device),
            'supported_formats': ['wav', 'mp3', 'flac', 'ogg'],
            'max_speakers': 10,
            'min_segment_duration': 0.5,
            'embedding_dimension': 512 if self.embedding_model else None
        }


class TritonDiarizationInference(TritonModelWrapper):
    """Triton-based diarization inference wrapper"""
    
    def __init__(self, triton_client: TritonClient):
        super().__init__(
            model_name=model_config.triton_diarization_model,
            triton_client=triton_client,
            input_names=['audio', 'sample_rate'],
            output_names=['segments', 'embeddings', 'num_speakers']
        )
    
    async def diarize(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run diarization using Triton server"""
        try:
            # Prepare inputs
            inputs = {
                'audio': audio_data.astype(np.float32),
                'sample_rate': np.array([sample_rate], dtype=np.int32)
            }
            
            # Add num_speakers if specified
            if num_speakers:
                inputs['num_speakers'] = np.array([num_speakers], dtype=np.int32)
                self.output_names.append('num_speakers')
            
            # Run inference
            results = await self.predict(inputs)
            
            # Process results
            segments_data = results['segments']
            embeddings_data = results['embeddings']
            num_speakers_detected = int(results['num_speakers'][0])
            
            # Convert to our format
            segments = []
            for i in range(len(segments_data)):
                segment = {
                    'start': float(segments_data[i][0]),
                    'end': float(segments_data[i][1]),
                    'speaker': f"speaker_{int(segments_data[i][2])}",
                    'confidence': float(segments_data[i][3]) if len(segments_data[i]) > 3 else 1.0
                }
                segment['duration'] = segment['end'] - segment['start']
                segments.append(segment)
            
            return {
                'segments': segments,
                'embeddings': embeddings_data,
                'num_speakers': num_speakers_detected,
                'speaker_labels': [f"speaker_{i}" for i in range(num_speakers_detected)]
            }
            
        except Exception as e:
            logger.exception(f"Error in Triton diarization: {e}")
            raise
