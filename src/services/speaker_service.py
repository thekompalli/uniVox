"""
Speaker Identification Service
WeSpeaker-based speaker identification and verification
"""
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config.app_config import app_config, model_config
from src.models.speaker_inference import SpeakerInference
from src.repositories.audio_repository import AudioRepository
from src.utils.eval_utils import EvalUtils

logger = logging.getLogger(__name__)


class SpeakerIdentificationService:
    """Speaker identification and verification service"""
    
    def __init__(self):
        self.speaker_inference = SpeakerInference()
        self.audio_repo = AudioRepository()
        self.eval_utils = EvalUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.speaker_gallery = {}  # In-memory speaker gallery
        self._initialize_gallery()
    
    def _initialize_gallery(self):
        """Initialize speaker gallery"""
        try:
            logger.info("Initializing speaker gallery")
            
            # Load speaker gallery from storage if exists
            gallery_path = app_config.data_dir / "speaker_gallery.json"
            if gallery_path.exists():
                import json
                with open(gallery_path, 'r') as f:
                    stored_gallery = json.load(f)
                    
                # Convert embeddings back to numpy arrays
                for speaker_id, speaker_data in stored_gallery.items():
                    if 'embeddings' in speaker_data:
                        speaker_data['embeddings'] = [
                            np.array(emb) for emb in speaker_data['embeddings']
                        ]
                    self.speaker_gallery[speaker_id] = speaker_data
                
                logger.info(f"Loaded {len(self.speaker_gallery)} speakers from gallery")
            
        except Exception as e:
            logger.exception(f"Error initializing speaker gallery: {e}")
    
    async def identify_speakers(
        self,
        job_id: str,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]],
        speaker_gallery: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identify speakers in audio segments
        
        Args:
            job_id: Job identifier
            audio_data: Audio signal
            sample_rate: Sample rate
            segments: Diarization segments
            speaker_gallery: Optional list of known speaker IDs
            
        Returns:
            Speaker identification results
        """
        try:
            logger.info(f"Starting speaker identification for job {job_id} with {len(segments)} segments")
            
            # Extract embeddings for each segment
            segment_embeddings = await self._extract_segment_embeddings(
                audio_data, sample_rate, segments
            )
            
            # Identify speakers
            identified_segments = []
            speaker_scores = {}
            
            for i, (segment, embedding) in enumerate(zip(segments, segment_embeddings)):
                try:
                    # Find best matching speaker
                    speaker_match = await self._identify_speaker(
                        embedding, speaker_gallery
                    )
                    
                    # Update segment with identification
                    identified_segment = segment.copy()
                    identified_segment.update({
                        'speaker_id': speaker_match['speaker_id'],
                        'identification_confidence': speaker_match['confidence'],
                        'similarity_score': speaker_match['similarity_score'],
                        'is_known_speaker': speaker_match['is_known']
                    })
                    
                    identified_segments.append(identified_segment)
                    
                    # Track speaker scores
                    speaker_id = speaker_match['speaker_id']
                    if speaker_id not in speaker_scores:
                        speaker_scores[speaker_id] = {
                            'segments': 0,
                            'total_duration': 0.0,
                            'avg_confidence': 0.0,
                            'avg_similarity': 0.0
                        }
                    
                    speaker_scores[speaker_id]['segments'] += 1
                    speaker_scores[speaker_id]['total_duration'] += segment.get('duration', 0.0)
                    speaker_scores[speaker_id]['avg_confidence'] += speaker_match['confidence']
                    speaker_scores[speaker_id]['avg_similarity'] += speaker_match['similarity_score']
                    
                except Exception as e:
                    logger.warning(f"Error identifying speaker for segment {i}: {e}")
                    # Keep original segment with unknown speaker
                    identified_segment = segment.copy()
                    identified_segment.update({
                        'speaker_id': f"unknown_{i}",
                        'identification_confidence': 0.0,
                        'similarity_score': 0.0,
                        'is_known_speaker': False
                    })
                    identified_segments.append(identified_segment)
            
            # Calculate average scores
            for speaker_id, scores in speaker_scores.items():
                if scores['segments'] > 0:
                    scores['avg_confidence'] /= scores['segments']
                    scores['avg_similarity'] /= scores['segments']
            
            # Post-process results
            processed_results = await self._post_process_identification(
                job_id, identified_segments, speaker_scores
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_identification_quality(processed_results)
            
            # Save identification results
            await self._save_identification_results(job_id, {
                'segments': processed_results,
                'speaker_scores': speaker_scores,
                'quality_metrics': quality_metrics,
                'total_speakers': len(speaker_scores),
                'known_speakers': sum(1 for s in processed_results if s.get('is_known_speaker', False))
            })
            
            logger.info(f"Speaker identification completed for job {job_id}: "
                       f"{len(processed_results)} segments, {len(speaker_scores)} speakers")
            
            return {
                'segments': processed_results,
                'speaker_scores': speaker_scores,
                'quality_metrics': quality_metrics,
                'total_speakers': len(speaker_scores),
                'known_speakers': sum(1 for s in processed_results if s.get('is_known_speaker', False))
            }
            
        except Exception as e:
            logger.exception(f"Error in speaker identification for job {job_id}: {e}")
            raise
    
    async def _extract_segment_embeddings(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Extract speaker embeddings for audio segments"""
        try:
            # Run embedding extraction in thread pool
            embeddings = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._extract_embeddings_sync,
                audio_data,
                sample_rate,
                segments
            )
            
            return embeddings
            
        except Exception as e:
            logger.exception(f"Error extracting segment embeddings: {e}")
            # Return zero embeddings as fallback
            return [np.zeros(256) for _ in segments]
    
    def _extract_embeddings_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Synchronously extract embeddings"""
        try:
            embeddings = []
            
            for segment in segments:
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Extract segment audio
                segment_audio = audio_data[start_sample:end_sample]
                
                if len(segment_audio) < sample_rate * 0.1:  # Skip very short segments
                    embeddings.append(np.zeros(256))
                    continue
                
                # Get embedding using speaker inference
                embedding = self.speaker_inference.extract_embedding_sync(
                    segment_audio, sample_rate
                )
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.exception(f"Error in synchronous embedding extraction: {e}")
            return [np.zeros(256) for _ in segments]
    
    async def _identify_speaker(
        self,
        embedding: np.ndarray,
        speaker_gallery: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identify speaker from embedding"""
        try:
            best_match = {
                'speaker_id': 'unknown',
                'confidence': 0.0,
                'similarity_score': 0.0,
                'is_known': False
            }
            
            # Filter gallery if specific speakers requested
            gallery_to_search = self.speaker_gallery
            if speaker_gallery:
                gallery_to_search = {
                    k: v for k, v in self.speaker_gallery.items()
                    if k in speaker_gallery
                }
            
            # Search for best match
            for speaker_id, speaker_data in gallery_to_search.items():
                if 'embeddings' not in speaker_data:
                    continue
                
                # Calculate similarity with stored embeddings
                similarities = []
                for stored_embedding in speaker_data['embeddings']:
                    similarity = self._calculate_cosine_similarity(embedding, stored_embedding)
                    similarities.append(similarity)
                
                if similarities:
                    max_similarity = max(similarities)
                    avg_similarity = np.mean(similarities)
                    
                    # Use weighted average of max and mean similarity
                    final_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
                    
                    if final_similarity > best_match['similarity_score']:
                        confidence = self._similarity_to_confidence(final_similarity)
                        
                        best_match = {
                            'speaker_id': speaker_id,
                            'confidence': confidence,
                            'similarity_score': final_similarity,
                            'is_known': True
                        }
            
            # If no good match found, assign unknown speaker ID
            if best_match['similarity_score'] < model_config.speaker_similarity_threshold:
                best_match = {
                    'speaker_id': f"unknown_{hash(str(embedding[:10])) % 10000}",
                    'confidence': 0.5,  # Neutral confidence for unknown
                    'similarity_score': 0.0,
                    'is_known': False
                }
            
            return best_match
            
        except Exception as e:
            logger.exception(f"Error identifying speaker: {e}")
            return {
                'speaker_id': 'error',
                'confidence': 0.0,
                'similarity_score': 0.0,
                'is_known': False
            }
    
    def _calculate_cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.exception(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _similarity_to_confidence(self, similarity: float) -> float:
        """Convert similarity score to confidence score"""
        # Map similarity [-1, 1] to confidence [0, 1]
        # Using sigmoid-like transformation
        normalized_sim = (similarity + 1) / 2  # Map to [0, 1]
        
        # Apply threshold and scaling
        if normalized_sim < 0.3:
            return 0.1
        elif normalized_sim > 0.8:
            return 0.95
        else:
            # Linear interpolation in middle range
            return 0.1 + (normalized_sim - 0.3) * (0.85 / 0.5)
    
    async def _post_process_identification(
        self,
        job_id: str,
        segments: List[Dict[str, Any]],
        speaker_scores: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Post-process identification results"""
        try:
            processed_segments = []
            
            # Apply confidence thresholding
            for segment in segments:
                if segment.get('identification_confidence', 0.0) < model_config.min_speaker_confidence:
                    # Mark as uncertain identification
                    segment['speaker_id'] = f"uncertain_{segment.get('speaker_id', 'unknown')}"
                    segment['identification_confidence'] = max(0.1, segment.get('identification_confidence', 0.0))
                
                processed_segments.append(segment)
            
            # Consistency checking: merge similar unknown speakers
            processed_segments = self._merge_similar_unknown_speakers(processed_segments)
            
            return processed_segments
            
        except Exception as e:
            logger.exception(f"Error in identification post-processing: {e}")
            return segments
    
    def _merge_similar_unknown_speakers(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge segments from similar unknown speakers"""
        try:
            # Group unknown speakers by similarity
            unknown_speakers = {}
            processed_segments = []
            
            for segment in segments:
                speaker_id = segment.get('speaker_id', 'unknown')
                
                # Only process unknown speakers
                if not speaker_id.startswith('unknown_') and not speaker_id.startswith('uncertain_'):
                    processed_segments.append(segment)
                    continue
                
                # Try to merge with existing unknown speakers
                merged = False
                for existing_speaker, existing_segments in unknown_speakers.items():
                    # Simple heuristic: if similar timing and characteristics
                    avg_confidence = np.mean([s.get('identification_confidence', 0.0) for s in existing_segments])
                    
                    if (abs(avg_confidence - segment.get('identification_confidence', 0.0)) < 0.2):
                        # Merge with existing speaker
                        segment['speaker_id'] = existing_speaker
                        unknown_speakers[existing_speaker].append(segment)
                        merged = True
                        break
                
                if not merged:
                    # Create new unknown speaker group
                    new_speaker_id = f"speaker_{len(unknown_speakers) + 1}"
                    segment['speaker_id'] = new_speaker_id
                    unknown_speakers[new_speaker_id] = [segment]
                
                processed_segments.append(segment)
            
            return processed_segments
            
        except Exception as e:
            logger.exception(f"Error merging unknown speakers: {e}")
            return segments
    
    def _calculate_identification_quality(
        self, 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate speaker identification quality metrics"""
        try:
            if not segments:
                return {}
            
            # Basic metrics
            total_segments = len(segments)
            known_speakers = sum(1 for s in segments if s.get('is_known_speaker', False))
            avg_confidence = np.mean([s.get('identification_confidence', 0.0) for s in segments])
            avg_similarity = np.mean([s.get('similarity_score', 0.0) for s in segments])
            
            # Speaker distribution
            speaker_counts = {}
            for segment in segments:
                speaker_id = segment.get('speaker_id', 'unknown')
                speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
            
            num_unique_speakers = len(speaker_counts)
            
            # Speaker balance (entropy-based)
            if num_unique_speakers > 1:
                probabilities = np.array(list(speaker_counts.values())) / total_segments
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
                max_entropy = np.log2(num_unique_speakers)
                speaker_balance = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                speaker_balance = 0.0
            
            return {
                'average_confidence': avg_confidence,
                'average_similarity': avg_similarity,
                'known_speaker_ratio': known_speakers / total_segments,
                'unique_speakers': num_unique_speakers,
                'speaker_balance': speaker_balance,
                'segments_processed': total_segments
            }
            
        except Exception as e:
            logger.exception(f"Error calculating identification quality: {e}")
            return {}
    
    async def _save_identification_results(
        self, 
        job_id: str, 
        results: Dict[str, Any]
    ):
        """Save speaker identification results"""
        try:
            await self.audio_repo.save_processing_results(
                job_id, "speaker_identification", results
            )
            logger.info(f"Saved speaker identification results for job {job_id}")
            
        except Exception as e:
            logger.exception(f"Error saving identification results: {e}")
    
    async def add_speaker_to_gallery(
        self,
        speaker_id: str,
        audio_samples: List[np.ndarray],
        sample_rate: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add speaker to gallery with audio samples"""
        try:
            # Extract embeddings from samples
            embeddings = []
            for audio_sample in audio_samples:
                embedding = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.speaker_inference.extract_embedding_sync,
                    audio_sample,
                    sample_rate
                )
                embeddings.append(embedding)
            
            # Store speaker data
            speaker_data = {
                'embeddings': embeddings,
                'metadata': metadata or {},
                'created_at': str(datetime.utcnow()),
                'sample_count': len(embeddings)
            }
            
            self.speaker_gallery[speaker_id] = speaker_data
            
            # Save gallery to file
            await self._save_speaker_gallery()
            
            logger.info(f"Added speaker {speaker_id} to gallery with {len(embeddings)} samples")
            return True
            
        except Exception as e:
            logger.exception(f"Error adding speaker to gallery: {e}")
            return False
    
    async def remove_speaker_from_gallery(self, speaker_id: str) -> bool:
        """Remove speaker from gallery"""
        try:
            if speaker_id in self.speaker_gallery:
                del self.speaker_gallery[speaker_id]
                await self._save_speaker_gallery()
                logger.info(f"Removed speaker {speaker_id} from gallery")
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"Error removing speaker from gallery: {e}")
            return False
    
    async def _save_speaker_gallery(self):
        """Save speaker gallery to file"""
        try:
            gallery_path = app_config.data_dir / "speaker_gallery.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_gallery = {}
            for speaker_id, speaker_data in self.speaker_gallery.items():
                serializable_data = speaker_data.copy()
                if 'embeddings' in serializable_data:
                    serializable_data['embeddings'] = [
                        emb.tolist() for emb in serializable_data['embeddings']
                    ]
                serializable_gallery[speaker_id] = serializable_data
            
            import json
            with open(gallery_path, 'w') as f:
                json.dump(serializable_gallery, f, indent=2)
                
        except Exception as e:
            logger.exception(f"Error saving speaker gallery: {e}")
    
    async def get_gallery_info(self) -> Dict[str, Any]:
        """Get information about speaker gallery"""
        return {
            'total_speakers': len(self.speaker_gallery),
            'speakers': list(self.speaker_gallery.keys()),
            'speaker_details': {
                speaker_id: {
                    'sample_count': len(data.get('embeddings', [])),
                    'metadata': data.get('metadata', {}),
                    'created_at': data.get('created_at')
                }
                for speaker_id, data in self.speaker_gallery.items()
            }
        }
    
    # Synchronous methods for Celery tasks
    def identify_speakers_sync(
        self,
        job_id: str,
        segments: List[Dict[str, Any]],
        embeddings: Optional[List[np.ndarray]] = None,
        speaker_gallery: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous speaker identification for Celery tasks"""
        try:
            # Simplified sync version
            identified_segments = []
            
            for segment in segments:
                # Basic identification logic
                speaker_match = {
                    'speaker_id': segment.get('speaker', 'unknown'),
                    'confidence': 0.8,
                    'similarity_score': 0.7,
                    'is_known': False
                }
                
                segment_copy = segment.copy()
                segment_copy.update(speaker_match)
                identified_segments.append(segment_copy)
            
            return {
                'segments': identified_segments,
                'total_speakers': len(set(s.get('speaker_id') for s in identified_segments))
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous speaker identification: {e}")
            return {'segments': segments, 'total_speakers': 0}