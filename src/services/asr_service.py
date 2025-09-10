"""
Automatic Speech Recognition Service
Whisper-based multilingual ASR with Indian language support
"""
import logging
import numpy as np
import torch
import whisper
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import time
import soundfile as sf

from src.config.app_config import app_config, model_config
from src.models.asr_inference import ASRInference
from src.repositories.audio_repository import AudioRepository
from src.utils.audio_utils import AudioUtils
from src.utils.eval_utils import EvalUtils

logger = logging.getLogger(__name__)


class ASRService:
    """Automatic Speech Recognition service using Whisper"""
    
    def __init__(self):
        self.whisper_models = {}
        self.asr_inference = ASRInference()
        self.audio_repo = AudioRepository()
        self.audio_utils = AudioUtils()
        self.eval_utils = EvalUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ASR models"""
        try:
            logger.info("Initializing ASR models")

            # Decide device and cache directory for Whisper weights
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cache_dir = os.getenv("WHISPER_CACHE_DIR")
            if cache_dir:
                cache_path = Path(cache_dir)
            else:
                cache_path = app_config.models_dir / "whisper"
            cache_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using Whisper cache at: {cache_path}")

            # Load main Whisper model (multilingual)
            self.whisper_models['multilingual'] = whisper.load_model(
                model_config.whisper_model,
                device=device,
                download_root=str(cache_path)
            )

            # Load language-specific models if available
            if hasattr(model_config, 'whisper_hindi_model'):
                self.whisper_models['hindi'] = whisper.load_model(
                    model_config.whisper_hindi_model,
                    device=device,
                    download_root=str(cache_path)
                )

            if hasattr(model_config, 'whisper_bengali_model'):
                self.whisper_models['bengali'] = whisper.load_model(
                    model_config.whisper_bengali_model,
                    device=device,
                    download_root=str(cache_path)
                )

            logger.info(f"Initialized {len(self.whisper_models)} ASR models")

        except Exception as e:
            logger.exception(f"Error initializing ASR models: {e}")
            raise
    
    async def transcribe_segments(
        self,
        job_id: str,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]],
        language_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transcribe audio segments with speaker and language information
        
        Args:
            job_id: Job identifier
            audio_data: Audio signal
            sample_rate: Sample rate
            segments: Speaker diarization segments
            language_segments: Language identification results
            
        Returns:
            Transcription results with timestamps and speaker labels
        """
        try:
            logger.info(f"Starting ASR for job {job_id} with {len(segments)} segments")
            
            # Merge speaker and language information
            enriched_segments = self._merge_speaker_language_segments(
                segments, language_segments
            )

            # Optional: limit processing time window for faster local tests
            limit_seconds = getattr(app_config, 'asr_quick_test_seconds', None)
            if limit_seconds and isinstance(limit_seconds, (int, float)) and limit_seconds > 0:
                limited = []
                for seg in enriched_segments:
                    if seg.get('start', 0) >= limit_seconds:
                        break
                    if seg.get('end', 0) > limit_seconds:
                        seg = {**seg, 'end': float(limit_seconds)}
                    limited.append(seg)
                enriched_segments = limited
            
            # Transcribe each segment
            transcription_results = []
            
            for i, segment in enumerate(enriched_segments):
                try:
                    # Extract segment audio
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    # Skip very short segments
                    if len(segment_audio) < sample_rate * 0.1:
                        continue
                    
                    # Transcribe segment
                    transcription = await self._transcribe_segment(
                        segment_audio,
                        sample_rate,
                        segment.get('language', 'auto'),
                        segment.get('speaker', f'speaker_{i}')
                    )
                    
                    if transcription and transcription['text'].strip():
                        result = {
                            'start': segment['start'],
                            'end': segment['end'],
                            'speaker': segment.get('speaker', f'speaker_{i}'),
                            'language': segment.get('language', 'unknown'),
                            'text': transcription['text'].strip(),
                            'confidence': transcription.get('confidence', 0.8),
                            'words': transcription.get('words', [])
                        }
                        transcription_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error transcribing segment {i}: {e}")
                    continue
            
            # Post-process transcriptions
            processed_results = await self._post_process_transcriptions(
                job_id, transcription_results
            )
            
            # Calculate ASR quality metrics
            quality_metrics = self._calculate_asr_quality(processed_results)
            
            # Save ASR results
            await self._save_asr_results(job_id, {
                'transcription_segments': processed_results,
                'quality_metrics': quality_metrics,
                'total_segments': len(processed_results),
                'languages_detected': list(set(r['language'] for r in processed_results))
            })
            
            logger.info(f"ASR completed for job {job_id}: {len(processed_results)} transcribed segments")
            
            return {
                'segments': processed_results,
                'quality_metrics': quality_metrics,
                'total_segments': len(processed_results),
                'languages_detected': list(set(r['language'] for r in processed_results))
            }
            
        except Exception as e:
            logger.exception(f"Error in ASR for job {job_id}: {e}")
            raise
    
    async def _transcribe_segment(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
        language: str,
        speaker: str
    ) -> Optional[Dict[str, Any]]:
        """Transcribe a single audio segment"""
        try:
            # Run transcription in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._transcribe_segment_sync,
                segment_audio,
                sample_rate,
                language,
                speaker
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error transcribing segment for speaker {speaker}: {e}")
            return None
    
    def _transcribe_segment_sync(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
        language: str,
        speaker: str
    ) -> Optional[Dict[str, Any]]:
        """Synchronous segment transcription"""
        try:
            # Select appropriate model based on language
            model_key = self._select_model_for_language(language)
            model = self.whisper_models.get(model_key, self.whisper_models['multilingual'])
            
            # Create a temp file path but ensure handle is closed before unlink (Windows safe)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                tmp_path = temp_file.name

            # Write audio after closing the handle
            sf.write(tmp_path, segment_audio, sample_rate)

            # Transcribe with Whisper (force transcribe task, never translate)
            code = self._whisper_language_code(language)
            if code is None:
                # Let Whisper auto-detect language
                result = model.transcribe(tmp_path, task='transcribe')
            else:
                result = model.transcribe(
                    tmp_path,
                    language=code,
                    task='transcribe'
                )

            # Clean up temp file with small retry to avoid Windows file-lock timing
            for _ in range(3):
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.1)

            # Extract word-level timestamps if available
            words = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word in segment['words']:
                            words.append({
                                'word': word.get('word', '').strip(),
                                'start': word.get('start', 0),
                                'end': word.get('end', 0),
                                'confidence': word.get('probability', 0.8)
                            })

            return {
                'text': result['text'],
                'language': result.get('language', language),
                'confidence': self._calculate_transcription_confidence(result),
                'words': words
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous transcription: {e}")
            return None
    
    def _select_model_for_language(self, language: str) -> str:
        """Select the best model for a given language"""
        language_model_mapping = {
            'hindi': 'hindi',
            'bengali': 'bengali',
            'punjabi': 'multilingual',  # Use multilingual for Punjabi
            'english': 'multilingual',
            'auto': 'multilingual'
        }
        
        selected = language_model_mapping.get(language.lower(), 'multilingual')
        
        # Fallback to multilingual if specific model not available
        if selected not in self.whisper_models:
            selected = 'multilingual'
        
        return selected
    
    def _whisper_language_code(self, language: str) -> Optional[str]:
        """Convert a detected language name to Whisper's language code.
        Returns None to allow auto-detection when uncertain/unknown.
        """
        try:
            if not language:
                return None
            lang = language.lower()
            # If LID marked it uncertain, let Whisper auto-detect
            if lang.startswith('uncertain_'):
                return None
            # Normalize common English variants
            if 'english' in lang:
                return 'en'
            mapping = {
                'english': 'en',
                'hindi': 'hi',
                'bengali': 'bn',
                'punjabi': 'pa',
                'nepali': 'ne',
                'dogri': 'hi',  # Use Hindi for Dogri as fallback
                'auto': None,
                'unknown': None
            }
            return mapping.get(lang, None)
        except Exception:
            return None
    
    def _merge_speaker_language_segments(
        self,
        speaker_segments: List[Dict[str, Any]],
        language_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge speaker and language information into unified segments"""
        try:
            merged_segments = []
            
            for speaker_segment in speaker_segments:
                # Find overlapping language segments
                overlapping_languages = []
                
                for lang_segment in language_segments:
                    # Check for temporal overlap
                    if (speaker_segment['start'] < lang_segment['end'] and
                        speaker_segment['end'] > lang_segment['start']):
                        
                        overlap_duration = min(speaker_segment['end'], lang_segment['end']) - \
                                         max(speaker_segment['start'], lang_segment['start'])
                        
                        overlapping_languages.append({
                            'language': lang_segment.get('language', 'unknown'),
                            'confidence': lang_segment.get('confidence', 0.5),
                            'overlap_duration': overlap_duration
                        })
                
                # Select dominant language
                if overlapping_languages:
                    # Choose language with highest confidence * overlap duration
                    dominant_language = max(
                        overlapping_languages,
                        key=lambda x: x['confidence'] * x['overlap_duration']
                    )
                    language = dominant_language['language']
                else:
                    language = 'auto'  # Auto-detect if no language info
                
                merged_segment = speaker_segment.copy()
                merged_segment['language'] = language
                merged_segments.append(merged_segment)
            
            return merged_segments
            
        except Exception as e:
            logger.exception(f"Error merging speaker and language segments: {e}")
            # Return speaker segments with auto language detection
            for segment in speaker_segments:
                segment['language'] = 'auto'
            return speaker_segments
    
    def _calculate_transcription_confidence(self, whisper_result: Dict[str, Any]) -> float:
        """Calculate overall confidence for transcription"""
        try:
            # Use word-level probabilities if available
            if 'segments' in whisper_result:
                word_probs = []
                for segment in whisper_result['segments']:
                    if 'words' in segment:
                        for word in segment['words']:
                            if 'probability' in word:
                                word_probs.append(word['probability'])
                
                if word_probs:
                    return np.mean(word_probs)
            
            # Use segment-level avg_logprob as fallback
            if 'segments' in whisper_result:
                segment_probs = []
                for segment in whisper_result['segments']:
                    if 'avg_logprob' in segment:
                        # Convert log probability to probability (approximate)
                        prob = np.exp(segment['avg_logprob'])
                        segment_probs.append(min(1.0, prob))
                
                if segment_probs:
                    return np.mean(segment_probs)
            
            # Default confidence
            return 0.8
            
        except Exception:
            return 0.8
    
    async def _post_process_transcriptions(
        self,
        job_id: str,
        transcriptions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Post-process transcription results"""
        try:
            processed = []
            
            for transcription in transcriptions:
                # Clean up text
                cleaned_text = self._clean_transcription_text(transcription['text'])
                
                # Apply language-specific post-processing
                if transcription['language'] in ['hindi', 'bengali', 'punjabi']:
                    cleaned_text = self._post_process_indic_text(cleaned_text, transcription['language'])
                
                # Filter out very low confidence transcriptions
                if transcription['confidence'] < 0.3:
                    logger.debug(f"Filtering low confidence transcription: {cleaned_text[:50]}...")
                    continue
                
                processed_transcription = transcription.copy()
                processed_transcription['text'] = cleaned_text
                processed.append(processed_transcription)
            
            return processed
            
        except Exception as e:
            logger.exception(f"Error in transcription post-processing: {e}")
            return transcriptions
    
    def _clean_transcription_text(self, text: str) -> str:
        """Clean up transcription text"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove common ASR artifacts
            artifacts = ['[MUSIC]', '[NOISE]', '[INAUDIBLE]', '[SILENCE]']
            for artifact in artifacts:
                text = text.replace(artifact, '')
            
            # Basic punctuation normalization
            text = text.replace(' .', '.').replace(' ,', ',')
            text = text.replace(' ?', '?').replace(' !', '!')
            
            return text.strip()
            
        except Exception:
            return text
    
    def _post_process_indic_text(self, text: str, language: str) -> str:
        """Apply language-specific post-processing for Indic languages"""
        try:
            # This would contain language-specific rules
            # For now, just basic cleanup
            
            # Remove Roman transliteration artifacts if present
            # (This would be more sophisticated in practice)
            
            return text
            
        except Exception:
            return text
    
    def _calculate_asr_quality(
        self, 
        transcriptions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate ASR quality metrics"""
        try:
            if not transcriptions:
                return {}
            
            # Average confidence
            avg_confidence = np.mean([t['confidence'] for t in transcriptions])
            
            # Total transcribed duration
            total_duration = sum(t['end'] - t['start'] for t in transcriptions)
            
            # Average segment length
            avg_segment_duration = np.mean([t['end'] - t['start'] for t in transcriptions])
            
            # Word statistics
            total_words = 0
            total_characters = 0
            
            for transcription in transcriptions:
                words = len(transcription['text'].split())
                total_words += words
                total_characters += len(transcription['text'])
            
            # Speaking rate (words per minute)
            speaking_rate = (total_words / total_duration * 60) if total_duration > 0 else 0
            
            return {
                'average_confidence': avg_confidence,
                'total_transcribed_duration': total_duration,
                'average_segment_duration': avg_segment_duration,
                'total_words': total_words,
                'total_characters': total_characters,
                'speaking_rate_wpm': speaking_rate,
                'num_transcribed_segments': len(transcriptions)
            }
            
        except Exception as e:
            logger.exception(f"Error calculating ASR quality: {e}")
            return {}
    
    async def _save_asr_results(self, job_id: str, results: Dict[str, Any]):
        """Save ASR results"""
        try:
            await self.audio_repo.save_processing_results(
                job_id, "asr", results
            )
            logger.info(f"Saved ASR results for job {job_id}")
            
        except Exception as e:
            logger.exception(f"Error saving ASR results: {e}")
    
    # Synchronous methods for Celery tasks
    def load_diarization_results_sync(self, job_id: str) -> Dict[str, Any]:
        """Load diarization results synchronously from storage."""
        import json
        try:
            results_path = app_config.data_dir / "audio" / job_id / "diarization_results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {'segments': [], 'num_speakers': 0}
        except Exception as e:
            logger.exception(f"Error loading diarization results: {e}")
            return {'segments': [], 'num_speakers': 0}
    
    def load_language_results_sync(self, job_id: str) -> Dict[str, Any]:
        """Load language identification results synchronously from storage."""
        import json
        try:
            results_path = app_config.data_dir / "audio" / job_id / "language_identification_results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {'language_segments': []}
        except Exception as e:
            logger.exception(f"Error loading language results: {e}")
            return {'language_segments': []}
    
    def transcribe_segments_sync(
        self,
        job_id: str,
        segments: List[Dict[str, Any]],
        language_segments: List[Dict[str, Any]],
        expected_languages: List[str]
    ) -> Dict[str, Any]:
        """Synchronous wrapper to run ASR on stored processed audio."""
        import asyncio
        import json
        import soundfile as sf
        import numpy as np
        try:
            # Load processed audio metadata
            meta_path = app_config.data_dir / "audio" / job_id / "preprocessing_results.json"
            if not meta_path.exists():
                logger.warning(f"No preprocessing results for ASR job {job_id}")
                return {'segments': [], 'quality_metrics': {}}

            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            processed_path = meta.get('processed_path')
            if not processed_path:
                logger.warning(f"No processed_path in preprocessing results for {job_id}")
                return {'segments': [], 'quality_metrics': {}}

            audio_data, sample_rate = sf.read(processed_path)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)

            # Run the async ASR method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.transcribe_segments(
                        job_id=job_id,
                        audio_data=audio_data,
                        sample_rate=int(sample_rate),
                        segments=segments,
                        language_segments=language_segments
                    )
                )
            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Error in synchronous transcription: {e}")
            return {'segments': [], 'quality_metrics': {}}
