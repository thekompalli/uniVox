"""
Automatic Speech Recognition Service
Whisper-based multilingual ASR with Indian language support
"""
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import time
import soundfile as sf

from src.config.app_config import app_config, model_config
from src.repositories.audio_repository import AudioRepository
from src.utils.audio_utils import AudioUtils
from src.utils.eval_utils import EvalUtils

logger = logging.getLogger(__name__)


class ASRService:
    """Automatic Speech Recognition service using Whisper"""
    
    def __init__(self):
        backend = getattr(model_config, 'whisper_backend', 'openai') or 'openai'
        backend_normalized = str(backend).lower()
        if backend_normalized in {'fast', 'fast-whisper', 'faster-whisper'}:
            self.whisper_backend = 'fast'
        else:
            self.whisper_backend = 'openai'
        logger.info("Using Whisper backend: %s", self.whisper_backend)
        self.whisper_models = {}
        self.audio_repo = AudioRepository()
        self.audio_utils = AudioUtils()
        self.eval_utils = EvalUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.whisperx_aligners = {}
        self.fast_whisper_model = None
        self.fast_whisper_options = {}
        self.custom_models = {}  # Store custom models (e.g., Hindi2Hinglish)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ASR models"""
        try:
            if self.whisper_backend == 'fast':
                self._initialize_fast_whisper()
                return
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

            # Import whisper only when needed for OpenAI backend
            import whisper

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
    
    def _initialize_fast_whisper(self):
        """Initialize fast-whisper backend."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "fast-whisper backend requested but faster-whisper is not installed. Install faster-whisper to use this backend."
            ) from exc

        try:
            logger.info("Initializing Fast Whisper ASR model")

            cache_env = os.getenv('WHISPER_CACHE_DIR')
            cache_path = Path(cache_env) if cache_env else (app_config.models_dir / "whisper")
            cache_path.mkdir(parents=True, exist_ok=True)

            model_identifier = getattr(model_config, 'fast_whisper_model_path', '') or model_config.whisper_model
            model_identifier = (model_identifier or model_config.whisper_model).strip()

            device = getattr(model_config, 'fast_whisper_device', '') or ('cuda' if torch.cuda.is_available() else 'cpu')
            compute_type = getattr(model_config, 'fast_whisper_compute_type', '')
            if not compute_type:
                compute_type = 'float16' if str(device).startswith('cuda') else 'auto'

            model_path = Path(model_identifier)
            kwargs = {
                'device': device,
                'compute_type': compute_type,
            }
            if model_path.exists():
                model_arg = str(model_path)
            else:
                model_arg = model_identifier
                kwargs['download_root'] = str(cache_path)

            self.fast_whisper_model = WhisperModel(model_arg, **kwargs)
            self.fast_whisper_options = {
                'beam_size': getattr(model_config, 'whisper_beam_size', 5),
                'temperature': getattr(model_config, 'whisper_temperature', 0.0),
                'vad_filter': getattr(model_config, 'fast_whisper_vad_filter', False),
            }

            logger.info(
                "Fast Whisper model loaded from %s on %s (compute_type=%s)",
                model_arg,
                device,
                compute_type,
            )

            # Load custom models if configured
            self._initialize_custom_models()
        except Exception as exc:
            logger.exception("Error initializing fast-whisper model: %s", exc)
            raise

    def _initialize_custom_models(self):
        """Initialize custom models (e.g., Hindi2Hinglish)"""
        try:
            # Load Hindi2Hinglish model if configured
            if (getattr(model_config, 'use_custom_hindi_model', False) and
                getattr(model_config, 'hindi2hinglish_model_path', None)):

                hindi_model_path = model_config.hindi2hinglish_model_path
                device = getattr(model_config, 'fast_whisper_device', 'cuda')
                compute_type = getattr(model_config, 'fast_whisper_compute_type', 'float16')

                logger.info(f"Loading Hindi2Hinglish model from: {hindi_model_path}")

                from faster_whisper import WhisperModel
                self.custom_models['hindi2hinglish'] = WhisperModel(
                    hindi_model_path,
                    device=device,
                    compute_type=compute_type
                )

                logger.info("Hindi2Hinglish model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load custom models: {e}")
            # Don't raise - continue with default model

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
                        segment.get('speaker', f'speaker_{i}'),
                        segment.get('lid_confidence')
                    )
                    
                    if transcription and transcription['text'].strip():
                        result = {
                            'start': segment['start'],
                            'end': segment['end'],
                            'speaker': segment.get('speaker', f'speaker_{i}'),
                            'language': transcription.get('language', segment.get('language', 'unknown')),
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
        speaker: str,
        lid_confidence: Optional[float] = None
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
                speaker,
                lid_confidence
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error transcribing segment for speaker {speaker}: {e}")
            return None
    
    def _get_whisperx_aligner(
        self,
        language_code: Optional[str],
        device: torch.device
    ):
        """Return cached WhisperX alignment model if available"""
        lang_key = (language_code or 'en').lower()
        device_str = str(device)
        device_key = 'cuda' if device_str.startswith('cuda') else 'cpu'
        cache_key = (lang_key, device_key)
        if cache_key in self.whisperx_aligners:
            return self.whisperx_aligners[cache_key]
        try:
            import whisperx
            align_model, metadata = whisperx.load_align_model(
                language_code=lang_key,
                device=device_key
            )
            self.whisperx_aligners[cache_key] = (align_model, metadata)
            return self.whisperx_aligners[cache_key]
        except Exception as align_err:
            logger.debug(
                "WhisperX aligner unavailable for %s on %s: %s",
                language_code or 'en',
                device,
                align_err
            )
            return None

    def _transcribe_segment_sync(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
        language: str,
        speaker: str,
        lid_confidence: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Synchronous segment transcription"""
        try:
            if self.whisper_backend == 'fast':
                return self._transcribe_segment_fast(
                    segment_audio,
                    sample_rate,
                    language,
                    lid_confidence,
                )
            # Select appropriate model based on language
            model_key = self._select_model_for_language(language)
            model = self.whisper_models.get(model_key, self.whisper_models['multilingual'])
            
            # Create a temp file path but ensure handle is closed before unlink (Windows safe)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                tmp_path = temp_file.name

            # Write audio after closing the handle
            sf.write(tmp_path, segment_audio, sample_rate)

            # Transcribe with Whisper (force transcribe task, never translate)
            # Only force language when LID is confident; otherwise allow auto-detect
            threshold = getattr(model_config, 'language_confidence_threshold', 0.8)
            force_lang = (lid_confidence is not None and float(lid_confidence) >= float(threshold))
            code = self._whisper_language_code(language) if force_lang else None
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

            # If words are not present, try WhisperX alignment to obtain word timestamps
            try:
                transcript_text = result.get('text', '') or ''
                if (
                    not words
                    and 'segments' in result
                    and transcript_text
                    and any(ch.isalpha() for ch in transcript_text)
                ):
                    import whisperx  # optional dependency present in requirements
                    import torchaudio
                    audio_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0)
                    if sample_rate != 16000:
                        import torchaudio.transforms as T
                        audio_tensor = T.Resample(sample_rate, 16000)(audio_tensor)
                        sr_align = 16000
                    else:
                        sr_align = sample_rate
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpw:
                        wav_path = tmpw.name
                    torchaudio.save(wav_path, audio_tensor, sr_align)
                    try:
                        align_pair = self._get_whisperx_aligner(code or 'en', model.device)
                        if align_pair:
                            align_model, metadata = align_pair
                            audio_wav = whisperx.load_audio(wav_path)
                            segments_for_align = result.get('segments', [])
                            device_for_align = 'cuda' if str(model.device).startswith('cuda') else 'cpu'
                            aligned = whisperx.align(
                                segments_for_align,
                                align_model,
                                metadata,
                                audio_wav,
                                device_for_align,
                                return_char_alignments=False
                            )
                            aligned_words = []
                            for seg in aligned.get('segments', []):
                                for w in seg.get('words', []) or []:
                                    aligned_words.append({
                                        'word': (w.get('word') or '').strip(),
                                        'start': float(w.get('start', 0.0)),
                                        'end': float(w.get('end', 0.0)),
                                        'confidence': float(w.get('score', 0.8)),
                                    })
                            if aligned_words:
                                words = aligned_words
                        else:
                            logger.debug(
                                "WhisperX alignment unavailable for %s; using Whisper timestamps",
                                code or 'en'
                            )
                    finally:
                        try:
                            Path(wav_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    if transcript_text and not any(ch.isalpha() for ch in transcript_text):
                        logger.debug("WhisperX alignment skipped: no alphabetic content")
            except Exception as _align_err:
                logger.debug(f"WhisperX alignment skipped: {_align_err}")

            # Normalize Whisper language code to canonical name when available
            detected_code = result.get('language')
            code_map = {
                'en': 'english',
                'hi': 'hindi',
                'bn': 'bengali',
                'pa': 'punjabi',
                'ur': 'urdu',
                'ne': 'nepali',
                'gu': 'gujarati',
                'mr': 'marathi'
            }
            detected_lang = code_map.get(str(detected_code).lower(), language)

            return {
                'text': result.get('text', ''),
                'language': detected_lang,
                'confidence': self._calculate_transcription_confidence(result),
                'words': words
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous transcription: {e}")
            return None
    

    def _transcribe_segment_fast(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
        language: str,
        lid_confidence: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Transcribe segment using faster-whisper backend."""
        # Select appropriate model based on language
        model = self._select_fast_whisper_model(language)

        if model is None:
            logger.error('No suitable Fast Whisper model is available')
            return None
        try:
            threshold = getattr(model_config, 'language_confidence_threshold', 0.8)
            force_lang = lid_confidence is not None and float(lid_confidence) >= float(threshold)
            code = self._whisper_language_code(language) if force_lang else None

            audio = segment_audio.astype(np.float32)
            target_sr = 16000
            if sample_rate != target_sr:
                import librosa
                audio = librosa.resample(segment_audio, orig_sr=sample_rate, target_sr=target_sr)
            audio = np.asarray(audio, dtype=np.float32)

            transcribe_kwargs = {
                'language': code,
                'beam_size': self.fast_whisper_options.get('beam_size'),
                'temperature': self.fast_whisper_options.get('temperature'),
                'vad_filter': self.fast_whisper_options.get('vad_filter', False),
                'word_timestamps': True,
            }
            if transcribe_kwargs['language'] is None:
                transcribe_kwargs.pop('language')
            if transcribe_kwargs['beam_size'] is None:
                transcribe_kwargs.pop('beam_size')
            if transcribe_kwargs['temperature'] is None:
                transcribe_kwargs.pop('temperature')

            segments_iter, info = model.transcribe(
                audio,
                **transcribe_kwargs,
            )

            segments = list(segments_iter)
            text_parts = []
            whisper_segments: List[Dict[str, Any]] = []
            aggregated_words: List[Dict[str, Any]] = []

            for seg in segments:
                seg_text = (seg.text or '').strip()
                text_parts.append(seg_text)
                seg_dict: Dict[str, Any] = {
                    'start': float(seg.start or 0.0),
                    'end': float(seg.end or 0.0),
                    'text': seg_text,
                }
                if getattr(seg, 'avg_logprob', None) is not None:
                    seg_dict['avg_logprob'] = float(seg.avg_logprob)
                if getattr(seg, 'no_speech_prob', None) is not None:
                    seg_dict['no_speech_prob'] = float(seg.no_speech_prob)
                if getattr(seg, 'compression_ratio', None) is not None:
                    seg_dict['compression_ratio'] = float(seg.compression_ratio)

                word_entries: List[Dict[str, Any]] = []
                for word in getattr(seg, 'words', []) or []:
                    word_dict = {
                        'word': (word.word or '').strip(),
                        'start': float(word.start or 0.0) if word.start is not None else 0.0,
                        'end': float(word.end or 0.0) if word.end is not None else 0.0,
                        'probability': float(word.probability or 0.0),
                    }
                    word_entries.append(word_dict)
                    aggregated_words.append(word_dict)
                if word_entries:
                    seg_dict['words'] = word_entries
                whisper_segments.append(seg_dict)

            combined_text = ' '.join(part for part in text_parts if part).strip()
            detected_code = None
            if info is not None and getattr(info, 'language', None):
                detected_code = info.language
            elif code:
                detected_code = code

            whisper_result = {
                'text': combined_text,
                'language': detected_code,
                'segments': whisper_segments,
            }

            code_map = {
                'en': 'english',
                'hi': 'hindi',
                'bn': 'bengali',
                'pa': 'punjabi',
                'ur': 'urdu',
                'ne': 'nepali',
                'gu': 'gujarati',
                'mr': 'marathi',
            }
            detected_lang = code_map.get(str(detected_code).lower(), language) if detected_code else language

            confidence = self._calculate_transcription_confidence(whisper_result)

            return {
                'text': combined_text,
                'language': detected_lang,
                'confidence': confidence,
                'words': aggregated_words,
            }
        except Exception as exc:
            logger.exception('Error in fast-whisper transcription: %s', exc)
            return None

    def _select_fast_whisper_model(self, language: str):
        """Select the appropriate faster-whisper model based on language"""
        language = language.lower()

        # Use custom Hindi2Hinglish model for Hindi content
        if (language in ['hindi', 'hi'] and
            'hindi2hinglish' in self.custom_models):
            logger.debug(f"Using Hindi2Hinglish model for language: {language}")
            return self.custom_models['hindi2hinglish']

        # Use default model for other languages
        if self.fast_whisper_model is not None:
            logger.debug(f"Using default model for language: {language}")
            return self.fast_whisper_model

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
                    lid_confidence = float(dominant_language.get('confidence', 0.0))
                else:
                    language = 'auto'  # Auto-detect if no language info
                    lid_confidence = 0.0
                
                merged_segment = speaker_segment.copy()
                merged_segment['language'] = language
                merged_segment['lid_confidence'] = lid_confidence
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
