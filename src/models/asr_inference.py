"""
ASR Inference
Whisper model wrapper for automatic speech recognition
"""
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
import soundfile as sf

import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.config.app_config import app_config, model_config
from src.models.triton_client import TritonClient, TritonModelWrapper

logger = logging.getLogger(__name__)


class ASRInference:
    """Whisper ASR model wrapper"""
    
    def __init__(self, triton_client: Optional[TritonClient] = None):
        self.models = {}
        self.processors = {}
        self.triton_client = triton_client
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.temp_dir = Path(app_config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Whisper models"""
        try:
            logger.info("Initializing Whisper ASR models")
            
            # Load main multilingual model
            model_size = getattr(model_config, 'whisper_model', 'large-v3')
            self.models['multilingual'] = whisper.load_model(
                model_size,
                device=self.device
            )
            
            logger.info(f"Loaded Whisper {model_size} model on {self.device}")
            
            # Load language-specific models if available
            if hasattr(model_config, 'whisper_hindi_model'):
                try:
                    self.models['hindi'] = whisper.load_model(
                        model_config.whisper_hindi_model,
                        device=self.device
                    )
                    logger.info("Loaded Hindi-specific Whisper model")
                except Exception as e:
                    logger.warning(f"Failed to load Hindi model: {e}")
            
            if hasattr(model_config, 'whisper_bengali_model'):
                try:
                    self.models['bengali'] = whisper.load_model(
                        model_config.whisper_bengali_model,
                        device=self.device
                    )
                    logger.info("Loaded Bengali-specific Whisper model")
                except Exception as e:
                    logger.warning(f"Failed to load Bengali model: {e}")
            
            # Initialize HuggingFace models if specified
            if hasattr(model_config, 'whisper_hf_model'):
                try:
                    self.processors['hf'] = WhisperProcessor.from_pretrained(
                        model_config.whisper_hf_model
                    )
                    self.models['hf'] = WhisperForConditionalGeneration.from_pretrained(
                        model_config.whisper_hf_model
                    ).to(self.device)
                    logger.info("Loaded HuggingFace Whisper model")
                except Exception as e:
                    logger.warning(f"Failed to load HF model: {e}")
            
            logger.info(f"Initialized {len(self.models)} ASR models")
            
        except Exception as e:
            logger.exception(f"Error initializing ASR models: {e}")
            raise
    
    async def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            language: Language hint (optional)
            task: "transcribe" or "translate"
            word_timestamps: Return word-level timestamps
            model_preference: Preferred model to use
            
        Returns:
            Transcription results with text, language, and timestamps
        """
        try:
            # Select appropriate model
            model_key = self._select_model(language, model_preference)
            
            # Run transcription in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_data,
                sample_rate,
                language,
                task,
                word_timestamps,
                model_key
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in transcription: {e}")
            raise
    
    def _transcribe_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        task: str,
        word_timestamps: bool,
        model_key: str
    ) -> Dict[str, Any]:
        """Synchronous transcription implementation"""
        try:
            model = self.models[model_key]
            
            # Resample to 16kHz if needed (Whisper requirement)
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            # Use HuggingFace model if selected
            if model_key == 'hf':
                return self._transcribe_hf_sync(
                    audio_data, language, task, word_timestamps
                )
            
            # Create temporary file for Whisper
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                dir=self.temp_dir,
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                
                # Save audio to temporary file
                sf.write(temp_path, audio_data, 16000)
                
                # Prepare transcription options
                options = {
                    "task": task,
                    "word_timestamps": word_timestamps
                }
                
                if language and language != 'auto':
                    whisper_lang = self._get_whisper_language_code(language)
                    if whisper_lang:
                        options["language"] = whisper_lang
                
                # Run transcription
                result = model.transcribe(str(temp_path), **options)
                
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
            
            # Process results
            processed_result = self._process_whisper_result(result, model_key)
            
            return processed_result
            
        except Exception as e:
            logger.exception(f"Error in synchronous transcription: {e}")
            raise
    
    def _transcribe_hf_sync(
        self,
        audio_data: np.ndarray,
        language: Optional[str],
        task: str,
        word_timestamps: bool
    ) -> Dict[str, Any]:
        """Synchronous transcription using HuggingFace model"""
        try:
            processor = self.processors['hf']
            model = self.models['hf']
            
            # Process audio
            inputs = processor(
                audio_data,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                if language and language != 'auto':
                    # Force language if specified
                    language_token = processor.tokenizer.convert_tokens_to_ids(
                        f"<|{self._get_whisper_language_code(language)}|>"
                    )
                    forced_decoder_ids = [[1, language_token]]
                    
                    generated_ids = model.generate(
                        inputs["input_features"],
                        forced_decoder_ids=forced_decoder_ids,
                        return_timestamps=word_timestamps
                    )
                else:
                    generated_ids = model.generate(
                        inputs["input_features"],
                        return_timestamps=word_timestamps
                    )
            
            # Decode transcription
            transcription = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Extract language from generated tokens
            detected_language = self._extract_language_from_tokens(
                generated_ids[0], processor
            )
            
            return {
                'text': transcription.strip(),
                'language': detected_language or language or 'unknown',
                'segments': [],
                'words': [],
                'confidence': 0.9,  # HF model doesn't provide confidence
                'model_used': 'hf'
            }
            
        except Exception as e:
            logger.exception(f"Error in HF transcription: {e}")
            raise
    
    def _process_whisper_result(
        self,
        whisper_result: Dict[str, Any],
        model_key: str
    ) -> Dict[str, Any]:
        """Process Whisper result into our format"""
        try:
            # Extract basic information
            text = whisper_result.get('text', '').strip()
            language = whisper_result.get('language', 'unknown')
            
            # Process segments
            segments = []
            words = []
            
            if 'segments' in whisper_result:
                for segment in whisper_result['segments']:
                    processed_segment = {
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'text': segment.get('text', '').strip(),
                        'confidence': segment.get('avg_logprob', -1.0)
                    }
                    
                    # Convert log probability to confidence score
                    if processed_segment['confidence'] != -1.0:
                        processed_segment['confidence'] = min(1.0, np.exp(processed_segment['confidence']))
                    else:
                        processed_segment['confidence'] = 0.8
                    
                    segments.append(processed_segment)
                    
                    # Extract words if available
                    if 'words' in segment:
                        for word in segment['words']:
                            word_info = {
                                'word': word.get('word', '').strip(),
                                'start': word.get('start', 0.0),
                                'end': word.get('end', 0.0),
                                'confidence': min(1.0, word.get('probability', 0.8))
                            }
                            words.append(word_info)
            
            # Calculate overall confidence
            overall_confidence = 0.8
            if segments:
                overall_confidence = np.mean([s['confidence'] for s in segments])
            
            return {
                'text': text,
                'language': language,
                'segments': segments,
                'words': words,
                'confidence': float(overall_confidence),
                'model_used': model_key
            }
            
        except Exception as e:
            logger.exception(f"Error processing Whisper result: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': [],
                'words': [],
                'confidence': 0.0,
                'model_used': model_key
            }
    
    async def transcribe_batch(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int,
        languages: Optional[List[str]] = None,
        task: str = "transcribe"
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio segments
        
        Args:
            audio_segments: List of audio segments
            sample_rate: Sample rate
            languages: Language hints for each segment
            task: Transcription task
            
        Returns:
            List of transcription results
        """
        try:
            if languages is None:
                languages = [None] * len(audio_segments)
            
            # Process segments concurrently
            tasks = []
            for i, (segment, language) in enumerate(zip(audio_segments, languages)):
                task_coro = self.transcribe(
                    audio_data=segment,
                    sample_rate=sample_rate,
                    language=language,
                    task=task
                )
                tasks.append(task_coro)
            
            # Wait for all transcriptions to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error transcribing segment {i}: {result}")
                    processed_results.append({
                        'text': '',
                        'language': 'unknown',
                        'segments': [],
                        'words': [],
                        'confidence': 0.0,
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.exception(f"Error in batch transcription: {e}")
            raise
    
    async def detect_language(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        duration_limit: float = 30.0
    ) -> Dict[str, Any]:
        """
        Detect language of audio using Whisper
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            duration_limit: Maximum duration to analyze
            
        Returns:
            Language detection results
        """
        try:
            # Limit audio length for faster processing
            max_samples = int(duration_limit * sample_rate)
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            # Run language detection in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._detect_language_sync,
                audio_data,
                sample_rate
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in language detection: {e}")
            return {'language': 'unknown', 'confidence': 0.0}
    
    def _detect_language_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """Synchronous language detection"""
        try:
            model = self.models['multilingual']
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            # Use Whisper's language detection
            mel = whisper.log_mel_spectrogram(audio_data).to(model.device)
            _, probs = model.detect_language(mel)
            
            # Get top language
            detected_language = max(probs, key=probs.get)
            confidence = float(probs[detected_language])
            
            return {
                'language': detected_language,
                'confidence': confidence,
                'all_probabilities': {k: float(v) for k, v in probs.items()}
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous language detection: {e}")
            return {'language': 'unknown', 'confidence': 0.0}
    
    def _select_model(
        self,
        language: Optional[str],
        model_preference: Optional[str]
    ) -> str:
        """Select the best model for given language"""
        if model_preference and model_preference in self.models:
            return model_preference
        
        if language:
            # Language-specific models
            if language.lower() == 'hindi' and 'hindi' in self.models:
                return 'hindi'
            elif language.lower() == 'bengali' and 'bengali' in self.models:
                return 'bengali'
        
        # Default to multilingual model
        return 'multilingual'
    
    def _get_whisper_language_code(self, language: str) -> Optional[str]:
        """Convert language name to Whisper language code"""
        language_codes = {
            'english': 'en',
            'hindi': 'hi',
            'bengali': 'bn',
            'punjabi': 'pa',
            'nepali': 'ne',
            'dogri': 'hi',  # Use Hindi for Dogri
            'urdu': 'ur',
            'gujarati': 'gu',
            'marathi': 'mr',
            'tamil': 'ta',
            'telugu': 'te',
            'kannada': 'kn',
            'malayalam': 'ml',
            'assamese': 'as',
            'odia': 'or'
        }
        
        return language_codes.get(language.lower())
    
    def _extract_language_from_tokens(
        self,
        generated_ids: torch.Tensor,
        processor
    ) -> Optional[str]:
        """Extract language from generated token IDs"""
        try:
            # Convert first few tokens to check for language token
            tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids[:10])
            
            for token in tokens:
                if token.startswith('<|') and token.endswith('|>') and len(token) == 5:
                    return token[2:-2]  # Extract language code
            
            return None
            
        except Exception:
            return None
    
    async def get_transcription_confidence(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        reference_text: str,
        language: Optional[str] = None
    ) -> float:
        """
        Calculate transcription confidence by comparing with reference
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            reference_text: Reference transcription
            language: Language hint
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Transcribe audio
            result = await self.transcribe(
                audio_data=audio_data,
                sample_rate=sample_rate,
                language=language
            )
            
            transcribed_text = result['text'].lower().strip()
            reference_text = reference_text.lower().strip()
            
            # Calculate similarity (simplified)
            if not transcribed_text or not reference_text:
                return 0.0
            
            # Word-level similarity
            transcribed_words = set(transcribed_text.split())
            reference_words = set(reference_text.split())
            
            if not reference_words:
                return 0.0
            
            intersection = len(transcribed_words.intersection(reference_words))
            union = len(transcribed_words.union(reference_words))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Combine with model confidence
            model_confidence = result.get('confidence', 0.8)
            final_confidence = (jaccard_similarity + model_confidence) / 2
            
            return final_confidence
            
        except Exception as e:
            logger.exception(f"Error calculating transcription confidence: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ASR model information"""
        return {
            'models_loaded': list(self.models.keys()),
            'device': str(self.device),
            'supported_languages': [
                'english', 'hindi', 'bengali', 'punjabi', 'nepali',
                'urdu', 'gujarati', 'marathi', 'tamil', 'telugu',
                'kannada', 'malayalam', 'assamese', 'odia'
            ],
            'supported_tasks': ['transcribe', 'translate'],
            'max_audio_length': 30 * 60,  # 30 minutes
            'sample_rate': 16000
        }


class TritonASRInference(TritonModelWrapper):
    """Triton-based ASR inference wrapper"""
    
    def __init__(self, triton_client: TritonClient):
        super().__init__(
            model_name=model_config.triton_asr_model,
            triton_client=triton_client,
            input_names=['audio', 'sample_rate'],
            output_names=['transcription', 'language', 'confidence']
        )
    
    async def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run ASR using Triton server"""
        try:
            # Prepare inputs
            inputs = {
                'audio': audio_data.astype(np.float32),
                'sample_rate': np.array([sample_rate], dtype=np.int32)
            }
            
            if language:
                inputs['language'] = np.array([language], dtype='<U10')
                self.input_names.append('language')
            
            # Run inference
            results = await self.predict(inputs)
            
            # Process results
            transcription = results['transcription'][0].decode('utf-8')
            detected_language = results['language'][0].decode('utf-8')
            confidence = float(results['confidence'][0])
            
            return {
                'text': transcription,
                'language': detected_language,
                'confidence': confidence,
                'segments': [],
                'words': [],
                'model_used': 'triton'
            }
            
        except Exception as e:
            logger.exception(f"Error in Triton ASR: {e}")
            raise