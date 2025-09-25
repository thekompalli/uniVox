"""
Neural Machine Translation Service
IndicTrans2 and NLLB-based translation for Indian languages
"""
import logging
import os
from pathlib import Path
import torch
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

from src.config.app_config import app_config, model_config
from src.repositories.audio_repository import AudioRepository
from src.utils.eval_utils import EvalUtils

logger = logging.getLogger(__name__)


class TranslationService:
    """Neural Machine Translation service for Indian languages"""
    
    def __init__(self):
        self.translation_models = {}
        self.tokenizers = {}
        self.pipeline_specs = {
            ('hindi', 'english'): 'Helsinki-NLP/opus-mt-hi-en',
            ('english', 'hindi'): 'Helsinki-NLP/opus-mt-en-hi',
            ('punjabi', 'english'): 'Helsinki-NLP/opus-mt-pa-en',
            ('english', 'punjabi'): 'Helsinki-NLP/opus-mt-en-pa',
        }
        self.pipelines = {}
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.audio_repo = AudioRepository()
        self.eval_utils = EvalUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize translation models"""
        try:
            logger.info("Initializing translation models")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Set and use a deterministic cache for HF models
            cache_dir = os.getenv("HF_CACHE_DIR") or os.getenv("TRANSFORMERS_CACHE") or str(app_config.models_dir / "huggingface")
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_path))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_path))

            # Prefer local model directories if present (offline-friendly)
            local_indic_en = app_config.models_dir / "indictrans2" / "indic_en_1b"
            local_en_indic = app_config.models_dir / "indictrans2" / "en_indic_1b"
            local_nllb = app_config.models_dir / "nllb" / "distilled_600m"
            indic_en_src = str(local_indic_en) if local_indic_en.exists() else (getattr(model_config, 'indictrans_indic_en_model', None) or 'ai4bharat/indictrans2-indic-en-1B')
            en_indic_src = str(local_en_indic) if local_en_indic.exists() else getattr(model_config, 'indictrans_model', None)
            nllb_src = str(local_nllb) if local_nllb.exists() else model_config.nllb_model

            # Initialize IndicTrans2 for Indian languages (both directions if available)
            if en_indic_src:
                try:
                    self.tokenizers['indictrans2_en_indic'] = AutoTokenizer.from_pretrained(
                        en_indic_src,
                        trust_remote_code=True,
                        cache_dir=str(cache_path)
                    )
                    self.translation_models['indictrans2_en_indic'] = AutoModelForSeq2SeqLM.from_pretrained(
                        en_indic_src,
                        trust_remote_code=True,
                        cache_dir=str(cache_path)
                    ).to(device)
                    logger.info(f"IndicTrans2 en->indic model loaded from {'local' if local_en_indic.exists() else 'hub'}")
                except Exception as e:
                    logger.warning(f"Failed to load IndicTrans2 en->indic: {e}")
            if indic_en_src:
                try:
                    self.tokenizers['indictrans2_indic_en'] = AutoTokenizer.from_pretrained(
                        indic_en_src,
                        trust_remote_code=True,
                        cache_dir=str(cache_path)
                    )
                    self.translation_models['indictrans2_indic_en'] = AutoModelForSeq2SeqLM.from_pretrained(
                        indic_en_src,
                        trust_remote_code=True,
                        cache_dir=str(cache_path)
                    ).to(device)
                    logger.info(f"IndicTrans2 indic->en model loaded from {'local' if local_indic_en.exists() else 'hub'}")
                except Exception as e:
                    logger.warning(f"Failed to load IndicTrans2 indic->en: {e}")
            
            # Initialize NLLB (tokenizer + model) for robust offline translation
            try:
                self.nllb_tokenizer = AutoTokenizer.from_pretrained(
                    nllb_src,
                    cache_dir=str(cache_path)
                )
                self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
                    nllb_src,
                    cache_dir=str(cache_path)
                ).to(device)
                logger.info(f"NLLB model loaded successfully from {'local' if local_nllb.exists() else 'hub'}")
            except Exception as e:
                logger.warning(f"Failed to load NLLB: {e}")
            
            logger.info(f"Translation service initialized with {len(self.translation_models) + len(self.pipelines)} models")
            
        except Exception as e:
            logger.exception(f"Error initializing translation models: {e}")
            raise
    
    async def translate_segments(
        self,
        job_id: str,
        transcription_segments: List[Dict[str, Any]],
        target_language: str = "english"
    ) -> Dict[str, Any]:
        """
        Translate transcribed segments to target language
        
        Args:
            job_id: Job identifier
            transcription_segments: ASR output segments
            target_language: Target language for translation
            
        Returns:
            Translation results with timestamps and quality metrics
        """
        try:
            logger.info(f"Starting translation for job {job_id} with {len(transcription_segments)} segments")
            
            translation_results = []
            
            for segment in transcription_segments:
                try:
                    source_language_raw = segment.get('language', 'unknown')
                    source_text = segment.get('text', '').strip()
                    
                    if not source_text:
                        continue
                    
                    # Infer source language more robustly using script detection + normalization
                    script = self._detect_script(source_text)
                    effective_source = self._infer_source_language(source_language_raw, script)
                    norm_target = self._normalize_language(target_language)

                    # Debug logging for language detection
                    logger.debug(f"Text: '{source_text[:50]}...' | Raw lang: {source_language_raw} | Script: {script} | Effective: {effective_source} | Target: {norm_target}")

                    if effective_source == norm_target:
                        script_overrides = {
                            'devanagari': 'hindi',
                            'arabic': 'urdu',
                            'gurmukhi': 'punjabi',
                            'bengali': 'bengali'
                        }
                        override_lang = script_overrides.get(script)
                        if override_lang and override_lang != norm_target:
                            effective_source = override_lang
                        else:
                            # Direct Unicode range checks to catch mixed-script misdetections
                            if any('\u0600' <= ch <= '\u06FF' for ch in source_text):
                                override_lang = 'urdu'
                            elif any('\u0900' <= ch <= '\u097F' for ch in source_text):
                                override_lang = 'hindi'
                            elif any('\u0A00' <= ch <= '\u0A7F' for ch in source_text):
                                override_lang = 'punjabi'
                            elif any('\u0980' <= ch <= '\u09FF' for ch in source_text):
                                override_lang = 'bengali'
                            else:
                                override_lang = None
                            if override_lang and override_lang != norm_target:
                                effective_source = override_lang
                    
                    # Skip if already in target language - but be more strict about what constitutes "same language"
                    skip_translation = False
                    if effective_source == norm_target:
                        if norm_target == 'english' and script == 'latin':
                            # Only skip if source is definitely English (detected as English AND uses Latin script)
                            skip_translation = True
                        elif norm_target != 'english':
                            # For non-English targets, also check if source matches target
                            skip_translation = True

                    if skip_translation:
                        logger.debug(f"Skipping translation - same language detected: {effective_source} -> {norm_target}")
                        translation_result = {
                            'start': segment['start'],
                            'end': segment['end'],
                            'speaker': segment.get('speaker', 'unknown'),
                            'source_language': effective_source,
                            'target_language': norm_target,
                            'source_text': source_text,
                            'translated_text': source_text,
                            'translation_confidence': 1.0,
                            'translation_method': 'no_translation'
                        }
                    else:
                        # Perform translation
                        logger.debug(f"Attempting translation: {effective_source} -> {norm_target}")
                        translation = await self._translate_text(
                            source_text,
                            effective_source,
                            norm_target
                        )

                        # If translation failed but returned original text, mark it appropriately
                        if translation['translated_text'] == source_text and translation['method'] in ['error_fallback', 'no_translation']:
                            logger.warning(f"Translation failed for {effective_source} -> {norm_target}: {source_text[:50]}...")

                        translation_result = {
                            'start': segment['start'],
                            'end': segment['end'],
                            'speaker': segment.get('speaker', 'unknown'),
                            'source_language': effective_source,
                            'target_language': norm_target,
                            'source_text': source_text,
                            'translated_text': translation['translated_text'],
                            'translation_confidence': translation['confidence'],
                            'translation_method': translation['method']
                        }
                    
                    translation_results.append(translation_result)
                    
                except Exception as e:
                    logger.warning(f"Error translating segment: {e}")
                    continue
            
            # Post-process translations
            processed_results = await self._post_process_translations(
                job_id, translation_results
            )
            
            # Calculate translation quality metrics
            quality_metrics = self._calculate_translation_quality(processed_results)
            
            # Save translation results
            await self._save_translation_results(job_id, {
                # Use canonical key expected by downstream (and keep legacy key for compatibility)
                'segments': processed_results,
                'translation_segments': processed_results,
                'quality_metrics': quality_metrics,
                'total_segments': len(processed_results),
                'languages_translated': list(set(r['source_language'] for r in processed_results))
            })
            
            logger.info(f"Translation completed for job {job_id}: {len(processed_results)} segments")
            
            return {
                'segments': processed_results,
                'quality_metrics': quality_metrics,
                'total_segments': len(processed_results),
                'languages_translated': list(set(r['source_language'] for r in processed_results))
            }
            
        except Exception as e:
            logger.exception(f"Error in translation for job {job_id}: {e}")
            raise
    
    async def _translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """Translate a single text segment"""
        try:
            # Run translation in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._translate_text_sync,
                text,
                source_language,
                target_language
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error translating text: {e}")
            return {
                'translated_text': text,
                'confidence': 0.1,
                'method': 'error_fallback'
            }
    
    def _get_translation_pipeline(self, source_language: str, target_language: str):
        """Load or retrieve a lightweight HF translation pipeline for the pair."""
        try:
            key = (source_language.lower(), target_language.lower())
            model_name = self.pipeline_specs.get(key)
            if not model_name:
                return None
            if key in self.pipelines:
                return self.pipelines[key]
            cache_dir = Path(model_config.huggingface_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline('translation', model=model_name, device=device, cache_dir=str(cache_dir))
            self.pipelines[key] = pipe
            return pipe
        except Exception as exc:
            logger.warning(f"Failed to load translation pipeline {source_language}->{target_language}: {exc}")
            return None

    def _translate_text_sync(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """Synchronous text translation"""
        try:
            # Select best model for language pair
            model_method = self._select_translation_method(source_language, target_language)
            
            if model_method == 'indictrans2':
                return self._translate_with_indictrans2(text, source_language, target_language)
            elif model_method == 'nllb':
                return self._translate_with_nllb(text, source_language, target_language)
            elif model_method.startswith('pipeline::'):
                _, src_key, tgt_key = model_method.split('::', 2)
                pipe = self._get_translation_pipeline(src_key, tgt_key)
                if pipe:
                    try:
                        outputs = pipe(text, max_length=512)
                        translated = outputs[0]['translation_text'] if outputs else text
                        return {
                            'translated_text': translated,
                            'confidence': 0.7,
                            'method': 'hf_pipeline'
                        }
                    except Exception as pe:
                        logger.warning(f"Pipeline translation failed for {src_key}->{tgt_key}: {pe}")
                # Fallback to no translation if pipeline unavailable
                return {
                    'translated_text': text,
                    'confidence': 0.6,
                    'method': 'no_translation'
                }
            else:
                # Fallback: return original text
                return {
                    'translated_text': text,
                    'confidence': 0.6,
                    'method': 'no_translation'
                }
                
        except Exception as e:
            logger.exception(f"Error in synchronous translation: {e}")
            return {
                'translated_text': text,
                'confidence': 0.6,
                'method': 'no_translation'
            }
    
    def _select_translation_method(
        self, 
        source_language: str, 
        target_language: str
    ) -> str:
        """Select best translation method for language pair"""
        try:
            # Prefer IndicTrans2 when available; use NLLB for Urdu and as fallback
            src = source_language.lower()
            tgt = target_language.lower()
            if src != 'english' and tgt == 'english':
                if src == 'urdu' and self.nllb_model:
                    return 'nllb'
                if 'indictrans2_indic_en' in self.translation_models:
                    return 'indictrans2'
                if self.nllb_model:
                    return 'nllb'
            if src == 'english' and tgt != 'english':
                if 'indictrans2_en_indic' in self.translation_models:
                    return 'indictrans2'
                if self.nllb_model:
                    return 'nllb'

            pipeline_key = (src, tgt)
            if pipeline_key in self.pipeline_specs:
                return f"pipeline::{pipeline_key[0]}::{pipeline_key[1]}"

            return 'none'
            
        except Exception:
            return 'none'
    
    def _translate_with_indictrans2(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """Translate using IndicTrans2 (supports both directions when models present)."""
        try:
            src = source_language.lower()
            tgt = target_language.lower()
            if src != 'english' and tgt == 'english' and 'indictrans2_indic_en' in self.translation_models:
                model = self.translation_models['indictrans2_indic_en']
                tokenizer = self.tokenizers['indictrans2_indic_en']
            elif src == 'english' and tgt != 'english' and 'indictrans2_en_indic' in self.translation_models:
                model = self.translation_models['indictrans2_en_indic']
                tokenizer = self.tokenizers['indictrans2_en_indic']
            else:
                raise RuntimeError('Requested IndicTrans2 direction not available')

            src_tag = self._get_indictrans_lang_code(source_language)
            tgt_tag = self._get_indictrans_lang_code(target_language)
            if not src_tag or not tgt_tag:
                raise ValueError(f"Unsupported IndicTrans2 language tags: {source_language} -> {target_language}")
            # IndicTrans2 tokenizer in this build expects: "SRC_TAG TGT_TAG <text>" (no angle brackets)
            tagged_input = f"{src_tag} {tgt_tag} {text}"

            inputs = tokenizer(
                tagged_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=model_config.indictrans_beam_size,
                    early_stopping=True,
                    do_sample=False
                )

            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            confidence = self._estimate_translation_confidence(text, translated_text, 'indictrans2')
            return {'translated_text': translated_text, 'confidence': confidence, 'method': 'indictrans2'}
        except Exception as e:
            logger.exception(f"Error with IndicTrans2 translation: {e}")
            if self.nllb_model:
                return self._translate_with_nllb(text, source_language, target_language)
            raise
    
    def _translate_with_nllb(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Dict[str, Any]:
        """Translate using NLLB"""
        try:
            if not self.nllb_model or not self.nllb_tokenizer:
                raise RuntimeError("NLLB model/tokenizer not available")

            # Get NLLB language codes
            src_code = self._get_nllb_lang_code(source_language)
            tgt_code = self._get_nllb_lang_code(target_language)
            if not tgt_code:
                raise ValueError(f"Unsupported language pair: {source_language} -> {target_language}")

            # Configure tokenizer for source language when available
            if src_code:
                try:
                    self.nllb_tokenizer.src_lang = src_code
                except Exception:
                    pass

            inputs = self.nllb_tokenizer(text, return_tensors="pt").to(self.nllb_model.device)
            forced_bos = None
            try:
                forced_bos = self.nllb_tokenizer.lang_code_to_id.get(tgt_code)
            except Exception:
                forced_bos = None
            if forced_bos is None:
                # Fallback: try convert_tokens_to_ids
                forced_bos = self.nllb_tokenizer.convert_tokens_to_ids(tgt_code)

            outputs = self.nllb_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=model_config.nllb_max_length
            )
            translated_text = self.nllb_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

            confidence = self._estimate_translation_confidence(text, translated_text, 'nllb')
            return {'translated_text': translated_text, 'confidence': confidence, 'method': 'nllb'}
        except Exception as e:
            logger.exception(f"Error with NLLB translation: {e}")
            raise
    
    def _get_indictrans_lang_code(self, language: str) -> Optional[str]:
        """Get IndicTrans2 language code"""
        lang_codes = {
            'english': 'eng_Latn',
            'hindi': 'hin_Deva',
            'bengali': 'ben_Beng',
            'punjabi': 'pan_Guru',
            'nepali': 'nep_Deva',
            'dogri': 'doi_Deva'
        }
        return lang_codes.get(language.lower())
    
    def _get_nllb_lang_code(self, language: str) -> Optional[str]:
        """Get NLLB language code"""
        lang_codes = {
            'english': 'eng_Latn',
            'hindi': 'hin_Deva',
            'bengali': 'ben_Beng',
            'punjabi': 'pan_Guru',
            'nepali': 'nep_Deva',
            'dogri': 'hin_Deva',  # Use Hindi for Dogri
            'urdu': 'urd_Arab'
        }
        return lang_codes.get(language.lower())

    def _normalize_language(self, language: str) -> str:
        """Normalize language tags to a canonical set used by translation.
        Maps variants like 'uncertain_english', 'en', 'eng', 'auto' (when likely English)
        to 'english' so we can skip unnecessary translation and avoid code mismatches.
        """
        try:
            if not language:
                return 'unknown'
            lang = language.strip().lower()

            # Language mappings
            lang_mappings = {
                'en': 'english',
                'eng': 'english',
                'english': 'english',
                'uncertain_english': 'english',
                'auto_english': 'english',
                'hi': 'hindi',
                'hin': 'hindi',
                'hindi': 'hindi',
                'pa': 'punjabi',
                'pan': 'punjabi',
                'punjabi': 'punjabi',
                'ur': 'urdu',
                'urd': 'urdu',
                'urdu': 'urdu',
                'bn': 'bengali',
                'ben': 'bengali',
                'bengali': 'bengali'
            }

            # Check direct mapping first
            if lang in lang_mappings:
                return lang_mappings[lang]

            # Check if it starts with or contains english
            if lang.startswith('english') or 'english' in lang:
                return 'english'

            return lang
        except Exception:
            return language or 'unknown'

    def _detect_script(self, text: str) -> str:
        """Detect dominant script of the text to assist language inference.
        Returns one of: 'devanagari', 'arabic', 'latin', 'other'.
        """
        try:
            if not text:
                return 'other'
            # Unicode ranges
            has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
            has_bengali = any('\u0980' <= ch <= '\u09FF' for ch in text)
            has_gurmukhi = any('\u0A00' <= ch <= '\u0A7F' for ch in text)
            has_arabic = any('\u0600' <= ch <= '\u06FF' for ch in text)
            has_latin = any('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in text)
            if has_devanagari:
                return 'devanagari'
            if has_bengali:
                return 'bengali'
            if has_gurmukhi:
                return 'gurmukhi'
            if has_arabic:
                return 'arabic'
            if has_latin:
                return 'latin'
            return 'other'
        except Exception:
            return 'other'

    def _infer_source_language(self, lang_hint: str, script: str) -> str:
        """Combine LID hint with script to infer a better source language label.
        - Devanagari script -> hindi
        - Arabic script -> urdu
        - Script takes precedence over potentially incorrect LID hints
        """
        try:
            # Script detection is more reliable than LID for these scripts
            if script == 'devanagari':
                return 'hindi'
            if script == 'arabic':
                return 'urdu'
            if script == 'gurmukhi':
                return 'punjabi'
            if script == 'bengali':
                return 'bengali'

            normalized = self._normalize_language(lang_hint)

            # Only return 'english' if we have Latin script AND reasonable LID confidence
            if normalized in ('auto', 'unknown', 'english'):
                if script == 'latin':
                    return 'english'
                # If non-Latin script but LID says English, assume it's wrong
                elif script in ['devanagari', 'arabic', 'gurmukhi', 'bengali']:
                    # Fallback based on script
                    script_mapping = {
                        'devanagari': 'hindi',
                        'arabic': 'urdu',
                        'gurmukhi': 'punjabi',
                        'bengali': 'bengali'
                    }
                    return script_mapping.get(script, 'unknown')

            return normalized
        except Exception:
            return self._normalize_language(lang_hint)
    
    def _estimate_translation_confidence(
        self, 
        source_text: str, 
        translated_text: str,
        method: str
    ) -> float:
        """Estimate translation confidence"""
        try:
            # Basic heuristics for translation quality
            confidence = 0.8  # Base confidence
            
            # Penalize very short translations
            if len(translated_text.split()) < len(source_text.split()) * 0.3:
                confidence *= 0.7
            
            # Penalize very long translations (possible hallucination)
            if len(translated_text.split()) > len(source_text.split()) * 3:
                confidence *= 0.8
            
            # Boost confidence for IndicTrans2 on Indic languages
            if method == 'indictrans2':
                confidence = min(1.0, confidence * 1.2)
            
            # Check for repeated patterns (sign of poor translation)
            words = translated_text.split()
            if len(words) > 3:
                repeated_words = len(words) - len(set(words))
                if repeated_words > len(words) * 0.3:
                    confidence *= 0.6
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.8
    
    async def _post_process_translations(
        self,
        job_id: str,
        translations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Post-process translation results"""
        try:
            processed = []
            
            for translation in translations:
                # Clean up translated text
                cleaned_text = self._clean_translation_text(translation['translated_text'])
                
                # Apply post-editing rules
                cleaned_text = self._apply_post_editing_rules(
                    cleaned_text, 
                    translation['source_language'],
                    translation['target_language']
                )
                
                # Filter very low quality translations
                if translation['translation_confidence'] < 0.2:
                    logger.debug(f"Filtering low quality translation: {cleaned_text[:50]}...")
                    continue
                
                processed_translation = translation.copy()
                processed_translation['translated_text'] = cleaned_text
                processed.append(processed_translation)
            
            return processed
            
        except Exception as e:
            logger.exception(f"Error in translation post-processing: {e}")
            return translations
    
    def _clean_translation_text(self, text: str) -> str:
        """Clean up translation text"""
        try:
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove common translation artifacts
            artifacts = ['<unk>', '[UNK]', '<s>', '</s>', '<pad>']
            for artifact in artifacts:
                text = text.replace(artifact, '')
            
            # Fix punctuation spacing
            text = re.sub(r'\s+([.!?,:;])', r'\1', text)
            text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
            
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception:
            return text
    
    def _apply_post_editing_rules(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> str:
        """Apply language-specific post-editing rules"""
        try:
            # English-specific post-editing
            if target_language.lower() == 'english':
                # Capitalize first word of sentences
                text = re.sub(r'(^|[.!?]\s+)([a-z])', 
                             lambda m: m.group(1) + m.group(2).upper(), text)
                
                # Fix common translation errors for Indian languages
                if source_language.lower() in ['hindi', 'bengali', 'punjabi']:
                    # Fix honorifics
                    text = text.replace('ji ', ' ji ')
                    text = text.replace('sahib', 'sahib')
                    
                    # Fix common mistranslations
                    corrections = {
                        'what what': 'what',
                        'that that': 'that',
                        'is is': 'is'
                    }
                    for wrong, correct in corrections.items():
                        text = text.replace(wrong, correct)
            
            return text
            
        except Exception:
            return text
    
    def _calculate_translation_quality(
        self, 
        translations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate translation quality metrics"""
        try:
            if not translations:
                return {}
            
            # Average confidence
            avg_confidence = np.mean([t['translation_confidence'] for t in translations])
            
            # Method distribution
            methods = [t['translation_method'] for t in translations]
            method_counts = {method: methods.count(method) for method in set(methods)}
            
            # Length statistics
            source_lengths = [len(t['source_text'].split()) for t in translations]
            target_lengths = [len(t['translated_text'].split()) for t in translations]
            
            avg_source_length = np.mean(source_lengths) if source_lengths else 0
            avg_target_length = np.mean(target_lengths) if target_lengths else 0
            
            # Translation ratio
            length_ratio = avg_target_length / avg_source_length if avg_source_length > 0 else 1.0
            
            return {
                'average_confidence': avg_confidence,
                'translation_methods': method_counts,
                'average_source_length': avg_source_length,
                'average_target_length': avg_target_length,
                'length_ratio': length_ratio,
                'total_translations': len(translations)
            }
            
        except Exception as e:
            logger.exception(f"Error calculating translation quality: {e}")
            return {}
    
    async def _save_translation_results(self, job_id: str, results: Dict[str, Any]):
        """Save translation results"""
        try:
            await self.audio_repo.save_processing_results(
                job_id, "translation", results
            )
            logger.info(f"Saved translation results for job {job_id}")
            
        except Exception as e:
            logger.exception(f"Error saving translation results: {e}")
    
    # Synchronous methods for Celery tasks
    def load_transcription_results_sync(self, job_id: str) -> Dict[str, Any]:
        """Load transcription results synchronously from storage."""
        import json
        try:
            results_path = app_config.data_dir / "audio" / job_id / "asr_results.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Normalize key name for downstream tasks
                    if 'segments' not in data and 'transcription_segments' in data:
                        return {'segments': data['transcription_segments'], **{k: v for k, v in data.items() if k != 'transcription_segments'}}
                    return data
            return {'segments': []}
        except Exception as e:
            logger.exception(f"Error loading transcription results: {e}")
            return {'segments': []}
    
    def translate_segments_sync(
        self,
        job_id: str,
        segments: List[Dict[str, Any]],
        target_language: str = 'english'
    ) -> Dict[str, Any]:
        """Synchronous wrapper around async translate_segments for Celery tasks."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.translate_segments(job_id=job_id, transcription_segments=segments, target_language=target_language)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.exception(f"Error in synchronous translation: {e}")
            return {'segments': [], 'quality_metrics': {}}
