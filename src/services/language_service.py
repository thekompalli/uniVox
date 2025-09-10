"""
Language Identification Service  
Wav2Vec2-based multilingual language identification
"""
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.config.app_config import app_config, model_config
from src.models.language_inference import LanguageInference
from src.repositories.audio_repository import AudioRepository
from src.utils.eval_utils import EvalUtils
import json
import soundfile as sf

logger = logging.getLogger(__name__)


class LanguageIdentificationService:
    """Language identification service for Indian languages"""
    
    def __init__(self):
        self.language_inference = LanguageInference()
        self.audio_repo = AudioRepository()
        self.eval_utils = EvalUtils()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Supported languages for PS-06 competition
        self.supported_languages = [
            'english', 'hindi', 'punjabi', 'bengali', 'nepali', 'dogri'
        ]
        
        # Language confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    async def identify_languages(
        self,
        job_id: str,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: Optional[List[Dict[str, Any]]] = None,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identify languages in audio segments
        
        Args:
            job_id: Job identifier
            audio_data: Audio signal
            sample_rate: Sample rate
            segments: Optional segments to process (if None, process entire audio)
            expected_languages: Optional list of expected languages
            
        Returns:
            Language identification results
        """
        try:
            logger.info(f"Starting language identification for job {job_id}")
            
            # If no segments provided, create segments from speech regions
            if segments is None:
                segments = await self._create_speech_segments(audio_data, sample_rate)
            
            # Filter expected languages
            if expected_languages:
                expected_languages = [
                    lang for lang in expected_languages 
                    if lang.lower() in self.supported_languages
                ]
            
            # Process segments for language identification
            language_segments = []
            language_distribution = {}
            
            for i, segment in enumerate(segments):
                try:
                    # Extract segment audio
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    # Skip very short segments
                    if len(segment_audio) < sample_rate * 0.5:  # 0.5 second minimum
                        continue
                    
                    # Identify language for segment
                    language_result = await self._identify_segment_language(
                        segment_audio, sample_rate, expected_languages
                    )
                    
                    # Create language segment
                    lang_segment = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'duration': segment.get('duration', segment['end'] - segment['start']),
                        'language': language_result['language'],
                        'confidence': language_result['confidence'],
                        'all_probabilities': language_result['probabilities'],
                        'confidence_level': self._get_confidence_level(language_result['confidence']),
                        'speaker': segment.get('speaker', f'speaker_{i}')
                    }
                    
                    language_segments.append(lang_segment)
                    
                    # Update language distribution
                    lang = language_result['language']
                    if lang not in language_distribution:
                        language_distribution[lang] = {
                            'segments': 0,
                            'total_duration': 0.0,
                            'avg_confidence': 0.0
                        }
                    
                    language_distribution[lang]['segments'] += 1
                    language_distribution[lang]['total_duration'] += lang_segment['duration']
                    language_distribution[lang]['avg_confidence'] += language_result['confidence']
                    
                except Exception as e:
                    logger.warning(f"Error processing segment {i}: {e}")
                    continue
            
            # Calculate average confidences
            for lang_data in language_distribution.values():
                if lang_data['segments'] > 0:
                    lang_data['avg_confidence'] /= lang_data['segments']
            
            # Post-process language segments
            processed_segments = await self._post_process_language_segments(
                job_id, language_segments
            )
            
            # Determine dominant languages
            dominant_languages = self._determine_dominant_languages(language_distribution)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_language_quality(
                processed_segments, language_distribution
            )
            
            # Save results
            results = {
                'segments': processed_segments,
                'language_distribution': language_distribution,
                'dominant_languages': dominant_languages,
                'quality_metrics': quality_metrics,
                'languages_detected': list(language_distribution.keys()),
                'total_segments': len(processed_segments)
            }
            
            await self._save_language_results(job_id, results)
            
            logger.info(f"Language identification completed for job {job_id}: "
                       f"{len(processed_segments)} segments, "
                       f"{len(language_distribution)} languages detected")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in language identification for job {job_id}: {e}")
            raise
    
    async def _create_speech_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segment_duration: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Create segments from audio for language identification"""
        try:
            segments = []
            audio_duration = len(audio_data) / sample_rate
            
            # Create fixed-duration segments
            current_time = 0.0
            segment_id = 0
            
            while current_time < audio_duration:
                end_time = min(current_time + segment_duration, audio_duration)
                
                # Only create segment if it's long enough
                if end_time - current_time >= 1.0:  # At least 1 second
                    segment = {
                        'start': current_time,
                        'end': end_time,
                        'duration': end_time - current_time,
                        'segment_id': segment_id
                    }
                    segments.append(segment)
                    segment_id += 1
                
                current_time += segment_duration
            
            return segments
            
        except Exception as e:
            logger.exception(f"Error creating speech segments: {e}")
            return []
    
    async def _identify_segment_language(
        self,
        segment_audio: np.ndarray,
        sample_rate: int,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identify language for a single segment"""
        try:
            # Run language identification in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._identify_language_sync,
                segment_audio,
                sample_rate,
                expected_languages
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error identifying segment language: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def _identify_language_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous language identification"""
        try:
            # Use language inference model
            result = self.language_inference.identify_language_sync(
                audio_data, sample_rate, expected_languages
            )
            
            # Validate result
            if result['language'] not in self.supported_languages:
                # Map to closest supported language or unknown
                result['language'] = self._map_to_supported_language(result['language'])
                result['confidence'] *= 0.7  # Reduce confidence for mapped languages
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in synchronous language identification: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def _map_to_supported_language(self, detected_language: str) -> str:
        """Map detected language to supported language"""
        language_mapping = {
            'urdu': 'hindi',  # Map Urdu to Hindi as they're similar
            'gujarati': 'hindi',  # Map other Indian languages to Hindi
            'marathi': 'hindi',
            'tamil': 'hindi',
            'telugu': 'hindi',
            'kannada': 'hindi',
            'malayalam': 'hindi',
            'assamese': 'bengali',  # Map Assamese to Bengali (similar script)
            'odia': 'bengali',
            'unknown': 'unknown'
        }
        
        return language_mapping.get(detected_language.lower(), 'unknown')
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category"""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'
    
    async def _post_process_language_segments(
        self,
        job_id: str,
        language_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Post-process language identification results"""
        try:
            if not language_segments:
                return language_segments
            
            # Smooth language transitions
            smoothed_segments = self._smooth_language_transitions(language_segments)
            
            # Filter low-confidence segments
            filtered_segments = self._filter_low_confidence_segments(smoothed_segments)
            
            # Merge adjacent segments with same language
            merged_segments = self._merge_adjacent_same_language(filtered_segments)
            
            return merged_segments
            
        except Exception as e:
            logger.exception(f"Error in language post-processing: {e}")
            return language_segments
    
    def _smooth_language_transitions(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Smooth rapid language transitions"""
        try:
            if len(segments) < 3:
                return segments
            
            smoothed_segments = segments.copy()
            
            # Look for single-segment outliers
            for i in range(1, len(smoothed_segments) - 1):
                current = smoothed_segments[i]
                prev_lang = smoothed_segments[i-1]['language']
                next_lang = smoothed_segments[i+1]['language']
                current_lang = current['language']
                
                # If current segment is different but neighbors are same
                if (prev_lang == next_lang and 
                    current_lang != prev_lang and 
                    current['confidence'] < 0.7):
                    
                    # Change to neighbor language if confidence is low
                    logger.debug(f"Smoothing language transition: {current_lang} -> {prev_lang}")
                    smoothed_segments[i]['language'] = prev_lang
                    smoothed_segments[i]['confidence'] *= 0.9  # Reduce confidence
                    smoothed_segments[i]['smoothed'] = True
            
            return smoothed_segments
            
        except Exception as e:
            logger.exception(f"Error smoothing language transitions: {e}")
            return segments
    
    def _filter_low_confidence_segments(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter or adjust low-confidence language predictions"""
        try:
            filtered_segments = []
            
            for segment in segments:
                confidence = segment.get('confidence', 0.0)
                
                if confidence < 0.3:
                    # Mark as uncertain
                    segment['language'] = f"uncertain_{segment['language']}"
                    segment['confidence_level'] = 'very_low'
                elif confidence < 0.5:
                    # Reduce confidence but keep
                    segment['confidence'] *= 0.8
                    segment['confidence_level'] = 'low'
                
                filtered_segments.append(segment)
            
            return filtered_segments
            
        except Exception as e:
            logger.exception(f"Error filtering low confidence segments: {e}")
            return segments
    
    def _merge_adjacent_same_language(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge adjacent segments with the same language"""
        try:
            if not segments:
                return segments
            
            merged_segments = [segments[0]]
            
            for current in segments[1:]:
                last_merged = merged_segments[-1]
                
                # Check if same language and adjacent
                if (current['language'] == last_merged['language'] and
                    abs(current['start'] - last_merged['end']) < 0.1):  # 100ms tolerance
                    
                    # Merge segments
                    merged_segments[-1] = {
                        'start': last_merged['start'],
                        'end': current['end'],
                        'duration': current['end'] - last_merged['start'],
                        'language': current['language'],
                        'confidence': (last_merged['confidence'] + current['confidence']) / 2,
                        'confidence_level': self._get_confidence_level(
                            (last_merged['confidence'] + current['confidence']) / 2
                        ),
                        'speaker': last_merged.get('speaker', current.get('speaker')),
                        'merged': True
                    }
                else:
                    merged_segments.append(current)
            
            logger.debug(f"Merged {len(segments)} segments to {len(merged_segments)}")
            return merged_segments
            
        except Exception as e:
            logger.exception(f"Error merging segments: {e}")
            return segments
    
    def _determine_dominant_languages(
        self, 
        language_distribution: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Determine dominant languages from distribution"""
        try:
            # Sort languages by total duration and confidence
            sorted_languages = sorted(
                language_distribution.items(),
                key=lambda x: x[1]['total_duration'] * x[1]['avg_confidence'],
                reverse=True
            )
            
            dominant_languages = []
            total_duration = sum(data['total_duration'] for data in language_distribution.values())
            
            for language, data in sorted_languages:
                dominance_score = (data['total_duration'] / total_duration) * data['avg_confidence']
                
                dominant_lang = {
                    'language': language,
                    'total_duration': data['total_duration'],
                    'segments': data['segments'],
                    'avg_confidence': data['avg_confidence'],
                    'dominance_score': dominance_score,
                    'percentage': (data['total_duration'] / total_duration) * 100
                }
                
                dominant_languages.append(dominant_lang)
            
            return dominant_languages
            
        except Exception as e:
            logger.exception(f"Error determining dominant languages: {e}")
            return []
    
    def _calculate_language_quality(
        self,
        segments: List[Dict[str, Any]],
        language_distribution: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate language identification quality metrics"""
        try:
            if not segments:
                return {}
            
            # Basic metrics
            avg_confidence = np.mean([s.get('confidence', 0.0) for s in segments])
            total_duration = sum(s.get('duration', 0.0) for s in segments)
            num_languages = len(language_distribution)
            
            # Confidence distribution
            high_conf_segments = sum(1 for s in segments if s.get('confidence', 0.0) >= 0.8)
            medium_conf_segments = sum(1 for s in segments if 0.6 <= s.get('confidence', 0.0) < 0.8)
            low_conf_segments = sum(1 for s in segments if s.get('confidence', 0.0) < 0.6)
            
            # Language consistency (less switching is better for most cases)
            language_switches = 0
            if len(segments) > 1:
                for i in range(1, len(segments)):
                    if segments[i]['language'] != segments[i-1]['language']:
                        language_switches += 1
            
            consistency_score = 1.0 - (language_switches / max(1, len(segments) - 1))
            
            # Dominant language strength
            if language_distribution:
                max_duration = max(data['total_duration'] for data in language_distribution.values())
                dominant_strength = max_duration / total_duration if total_duration > 0 else 0
            else:
                dominant_strength = 0.0
            
            return {
                'average_confidence': avg_confidence,
                'total_duration': total_duration,
                'num_languages_detected': num_languages,
                'high_confidence_ratio': high_conf_segments / len(segments),
                'medium_confidence_ratio': medium_conf_segments / len(segments),
                'low_confidence_ratio': low_conf_segments / len(segments),
                'language_consistency': consistency_score,
                'dominant_language_strength': dominant_strength,
                'segments_processed': len(segments)
            }
            
        except Exception as e:
            logger.exception(f"Error calculating language quality: {e}")
            return {}
    
    async def _save_language_results(self, job_id: str, results: Dict[str, Any]):
        """Save language identification results"""
        try:
            await self.audio_repo.save_processing_results(
                job_id, "language_identification", results
            )
            logger.info(f"Saved language identification results for job {job_id}")
            
        except Exception as e:
            logger.exception(f"Error saving language results: {e}")
    
    async def validate_language_detection(
        self,
        job_id: str,
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate language detection against ground truth"""
        try:
            # Load results
            results = await self.audio_repo.load_processing_results(job_id, "language_identification")
            if not results:
                return {}
            
            segments = results.get('segments', [])
            
            # Calculate accuracy metrics
            correct_predictions = 0
            total_predictions = 0
            
            for gt_segment in ground_truth:
                # Find overlapping predicted segments
                overlapping_segments = [
                    s for s in segments
                    if s['start'] < gt_segment['end'] and s['end'] > gt_segment['start']
                ]
                
                if overlapping_segments:
                    # Use segment with maximum overlap
                    best_segment = max(
                        overlapping_segments,
                        key=lambda s: min(s['end'], gt_segment['end']) - max(s['start'], gt_segment['start'])
                    )
                    
                    if best_segment['language'] == gt_segment['language']:
                        correct_predictions += 1
                    
                    total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions
            }
            
        except Exception as e:
            logger.exception(f"Error validating language detection: {e}")
            return {}
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get language identification model information"""
        return {
            'supported_languages': self.supported_languages,
            'confidence_thresholds': self.confidence_thresholds,
            'model_type': 'wav2vec2_xlsr',
            'min_segment_duration': 0.5,
            'max_segment_duration': 30.0,
            'sample_rate': 16000
        }

    # ---------- Refinement helpers using transcripts ----------
    def _detect_script(self, text: str) -> str:
        """Detect dominant script in text."""
        try:
            if not text:
                return 'other'
            has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
            has_bengali = any('\u0980' <= ch <= '\u09FF' for ch in text)
            has_gurmukhi = any('\u0A00' <= ch <= '\u0A7F' for ch in text)
            has_arabic = any('\u0600' <= ch <= '\u06FF' for ch in text)
            if has_devanagari:
                return 'devanagari'
            if has_bengali:
                return 'bengali'
            if has_gurmukhi:
                return 'gurmukhi'
            if has_arabic:
                return 'arabic'
            return 'latin' if any(ch.isascii() and ch.isalpha() for ch in text) else 'other'
        except Exception:
            return 'other'

    def refine_with_transcripts_sync(self, job_id: str, transcription_segments: List[Dict[str, Any]]) -> bool:
        """Refine low-confidence language segments using transcript scripts.

        Updates language_identification_results.json in-place if improvements are found.
        """
        try:
            job_dir = app_config.data_dir / "audio" / job_id
            lid_path = job_dir / "language_identification_results.json"
            if not lid_path.exists():
                return False

            with open(lid_path, 'r', encoding='utf-8') as f:
                lid = json.load(f)

            lid_segments = lid.get('segments', [])
            if not lid_segments or not transcription_segments:
                return False

            def overlap(a, b):
                return (a['start'] < b['end']) and (a['end'] > b['start'])

            changed = False
            for seg in lid_segments:
                conf = float(seg.get('confidence', 0.0))
                level = seg.get('confidence_level', 'very_low')
                if conf >= 0.5 and level not in ('low', 'very_low'):
                    continue
                # Gather overlapping transcript text
                texts = [ts.get('text', '') for ts in transcription_segments if overlap(seg, ts) and ts.get('text')]
                if not texts:
                    continue
                joined = ' '.join(texts)
                script = self._detect_script(joined)
                new_lang = None
                if script == 'devanagari':
                    new_lang = 'hindi'
                elif script == 'arabic':
                    new_lang = 'urdu'
                elif script == 'gurmukhi':
                    new_lang = 'punjabi'
                elif script == 'bengali':
                    new_lang = 'bengali'
                elif script == 'latin':
                    new_lang = 'english'

                if new_lang and new_lang in self.supported_languages and new_lang != seg.get('language'):
                    seg['language'] = new_lang
                    seg['confidence'] = max(conf, 0.75)
                    seg['confidence_level'] = self._get_confidence_level(seg['confidence'])
                    changed = True

            if not changed:
                return False

            # Recompute simple distribution/metrics
            dist = {}
            total_dur = 0.0
            conf_sum = 0.0
            for s in lid_segments:
                lang = s.get('language', 'unknown')
                dur = float(s.get('duration', s.get('end', 0) - s.get('start', 0)))
                total_dur += dur
                conf_sum += float(s.get('confidence', 0.0))
                if lang not in dist:
                    dist[lang] = {'segments': 0, 'total_duration': 0.0, 'conf_sum': 0.0}
                dist[lang]['segments'] += 1
                dist[lang]['total_duration'] += dur
                dist[lang]['conf_sum'] += float(s.get('confidence', 0.0))

            lid['language_distribution'] = {
                lang: {
                    'segments': v['segments'],
                    'total_duration': v['total_duration'],
                    'avg_confidence': (v['conf_sum'] / v['segments']) if v['segments'] else 0.0
                } for lang, v in dist.items()
            }
            if dist:
                dominant = max(dist.items(), key=lambda kv: kv[1]['total_duration'])
                lang, v = dominant
                lid['dominant_languages'] = [{
                    'language': lang,
                    'total_duration': v['total_duration'],
                    'segments': v['segments'],
                    'avg_confidence': (v['conf_sum'] / v['segments']) if v['segments'] else 0.0,
                    'dominance_score': (v['conf_sum'] / v['segments']) if v['segments'] else 0.0,
                    'percentage': (v['total_duration'] / total_dur * 100.0) if total_dur > 0 else 0.0
                }]
            lid['quality_metrics'] = {
                'average_confidence': (conf_sum / len(lid_segments)) if lid_segments else 0.0,
                'total_duration': total_dur,
                'num_languages_detected': len(dist),
                'high_confidence_ratio': sum(1 for s in lid_segments if s.get('confidence', 0) >= 0.8) / len(lid_segments),
                'medium_confidence_ratio': sum(1 for s in lid_segments if 0.6 <= s.get('confidence', 0) < 0.8) / len(lid_segments),
                'low_confidence_ratio': sum(1 for s in lid_segments if s.get('confidence', 0) < 0.6) / len(lid_segments),
                'language_consistency': 1.0 if len(dist) == 1 else 0.0,
                'dominant_language_strength': max((v['conf_sum'] / v['segments']) for v in dist.values()) if dist else 0.0,
                'segments_processed': len(lid_segments)
            }
            lid['languages_detected'] = list(dist.keys())
            lid['total_segments'] = len(lid_segments)

            with open(lid_path, 'w', encoding='utf-8') as f:
                json.dump(lid, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.exception(f"Error refining language results for {job_id}: {e}")
            return False
    
    # Synchronous methods for Celery tasks
    def load_processed_data_sync(self, job_id: str) -> Dict[str, Any]:
        """Load processed audio data synchronously (from preprocessing results)."""
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
                audio_data = audio_data.mean(axis=1)

            return {
                'audio_data': audio_data.astype(np.float32),
                'sample_rate': int(sample_rate),
                'speech_segments': meta.get('speech_segments', [])
            }

        except Exception as e:
            logger.exception(f"Error loading processed data sync: {e}")
            return {'audio_data': np.array([]), 'sample_rate': 16000, 'speech_segments': []}
    
    def identify_languages_sync(
        self,
        job_id: str,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict[str, Any]],
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper around async identify_languages for Celery tasks."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.identify_languages(
                        job_id=job_id,
                        audio_data=audio_data,
                        sample_rate=sample_rate,
                        segments=segments,
                        expected_languages=expected_languages
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            logger.exception(f"Error in synchronous language identification: {e}")
            return {'segments': [], 'languages_detected': [], 'total_segments': 0}
