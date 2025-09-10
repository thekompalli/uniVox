"""
Evaluation Utilities
Metrics calculation and evaluation functions for PS-06 competition
"""
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class EvalUtils:
    """Evaluation utilities for competition metrics"""
    
    def __init__(self):
        pass
    
    # Speaker Diarization Evaluation
    def calculate_der(
        self,
        reference_segments: List[Dict[str, Any]],
        hypothesis_segments: List[Dict[str, Any]],
        collar: float = 0.25
    ) -> Dict[str, float]:
        """
        Calculate Diarization Error Rate (DER)
        
        Args:
            reference_segments: Ground truth segments
            hypothesis_segments: System output segments  
            collar: Collar size in seconds for matching
            
        Returns:
            DER metrics dictionary
        """
        try:
            # Convert segments to time intervals
            ref_intervals = self._segments_to_intervals(reference_segments)
            hyp_intervals = self._segments_to_intervals(hypothesis_segments)
            
            # Calculate total reference time
            total_ref_time = sum(end - start for start, end, _ in ref_intervals)
            
            if total_ref_time == 0:
                return {'der': 1.0, 'miss': 1.0, 'false_alarm': 0.0, 'confusion': 0.0}
            
            # Calculate speech/non-speech errors
            speech_errors = self._calculate_speech_errors(ref_intervals, hyp_intervals, collar)
            
            # Calculate speaker confusion errors
            confusion_errors = self._calculate_speaker_confusion(
                ref_intervals, hyp_intervals, collar
            )
            
            # Calculate metrics
            miss_time = speech_errors['miss']
            fa_time = speech_errors['false_alarm']
            confusion_time = confusion_errors['confusion']
            
            der = (miss_time + fa_time + confusion_time) / total_ref_time
            miss_rate = miss_time / total_ref_time
            fa_rate = fa_time / total_ref_time
            confusion_rate = confusion_time / total_ref_time
            
            return {
                'der': der,
                'miss': miss_rate,
                'false_alarm': fa_rate,
                'confusion': confusion_rate,
                'total_error_time': miss_time + fa_time + confusion_time,
                'total_reference_time': total_ref_time
            }
            
        except Exception as e:
            logger.exception(f"Error calculating DER: {e}")
            return {'der': 1.0, 'error': str(e)}
    
    def _segments_to_intervals(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Tuple[float, float, str]]:
        """Convert segments to (start, end, speaker) intervals"""
        intervals = []
        for segment in segments:
            start = float(segment.get('start', 0))
            end = float(segment.get('end', 0))
            speaker = str(segment.get('speaker', 'unknown'))
            if end > start:
                intervals.append((start, end, speaker))
        
        # Sort by start time
        return sorted(intervals, key=lambda x: x[0])
    
    def _calculate_speech_errors(
        self,
        ref_intervals: List[Tuple[float, float, str]],
        hyp_intervals: List[Tuple[float, float, str]],
        collar: float
    ) -> Dict[str, float]:
        """Calculate speech/non-speech detection errors"""
        # Create time grids
        all_times = set()
        for start, end, _ in ref_intervals + hyp_intervals:
            all_times.update([start, end])
        
        sorted_times = sorted(all_times)
        
        miss_time = 0.0
        fa_time = 0.0
        
        # Check each time segment
        for i in range(len(sorted_times) - 1):
            seg_start = sorted_times[i]
            seg_end = sorted_times[i + 1]
            seg_mid = (seg_start + seg_end) / 2
            
            # Check if reference has speech at this time
            ref_has_speech = any(
                start - collar <= seg_mid <= end + collar
                for start, end, _ in ref_intervals
            )
            
            # Check if hypothesis has speech at this time  
            hyp_has_speech = any(
                start <= seg_mid <= end
                for start, end, _ in hyp_intervals
            )
            
            seg_duration = seg_end - seg_start
            
            if ref_has_speech and not hyp_has_speech:
                miss_time += seg_duration
            elif not ref_has_speech and hyp_has_speech:
                fa_time += seg_duration
        
        return {'miss': miss_time, 'false_alarm': fa_time}
    
    def _calculate_speaker_confusion(
        self,
        ref_intervals: List[Tuple[float, float, str]],
        hyp_intervals: List[Tuple[float, float, str]],
        collar: float
    ) -> Dict[str, float]:
        """Calculate speaker confusion errors"""
        confusion_time = 0.0
        
        for ref_start, ref_end, ref_speaker in ref_intervals:
            # Find overlapping hypothesis segments
            overlapping_hyp = []
            for hyp_start, hyp_end, hyp_speaker in hyp_intervals:
                # Check for overlap considering collar
                if (ref_start - collar < hyp_end and hyp_start < ref_end + collar):
                    overlap_start = max(ref_start - collar, hyp_start)
                    overlap_end = min(ref_end + collar, hyp_end)
                    if overlap_end > overlap_start:
                        overlapping_hyp.append((overlap_start, overlap_end, hyp_speaker))
            
            # Calculate confusion for this reference segment
            ref_duration = ref_end - ref_start
            
            # Group overlapping segments by speaker
            speaker_times = defaultdict(float)
            for start, end, speaker in overlapping_hyp:
                # Calculate actual overlap with reference segment
                actual_start = max(start, ref_start)
                actual_end = min(end, ref_end)
                if actual_end > actual_start:
                    speaker_times[speaker] += actual_end - actual_start
            
            # Find dominant hypothesis speaker
            if speaker_times:
                dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
                dominant_time = speaker_times[dominant_speaker]
                
                # If dominant speaker is wrong, count as confusion
                if dominant_speaker != ref_speaker:
                    confusion_time += min(dominant_time, ref_duration)
        
        return {'confusion': confusion_time}
    
    # Speaker Identification Evaluation
    def calculate_speaker_accuracy(
        self,
        reference_segments: List[Dict[str, Any]],
        hypothesis_segments: List[Dict[str, Any]],
        time_tolerance: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate speaker identification accuracy
        
        Args:
            reference_segments: Ground truth with speaker IDs
            hypothesis_segments: System output with speaker IDs
            time_tolerance: Time tolerance for matching segments
            
        Returns:
            Accuracy metrics
        """
        try:
            if not reference_segments:
                return {'accuracy': 0.0, 'total_segments': 0}
            
            correct_identifications = 0
            total_segments = len(reference_segments)
            matched_segments = 0
            
            for ref_seg in reference_segments:
                ref_start = ref_seg.get('start', 0)
                ref_end = ref_seg.get('end', 0)
                ref_speaker = ref_seg.get('speaker_id', ref_seg.get('speaker', 'unknown'))
                
                # Find best matching hypothesis segment
                best_match = None
                best_overlap = 0
                
                for hyp_seg in hypothesis_segments:
                    hyp_start = hyp_seg.get('start', 0)
                    hyp_end = hyp_seg.get('end', 0)
                    
                    # Calculate overlap
                    overlap_start = max(ref_start, hyp_start)
                    overlap_end = min(ref_end, hyp_end)
                    
                    if overlap_end > overlap_start:
                        overlap = overlap_end - overlap_start
                        ref_duration = ref_end - ref_start
                        
                        # Check if significant overlap (>50% of reference segment)
                        if overlap / ref_duration > 0.5 and overlap > best_overlap:
                            best_overlap = overlap
                            best_match = hyp_seg
                
                if best_match:
                    matched_segments += 1
                    hyp_speaker = best_match.get('speaker_id', best_match.get('speaker', 'unknown'))
                    
                    if ref_speaker == hyp_speaker:
                        correct_identifications += 1
            
            accuracy = correct_identifications / total_segments if total_segments > 0 else 0.0
            recall = matched_segments / total_segments if total_segments > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'recall': recall,
                'correct_identifications': correct_identifications,
                'matched_segments': matched_segments,
                'total_segments': total_segments
            }
            
        except Exception as e:
            logger.exception(f"Error calculating speaker accuracy: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    # Language Identification Evaluation  
    def calculate_language_accuracy(
        self,
        reference_segments: List[Dict[str, Any]],
        hypothesis_segments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate language identification accuracy"""
        try:
            if not reference_segments:
                return {'accuracy': 0.0, 'total_duration': 0.0}
            
            total_duration = 0.0
            correct_duration = 0.0
            
            for ref_seg in reference_segments:
                ref_start = ref_seg.get('start', 0)
                ref_end = ref_seg.get('end', 0)
                ref_language = ref_seg.get('language', 'unknown')
                ref_duration = ref_end - ref_start
                
                total_duration += ref_duration
                
                # Find overlapping hypothesis segments
                for hyp_seg in hypothesis_segments:
                    hyp_start = hyp_seg.get('start', 0)
                    hyp_end = hyp_seg.get('end', 0)
                    hyp_language = hyp_seg.get('language', 'unknown')
                    
                    # Calculate overlap
                    overlap_start = max(ref_start, hyp_start)
                    overlap_end = min(ref_end, hyp_end)
                    
                    if overlap_end > overlap_start:
                        overlap_duration = overlap_end - overlap_start
                        
                        if ref_language == hyp_language:
                            correct_duration += overlap_duration
            
            accuracy = correct_duration / total_duration if total_duration > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'correct_duration': correct_duration,
                'total_duration': total_duration
            }
            
        except Exception as e:
            logger.exception(f"Error calculating language accuracy: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    # ASR Evaluation
    def calculate_wer(
        self,
        reference_text: str,
        hypothesis_text: str,
        normalize: bool = True
    ) -> Dict[str, Union[float, int]]:
        """
        Calculate Word Error Rate (WER)
        
        Args:
            reference_text: Ground truth text
            hypothesis_text: System output text
            normalize: Whether to normalize text before comparison
            
        Returns:
            WER metrics
        """
        try:
            if normalize:
                reference_text = self._normalize_text(reference_text)
                hypothesis_text = self._normalize_text(hypothesis_text)
            
            ref_words = reference_text.split()
            hyp_words = hypothesis_text.split()
            
            # Calculate edit distance
            edit_ops = self._calculate_edit_operations(ref_words, hyp_words)
            
            substitutions = edit_ops['substitutions']
            insertions = edit_ops['insertions'] 
            deletions = edit_ops['deletions']
            
            total_errors = substitutions + insertions + deletions
            total_words = len(ref_words)
            
            wer = total_errors / total_words if total_words > 0 else 1.0 if hyp_words else 0.0
            
            return {
                'wer': wer,
                'substitutions': substitutions,
                'insertions': insertions,
                'deletions': deletions,
                'total_errors': total_errors,
                'total_words': total_words,
                'hypothesis_words': len(hyp_words)
            }
            
        except Exception as e:
            logger.exception(f"Error calculating WER: {e}")
            return {'wer': 1.0, 'error': str(e)}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for evaluation"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_edit_operations(
        self, 
        ref_words: List[str], 
        hyp_words: List[str]
    ) -> Dict[str, int]:
        """Calculate edit operations using dynamic programming"""
        m, n = len(ref_words), len(hyp_words)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        ops = [[None] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
            ops[i][0] = 'D' if i > 0 else None
            
        for j in range(n + 1):
            dp[0][j] = j  
            ops[0][j] = 'I' if j > 0 else None
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                    ops[i][j] = 'C'  # Correct
                else:
                    # Find minimum cost operation
                    costs = [
                        (dp[i-1][j] + 1, 'D'),      # Deletion
                        (dp[i][j-1] + 1, 'I'),      # Insertion  
                        (dp[i-1][j-1] + 1, 'S')     # Substitution
                    ]
                    
                    min_cost, min_op = min(costs, key=lambda x: x[0])
                    dp[i][j] = min_cost
                    ops[i][j] = min_op
        
        # Backtrack to count operations
        substitutions = insertions = deletions = 0
        i, j = m, n
        
        while i > 0 or j > 0:
            op = ops[i][j]
            
            if op == 'S':
                substitutions += 1
                i -= 1
                j -= 1
            elif op == 'I':
                insertions += 1
                j -= 1
            elif op == 'D':
                deletions += 1
                i -= 1
            else:  # op == 'C'
                i -= 1
                j -= 1
        
        return {
            'substitutions': substitutions,
            'insertions': insertions, 
            'deletions': deletions
        }
    
    # Translation Evaluation
    def calculate_bleu(
        self,
        reference_text: str,
        hypothesis_text: str,
        n_grams: int = 4
    ) -> Dict[str, float]:
        """
        Calculate BLEU score for translation evaluation
        
        Args:
            reference_text: Ground truth translation
            hypothesis_text: System output translation
            n_grams: Maximum n-gram order
            
        Returns:
            BLEU metrics
        """
        try:
            if not reference_text.strip() or not hypothesis_text.strip():
                return {'bleu': 0.0, 'brevity_penalty': 0.0}
            
            ref_words = reference_text.lower().split()
            hyp_words = hypothesis_text.lower().split()
            
            # Calculate n-gram precisions
            precisions = []
            
            for n in range(1, n_grams + 1):
                ref_ngrams = self._get_ngrams(ref_words, n)
                hyp_ngrams = self._get_ngrams(hyp_words, n)
                
                if not hyp_ngrams:
                    precisions.append(0.0)
                    continue
                
                # Count matches
                matches = 0
                for ngram in hyp_ngrams:
                    if ngram in ref_ngrams:
                        matches += min(hyp_ngrams[ngram], ref_ngrams[ngram])
                
                precision = matches / sum(hyp_ngrams.values())
                precisions.append(precision)
            
            # Calculate brevity penalty
            ref_len = len(ref_words)
            hyp_len = len(hyp_words)
            
            if hyp_len > ref_len:
                brevity_penalty = 1.0
            else:
                brevity_penalty = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
            
            # Calculate BLEU score
            if all(p > 0 for p in precisions):
                log_precisions = [np.log(p) for p in precisions]
                bleu = brevity_penalty * np.exp(np.mean(log_precisions))
            else:
                bleu = 0.0
            
            return {
                'bleu': bleu,
                'brevity_penalty': brevity_penalty,
                'precisions': precisions,
                'reference_length': ref_len,
                'hypothesis_length': hyp_len
            }
            
        except Exception as e:
            logger.exception(f"Error calculating BLEU: {e}")
            return {'bleu': 0.0, 'error': str(e)}
    
    def _get_ngrams(self, words: List[str], n: int) -> Dict[tuple, int]:
        """Extract n-grams from word list"""
        ngrams = defaultdict(int)
        
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            ngrams[ngram] += 1
        
        return dict(ngrams)
    
    # Overall System Evaluation
    def calculate_system_performance(
        self,
        diarization_results: Dict[str, Any],
        speaker_id_results: Dict[str, Any], 
        language_id_results: Dict[str, Any],
        asr_results: Dict[str, Any],
        translation_results: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate overall system performance score
        
        Args:
            diarization_results: DER results
            speaker_id_results: Speaker ID accuracy results
            language_id_results: Language ID accuracy results
            asr_results: WER results
            translation_results: BLEU results
            weights: Component weights (default: competition weights)
            
        Returns:
            Overall performance metrics
        """
        try:
            # Default PS-06 competition weights
            if weights is None:
                weights = {
                    'diarization': 0.20,    # 20%
                    'speaker_id': 0.15,     # 15% 
                    'language_id': 0.20,    # 20%
                    'asr': 0.30,           # 30%
                    'translation': 0.15     # 15%
                }
            
            # Convert metrics to scores (0-1, higher is better)
            diar_score = max(0, 1 - diarization_results.get('der', 1.0))
            speaker_score = speaker_id_results.get('accuracy', 0.0)
            lang_score = language_id_results.get('accuracy', 0.0)
            asr_score = max(0, 1 - asr_results.get('wer', 1.0))
            trans_score = translation_results.get('bleu', 0.0)
            
            # Calculate weighted average
            overall_score = (
                weights['diarization'] * diar_score +
                weights['speaker_id'] * speaker_score +
                weights['language_id'] * lang_score + 
                weights['asr'] * asr_score +
                weights['translation'] * trans_score
            )
            
            return {
                'overall_score': overall_score,
                'component_scores': {
                    'diarization': diar_score,
                    'speaker_identification': speaker_score,
                    'language_identification': lang_score,
                    'asr': asr_score,
                    'translation': trans_score
                },
                'weights_used': weights
            }
            
        except Exception as e:
            logger.exception(f"Error calculating system performance: {e}")
            return {'overall_score': 0.0, 'error': str(e)}
    
    # Utility functions
    def format_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        job_id: str
    ) -> str:
        """Format evaluation results into readable report"""
        try:
            report = [
                f"PS-06 Evaluation Report",
                f"Job ID: {job_id}",
                f"Generated: {np.datetime64('now')}",
                "=" * 50,
                ""
            ]
            
            # Overall performance
            if 'overall_score' in evaluation_results:
                score = evaluation_results['overall_score'] * 100
                report.append(f"Overall System Score: {score:.2f}%")
                report.append("")
            
            # Component details
            components = [
                ('diarization', 'Speaker Diarization (DER)', '%'),
                ('speaker_id', 'Speaker Identification', '%'), 
                ('language_id', 'Language Identification', '%'),
                ('asr', 'Speech Recognition (WER)', '%'),
                ('translation', 'Translation (BLEU)', '')
            ]
            
            for comp_key, comp_name, unit in components:
                if comp_key in evaluation_results:
                    comp_results = evaluation_results[comp_key]
                    report.append(f"{comp_name}:")
                    
                    for metric, value in comp_results.items():
                        if isinstance(value, float):
                            if unit == '%':
                                report.append(f"  {metric}: {value*100:.2f}%")
                            else:
                                report.append(f"  {metric}: {value:.4f}")
                        else:
                            report.append(f"  {metric}: {value}")
                    
                    report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.exception(f"Error formatting evaluation report: {e}")
            return f"Error generating report: {e}"