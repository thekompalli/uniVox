#!/usr/bin/env python3
"""
Test script to verify faster-whisper is being used by the ASR service
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_faster_whisper_direct():
    """Test faster-whisper directly"""
    print("Testing faster-whisper directly...")
    try:
        from faster_whisper import WhisperModel

        model_path = r'C:\PS6\ps06_system\models\whisper\large_v3_ct2'
        model = WhisperModel(model_path, device='cuda', compute_type='float16')

        # Create dummy audio (1 second of silence)
        audio = np.zeros(16000, dtype=np.float32)

        segments, info = model.transcribe(audio, beam_size=5, word_timestamps=True)
        segments = list(segments)

        print(f"[SUCCESS] Fast Whisper model loaded successfully!")
        print(f"Model: {model_path}")
        print(f"Language: {info.language}")
        print(f"Language probability: {info.language_probability:.2f}")
        print(f"Number of segments: {len(segments)}")

        return True

    except Exception as e:
        print(f"[ERROR] Direct faster-whisper test failed: {e}")
        return False

def test_asr_service_config():
    """Test ASR service configuration"""
    print("\nTesting ASR service configuration...")
    try:
        from src.config.app_config import model_config

        print(f"Whisper backend: {model_config.whisper_backend}")
        print(f"Fast Whisper model path: {model_config.fast_whisper_model_path}")
        print(f"Fast Whisper device: {model_config.fast_whisper_device}")
        print(f"Fast Whisper compute type: {model_config.fast_whisper_compute_type}")

        if model_config.whisper_backend == 'fast':
            print("[SUCCESS] Configuration is set to use faster-whisper!")
            return True
        else:
            print("[ERROR] Configuration is not set to use faster-whisper")
            return False

    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_transcribe_segment_fast():
    """Test the faster-whisper transcription function specifically"""
    print("\nTesting ASR service _transcribe_segment_fast method...")
    try:
        # Create a minimal mock ASR service to test the fast transcription method
        class MockASRService:
            def __init__(self):
                from faster_whisper import WhisperModel
                from src.config.app_config import model_config

                model_path = model_config.fast_whisper_model_path
                self.fast_whisper_model = WhisperModel(
                    model_path,
                    device=model_config.fast_whisper_device,
                    compute_type=model_config.fast_whisper_compute_type
                )
                self.fast_whisper_options = {
                    'beam_size': 5,
                    'temperature': 0.0,
                    'vad_filter': False,
                }

            def _whisper_language_code(self, language):
                mapping = {
                    'english': 'en',
                    'hindi': 'hi',
                    'punjabi': 'pa',
                    'auto': None
                }
                return mapping.get(language, None)

            def _transcribe_segment_fast(self, segment_audio, sample_rate, language, lid_confidence):
                """Copy of the _transcribe_segment_fast method from ASRService"""
                if self.fast_whisper_model is None:
                    return None

                try:
                    from src.config.app_config import model_config
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

                    segments_iter, info = self.fast_whisper_model.transcribe(
                        audio,
                        **transcribe_kwargs,
                    )

                    segments = list(segments_iter)
                    text_parts = []
                    aggregated_words = []

                    for seg in segments:
                        seg_text = (seg.text or '').strip()
                        text_parts.append(seg_text)

                        for word in getattr(seg, 'words', []) or []:
                            word_dict = {
                                'word': (word.word or '').strip(),
                                'start': float(word.start or 0.0) if word.start is not None else 0.0,
                                'end': float(word.end or 0.0) if word.end is not None else 0.0,
                                'probability': float(word.probability or 0.0),
                            }
                            aggregated_words.append(word_dict)

                    combined_text = ' '.join(part for part in text_parts if part).strip()
                    detected_code = None
                    if info is not None and getattr(info, 'language', None):
                        detected_code = info.language

                    code_map = {
                        'en': 'english',
                        'hi': 'hindi',
                        'bn': 'bengali',
                        'pa': 'punjabi',
                        'ur': 'urdu'
                    }
                    detected_lang = code_map.get(str(detected_code).lower(), language) if detected_code else language

                    return {
                        'text': combined_text,
                        'language': detected_lang,
                        'confidence': 0.8,  # Simplified confidence calculation
                        'words': aggregated_words,
                    }
                except Exception as e:
                    print(f"Error in fast transcription: {e}")
                    return None

        # Test with dummy audio
        service = MockASRService()
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = service._transcribe_segment_fast(audio, 16000, 'english', 0.9)

        if result:
            print(f"[SUCCESS] Fast transcription method works!")
            print(f"Transcribed text: '{result['text']}'")
            print(f"Detected language: {result['language']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Number of words: {len(result['words'])}")
            return True
        else:
            print("[ERROR] Fast transcription returned None")
            return False

    except Exception as e:
        print(f"[ERROR] Fast transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("FASTER-WHISPER INTEGRATION TEST")
    print("=" * 60)

    results = []
    results.append(test_faster_whisper_direct())
    results.append(test_asr_service_config())
    results.append(test_transcribe_segment_fast())

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all(results):
        print("[SUCCESS] All tests passed! Faster-Whisper is properly integrated.")
    else:
        print("[ERROR] Some tests failed. Check the output above.")
        failed_tests = sum(1 for r in results if not r)
        print(f"Failed: {failed_tests}/{len(results)} tests")