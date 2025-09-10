"""
Tests for ASR Service
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import os

from src.services.asr_service import ASRService
from src.config.app_config import app_config


@pytest.fixture
def mock_whisper_models():
    """Mock Whisper models"""
    mock_model = Mock()
    mock_model.transcribe.return_value = {
        'text': 'Hello world',
        'segments': [
            {
                'start': 0.0,
                'end': 2.0,
                'text': 'Hello world',
                'confidence': 0.95
            }
        ],
        'language': 'en'
    }
    return {'multilingual': mock_model}


@pytest.fixture
def mock_audio_data():
    """Mock audio data"""
    # 16kHz, 2 seconds of sine wave
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio.astype(np.float32), sample_rate


@pytest.fixture
def asr_service():
    """ASR service fixture"""
    with patch('src.services.asr_service.whisper.load_model'):
        service = ASRService()
        return service


class TestASRServiceInitialization:
    """Test ASR service initialization"""
    
    def test_init_creates_service(self):
        """Test ASR service can be created"""
        with patch('src.services.asr_service.whisper.load_model'):
            service = ASRService()
            assert service is not None
            assert hasattr(service, 'whisper_models')
            assert hasattr(service, 'asr_inference')
    
    @patch('src.services.asr_service.whisper.load_model')
    def test_model_initialization(self, mock_load_model):
        """Test model initialization"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        service = ASRService()
        
        # Should load at least the multilingual model
        assert mock_load_model.called
        assert 'multilingual' in service.whisper_models


class TestASRTranscription:
    """Test ASR transcription functionality"""
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_basic(self, asr_service, mock_audio_data):
        """Test basic audio transcription"""
        audio_data, sample_rate = mock_audio_data
        
        # Mock the whisper model
        with patch.object(asr_service, 'whisper_models', mock_whisper_models()):
            result = await asr_service.transcribe_audio(
                job_id="test_job",
                audio_data=audio_data,
                sample_rate=sample_rate,
                language="english"
            )
        
        assert result is not None
        assert 'transcriptions' in result
        assert len(result['transcriptions']) > 0
        assert 'text' in result['transcriptions'][0]
    
    @pytest.mark.asyncio
    async def test_transcribe_segments(self, asr_service, mock_audio_data):
        """Test transcribing audio segments"""
        audio_data, sample_rate = mock_audio_data
        
        segments = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'speaker1'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'speaker2'}
        ]
        
        language_segments = [
            {'start': 0.0, 'end': 1.0, 'language': 'english'},
            {'start': 1.0, 'end': 2.0, 'language': 'english'}
        ]
        
        with patch.object(asr_service, 'whisper_models', mock_whisper_models()):
            result = await asr_service.transcribe_segments(
                job_id="test_job",
                audio_data=audio_data,
                sample_rate=sample_rate,
                segments=segments,
                language_segments=language_segments
            )
        
        assert result is not None
        assert 'transcriptions' in result
        assert len(result['transcriptions']) == len(segments)
    
    @pytest.mark.asyncio
    async def test_transcribe_multilingual(self, asr_service, mock_audio_data):
        """Test multilingual transcription"""
        audio_data, sample_rate = mock_audio_data
        
        # Mock different language outputs
        mock_model = Mock()
        mock_model.transcribe.side_effect = [
            {
                'text': 'Hello world',
                'segments': [{'start': 0.0, 'end': 1.0, 'text': 'Hello world', 'confidence': 0.95}],
                'language': 'en'
            },
            {
                'text': 'नमस्ते दुनिया',
                'segments': [{'start': 1.0, 'end': 2.0, 'text': 'नमस्ते दुनिया', 'confidence': 0.90}],
                'language': 'hi'
            }
        ]
        
        segments = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'speaker1'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'speaker2'}
        ]
        
        language_segments = [
            {'start': 0.0, 'end': 1.0, 'language': 'english'},
            {'start': 1.0, 'end': 2.0, 'language': 'hindi'}
        ]
        
        with patch.object(asr_service, 'whisper_models', {'multilingual': mock_model}):
            result = await asr_service.transcribe_segments(
                job_id="test_job",
                audio_data=audio_data,
                sample_rate=sample_rate,
                segments=segments,
                language_segments=language_segments
            )
        
        assert len(result['transcriptions']) == 2
        # Check that different languages were processed
        transcriptions = result['transcriptions']
        assert any('Hello' in t['text'] for t in transcriptions)
        assert any('नमस्ते' in t['text'] for t in transcriptions)


class TestASRLanguageHandling:
    """Test ASR language-specific handling"""
    
    def test_select_model_for_language(self, asr_service):
        """Test model selection for different languages"""
        # Test with available models
        asr_service.whisper_models = {
            'multilingual': Mock(),
            'hindi': Mock(),
            'bengali': Mock()
        }
        
        assert asr_service._select_model_for_language('english') == 'multilingual'
        assert asr_service._select_model_for_language('hindi') == 'hindi'
        assert asr_service._select_model_for_language('bengali') == 'bengali'
        assert asr_service._select_model_for_language('punjabi') == 'multilingual'
        assert asr_service._select_model_for_language('unknown') == 'multilingual'
    
    def test_whisper_language_code(self, asr_service):
        """Test Whisper language code conversion"""
        assert asr_service._whisper_language_code('english') == 'en'
        assert asr_service._whisper_language_code('hindi') == 'hi'
        assert asr_service._whisper_language_code('bengali') == 'bn'
        assert asr_service._whisper_language_code('punjabi') == 'pa'
        assert asr_service._whisper_language_code('auto') is None
        assert asr_service._whisper_language_code('unknown') == 'en'
    
    def test_merge_speaker_language_segments(self, asr_service):
        """Test merging speaker and language segments"""
        speaker_segments = [
            {'start': 0.0, 'end': 2.0, 'speaker': 'speaker1'},
            {'start': 2.0, 'end': 4.0, 'speaker': 'speaker2'}
        ]
        
        language_segments = [
            {'start': 0.0, 'end': 1.5, 'language': 'english'},
            {'start': 1.5, 'end': 4.0, 'language': 'hindi'}
        ]
        
        merged = asr_service._merge_speaker_language_segments(
            speaker_segments, language_segments
        )
        
        assert len(merged) >= len(speaker_segments)
        # Check that language information is added
        assert all('language' in segment for segment in merged)


class TestASRTextProcessing:
    """Test ASR text processing and postprocessing"""
    
    def test_post_process_text(self, asr_service):
        """Test text postprocessing"""
        # Test basic text cleanup
        text = " hello , world . how are you ? "
        processed = asr_service._post_process_text(text, 'english')
        
        assert processed.strip() == processed  # No leading/trailing spaces
        assert ',world' not in processed  # Spaces after punctuation
        assert '?' in processed  # Punctuation preserved
    
    def test_post_process_indic_text(self, asr_service):
        """Test Indic language text postprocessing"""
        hindi_text = "नमस्ते दुनिया"
        processed = asr_service._post_process_indic_text(hindi_text, 'hindi')
        
        assert isinstance(processed, str)
        assert len(processed) > 0
    
    def test_calculate_asr_quality(self, asr_service):
        """Test ASR quality calculation"""
        transcriptions = [
            {
                'text': 'Hello world',
                'confidence': 0.95,
                'start': 0.0,
                'end': 2.0
            },
            {
                'text': 'How are you',
                'confidence': 0.88,
                'start': 2.0,
                'end': 4.0
            }
        ]
        
        quality = asr_service._calculate_asr_quality(transcriptions)
        
        assert 'average_confidence' in quality
        assert 'total_transcribed_duration' in quality
        assert 'speaking_rate_wpm' in quality
        assert quality['average_confidence'] > 0.8
        assert quality['total_transcribed_duration'] == 4.0


class TestASRErrorHandling:
    """Test ASR error handling"""
    
    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self, asr_service):
        """Test handling of empty audio"""
        empty_audio = np.array([])
        
        with patch.object(asr_service, 'whisper_models', mock_whisper_models()):
            result = await asr_service.transcribe_audio(
                job_id="test_job",
                audio_data=empty_audio,
                sample_rate=16000,
                language="english"
            )
        
        # Should handle gracefully
        assert result is not None
        assert 'transcriptions' in result
    
    @pytest.mark.asyncio
    async def test_transcribe_invalid_segments(self, asr_service, mock_audio_data):
        """Test handling of invalid segments"""
        audio_data, sample_rate = mock_audio_data
        
        # Invalid segments (negative time, out of bounds)
        invalid_segments = [
            {'start': -1.0, 'end': 1.0, 'speaker': 'speaker1'},
            {'start': 1.0, 'end': 10.0, 'speaker': 'speaker2'},  # Beyond audio length
            {'start': 2.0, 'end': 1.5, 'speaker': 'speaker3'}    # End before start
        ]
        
        language_segments = [
            {'start': 0.0, 'end': 2.0, 'language': 'english'}
        ]
        
        with patch.object(asr_service, 'whisper_models', mock_whisper_models()):
            result = await asr_service.transcribe_segments(
                job_id="test_job",
                audio_data=audio_data,
                sample_rate=sample_rate,
                segments=invalid_segments,
                language_segments=language_segments
            )
        
        # Should filter out invalid segments
        assert result is not None
        assert 'transcriptions' in result
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test handling of model loading failure"""
        with patch('src.services.asr_service.whisper.load_model') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                ASRService()


class TestASRPerformance:
    """Test ASR performance characteristics"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_transcription_performance(self, asr_service, mock_audio_data):
        """Test transcription performance"""
        audio_data, sample_rate = mock_audio_data
        
        import time
        
        with patch.object(asr_service, 'whisper_models', mock_whisper_models()):
            start_time = time.time()
            
            result = await asr_service.transcribe_audio(
                job_id="test_job",
                audio_data=audio_data,
                sample_rate=sample_rate,
                language="english"
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
        
        # Should complete in reasonable time
        assert processing_time < 10.0  # 10 seconds max for test
        assert result is not None
    
    @pytest.mark.benchmark
    def test_transcription_benchmark(self, benchmark, asr_service, mock_audio_data):
        """Benchmark transcription performance"""
        audio_data, sample_rate = mock_audio_data
        
        async def transcribe():
            with patch.object(asr_service, 'whisper_models', mock_whisper_models()):
                return await asr_service.transcribe_audio(
                    job_id="test_job",
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    language="english"
                )
        
        # Use pytest-benchmark if available
        result = benchmark(lambda: asyncio.run(transcribe()))
        assert result is not None


class TestASRConfiguration:
    """Test ASR configuration handling"""
    
    def test_load_configuration(self, asr_service):
        """Test loading ASR configuration"""
        # Should use default configuration
        assert hasattr(asr_service, 'whisper_models')
        assert hasattr(asr_service, 'asr_inference')
    
    def test_model_variants(self, asr_service):
        """Test different model variants"""
        # Test that service can handle different model configurations
        models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v3']
        
        for model in models:
            # Should be able to select appropriate model
            selected = asr_service._select_model_for_language('english')
            assert selected in asr_service.whisper_models or selected == 'multilingual'


@pytest.mark.integration
class TestASRIntegration:
    """Integration tests for ASR service"""
    
    @pytest.mark.skipif(not os.path.exists('models/whisper'), reason="Whisper models not available")
    @pytest.mark.asyncio
    async def test_real_model_transcription(self):
        """Test with real Whisper model (if available)"""
        # This test would use actual models
        service = ASRService()
        
        # Create test audio (simple sine wave)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        
        result = await service.transcribe_audio(
            job_id="integration_test",
            audio_data=audio_data.astype(np.float32),
            sample_rate=16000,
            language="english"
        )
        
        assert result is not None
        assert 'transcriptions' in result


def test_asr_service_synchronous_methods(asr_service):
    """Test synchronous methods for Celery compatibility"""
    # Test that sync methods exist and work
    assert hasattr(asr_service, 'load_diarization_results_sync')
    assert hasattr(asr_service, 'load_language_results_sync')
    assert hasattr(asr_service, 'transcribe_segments_sync')
    
    # Test sync methods don't crash
    result1 = asr_service.load_diarization_results_sync("test_job")
    result2 = asr_service.load_language_results_sync("test_job")
    result3 = asr_service.transcribe_segments_sync("test_job", [], [], [])
    
    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert isinstance(result3, dict)