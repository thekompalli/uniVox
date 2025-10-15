"""
Model Configuration
Configuration for ML models and inference parameters
"""
import os
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = '') -> str:
    value = os.getenv(key, default)
    if value is None:
        return default
    cleaned = value.split('#', 1)[0].strip()
    return cleaned or default

class ModelConfig:
    """Configuration for ML models"""
    
    # Base model directory
    models_dir = Path(os.getenv('MODELS_DIR', 'models'))
    
    # Whisper ASR Models
    whisper_model = _env('WHISPER_MODEL', 'large-v3')
    whisper_hindi_model = _env('WHISPER_HINDI_MODEL', '')  # Optional fine-tuned model
    whisper_bengali_model = _env('WHISPER_BENGALI_MODEL', '')  # Optional fine-tuned model
    whisper_hf_model = _env('WHISPER_HF_MODEL', 'openai/whisper-large-v3')
    whisper_backend = _env('WHISPER_BACKEND', 'openai').lower()
    fast_whisper_model_path = _env('FAST_WHISPER_MODEL_PATH', '')
    fast_whisper_device = _env('FAST_WHISPER_DEVICE', '')
    fast_whisper_compute_type = _env('FAST_WHISPER_COMPUTE_TYPE', '')
    fast_whisper_vad_filter = _env('FAST_WHISPER_VAD_FILTER', 'false').lower() == 'true'

    
    # Whisper inference parameters
    whisper_beam_size = int(os.getenv('WHISPER_BEAM_SIZE', '5'))
    whisper_temperature = float(os.getenv('WHISPER_TEMPERATURE', '0.0'))
    whisper_no_repeat_ngram_size = int(os.getenv('WHISPER_NO_REPEAT_NGRAM_SIZE', '2'))
    
    # pyannote.audio Models
    pyannote_model = os.getenv('PYANNOTE_MODEL', 'pyannote/speaker-diarization-3.1')
    pyannote_segmentation_model = os.getenv('PYANNOTE_SEGMENTATION_MODEL', 'pyannote/segmentation-3.0')
    pyannote_embedding_model = os.getenv('PYANNOTE_EMBEDDING_MODEL', 'pyannote/embedding')
    
    # Diarization parameters
    diarization_threshold = float(os.getenv('DIARIZATION_THRESHOLD', '0.5'))
    min_speakers = int(os.getenv('MIN_SPEAKERS', '1'))
    max_speakers = int(os.getenv('MAX_SPEAKERS', '10'))
    
    # Speaker Models
    wespeaker_model = os.getenv('WESPEAKER_MODEL', 'english')
    speaker_embedding_model = os.getenv('SPEAKER_EMBEDDING_MODEL', 'speechbrain/spkrec-ecapa-voxceleb')
    speaker_similarity_threshold = float(os.getenv('SPEAKER_SIMILARITY_THRESHOLD', '0.6'))
    min_speaker_confidence = float(os.getenv('MIN_SPEAKER_CONFIDENCE', '0.4'))
    
    # Language Models
    wav2vec2_lang_model = os.getenv('WAV2VEC2_LANG_MODEL', 'facebook/wav2vec2-large-xlsr-53')
    language_confidence_threshold = float(os.getenv('LANGUAGE_CONFIDENCE_THRESHOLD', '0.5'))
    
    # Translation Models
    indictrans_model = os.getenv('INDICTRANS_MODEL', 'ai4bharat/indictrans2-en-indic-1B')
    nllb_model = os.getenv('NLLB_MODEL', 'facebook/nllb-200-1.3B')
    
    # Translation parameters
    indictrans_beam_size = int(os.getenv('INDICTRANS_BEAM_SIZE', '4'))
    indictrans_max_length = int(os.getenv('INDICTRANS_MAX_LENGTH', '512'))
    nllb_max_length = int(os.getenv('NLLB_MAX_LENGTH', '400'))
    
    # Triton Inference Server Models
    triton_diarization_model = os.getenv('TRITON_DIARIZATION_MODEL', 'diarization')
    triton_speaker_model = os.getenv('TRITON_SPEAKER_MODEL', 'speaker_embedding')
    triton_language_model = os.getenv('TRITON_LANGUAGE_MODEL', 'language_id')
    triton_asr_model = os.getenv('TRITON_ASR_MODEL', 'whisper_asr')
    triton_translation_model = os.getenv('TRITON_TRANSLATION_MODEL', 'indictrans2')
    
    # Model Loading Parameters
    model_cache_dir = models_dir / "cache"
    model_download_timeout = int(os.getenv('MODEL_DOWNLOAD_TIMEOUT', '3600'))  # 1 hour
    model_retry_attempts = int(os.getenv('MODEL_RETRY_ATTEMPTS', '3'))
    
    # GPU Memory Management
    gpu_memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
    mixed_precision = os.getenv('MIXED_PRECISION', 'true').lower() == 'true'
    gradient_checkpointing = os.getenv('GRADIENT_CHECKPOINTING', 'true').lower() == 'true'
    
    # Batch Processing
    max_batch_size_asr = int(os.getenv('MAX_BATCH_SIZE_ASR', '8'))
    max_batch_size_speaker = int(os.getenv('MAX_BATCH_SIZE_SPEAKER', '16'))
    max_batch_size_language = int(os.getenv('MAX_BATCH_SIZE_LANGUAGE', '16'))
    max_batch_size_translation = int(os.getenv('MAX_BATCH_SIZE_TRANSLATION', '8'))
    
    # Quality Thresholds
    min_audio_quality_snr = float(os.getenv('MIN_AUDIO_QUALITY_SNR', '5.0'))  # 5dB minimum SNR
    min_segment_confidence = float(os.getenv('MIN_SEGMENT_CONFIDENCE', '0.3'))
    min_transcription_confidence = float(os.getenv('MIN_TRANSCRIPTION_CONFIDENCE', '0.4'))
    min_translation_confidence = float(os.getenv('MIN_TRANSLATION_CONFIDENCE', '0.3'))
    
    # Model-specific parameters
    silero_vad_model = os.getenv('SILERO_VAD_MODEL', 'silero_vad')
    silero_confidence_threshold = float(os.getenv('SILERO_CONFIDENCE_THRESHOLD', '0.5'))
    
    # Competition-specific settings
    competition_languages = ['english', 'hindi', 'punjabi', 'bengali', 'nepali', 'dogri']
    primary_languages = ['english', 'hindi', 'punjabi']  # Stage 1
    secondary_languages = ['bengali', 'nepali', 'dogri']  # Stage 2
    
    # Performance targets (competition requirements)
    target_speaker_accuracy = 0.85  # >85%
    target_diarization_error_rate = 0.20  # <20%
    target_word_error_rate = 0.25  # <25%
    target_bleu_score = 30.0  # >30
    target_rtf = 2.0  # Real-time factor <2x
    
    # HuggingFace settings
    huggingface_token = os.getenv('HUGGINGFACE_TOKEN', '')
    huggingface_cache_dir = os.getenv('HUGGINGFACE_CACHE_DIR', str(models_dir / "huggingface"))
    
    # Model URLs and checksums for verification
    model_urls = {
        'whisper_large_v3': 'https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt',
        'pyannote_diarization': 'pyannote/speaker-diarization-3.1',
        'indictrans2': 'ai4bharat/indictrans2-en-indic-1B',
        'nllb_1.3b': 'facebook/nllb-200-1.3B'
    }
    
    model_checksums = {
        'whisper_large_v3': 'e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb',
    }
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get full path for a model"""
        return cls.models_dir / model_name
    
    @classmethod
    def is_model_available(cls, model_name: str) -> bool:
        """Check if model is available locally"""
        return cls.get_model_path(model_name).exists()
    
    @classmethod
    def get_language_model_for_task(cls, task: str, language: str) -> str:
        """Get best model for a specific task and language"""
        if task == 'asr':
            if language == 'hindi' and cls.whisper_hindi_model:
                return cls.whisper_hindi_model
            elif language == 'bengali' and cls.whisper_bengali_model:
                return cls.whisper_bengali_model
            else:
                return cls.whisper_model
        
        elif task == 'translation':
            if language in ['hindi', 'bengali', 'punjabi', 'nepali', 'dogri']:
                return cls.indictrans_model
            else:
                return cls.nllb_model
        
        return ""
    
    @classmethod
    def get_inference_config(cls, task: str) -> dict:
        """Get inference configuration for a specific task"""
        configs = {
            'asr': {
                'beam_size': cls.whisper_beam_size,
                'temperature': cls.whisper_temperature,
                'no_repeat_ngram_size': cls.whisper_no_repeat_ngram_size,
                'max_batch_size': cls.max_batch_size_asr,
                'confidence_threshold': cls.min_transcription_confidence
            },
            'diarization': {
                'threshold': cls.diarization_threshold,
                'min_speakers': cls.min_speakers,
                'max_speakers': cls.max_speakers,
                'min_segment_duration': 0.5
            },
            'speaker_id': {
                'similarity_threshold': cls.speaker_similarity_threshold,
                'confidence_threshold': cls.min_speaker_confidence,
                'max_batch_size': cls.max_batch_size_speaker,
                'embedding_dimension': 256
            },
            'language_id': {
                'confidence_threshold': cls.language_confidence_threshold,
                'max_batch_size': cls.max_batch_size_language,
                'supported_languages': cls.competition_languages
            },
            'translation': {
                'beam_size': cls.indictrans_beam_size,
                'max_length': cls.indictrans_max_length,
                'confidence_threshold': cls.min_translation_confidence,
                'max_batch_size': cls.max_batch_size_translation
            }
        }
        
        return configs.get(task, {})
    
    @classmethod
    def validate_config(cls) -> dict:
        """Validate model configuration"""
        issues = []
        
        # Check required directories
        if not cls.models_dir.exists():
            issues.append(f"Models directory does not exist: {cls.models_dir}")
        
        # Check GPU settings
        if cls.gpu_memory_fraction <= 0 or cls.gpu_memory_fraction > 1:
            issues.append(f"Invalid GPU memory fraction: {cls.gpu_memory_fraction}")
        
        # Check thresholds
        thresholds = {
            'diarization_threshold': cls.diarization_threshold,
            'speaker_similarity_threshold': cls.speaker_similarity_threshold,
            'language_confidence_threshold': cls.language_confidence_threshold
        }
        
        for name, value in thresholds.items():
            if value < 0 or value > 1:
                issues.append(f"Invalid threshold {name}: {value} (should be 0-1)")
        
        # Check batch sizes
        batch_sizes = {
            'max_batch_size_asr': cls.max_batch_size_asr,
            'max_batch_size_speaker': cls.max_batch_size_speaker,
            'max_batch_size_language': cls.max_batch_size_language,
            'max_batch_size_translation': cls.max_batch_size_translation
        }
        
        for name, size in batch_sizes.items():
            if size <= 0 or size > 64:
                issues.append(f"Invalid batch size {name}: {size}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


# Global model config instance
model_config = ModelConfig()