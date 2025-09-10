# PS-06 Competition System

**Language Agnostic Speaker Identification & Diarization; and subsequent Transcription & Translation System**

A comprehensive audio processing system designed for the PS-06 Grand Challenge competition, providing offline speaker identification, diarization, language identification, automatic speech recognition, and neural machine translation for multilingual audio files.

## 🎯 Overview

This system transforms spoken audio into structured, multilingual textual insights through an integrated pipeline that supports:

- **Speaker Identification**: Match speakers to known identities
- **Speaker Diarization**: Segment audio by different speakers  
- **Language Identification**: Detect languages in multilingual/code-switched audio
- **Automatic Speech Recognition**: Convert speech to text in original languages
- **Neural Machine Translation**: Translate transcripts to English

## 🏆 Competition Requirements

### Supported Languages
- **Stage 1**: English, Hindi, Punjabi
- **Stage 2**: English, Hindi, Punjabi, Bengali, Nepali, Dogri
- **Stage 3**: Above languages + 5 additional (to be disclosed)

### Audio Format Support
- **Sample Rate**: 8kHz to 48kHz
- **Bit Depth**: 8 to 32 bits  
- **File Types**: WAV, MP3, OGG, FLAC
- **SNR**: 5dB or better

### Performance Targets
- Speaker Identification Accuracy: >85%
- Diarization Error Rate: <20%
- Word Error Rate: <25%
- BLEU Score: >30
- Real-time Factor: <2x

## 🏗️ Architecture

```
ps06_system/
├── src/
│   ├── api/                    # FastAPI REST endpoints
│   ├── services/               # Core business logic
│   ├── models/                 # ML model wrappers
│   ├── repositories/           # Data access layer
│   ├── tasks/                  # Async task processing
│   ├── utils/                  # Utilities
│   └── config/                 # Configuration
├── models/                     # Pre-trained models
├── data/                       # Processing data
├── configs/                    # Configuration files
├── scripts/                    # Setup/deployment scripts
└── tests/                      # Test suite
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 100GB+ storage

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ps06_system
```

2. **Environment setup**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models**
```bash
./scripts/download_models.sh
```

5. **Start services**
```bash
docker-compose up -d
```

6. **Run the application**
```bash
python -m src.api.main
```

### Using Docker

```bash
# Build and start all services
docker-compose up --build

# Access the API
curl http://localhost:8000/api/v1/health
```

## 📖 API Usage

### Process Audio File

```python
import requests

# Upload and process audio
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'audio_file': f},
        data={
            'languages': 'english,hindi,punjabi',
            'quality_mode': 'balanced'
        }
    )

job_id = response.json()['data']['job_id']
```

### Check Processing Status

```python
# Check status
status = requests.get(f'http://localhost:8000/api/v1/status/{job_id}')
print(status.json())
```

### Get Results

```python
# Get competition format results
results = requests.get(f'http://localhost:8000/api/v1/result/{job_id}')
print(results.json())
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ps06_db
REDIS_URL=redis://localhost:6379

# Models
WHISPER_MODEL=large-v3
PYANNOTE_MODEL=pyannote/speaker-diarization-3.1
INDICTRANS_MODEL=ai4bharat/indictrans2-en-indic-1B

# Processing
MAX_FILE_SIZE=500000000
TARGET_SAMPLE_RATE=16000
CHUNK_DURATION=30

# GPU
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

Edit `configs/model_configs/` files to customize model parameters:

- `whisper_config.yaml` - ASR model settings
- `pyannote_config.yaml` - Diarization parameters  
- `indictrans_config.yaml` - Translation settings

## 🧪 Testing

```bash
# Run all tests
./scripts/run_tests.sh

# Run specific test suite
pytest tests/test_services/test_asr_service.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📊 Monitoring

### Health Checks

```bash
# Basic health
curl http://localhost:8000/api/v1/health

# Detailed system status
curl http://localhost:8000/api/v1/health/detailed

# Metrics
curl http://localhost:8000/api/v1/metrics
```

### Performance Monitoring

- **Logs**: `logs/ps06_system.log`
- **Metrics**: `logs/ps06_performance.log`
- **Audit**: `logs/ps06_audit.log`

## 🏁 Competition Submission

### Generate Submission Package

```python
from src.services.format_service import FormatService

format_service = FormatService()
package_path = await format_service.create_submission_package(
    job_id="your-job-id",
    evaluation_id="01"
)
```

### Output Files

The system generates competition-compliant output files:

- `SID_XX.csv` - Speaker identification results
- `SD_XX.csv` - Speaker diarization results
- `LID_XX.csv` - Language identification results
- `ASR_XX.trn` - Speech recognition transcripts
- `NMT_XX.txt` - Translation results
- `PS_06_<app_id>_<eval_id>.hash` - Solution verification hash

## 🔬 Model Details

### Core Models

| Component | Model | Purpose |
|-----------|-------|---------|
| ASR | Whisper Large-v3 | Multilingual speech recognition |
| Diarization | pyannote/speaker-diarization-3.1 | Speaker segmentation |
| Speaker ID | WeSpeaker | Speaker embedding extraction |
| Language ID | Wav2Vec2-XLSR-53 | Language identification |
| Translation | IndicTrans2, NLLB-200 | Neural machine translation |
| VAD | Silero VAD | Voice activity detection |

### Performance Optimizations

- **GPU acceleration** for all models
- **Triton Inference Server** support for production deployment
- **Async processing** with Celery
- **Model caching** and warm-up
- **Batch processing** for efficiency

## 🛠️ Development

### Project Structure

```python
# Core services
from src.services.orchestrator_service import OrchestratorService
from src.services.diarization_service import DiarizationService
from src.services.asr_service import ASRService

# Model inference
from src.models.asr_inference import ASRInference
from src.models.diarization_inference import DiarizationInference

# Utilities
from src.utils.audio_utils import AudioUtils
from src.utils.format_utils import FormatUtils
```

### Adding New Languages

1. Update language enums in `src/api/schemas/common_schemas.py`
2. Add language mappings in model inference classes
3. Update configuration files
4. Test with sample audio

### Custom Models

1. Implement model wrapper in `src/models/`
2. Add configuration in `src/config/model_config.py`
3. Update service layer integration
4. Add tests

## 📝 Troubleshooting

### Common Issues

**GPU Out of Memory**
```bash
# Reduce batch sizes in model configs
WHISPER_BATCH_SIZE=1
DIARIZATION_BATCH_SIZE=1
```

**Model Download Failures**
```bash
# Set HuggingFace token for private models
export HUGGINGFACE_TOKEN=your_token_here
```

**Processing Timeouts**
```bash
# Increase timeout settings
CELERY_TASK_TIMEOUT=3600
PROCESSING_TIMEOUT=1800
```

### Performance Tuning

- Use GPU-optimized Docker images
- Enable mixed precision training
- Optimize chunk sizes for your hardware
- Use faster storage (SSD) for model loading


## 🤝 Contributing (internal only with Flyers Soft)

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For technical support and questions:

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [contact information]

## 📈 Roadmap

- [ ] Support for additional Indian languages
- [ ] Real-time streaming processing
- [ ] Model quantization for edge deployment
- [ ] Integration with cloud platforms
- [ ] Advanced speaker adaptation
- [ ] Multi-modal processing (audio + text)

---