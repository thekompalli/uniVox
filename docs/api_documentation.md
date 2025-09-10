# PS-06 Competition System API Documentation

## Overview

The PS-06 Competition System provides a RESTful API for processing audio files through speaker identification, diarization, language identification, automatic speech recognition, and neural machine translation.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication for basic operations. In production environments, appropriate authentication mechanisms should be implemented.

## Rate Limiting

The API implements rate limiting to ensure fair usage:
- Default: 100 requests per minute per IP
- Configurable via environment variables

## Response Format

All API responses follow a consistent format:

```json
{
  "success": boolean,
  "data": object | null,
  "error": object | null,
  "timestamp": "ISO 8601 timestamp"
}
```

### Success Response
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "error": "ERROR_CODE",
    "message": "Human readable error message",
    "detail": "Detailed error information",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Endpoints

### Health Endpoints

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "uptime": "2 days, 3:45:12",
    "version": "1.0.0"
  }
}
```

#### GET /health/detailed
Detailed system health with component status.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "uptime": "2 days, 3:45:12",
    "version": "1.0.0",
    "components": {
      "database": {"healthy": true},
      "triton": {"healthy": true},
      "storage": {"healthy": true},
      "celery": {"healthy": true}
    },
    "system_info": {
      "platform": "Linux",
      "python_version": "3.10.0",
      "total_memory": 16.0,
      "available_memory": 8.5,
      "cpu_count": 8,
      "gpu_count": 1
    }
  }
}
```

#### GET /health/ready
Kubernetes readiness probe endpoint.

**Responses:**
- `200 OK`: System is ready
- `503 Service Unavailable`: System is not ready

#### GET /health/live
Kubernetes liveness probe endpoint.

**Response:** Always returns `200 OK` if service is running.

### Processing Endpoints

#### POST /process
Process an audio file through the complete pipeline.

**Request:**
- Content-Type: `multipart/form-data`

**Parameters:**
- `audio_file` (file, required): Audio file to process
- `languages` (string, optional): Comma-separated list of expected languages (default: "english,hindi,punjabi")
- `speaker_gallery` (string, optional): Comma-separated list of known speaker IDs
- `quality_mode` (string, optional): Processing quality mode - "fast", "balanced", "high" (default: "balanced")
- `enable_overlaps` (boolean, optional): Enable overlap detection (default: true)
- `min_segment_duration` (float, optional): Minimum segment duration in seconds (default: 0.5)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "audio_file=@sample.wav" \
  -F "languages=english,hindi" \
  -F "quality_mode=balanced" \
  -F "enable_overlaps=true"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "QUEUED",
    "progress": 0.0,
    "created_at": "2024-01-15T10:30:00Z",
    "estimated_completion": "2024-01-15T10:35:00Z",
    "audio_specs": {
      "format": "wav",
      "duration": 120.5,
      "sample_rate": 16000,
      "channels": 1,
      "size_bytes": 3849600
    }
  }
}
```

#### GET /status/{job_id}
Get the current status of a processing job.

**Parameters:**
- `job_id` (string, required): Job identifier from the process request

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "PROCESSING",
    "progress": 0.65,
    "current_step": "ASR",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:33:15Z",
    "estimated_completion": "2024-01-15T10:35:00Z",
    "error_msg": null
  }
}
```

**Status Values:**
- `QUEUED`: Job is queued for processing
- `PREPROCESSING`: Audio preprocessing
- `DIARIZATION`: Speaker diarization
- `LANGUAGE_ID`: Language identification
- `TRANSCRIPTION`: Speech recognition
- `TRANSLATION`: Neural machine translation
- `FORMATTING`: Output formatting
- `COMPLETED`: Job completed successfully
- `FAILED`: Job failed
- `CANCELLED`: Job was cancelled

#### GET /result/{job_id}
Get the complete results of a processed job.

**Parameters:**
- `job_id` (string, required): Job identifier

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "COMPLETED",
    "progress": 1.0,
    "processing_time": 145.2,
    "audio_specs": {
      "format": "wav",
      "duration": 120.5,
      "sample_rate": 16000,
      "channels": 1
    },
    "sid_csv": "/path/to/SID_01.csv",
    "sd_csv": "/path/to/SD_01.csv",
    "lid_csv": "/path/to/LID_01.csv",
    "asr_trn": "/path/to/ASR_01.trn",
    "nmt_txt": "/path/to/NMT_01.txt",
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "speaker": "speaker1",
        "speaker_id": "ID_001",
        "language": "english",
        "text": "Hello, welcome to our presentation",
        "translated_text": "Hello, welcome to our presentation",
        "confidence": 0.95
      }
    ],
    "speakers_detected": 3,
    "languages_detected": ["english", "hindi"],
    "metrics": {
      "speaker_identification_accuracy": 0.92,
      "diarization_error_rate": 0.15,
      "language_identification_accuracy": 0.88,
      "word_error_rate": 0.12,
      "bleu_score": 35.2,
      "real_time_factor": 1.8
    },
    "solution_hash": "a1b2c3d4e5f6..."
  }
}
```

#### POST /batch
Process multiple audio files in batch.

**Request:**
```json
{
  "files": [
    "/path/to/file1.wav",
    "/path/to/file2.mp3"
  ],
  "common_settings": {
    "languages": ["english", "hindi"],
    "quality_mode": "balanced",
    "enable_overlaps": true
  },
  "priority": 1
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_550e8400-e29b-41d4-a716-446655440000",
    "total_files": 2,
    "status": "PROCESSING",
    "jobs": [
      {
        "job_id": "job1_id",
        "file": "/path/to/file1.wav",
        "status": "QUEUED"
      },
      {
        "job_id": "job2_id", 
        "file": "/path/to/file2.mp3",
        "status": "QUEUED"
      }
    ]
  }
}
```

#### GET /batch/{batch_id}
Get batch processing status.

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_550e8400-e29b-41d4-a716-446655440000",
    "total_files": 2,
    "completed": 1,
    "failed": 0,
    "progress": 0.5,
    "results": [
      {
        "job_id": "job1_id",
        "status": "COMPLETED",
        "file": "/path/to/file1.wav"
      },
      {
        "job_id": "job2_id",
        "status": "PROCESSING", 
        "file": "/path/to/file2.mp3"
      }
    ]
  }
}
```

### Monitoring Endpoints

#### GET /metrics
Get system performance metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "requests_total": 15420,
    "requests_successful": 14876,
    "requests_failed": 544,
    "average_response_time": 0.145,
    "jobs_processed": 1247,
    "jobs_queued": 12,
    "jobs_failed": 23,
    "system_load": 0.68,
    "memory_usage": 0.72,
    "gpu_utilization": 0.85
  }
}
```

#### GET /version
Get system version information.

**Response:**
```json
{
  "success": true,
  "data": {
    "version": "1.0.0",
    "build_date": "2024-01-15T08:00:00Z",
    "git_commit": "abc123def456",
    "python_version": "3.10.0",
    "dependencies": {
      "fastapi": "0.104.1",
      "torch": "2.1.1",
      "transformers": "4.36.2"
    }
  }
}
```

## Competition Output Formats

The system generates competition-compliant output files:

### Speaker Identification (SID_XX.csv)
```csv
Audio File Name, speaker ID, confidence score (in %), start TS, end TS
ps6_01_001.wav, ID1, 95, 100.006, 120.002
ps6_01_001.wav, ID2, 93, 118.080, 200.256
```

### Speaker Diarization (SD_XX.csv)
```csv
Audio File Name, speaker, confidence score (in %), start TS, end TS
ps6_01_001.wav, speaker1, 95, 100.006, 120.002
ps6_01_001.wav, speaker2, 93, 118.080, 200.256
```

### Language Identification (LID_XX.csv)
```csv
Audio File Name, language, confidence score (in %), start TS, end TS
ps6_01_001.wav, english, 95, 100.006, 120.002
ps6_01_001.wav, hindi, 93, 118.080, 200.256
```

### Automatic Speech Recognition (ASR_XX.trn)
```
Audio File Name, start TS, end TS, Transcript
ps6_01_001.wav, 100.006, 120.002, hi how are you?
ps6_01_001.wav, 118.080, 200.256, आज थोड़ा काम है
```

### Neural Machine Translation (NMT_XX.txt)
```
Audio File Name, start TS, end TS, Translation
ps6_01_001.wav, 100.006, 120.002, hi how are you?
ps6_01_001.wav, 118.080, 200.256, I have some work today
```

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_FILE_FORMAT` | Unsupported audio file format |
| `FILE_TOO_LARGE` | Audio file exceeds size limit |
| `INVALID_PARAMETERS` | Invalid request parameters |
| `JOB_NOT_FOUND` | Job ID not found |
| `PROCESSING_FAILED` | Audio processing failed |
| `STORAGE_ERROR` | File storage error |
| `MODEL_ERROR` | ML model error |
| `SYSTEM_OVERLOAD` | System is overloaded |

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/process` | 10 requests per minute |
| `/status/*` | 100 requests per minute |
| `/result/*` | 50 requests per minute |
| `/health*` | 1000 requests per minute |

## File Format Support

### Supported Audio Formats
- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A

### Audio Specifications
- Sample Rate: 8kHz to 48kHz
- Bit Depth: 8 to 32 bits
- Channels: Mono or Stereo (converted to mono)
- Maximum file size: 500MB
- SNR: 5dB or better recommended

## Language Support

### Stage 1 Languages
- English
- Hindi  
- Punjabi

### Stage 2 Languages
- English
- Hindi
- Punjabi
- Bengali
- Nepali
- Dogri

## SDKs and Examples

### Python SDK Example
```python
import requests
import time

# Upload and process file
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/process',
        files={'audio_file': f},
        data={'languages': 'english,hindi'}
    )

job_data = response.json()['data']
job_id = job_data['job_id']

# Poll for completion
while True:
    status_response = requests.get(f'http://localhost:8000/api/v1/status/{job_id}')
    status = status_response.json()['data']['status']
    
    if status == 'COMPLETED':
        break
    elif status == 'FAILED':
        print("Processing failed")
        break
    
    time.sleep(5)

# Get results
result_response = requests.get(f'http://localhost:8000/api/v1/result/{job_id}')
results = result_response.json()['data']
print(f"Processing completed with {len(results['segments'])} segments")
```

### JavaScript/Node.js Example
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function processAudio(audioFilePath) {
    const form = new FormData();
    form.append('audio_file', fs.createReadStream(audioFilePath));
    form.append('languages', 'english,hindi');
    
    // Submit job
    const response = await axios.post(
        'http://localhost:8000/api/v1/process',
        form,
        { headers: form.getHeaders() }
    );
    
    const jobId = response.data.data.job_id;
    
    // Poll for completion
    while (true) {
        const statusResponse = await axios.get(
            `http://localhost:8000/api/v1/status/${jobId}`
        );
        
        const status = statusResponse.data.data.status;
        
        if (status === 'COMPLETED') {
            break;
        } else if (status === 'FAILED') {
            throw new Error('Processing failed');
        }
        
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    // Get results
    const resultResponse = await axios.get(
        `http://localhost:8000/api/v1/result/${jobId}`
    );
    
    return resultResponse.data.data;
}
```

## WebSocket Support

For real-time status updates, the API supports WebSocket connections:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/status/' + jobId);

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Progress:', update.progress);
    console.log('Step:', update.current_step);
};
```

## Troubleshooting

### Common Issues

1. **File Upload Fails**
   - Check file format is supported
   - Verify file size is under 500MB
   - Ensure audio file is not corrupted

2. **Processing Takes Too Long**
   - Large files may take 2x real-time to process
   - Check system load with `/metrics` endpoint
   - Consider using "fast" quality mode

3. **Poor Results Quality**
   - Ensure audio quality meets requirements (SNR > 5dB)
   - Check if correct languages are specified
   - Try "high" quality mode for better accuracy

4. **API Errors**
   - Check API response error codes
   - Verify request format matches documentation
   - Check system health with `/health` endpoint

### Support

For technical support:
- GitHub Issues: [repository-url]/issues
- API Status: Check `/health` endpoint
- Logs: Available in system logs (if accessible)