#!/usr/bin/env python3
"""
Simple local runner for PS-06 system
Runs without Docker, external databases, or cloud services
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for local development
os.environ.update({
    'DATABASE_URL': 'sqlite:///./local_ps06.db',
    'REDIS_URL': 'redis://localhost:6379/0',
    'DEBUG': 'true',
    'LOG_LEVEL': 'INFO',
    'USE_CELERY': 'false',  # Disable async processing for simplicity
    'MODELS_DIR': str(project_root / 'models'),
    'DATA_DIR': str(project_root / 'data'),
    'LOGS_DIR': str(project_root / 'logs'),
    'TEMP_DIR': str(project_root / 'temp'),
    'MAX_FILE_SIZE': '100000000',  # 100MB
    'PYTHONPATH': str(project_root)
})

# Create necessary directories
for dir_name in ['data', 'logs', 'temp', 'models']:
    (project_root / dir_name).mkdir(exist_ok=True)

if __name__ == "__main__":
    print("🚀 Starting PS-06 System Locally")
    print("=" * 40)
    print(f"📁 Project Root: {project_root}")
    print(f"🌐 API will be available at: http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print("=" * 40)
    
    # Start the FastAPI server
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )