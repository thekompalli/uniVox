#!/usr/bin/env python3
"""
Model Download Script for PS-06 System
Downloads all required models from HuggingFace Hub
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from datetime import datetime

# Load environment variables from local .env if present
try:
    from dotenv import load_dotenv
    # Ensure we load the .env that sits next to this script
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except Exception:
    # Safe to ignore if python-dotenv is not available
    pass

# Model configurations
MODELS = {
    "silero_vad": {
        "repo": "silero/silero-vad", 
        "local_dir": "models/silero/vad",
        "size": "0.1GB"
    },
    "pyannote_segmentation": {
        "repo": "pyannote/segmentation-3.0",
        "local_dir": "models/pyannote/segmentation", 
        "size": "0.2GB"
    },
    "wespeaker_voxceleb": {
        "repo": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "local_dir": "models/wespeaker/voxceleb",
        "size": "0.3GB"
    },
    "pyannote_diarization": {
        "repo": "pyannote/speaker-diarization-3.1", 
        "local_dir": "models/pyannote/diarization",
        "size": "0.5GB"
    },
    "speechbrain_lang_id": {
        "repo": "speechbrain/lang-id-voxlingua107-ecapa",
        "local_dir": "models/speechbrain/lang_id", 
        "size": "0.8GB"
    },
    "wav2vec2_xlsr": {
        "repo": "facebook/wav2vec2-large-xlsr-53",
        "local_dir": "models/wav2vec2/xlsr_53",
        "size": "1.2GB"
    },
    "nllb_distilled": {
        "repo": "facebook/nllb-200-distilled-600M",
        "local_dir": "models/nllb/distilled_600m",
        "size": "2.4GB"
    },
    "whisper_large_v3": {
        "repo": "openai/whisper-large-v3", 
        "local_dir": "models/whisper/large_v3",
        "size": "3.1GB"
    },
    "indictrans2_1b": {
        "repo": "ai4bharat/indictrans2-en-indic-1B",
        "local_dir": "models/indictrans2/en_indic_1b", 
        "size": "4.2GB"
    }
}

def download_model(model_name, model_info):
    """Download a model from HuggingFace Hub"""
    print(f"\nDownloading {model_name} ({model_info['size']})...")
    print(f"   Repository: {model_info['repo']}")
    print(f"   Local directory: {model_info['local_dir']}")
    
    try:
        # Create local directory
        local_dir = Path(model_info['local_dir'])
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        marker_file = local_dir / ".download_complete"
        if marker_file.exists():
            print(f"   {model_name} already downloaded")
            return True
        
        # Prefer a token from standard env var names
        hf_token = (
            os.getenv('HUGGINGFACE_HUB_TOKEN')
            or os.getenv('HUGGINGFACE_TOKEN')
            or os.getenv('HF_TOKEN')
            or os.getenv('PYANNOTE_ACCESS_TOKEN')
        )
        if hf_token:
            print(f"   Auth: using HuggingFace token {hf_token[:8]}…")

        # Download the model
        snapshot_download(
            repo_id=model_info['repo'],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token if hf_token else None
        )
        
        # Create completion marker
        with open(marker_file, 'w') as f:
            f.write(f"Downloaded: {model_info['repo']}\n")
            f.write(f"Size: {model_info['size']}\n")
        
        print(f"   Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        print(f"   Failed to download {model_name}: {e}")
        # Add a helpful hint for gated pyannote repositories
        if str(model_info['repo']).startswith('pyannote/'):
            print("   Note: pyannote models are gated on Hugging Face.")
            print("         Ensure you have accepted access on the model page:")
            print("           - https://huggingface.co/pyannote/segmentation-3.0")
            print("           - https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("         Then set HUGGINGFACE_HUB_TOKEN (or HUGGINGFACE_TOKEN) and retry.")
            print("         See ps06_system/PYANNOTE_ACCESS_INSTRUCTIONS.md for details.")
        return False

def download_silero_vad_special():
    """Special download for Silero VAD using torch.hub"""
    model_name = "silero_vad"
    local_dir = Path("models/silero/vad")
    marker_file = local_dir / ".download_complete"
    
    if marker_file.exists():
        print(f"   {model_name} already downloaded (torch.hub)")
        return True
        
    print(f"\nDownloading {model_name} via torch.hub...")
    
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download Silero VAD using torch.hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # Save model state dict
        model_path = local_dir / "silero_vad.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save utils
        import pickle
        utils_path = local_dir / "vad_utils.pkl" 
        with open(utils_path, 'wb') as f:
            pickle.dump(utils, f)
        
        # Create completion marker
        with open(marker_file, 'w') as f:
            f.write("Downloaded: silero/silero-vad (torch.hub)\n")
            f.write("Size: 0.1GB\n")
        
        print(f"   Successfully downloaded {model_name} via torch.hub")
        return True
        
    except Exception as e:
        print(f"   Failed to download {model_name}: {e}")
        return False

def main():
    print("PS-06 Model Download Script")
    print("=" * 50)
    print("Total download size: ~12GB")
    print("This will take a while depending on your internet connection...")
    
    # Get HuggingFace token if available
    hf_token = (
        os.getenv('HUGGINGFACE_HUB_TOKEN')
        or os.getenv('HUGGINGFACE_TOKEN')
        or os.getenv('HF_TOKEN')
        or os.getenv('PYANNOTE_ACCESS_TOKEN')
    )
    if hf_token:
        print(f"Using HuggingFace token: {hf_token[:8]}...")
    else:
        print("No HuggingFace token found. Some models might not be accessible.")
    
    successful = []
    failed = []
    
    # Download models in order of size (smallest first)
    model_order = sorted(MODELS.keys(), key=lambda x: float(MODELS[x]['size'].replace('GB', '')))
    
    for model_name in model_order:
        model_info = MODELS[model_name]
        
        if download_model(model_name, model_info):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Try special download for Silero VAD
    print("\n" + "="*50)
    print("Attempting special download for Silero VAD...")
    if download_silero_vad_special():
        if "silero_vad" not in successful:
            successful.append("silero_vad")
    
    # Summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Successfully downloaded: {len(successful)}/{len(MODELS)} models")
    
    if successful:
        print("\nSuccessful downloads:")
        for model in successful:
            size = MODELS.get(model, {}).get('size', 'Unknown')
            print(f"  {model} ({size})")
    
    if failed:
        print(f"\nFailed downloads: {len(failed)}")
        for model in failed:
            size = MODELS.get(model, {}).get('size', 'Unknown') 
            print(f"  {model} ({size})")
    
    # Create registry file
    registry_path = Path("models/model_registry.json")
    import json
    registry = {
        "updated": datetime.utcnow().isoformat(),
        "models": {}
    }
    
    for model in successful:
        registry["models"][model] = {
            "repo": MODELS.get(model, {}).get('repo', 'unknown'),
            "local_dir": MODELS.get(model, {}).get('local_dir', 'unknown'),
            "size": MODELS.get(model, {}).get('size', 'unknown'),
            "downloaded": True
        }
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\nModel registry saved to: {registry_path}")
    
    if len(failed) == 0:
        print("\nAll models downloaded successfully!")
        print("You can now run the full PS-06 system with real AI models!")
        return 0
    else:
        print(f"\n{len(failed)} models failed to download.")
        print("You can retry downloading failed models later.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
