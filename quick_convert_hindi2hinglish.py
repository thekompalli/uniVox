#!/usr/bin/env python3
"""
Quick conversion script for Hindi2Hinglish Whisper model
"""
import os
import subprocess
from pathlib import Path

def quick_convert():
    """Convert your existing Hindi2Hinglish model to CT2 format"""

    print("=" * 60)
    print("QUICK HINDI2HINGLISH MODEL CONVERSION")
    print("=" * 60)

    # Step 1: Install required packages
    print("Step 1: Installing required packages...")
    packages = ["ctranslate2", "transformers", "torch", "faster-whisper"]

    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run(["pip", "install", package], check=True)

    # Step 2: Provide conversion commands
    print("\nStep 2: Conversion Commands")
    print("=" * 40)

    # If you have a HuggingFace model
    print("\n🔹 If your model is in HuggingFace format:")
    print("```bash")
    print("ct2-transformers-converter \\")
    print("    --model /path/to/your/hindi2hinglish/model \\")
    print("    --output_dir models/whisper/hindi2hinglish_ct2 \\")
    print("    --copy_files tokenizer.json preprocessor_config.json config.json \\")
    print("    --quantization float16")
    print("```")

    # If you have a PyTorch .pt file (your case)
    print("\n🔹 If you have a .pt file (your case):")
    print("Since your model is in PyTorch .pt format, you'll need to:")
    print("1. First load it back into HuggingFace format")
    print("2. Then convert to CT2")

    print("\nStep 3: Loading your .pt model back to HuggingFace format...")

    conversion_code = '''
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

# Load your converted model
model_path = "Whisper-Hindi2Hinglish-Prime.pt"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Create a new model instance (you'll need to specify the config)
    # This assumes your model is based on Whisper large-v3 architecture
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",  # Base architecture
        torch_dtype=torch.float16
    )

    # Load your custom state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Save in HuggingFace format
    hf_output_dir = "models/whisper/hindi2hinglish_hf"
    os.makedirs(hf_output_dir, exist_ok=True)

    model.save_pretrained(hf_output_dir)

    # Also save the processor/tokenizer
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    processor.save_pretrained(hf_output_dir)

    print(f"✅ Model saved in HuggingFace format to: {hf_output_dir}")

    # Now convert to CT2
    print("Converting to CT2 format...")
    import subprocess

    cmd = [
        "ct2-transformers-converter",
        "--model", hf_output_dir,
        "--output_dir", "models/whisper/hindi2hinglish_ct2",
        "--copy_files", "tokenizer.json", "preprocessor_config.json", "config.json",
        "--quantization", "float16"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Successfully converted to CT2 format!")
        print("Your model is ready at: models/whisper/hindi2hinglish_ct2")
    else:
        print(f"❌ Conversion failed: {result.stderr}")
else:
    print(f"❌ Model file not found: {model_path}")
    print("Please ensure your .pt file is in the correct location")
'''

    # Save the conversion code
    with open("convert_pt_to_ct2.py", "w") as f:
        f.write(conversion_code)

    print("\nStep 4: Run the conversion")
    print("=" * 40)
    print("I've created a conversion script: convert_pt_to_ct2.py")
    print("Run it with: python convert_pt_to_ct2.py")

    # Step 5: Update configuration
    print("\nStep 5: Update your configuration")
    print("=" * 40)
    print("Once conversion is complete, update your .env file:")
    print("PS06_FAST_WHISPER_MODEL_PATH=C:\\\\PS6\\\\ps06_system\\\\models\\\\whisper\\\\hindi2hinglish_ct2")

    print("\n" + "=" * 60)
    print("📝 SUMMARY:")
    print("1. Run: python convert_pt_to_ct2.py")
    print("2. Update PS06_FAST_WHISPER_MODEL_PATH in .env")
    print("3. Restart your application")
    print("=" * 60)

if __name__ == "__main__":
    quick_convert()