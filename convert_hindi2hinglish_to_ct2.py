#!/usr/bin/env python3
"""
Convert your Hindi2Hinglish Whisper model to CT2 format for faster-whisper
"""
import os
import subprocess
from pathlib import Path

def convert_model_to_ct2():
    """Convert your custom model to CT2 format"""

    # Paths
    input_model = "path/to/your/hindi2hinglish/model"  # Replace with your model path
    output_dir = Path("models/whisper/hindi2hinglish_ct2")

    print("=" * 60)
    print("CONVERTING HINDI2HINGLISH MODEL TO CT2 FORMAT")
    print("=" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if ctranslate2 is installed
    try:
        result = subprocess.run(["ct2-transformers-converter", "--help"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Installing ctranslate2...")
            subprocess.run(["pip", "install", "ctranslate2[transformers]"], check=True)
    except FileNotFoundError:
        print("Installing ctranslate2...")
        subprocess.run(["pip", "install", "ctranslate2[transformers]"], check=True)

    # Convert model
    print(f"Converting model from: {input_model}")
    print(f"Output directory: {output_dir}")

    cmd = [
        "ct2-transformers-converter",
        "--model", str(input_model),
        "--output_dir", str(output_dir),
        "--copy_files", "tokenizer.json", "preprocessor_config.json", "config.json",
        "--quantization", "float16"
    ]

    try:
        print("Running conversion command:")
        print(" ".join(cmd))
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        print("✅ Conversion successful!")
        print(f"CT2 model saved to: {output_dir}")

        # List the converted files
        if output_dir.exists():
            print("\nConverted files:")
            for file in output_dir.iterdir():
                print(f"  - {file.name}")

        return str(output_dir)

    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

if __name__ == "__main__":
    convert_model_to_ct2()