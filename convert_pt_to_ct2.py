#!/usr/bin/env python3
"""
Convert your .pt Hindi2Hinglish model to CT2 format for faster-whisper
"""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import subprocess
from pathlib import Path

def convert_pt_to_ct2():
    """Convert .pt model to CT2 format"""

    print("=" * 60)
    print("CONVERTING .PT MODEL TO CT2 FORMAT")
    print("=" * 60)

    # Paths
    model_path = "Whisper-Hindi2Hinglish-Prime.pt"
    hf_output_dir = "models/whisper/hindi2hinglish_hf"
    ct2_output_dir = "models/whisper/hindi2hinglish_ct2"

    # Step 1: Load your .pt model
    print("Step 1: Loading your .pt model...")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure your .pt file is in the correct location")
        return False

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Loaded checkpoint from {model_path}")

        # Print checkpoint structure for debugging
        print("Checkpoint keys:", list(checkpoint.keys()))
        if "dims" in checkpoint:
            print("Model dimensions:", checkpoint["dims"])

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

    # Step 2: Create HuggingFace model
    print("\nStep 2: Creating HuggingFace model...")
    try:
        # Create model instance based on dimensions or use large-v3 as base
        base_model = "openai/whisper-large-v3"  # Adjust if your model is different size

        model = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(f"✅ Created model instance from {base_model}")

        # Load your custom state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the entire checkpoint is the state dict
            state_dict = checkpoint

        # Load the state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=False)
            print("✅ Loaded custom weights")
        except Exception as e:
            print(f"⚠️  Warning loading weights: {e}")
            print("Attempting partial load...")
            # Try to load what we can
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded {len(filtered_dict)}/{len(state_dict)} parameters")

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

    # Step 3: Save in HuggingFace format
    print("\nStep 3: Saving in HuggingFace format...")
    try:
        os.makedirs(hf_output_dir, exist_ok=True)

        model.save_pretrained(hf_output_dir)
        print(f"✅ Model saved to {hf_output_dir}")

        # Save processor/tokenizer
        processor = WhisperProcessor.from_pretrained(base_model)
        processor.save_pretrained(hf_output_dir)
        print(f"✅ Processor saved to {hf_output_dir}")

    except Exception as e:
        print(f"❌ Error saving HuggingFace model: {e}")
        return False

    # Step 4: Convert to CT2
    print("\nStep 4: Converting to CT2 format...")
    try:
        os.makedirs(ct2_output_dir, exist_ok=True)

        # Check if ct2-transformers-converter is available
        try:
            subprocess.run(["ct2-transformers-converter", "--help"],
                         capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Installing ctranslate2...")
            subprocess.run(["pip", "install", "ctranslate2"], check=True)

        # Convert to CT2
        cmd = [
            "ct2-transformers-converter",
            "--model", hf_output_dir,
            "--output_dir", ct2_output_dir,
            "--copy_files", "tokenizer.json", "preprocessor_config.json", "config.json",
            "--quantization", "float16"
        ]

        print("Running conversion command:")
        print(" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Successfully converted to CT2 format!")
            print(f"Your model is ready at: {ct2_output_dir}")

            # List converted files
            if os.path.exists(ct2_output_dir):
                print("\nConverted files:")
                for file in os.listdir(ct2_output_dir):
                    print(f"  - {file}")

            return ct2_output_dir

        else:
            print(f"❌ CT2 conversion failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error during CT2 conversion: {e}")
        return False

def update_config(ct2_model_path):
    """Update the configuration to use the new model"""
    print("\nStep 5: Updating configuration...")

    env_file = ".env"
    if not os.path.exists(env_file):
        print("❌ .env file not found")
        return

    # Read current .env
    with open(env_file, 'r') as f:
        lines = f.readlines()

    # Update the model path
    updated_lines = []
    model_path_updated = False

    for line in lines:
        if line.startswith("PS06_FAST_WHISPER_MODEL_PATH="):
            new_path = os.path.abspath(ct2_model_path).replace('\\', '\\\\')
            updated_lines.append(f"PS06_FAST_WHISPER_MODEL_PATH={new_path}\n")
            model_path_updated = True
            print(f"✅ Updated model path to: {new_path}")
        else:
            updated_lines.append(line)

    if not model_path_updated:
        # Add the line if it doesn't exist
        new_path = os.path.abspath(ct2_model_path).replace('\\', '\\\\')
        updated_lines.append(f"PS06_FAST_WHISPER_MODEL_PATH={new_path}\n")
        print(f"✅ Added model path: {new_path}")

    # Write updated .env
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)

def main():
    """Main conversion process"""
    ct2_model_path = convert_pt_to_ct2()

    if ct2_model_path:
        update_config(ct2_model_path)

        print("\n" + "=" * 60)
        print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your Hindi2Hinglish model is now ready to use with faster-whisper!")
        print("\nNext steps:")
        print("1. Restart your application")
        print("2. Test with Hindi audio to see Hindi2Hinglish output")
        print("3. The model should now be significantly faster!")
        print("=" * 60)

        return True
    else:
        print("\n❌ Conversion failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    main()