#!/usr/bin/env python3
"""
Convert custom HuggingFace Whisper model to OpenAI format, then to CT2 format for faster-whisper
"""
import torch
from transformers import AutoModelForSpeechSeq2Seq
import re
from tqdm import tqdm
from collections import OrderedDict
import json
import os
from pathlib import Path

def load_conversion_mapping():
    """Load or create parameter name mapping from HF to OpenAI format"""
    # You'll need to provide the convert_hf2openai.json file, or I can help create one
    mapping_file = 'convert_hf2openai.json'

    if not os.path.exists(mapping_file):
        # Create a basic mapping if file doesn't exist
        # This is a simplified version - you may need to adjust based on your specific model
        basic_mapping = {
            # Encoder mappings
            "encoder.layers.([0-9]+).self_attn.q_proj.weight": "encoder.blocks.\\1.attn.query.weight",
            "encoder.layers.([0-9]+).self_attn.k_proj.weight": "encoder.blocks.\\1.attn.key.weight",
            "encoder.layers.([0-9]+).self_attn.v_proj.weight": "encoder.blocks.\\1.attn.value.weight",
            "encoder.layers.([0-9]+).self_attn.out_proj.weight": "encoder.blocks.\\1.attn.out.weight",
            "encoder.layers.([0-9]+).self_attn_layer_norm.weight": "encoder.blocks.\\1.attn_ln.weight",
            "encoder.layers.([0-9]+).fc1.weight": "encoder.blocks.\\1.mlp.0.weight",
            "encoder.layers.([0-9]+).fc2.weight": "encoder.blocks.\\1.mlp.2.weight",
            "encoder.layers.([0-9]+).final_layer_norm.weight": "encoder.blocks.\\1.mlp_ln.weight",

            # Decoder mappings
            "decoder.layers.([0-9]+).self_attn.q_proj.weight": "decoder.blocks.\\1.attn.query.weight",
            "decoder.layers.([0-9]+).self_attn.k_proj.weight": "decoder.blocks.\\1.attn.key.weight",
            "decoder.layers.([0-9]+).self_attn.v_proj.weight": "decoder.blocks.\\1.attn.value.weight",
            "decoder.layers.([0-9]+).self_attn.out_proj.weight": "decoder.blocks.\\1.attn.out.weight",
            "decoder.layers.([0-9]+).self_attn_layer_norm.weight": "decoder.blocks.\\1.attn_ln.weight",

            # Cross attention
            "decoder.layers.([0-9]+).encoder_attn.q_proj.weight": "decoder.blocks.\\1.cross_attn.query.weight",
            "decoder.layers.([0-9]+).encoder_attn.k_proj.weight": "decoder.blocks.\\1.cross_attn.key.weight",
            "decoder.layers.([0-9]+).encoder_attn.v_proj.weight": "decoder.blocks.\\1.cross_attn.value.weight",
            "decoder.layers.([0-9]+).encoder_attn.out_proj.weight": "decoder.blocks.\\1.cross_attn.out.weight",
            "decoder.layers.([0-9]+).encoder_attn_layer_norm.weight": "decoder.blocks.\\1.cross_attn_ln.weight",

            # MLP
            "decoder.layers.([0-9]+).fc1.weight": "decoder.blocks.\\1.mlp.0.weight",
            "decoder.layers.([0-9]+).fc2.weight": "decoder.blocks.\\1.mlp.2.weight",
            "decoder.layers.([0-9]+).final_layer_norm.weight": "decoder.blocks.\\1.mlp_ln.weight",

            # Bias terms (similar pattern)
            "encoder.layers.([0-9]+).self_attn.q_proj.bias": "encoder.blocks.\\1.attn.query.bias",
            "encoder.layers.([0-9]+).self_attn.k_proj.bias": "encoder.blocks.\\1.attn.key.bias",
            "encoder.layers.([0-9]+).self_attn.v_proj.bias": "encoder.blocks.\\1.attn.value.bias",
            "encoder.layers.([0-9]+).self_attn.out_proj.bias": "encoder.blocks.\\1.attn.out.bias",
            "encoder.layers.([0-9]+).self_attn_layer_norm.bias": "encoder.blocks.\\1.attn_ln.bias",
            "encoder.layers.([0-9]+).fc1.bias": "encoder.blocks.\\1.mlp.0.bias",
            "encoder.layers.([0-9]+).fc2.bias": "encoder.blocks.\\1.mlp.2.bias",
            "encoder.layers.([0-9]+).final_layer_norm.bias": "encoder.blocks.\\1.mlp_ln.bias",

            # Root level
            "encoder.layer_norm.weight": "encoder.ln_post.weight",
            "encoder.layer_norm.bias": "encoder.ln_post.bias",
            "decoder.layer_norm.weight": "decoder.ln.weight",
            "decoder.layer_norm.bias": "decoder.ln.bias",
            "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
            "decoder.embed_positions.weight": "decoder.positional_embedding",
            "proj_out.weight": "decoder.token_embedding.weight",  # Often tied
        }

        with open(mapping_file, 'w') as f:
            json.dump(basic_mapping, f, indent=2)
        print(f"Created basic mapping file: {mapping_file}")

    with open(mapping_file, 'r') as f:
        reverse_translation = json.load(f)

    return OrderedDict(reverse_translation)

def save_model(model, save_path):
    """Convert HF model to OpenAI Whisper format"""
    reverse_translation = load_conversion_mapping()

    def reverse_translate(current_param):
        # Convert parameter names using regex patterns
        for pattern, repl in reverse_translation.items():
            if re.match(pattern, current_param):
                return re.sub(pattern, repl, current_param)
        return None  # Return None if no match found

    # Extract model dimensions from config
    config = model.config
    model_dims = {
        "n_mels": getattr(config, 'num_mel_bins', 80),           # Number of mel spectrogram bins
        "n_vocab": config.vocab_size,                            # Vocabulary size
        "n_audio_ctx": getattr(config, 'max_source_positions', 1500),    # Max audio context length
        "n_audio_state": config.d_model,                         # Audio encoder state dimension
        "n_audio_head": config.encoder_attention_heads,          # Audio encoder attention heads
        "n_audio_layer": config.encoder_layers,                  # Number of audio encoder layers
        "n_text_ctx": getattr(config, 'max_target_positions', 448),     # Max text context length
        "n_text_state": config.d_model,                          # Text decoder state dimension
        "n_text_head": config.decoder_attention_heads,           # Text decoder attention heads
        "n_text_layer": config.decoder_layers,                   # Number of text decoder layers
    }

    # Convert model state dict to Whisper format
    original_model_state_dict = model.state_dict()
    new_state_dict = {}

    for key, value in tqdm(original_model_state_dict.items(), desc="Converting parameters"):
        key = key.replace("model.", "")          # Remove 'model.' prefix
        new_key = reverse_translate(key)         # Convert parameter names
        if new_key is not None:
            new_state_dict[new_key] = value
        else:
            print(f"Warning: No mapping found for parameter: {key}")

    # Create final model dictionary
    pytorch_model = {
        "dims": model_dims,
        "model_state_dict": new_state_dict
    }

    # Save the converted model
    print(f"Saving converted model to: {save_path}")
    torch.save(pytorch_model, save_path)
    return pytorch_model

def convert_to_ct2(openai_model_path, ct2_output_dir, quantization="float16"):
    """Convert OpenAI format model to CT2 format for faster-whisper"""
    try:
        from ct2 import converters
        print(f"Converting {openai_model_path} to CT2 format...")

        converter = converters.TransformersConverter(
            model_name_or_path=openai_model_path,
            copy_files=["tokenizer.json", "preprocessor_config.json", "config.json"],
            quantization=quantization,
        )
        converter.convert(ct2_output_dir)
        print(f"CT2 model saved to: {ct2_output_dir}")

    except ImportError:
        print("CT2 not installed. Installing...")
        os.system("pip install ctranslate2")
        # Retry conversion
        from ct2 import converters
        converter = converters.TransformersConverter(
            model_name_or_path=openai_model_path,
            copy_files=["tokenizer.json", "preprocessor_config.json", "config.json"],
            quantization=quantization,
        )
        converter.convert(ct2_output_dir)
        print(f"CT2 model saved to: {ct2_output_dir}")

def main():
    """Main conversion workflow"""
    # Configuration
    hf_model_path = "your-hf-model-path"  # Replace with your HuggingFace model path
    openai_model_path = "models/whisper/Whisper-Hindi2Hinglish-Prime.pt"
    ct2_output_dir = "models/whisper/hindi2hinglish_ct2"

    print("=" * 60)
    print("CUSTOM WHISPER MODEL CONVERSION")
    print("=" * 60)

    # Step 1: Load HuggingFace model
    print("Step 1: Loading HuggingFace model...")
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        print(f"✅ Model loaded successfully from: {hf_model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Step 2: Convert to OpenAI format
    print("\nStep 2: Converting to OpenAI format...")
    try:
        os.makedirs(os.path.dirname(openai_model_path), exist_ok=True)
        save_model(model, openai_model_path)
        print(f"✅ OpenAI format model saved to: {openai_model_path}")
    except Exception as e:
        print(f"❌ Failed to convert to OpenAI format: {e}")
        return False

    # Step 3: Convert to CT2 format
    print("\nStep 3: Converting to CT2 format for faster-whisper...")
    try:
        os.makedirs(ct2_output_dir, exist_ok=True)
        convert_to_ct2(openai_model_path, ct2_output_dir)
        print(f"✅ CT2 model saved to: {ct2_output_dir}")
    except Exception as e:
        print(f"❌ Failed to convert to CT2 format: {e}")
        print("You may need to convert manually using:")
        print(f"ct2-transformers-converter --model {openai_model_path} --output_dir {ct2_output_dir} --quantization float16")
        return False

    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
    print(f"Your custom model is ready at: {ct2_output_dir}")
    print("=" * 60)

    return True

if __name__ == "__main__":
    # You need to set the correct path to your HuggingFace model
    # Replace "your-hf-model-path" with the actual path
    main()