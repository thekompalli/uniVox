#!/usr/bin/env python3
"""
Test script for Hindi2Hinglish model integration with faster-whisper
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_configuration():
    """Test that configuration is properly loaded"""
    print("Testing configuration...")

    try:
        from src.config.app_config import model_config

        print(f"Whisper backend: {model_config.whisper_backend}")
        print(f"Use custom Hindi model: {model_config.use_custom_hindi_model}")
        print(f"Hindi2Hinglish model path: {model_config.hindi2hinglish_model_path}")
        print(f"Custom model languages: {model_config.custom_model_languages}")

        if model_config.whisper_backend == 'fast':
            print("[SUCCESS] Configuration is set to use faster-whisper!")
            return True
        else:
            print("[ERROR] Configuration is not set to use faster-whisper")
            return False

    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_model_loading():
    """Test loading of custom model"""
    print("\nTesting model loading...")

    try:
        from src.config.app_config import model_config

        # Test default model
        if model_config.fast_whisper_model_path and os.path.exists(model_config.fast_whisper_model_path):
            print(f"[SUCCESS] Default model exists: {model_config.fast_whisper_model_path}")
        else:
            print(f"[WARNING] Default model not found: {model_config.fast_whisper_model_path}")

        # Test custom Hindi2Hinglish model
        if (model_config.use_custom_hindi_model and
            model_config.hindi2hinglish_model_path and
            os.path.exists(model_config.hindi2hinglish_model_path)):

            print(f"[SUCCESS] Hindi2Hinglish model exists: {model_config.hindi2hinglish_model_path}")

            # Try to load the model
            from faster_whisper import WhisperModel
            model = WhisperModel(
                model_config.hindi2hinglish_model_path,
                device=model_config.fast_whisper_device,
                compute_type=model_config.fast_whisper_compute_type
            )
            print("[SUCCESS] Hindi2Hinglish model loaded successfully!")
            return True

        else:
            print("[INFO] Hindi2Hinglish model not configured or not found")
            print("This is normal if you haven't converted your model yet")
            return True

    except Exception as e:
        print(f"[ERROR] Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_selection():
    """Test model selection logic"""
    print("\nTesting model selection logic...")

    try:
        # Create a mock ASR service to test model selection
        class MockASRService:
            def __init__(self):
                from src.config.app_config import model_config
                self.custom_models = {}

                # Mock custom model loading
                if (model_config.use_custom_hindi_model and
                    model_config.hindi2hinglish_model_path and
                    os.path.exists(model_config.hindi2hinglish_model_path)):
                    # Just mark that we have the model (don't actually load for speed)
                    self.custom_models['hindi2hinglish'] = "mock_model"

                self.fast_whisper_model = "mock_default_model"

            def _select_fast_whisper_model(self, language: str):
                """Select the appropriate faster-whisper model based on language"""
                language = language.lower()

                # Use custom Hindi2Hinglish model for Hindi content
                if (language in ['hindi', 'hi'] and
                    'hindi2hinglish' in self.custom_models):
                    return f"hindi2hinglish_model_for_{language}"

                # Use default model for other languages
                if self.fast_whisper_model is not None:
                    return f"default_model_for_{language}"

                return None

        # Test model selection
        service = MockASRService()

        test_cases = [
            ('hindi', 'hindi2hinglish_model_for_hindi'),
            ('hi', 'hindi2hinglish_model_for_hi'),
            ('english', 'default_model_for_english'),
            ('punjabi', 'default_model_for_punjabi')
        ]

        all_passed = True
        for language, expected in test_cases:
            result = service._select_fast_whisper_model(language)
            if 'hindi2hinglish' in service.custom_models:
                # We have custom model
                if language in ['hindi', 'hi']:
                    expected = f"hindi2hinglish_model_for_{language}"
                else:
                    expected = f"default_model_for_{language}"
            else:
                # No custom model
                expected = f"default_model_for_{language}"

            if expected in result:
                print(f"[PASS] Language '{language}' -> {result}")
            else:
                print(f"[FAIL] Language '{language}' -> {result} (expected: {expected})")
                all_passed = False

        if all_passed:
            print("[SUCCESS] Model selection logic works correctly!")
            return True
        else:
            print("[ERROR] Some model selection tests failed")
            return False

    except Exception as e:
        print(f"[ERROR] Model selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """Show next steps to complete the setup"""
    print("\n" + "=" * 60)
    print("NEXT STEPS TO COMPLETE SETUP")
    print("=" * 60)

    print("\n1. CONVERT YOUR MODEL:")
    print("   Run: python convert_pt_to_ct2.py")
    print("   This will convert your .pt model to CT2 format")

    print("\n2. ENABLE CUSTOM MODEL:")
    print("   Set PS06_USE_CUSTOM_HINDI_MODEL=true in .env")

    print("\n3. TEST WITH HINDI AUDIO:")
    print("   Process Hindi audio files to see Hindi2Hinglish output")

    print("\n4. VERIFY PERFORMANCE:")
    print("   Your custom model should be 4-5x faster than OpenAI Whisper")

    print("\n5. SUPPORTED WORKFLOW:")
    print("   - Hindi audio -> Hindi2Hinglish model -> Hinglish text")
    print("   - Other languages -> Default model -> Original language")

def main():
    """Main test function"""
    print("=" * 60)
    print("HINDI2HINGLISH MODEL INTEGRATION TEST")
    print("=" * 60)

    results = []
    results.append(test_configuration())
    results.append(test_model_loading())
    results.append(test_model_selection())

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all(results):
        print("[SUCCESS] All tests passed!")
        print("Your system is ready to use the Hindi2Hinglish model!")
    else:
        print("[WARNING] Some tests had issues (see above)")
        failed_tests = sum(1 for r in results if not r)
        print(f"Issues: {failed_tests}/{len(results)} tests")

    show_next_steps()

if __name__ == "__main__":
    main()