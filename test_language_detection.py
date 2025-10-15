#!/usr/bin/env python3
"""
Test script for language detection improvements
"""

class MockTranslationService:
    """Mock translation service to test language detection logic"""

    def _detect_script(self, text: str) -> str:
        """Detect dominant script of the text"""
        if not text:
            return 'other'
        # Unicode ranges
        has_devanagari = any('\u0900' <= ch <= '\u097F' for ch in text)
        has_bengali = any('\u0980' <= ch <= '\u09FF' for ch in text)
        has_gurmukhi = any('\u0A00' <= ch <= '\u0A7F' for ch in text)
        has_arabic = any('\u0600' <= ch <= '\u06FF' for ch in text)
        has_latin = any('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in text)

        if has_devanagari:
            return 'devanagari'
        if has_bengali:
            return 'bengali'
        if has_gurmukhi:
            return 'gurmukhi'
        if has_arabic:
            return 'arabic'
        if has_latin:
            return 'latin'
        return 'other'

    def _normalize_language(self, language: str) -> str:
        """Normalize language tags"""
        if not language:
            return 'unknown'
        lang = language.strip().lower()

        # Language mappings
        lang_mappings = {
            'en': 'english',
            'eng': 'english',
            'english': 'english',
            'uncertain_english': 'english',
            'auto_english': 'english',
            'hi': 'hindi',
            'hin': 'hindi',
            'hindi': 'hindi',
            'pa': 'punjabi',
            'pan': 'punjabi',
            'punjabi': 'punjabi',
            'ur': 'urdu',
            'urd': 'urdu',
            'urdu': 'urdu',
            'bn': 'bengali',
            'ben': 'bengali',
            'bengali': 'bengali'
        }

        # Check direct mapping first
        if lang in lang_mappings:
            return lang_mappings[lang]

        # Check if it starts with or contains english
        if lang.startswith('english') or 'english' in lang:
            return 'english'

        return lang

    def _infer_source_language(self, lang_hint: str, script: str) -> str:
        """Combine LID hint with script to infer better source language"""
        # Script detection is more reliable than LID for these scripts
        if script == 'devanagari':
            return 'hindi'
        if script == 'arabic':
            return 'urdu'
        if script == 'gurmukhi':
            return 'punjabi'
        if script == 'bengali':
            return 'bengali'

        normalized = self._normalize_language(lang_hint)

        # Only return 'english' if we have Latin script AND reasonable LID confidence
        if normalized in ('auto', 'unknown', 'english'):
            if script == 'latin':
                return 'english'
            # If non-Latin script but LID says English, assume it's wrong
            elif script in ['devanagari', 'arabic', 'gurmukhi', 'bengali']:
                # Fallback based on script
                script_mapping = {
                    'devanagari': 'hindi',
                    'arabic': 'urdu',
                    'gurmukhi': 'punjabi',
                    'bengali': 'bengali'
                }
                return script_mapping.get(script, 'unknown')

        return normalized

def test_language_detection():
    """Test language detection with problematic cases from your data"""
    service = MockTranslationService()

    # Test cases based on your JSON data
    test_cases = [
        # Gurmukhi script (Punjabi) that was detected as Hindi
        {'text': 'ਅਜੇਵੀ ਸਿਂਦਰਬਾਲ ਬੱਠਾ ਏਕ ਦੀਨ ਪੇਲਾ', 'raw_lang': 'hindi', 'expected': 'punjabi'},

        # Devanagari script (Hindi) that was correct
        {'text': 'वो नहीं कहा यार कोई गाल नहीं।', 'raw_lang': 'hindi', 'expected': 'hindi'},

        # Arabic script (Urdu)
        {'text': 'یہ نکال جائے', 'raw_lang': 'urdu', 'expected': 'urdu'},

        # Mixed Gurmukhi with some Devanagari influence
        {'text': 'ਚੇਡੇ ਮੇਨ ਕੇ ਸੀ ਆਪਲੇ ਉਠਿਗੇ', 'raw_lang': 'punjabi', 'expected': 'punjabi'},

        # True English text
        {'text': "I don't know what he means.", 'raw_lang': 'english', 'expected': 'english'},

        # English detected text but non-Latin script (common error)
        {'text': 'ਹੀ ਸੁ ਹੀ ਸ੍ത്തੇ ਹੀ', 'raw_lang': 'english', 'expected': 'punjabi'},

        # Bengali script
        {'text': 'ফাল্লো ফাল্দে কেয়নি', 'raw_lang': 'bengali', 'expected': 'bengali'},
    ]

    print("Testing Language Detection Fixes:")
    print("=" * 80)

    all_correct = True
    for i, case in enumerate(test_cases, 1):
        script = service._detect_script(case['text'])
        inferred = service._infer_source_language(case['raw_lang'], script)

        is_correct = inferred == case['expected']
        all_correct = all_correct and is_correct

        status = "[PASS]" if is_correct else "[FAIL]"

        print(f"{status} Test {i}: {case['text'][:40]}...")
        print(f"   Raw Lang: {case['raw_lang']} | Script: {script}")
        print(f"   Expected: {case['expected']} | Got: {inferred}")
        print()

    print("=" * 80)
    if all_correct:
        print("[SUCCESS] All tests passed! Language detection improvements are working correctly.")
    else:
        print("[ERROR] Some tests failed. Check the logic above.")

    return all_correct

def test_skip_logic():
    """Test the improved skip translation logic"""
    service = MockTranslationService()

    print("\nTesting Skip Translation Logic:")
    print("=" * 80)

    test_cases = [
        # Should skip: True English text
        {'text': "Hello world", 'raw_lang': 'english', 'target': 'english', 'should_skip': True},

        # Should NOT skip: Non-Latin script detected as English (error case)
        {'text': 'ਅਜੇਵੀ ਸਿਂਦਰਬਾਲ', 'raw_lang': 'english', 'target': 'english', 'should_skip': False},

        # Should NOT skip: Different languages
        {'text': 'वो नहीं कहा यार', 'raw_lang': 'hindi', 'target': 'english', 'should_skip': False},

        # Should skip: Same non-English language
        {'text': 'वो नहीं कहा यार', 'raw_lang': 'hindi', 'target': 'hindi', 'should_skip': True},
    ]

    for i, case in enumerate(test_cases, 1):
        script = service._detect_script(case['text'])
        effective_source = service._infer_source_language(case['raw_lang'], script)
        norm_target = service._normalize_language(case['target'])

        # Apply the new skip logic
        skip_translation = False
        if effective_source == norm_target:
            if norm_target == 'english' and script == 'latin':
                skip_translation = True
            elif norm_target != 'english':
                skip_translation = True

        is_correct = skip_translation == case['should_skip']
        status = "[PASS]" if is_correct else "[FAIL]"

        print(f"{status} Test {i}: {case['text'][:30]}...")
        print(f"   Source: {effective_source} | Target: {norm_target} | Script: {script}")
        print(f"   Expected Skip: {case['should_skip']} | Got Skip: {skip_translation}")
        print()

if __name__ == "__main__":
    test_language_detection()
    test_skip_logic()