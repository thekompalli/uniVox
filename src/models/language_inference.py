"""
Language Inference
Wav2Vec2-based multilingual language identification
"""
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
import soundfile as sf

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

from src.config.app_config import app_config, model_config
from src.models.triton_client import TritonClient, TritonModelWrapper

logger = logging.getLogger(__name__)


class LanguageInference:
    """Language identification model wrapper"""
    
    def __init__(self, triton_client: Optional[TritonClient] = None):
        self.model = None
        self.feature_extractor = None
        self.label_map = {}
        self.triton_client = triton_client
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.temp_dir = Path(app_config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Supported languages for PS-06
        self.supported_languages = [
            'english', 'hindi', 'punjabi', 'bengali', 'nepali', 'dogri'
        ]
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize language identification model"""
        try:
            logger.info("Initializing language identification model")
            
            # Try to load Wav2Vec2-based model
            try:
                model_name = getattr(model_config, 'wav2vec2_lang_model', 'facebook/wav2vec2-large-xlsr-53')
                
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.supported_languages)
                ).to(self.device)
                
                # Create label mapping
                self.label_map = {i: lang for i, lang in enumerate(self.supported_languages)}
                
                logger.info("Wav2Vec2 language model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load Wav2Vec2 model: {e}")
                self._initialize_fallback_model()
                
        except Exception as e:
            logger.exception(f"Error initializing language model: {e}")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback language detection model"""
        try:
            logger.warning("Using fallback language detection")
            
            # Simple rule-based fallback
            self.model = "fallback"
            self.label_map = {i: lang for i, lang in enumerate(self.supported_languages)}
            
        except Exception as e:
            logger.exception(f"Error initializing fallback model: {e}")
            raise
    
    async def identify_language(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identify language in audio
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            expected_languages: Optional list of expected languages
            
        Returns:
            Language identification results
        """
        try:
            # Run language identification in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.identify_language_sync,
                audio_data,
                sample_rate,
                expected_languages
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in language identification: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def identify_language_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous language identification"""
        try:
            # Handle fallback model
            if isinstance(self.model, str) and self.model == "fallback":
                return self._fallback_language_detection(audio_data, expected_languages)
            
            # Ensure minimum length
            min_samples = int(1.0 * sample_rate)  # 1 second minimum
            if len(audio_data) < min_samples:
                audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)), mode='constant')
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000
            
            # Extract features
            inputs = self.feature_extractor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
            # Map to language names
            language_probs = {}
            for idx, prob in enumerate(probabilities):
                if idx in self.label_map:
                    language_probs[self.label_map[idx]] = float(prob)
            
            # Filter by expected languages if provided
            if expected_languages:
                filtered_probs = {
                    lang: prob for lang, prob in language_probs.items()
                    if lang in expected_languages
                }
                if filtered_probs:
                    language_probs = filtered_probs
                    # Renormalize
                    total_prob = sum(language_probs.values())
                    if total_prob > 0:
                        language_probs = {
                            lang: prob / total_prob
                            for lang, prob in language_probs.items()
                        }
            
            # Get top prediction
            if language_probs:
                top_language = max(language_probs.items(), key=lambda x: x[1])
                predicted_language = top_language[0]
                confidence = top_language[1]
            else:
                predicted_language = 'unknown'
                confidence = 0.0
            
            return {
                'language': predicted_language,
                'confidence': confidence,
                'probabilities': language_probs
            }
            
        except Exception as e:
            logger.exception(f"Error in synchronous language identification: {e}")
            return self._fallback_language_detection(audio_data, expected_languages)
    
    def _fallback_language_detection(
        self,
        audio_data: np.ndarray,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fallback language detection using simple heuristics"""
        try:
            # Use simple spectral features for language detection
            import librosa
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=16000)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=16000)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
            
            # Calculate feature statistics
            centroid_mean = np.mean(spectral_centroids)
            rolloff_mean = np.mean(spectral_rolloff)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Simple rule-based classification
            language_scores = {}
            
            # Heuristic rules (these would be learned from data in practice)
            if expected_languages:
                valid_languages = expected_languages
            else:
                valid_languages = self.supported_languages
            
            for language in valid_languages:
                score = 0.5  # Base score
                
                # Add language-specific heuristics
                if language == 'english':
                    # English typically has higher spectral centroid
                    if centroid_mean > 2000:
                        score += 0.3
                elif language == 'hindi':
                    # Hindi has different spectral characteristics
                    if 1500 < centroid_mean < 2500:
                        score += 0.2
                elif language == 'punjabi':
                    # Punjabi characteristics
                    if centroid_mean > 1800:
                        score += 0.2
                elif language in ['bengali', 'nepali']:
                    # Similar characteristics for Bengali/Nepali
                    if centroid_mean < 2000:
                        score += 0.2
                
                language_scores[language] = score
            
            # Normalize scores
            total_score = sum(language_scores.values())
            if total_score > 0:
                language_probs = {
                    lang: score / total_score
                    for lang, score in language_scores.items()
                }
            else:
                # Uniform distribution as ultimate fallback
                language_probs = {
                    lang: 1.0 / len(valid_languages)
                    for lang in valid_languages
                }
            
            # Get top prediction
            top_language = max(language_probs.items(), key=lambda x: x[1])
            
            return {
                'language': top_language[0],
                'confidence': top_language[1],
                'probabilities': language_probs
            }
            
        except Exception as e:
            logger.exception(f"Error in fallback language detection: {e}")
            return {
                'language': 'hindi',  # Default to Hindi
                'confidence': 0.5,
                'probabilities': {'hindi': 0.5}
            }
    
    async def identify_batch_languages(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int,
        expected_languages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Identify languages for multiple audio segments"""
        try:
            # Process segments in parallel
            tasks = []
            for segment in audio_segments:
                task = self.identify_language(segment, sample_rate, expected_languages)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing segment {i}: {result}")
                    processed_results.append({
                        'language': 'unknown',
                        'confidence': 0.0,
                        'probabilities': {}
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.exception(f"Error in batch language identification: {e}")
            return [
                {'language': 'unknown', 'confidence': 0.0, 'probabilities': {}}
                for _ in audio_segments
            ]
    
    async def detect_language_switches(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_duration: float = 3.0,
        hop_duration: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Detect language switches in audio using sliding window
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            window_duration: Window duration in seconds
            hop_duration: Hop duration in seconds
            
        Returns:
            List of language detection results with timestamps
        """
        try:
            results = []
            
            window_samples = int(window_duration * sample_rate)
            hop_samples = int(hop_duration * sample_rate)
            
            for start in range(0, len(audio_data) - window_samples + 1, hop_samples):
                end = start + window_samples
                window_audio = audio_data[start:end]
                
                # Identify language for this window
                result = await self.identify_language(window_audio, sample_rate)
                
                # Add timing information
                result.update({
                    'start_time': start / sample_rate,
                    'end_time': end / sample_rate,
                    'window_duration': window_duration
                })
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error detecting language switches: {e}")
            return []
    
    async def validate_language_prediction(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        ground_truth_language: str
    ) -> Dict[str, float]:
        """Validate language prediction against ground truth"""
        try:
            result = await self.identify_language(audio_data, sample_rate)
            
            predicted_language = result['language']
            confidence = result['confidence']
            
            is_correct = predicted_language == ground_truth_language
            
            return {
                'is_correct': is_correct,
                'predicted_language': predicted_language,
                'ground_truth_language': ground_truth_language,
                'confidence': confidence,
                'accuracy': 1.0 if is_correct else 0.0
            }
            
        except Exception as e:
            logger.exception(f"Error validating language prediction: {e}")
            return {
                'is_correct': False,
                'predicted_language': 'unknown',
                'ground_truth_language': ground_truth_language,
                'confidence': 0.0,
                'accuracy': 0.0
            }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get language model information"""
        model_type = "unknown"
        if isinstance(self.model, str):
            model_type = self.model
        elif hasattr(self.model, '__class__'):
            model_type = self.model.__class__.__name__
        
        return {
            'model_type': model_type,
            'supported_languages': self.supported_languages,
            'num_labels': len(self.supported_languages),
            'sample_rate': 16000,
            'min_duration': 1.0,
            'device': str(self.device),
            'feature_extractor': 'Wav2Vec2FeatureExtractor' if self.feature_extractor else None
        }
    
    async def fine_tune_model(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: List[Dict[str, Any]],
        num_epochs: int = 10,
        learning_rate: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Fine-tune the language identification model
        
        Args:
            training_data: List of training samples with 'audio' and 'language' keys
            validation_data: List of validation samples
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training results and metrics
        """
        try:
            if isinstance(self.model, str):
                logger.error("Cannot fine-tune fallback model")
                return {'error': 'Fallback model cannot be fine-tuned'}
            
            from torch.utils.data import DataLoader, Dataset
            from torch.optim import AdamW
            from torch.nn import CrossEntropyLoss
            
            # Create dataset class
            class LanguageDataset(Dataset):
                def __init__(self, data, feature_extractor, label_map, sample_rate=16000):
                    self.data = data
                    self.feature_extractor = feature_extractor
                    self.label_map = {lang: idx for idx, lang in label_map.items()}
                    self.sample_rate = sample_rate
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    item = self.data[idx]
                    audio = item['audio']
                    language = item['language']
                    
                    # Extract features
                    inputs = self.feature_extractor(
                        audio,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Get label
                    label = self.label_map.get(language, 0)
                    
                    return {
                        'input_values': inputs.input_values.squeeze(),
                        'labels': torch.tensor(label, dtype=torch.long)
                    }
            
            # Create datasets and dataloaders
            train_dataset = LanguageDataset(training_data, self.feature_extractor, self.label_map)
            val_dataset = LanguageDataset(validation_data, self.feature_extractor, self.label_map)
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
            
            # Set up training
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            criterion = CrossEntropyLoss()
            
            # Training loop
            training_history = []
            
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    inputs = batch['input_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs.logits, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.logits, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch['input_values'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = criterion(outputs.logits, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # Record metrics
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss / len(train_loader),
                    'train_accuracy': train_correct / train_total,
                    'val_loss': val_loss / len(val_loader),
                    'val_accuracy': val_correct / val_total
                }
                
                training_history.append(epoch_metrics)
                
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                           f"Train Loss: {epoch_metrics['train_loss']:.4f}, "
                           f"Train Acc: {epoch_metrics['train_accuracy']:.4f}, "
                           f"Val Loss: {epoch_metrics['val_loss']:.4f}, "
                           f"Val Acc: {epoch_metrics['val_accuracy']:.4f}")
            
            return {
                'training_completed': True,
                'training_history': training_history,
                'final_train_accuracy': training_history[-1]['train_accuracy'],
                'final_val_accuracy': training_history[-1]['val_accuracy']
            }
            
        except Exception as e:
            logger.exception(f"Error fine-tuning model: {e}")
            return {'error': str(e), 'training_completed': False}


class TritonLanguageInference(TritonModelWrapper):
    """Triton-based language inference wrapper"""
    
    def __init__(self, triton_client: TritonClient):
        super().__init__(
            model_name=model_config.triton_language_model,
            triton_client=triton_client,
            input_names=['audio', 'sample_rate'],
            output_names=['language', 'probabilities', 'confidence']
        )
        
        self.supported_languages = [
            'english', 'hindi', 'punjabi', 'bengali', 'nepali', 'dogri'
        ]
    
    async def identify_language(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        expected_languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identify language using Triton server"""
        try:
            # Prepare inputs
            inputs = {
                'audio': audio_data.astype(np.float32),
                'sample_rate': np.array([sample_rate], dtype=np.int32)
            }
            
            if expected_languages:
                # Convert expected languages to indices
                expected_indices = [
                    i for i, lang in enumerate(self.supported_languages)
                    if lang in expected_languages
                ]
                inputs['expected_languages'] = np.array(expected_indices, dtype=np.int32)
                self.input_names.append('expected_languages')
            
            # Run inference
            results = await self.predict(inputs)
            
            # Process results
            language = results['language'][0].decode('utf-8')
            confidence = float(results['confidence'][0])
            
            # Parse probabilities
            prob_array = results['probabilities'][0]
            probabilities = {
                self.supported_languages[i]: float(prob)
                for i, prob in enumerate(prob_array)
                if i < len(self.supported_languages)
            }
            
            return {
                'language': language,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.exception(f"Error in Triton language inference: {e}")
            return {
                'language': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }