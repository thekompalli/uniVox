"""
Speaker Inference
WeSpeaker model wrapper for speaker embedding extraction
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
import os

from src.config.app_config import app_config, model_config
from src.models.triton_client import TritonClient, TritonModelWrapper

logger = logging.getLogger(__name__)


class SpeakerInference:
    """WeSpeaker model for speaker embedding extraction"""
    
    def __init__(self, triton_client: Optional[TritonClient] = None):
        self.model = None
        self.onnx_session = None
        self.use_onnx = False
        self.triton_client = triton_client
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.temp_dir = Path(app_config.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize WeSpeaker model"""
        try:
            logger.info("Initializing WeSpeaker model")
            
            # PRIORITY 1: Check for manually downloaded ONNX model
            onnx_path = os.path.expanduser('~/.wespeaker/english/voxceleb_resnet221_LM.onnx')
            
            if os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    logger.info(f"Loading manually downloaded ONNX model from {onnx_path}")
                    
                    # Load ONNX model
                    self.onnx_session = ort.InferenceSession(
                        onnx_path,
                        providers=['CPUExecutionProvider']  # Use CPU for stability
                    )
                    self.use_onnx = True
                    self.model = None
                    
                    logger.info("ONNX WeSpeaker model loaded successfully")
                    return
                    
                except Exception as onnx_error:
                    logger.error(f"Failed to load ONNX model: {onnx_error}")
                    # Continue to try other methods
            
            # PRIORITY 2: Try to load WeSpeaker model normally
            try:
                import wespeaker
                cfg_val = str(getattr(model_config, 'wespeaker_model', 'english')).strip().lower()
                alias_map = {
                    'voxceleb_resnet34': 'english',
                    'voxceleb_resnet34_lm': 'english',
                    'wespeaker/voxceleb_resnet34_lm': 'english',
                    'pyannote/wespeaker-voxceleb-resnet34-lm': 'english',
                    'voxceleb': 'english',
                }
                lang = alias_map.get(cfg_val, cfg_val)

                try:
                    self.model = wespeaker.load_model(lang)
                except Exception as e:
                    if 'Unsupported lang' in str(e) or 'unsupported' in str(e).lower():
                        logger.warning(f"WeSpeaker load_model({lang}) failed, falling back to 'english'")
                        self.model = wespeaker.load_model('english')
                    else:
                        raise

                use_cuda = torch.cuda.is_available()
                if hasattr(self.model, 'set_gpu'):
                    self.model.set_gpu(use_cuda)
                elif hasattr(self.model, 'use_gpu'):
                    self.model.use_gpu(use_cuda)
                elif hasattr(self.model, 'to'):
                    try:
                        self.model = self.model.to('cuda' if use_cuda else 'cpu')
                    except Exception:
                        pass

                logger.info("WeSpeaker model loaded successfully")
                
            except ImportError:
                logger.warning("WeSpeaker not available, trying alternative")
                self._initialize_alternative_model()
            
        except Exception as e:
            logger.exception(f"Error initializing speaker model: {e}")
            self._initialize_fallback_model()
    
    def _initialize_alternative_model(self):
        """Initialize alternative speaker model using SpeechBrain"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=app_config.models_dir / "speechbrain"
            )
            
            logger.info("SpeechBrain ECAPA-TDNN model initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize SpeechBrain model: {e}")
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize fallback model using simple MFCC features"""
        try:
            logger.warning("Using fallback MFCC-based speaker embeddings")
            self.model = "fallback"
            
        except Exception as e:
            logger.exception(f"Error initializing fallback model: {e}")
            raise
    
    async def extract_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio
        
        Args:
            audio_data: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Speaker embedding vector
        """
        try:
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.extract_embedding_sync,
                audio_data,
                sample_rate
            )
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error extracting embedding: {e}")
            return np.zeros(256)
    
    def extract_embedding_sync(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Synchronous embedding extraction"""
        try:
            # Use ONNX model if available
            if self.use_onnx and self.onnx_session:
                return self._extract_onnx_embedding(audio_data, sample_rate)
            
            # Fallback
            if isinstance(self.model, str) and self.model == "fallback":
                return self._extract_mfcc_embedding(audio_data, sample_rate)
            
            # Ensure minimum length
            min_samples = int(0.5 * sample_rate)
            if len(audio_data) < min_samples:
                padding = min_samples - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000
            
            # Extract embedding based on available model
            if hasattr(self.model, 'extract_embedding'):
                return self._extract_wespeaker_embedding(audio_data, sample_rate)
            elif hasattr(self.model, 'encode_batch'):
                return self._extract_speechbrain_embedding(audio_data, sample_rate)
            else:
                return self._extract_mfcc_embedding(audio_data, sample_rate)
                
        except Exception as e:
            logger.exception(f"Error in synchronous embedding extraction: {e}")
            return np.zeros(256)
    
    def _extract_onnx_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract embedding using ONNX model"""
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
            
            # Ensure minimum length (0.5 seconds)
            min_samples = 8000  # 0.5 seconds at 16kHz
            if len(audio_data) < min_samples:
                audio_data = np.pad(audio_data, (0, min_samples - len(audio_data)))
            
            # Prepare input (float32, add batch dimension)
            audio_input = audio_data.astype(np.float32)
            audio_input = np.expand_dims(audio_input, axis=0)
            
            # Get input name
            input_name = self.onnx_session.get_inputs()[0].name
            
            # Run ONNX inference
            ort_inputs = {input_name: audio_input}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            
            # Get embedding (first output)
            embedding = ort_outs[0].flatten()
            
            # Ensure 256 dimensions
            if len(embedding) > 256:
                embedding = embedding[:256]
            elif len(embedding) < 256:
                embedding = np.pad(embedding, (0, 256 - len(embedding)))
            
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error in ONNX embedding extraction: {e}")
            return self._extract_mfcc_embedding(audio_data, 16000)
    
    def _extract_wespeaker_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract embedding using WeSpeaker"""
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                dir=self.temp_dir,
                delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                
                sf.write(temp_path, audio_data, sample_rate)
                embedding = self.model.extract_embedding(str(temp_path))
                temp_path.unlink(missing_ok=True)
                
                if torch.is_tensor(embedding):
                    embedding = embedding.cpu().numpy()
                
                embedding = embedding.flatten()
                if len(embedding) != 256:
                    if len(embedding) > 256:
                        embedding = embedding[:256]
                    else:
                        embedding = np.pad(embedding, (0, 256 - len(embedding)))
                
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                return embedding
                
        except Exception as e:
            logger.exception(f"Error in WeSpeaker embedding extraction: {e}")
            return np.zeros(256)
    
    def _extract_speechbrain_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract embedding using SpeechBrain"""
        try:
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
            
            with torch.no_grad():
                embeddings = self.model.encode_batch(audio_tensor)
                embedding = embeddings.squeeze().cpu().numpy()
            
            if len(embedding) != 256:
                if len(embedding) > 256:
                    embedding = embedding[:256]
                else:
                    embedding = np.pad(embedding, (0, 256 - len(embedding)))
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error in SpeechBrain embedding extraction: {e}")
            return np.zeros(256)
    
    def _extract_mfcc_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract MFCC-based embedding as fallback"""
        try:
            import librosa
            
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=512,
                hop_length=256,
                win_length=400
            )
            
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            basic_embedding = np.concatenate([mfcc_mean, mfcc_std])
            
            delta_mfcc = librosa.feature.delta(mfccs)
            delta2_mfcc = librosa.feature.delta(mfccs, order=2)
            
            delta_mean = np.mean(delta_mfcc, axis=1)
            delta_std = np.mean(delta2_mfcc, axis=1)
            
            extended_embedding = np.concatenate([
                basic_embedding, delta_mean, delta_std
            ])
            
            if len(extended_embedding) > 256:
                embedding = extended_embedding[:256]
            else:
                embedding = np.pad(extended_embedding, (0, 256 - len(extended_embedding)))
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error in MFCC embedding extraction: {e}")
            return np.zeros(256)
    
    # Keep all other methods unchanged from original code
    async def extract_batch_embeddings(
        self,
        audio_segments: List[np.ndarray],
        sample_rate: int
    ) -> List[np.ndarray]:
        """Extract embeddings for multiple audio segments"""
        try:
            tasks = []
            for segment in audio_segments:
                task = self.extract_embedding(segment, sample_rate)
                tasks.append(task)
            
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_embeddings = []
            for i, embedding in enumerate(embeddings):
                if isinstance(embedding, Exception):
                    logger.error(f"Error processing segment {i}: {embedding}")
                    processed_embeddings.append(np.zeros(256))
                else:
                    processed_embeddings.append(embedding)
            
            return processed_embeddings
            
        except Exception as e:
            logger.exception(f"Error in batch embedding extraction: {e}")
            return [np.zeros(256) for _ in audio_segments]
    
    async def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.exception(f"Error calculating similarity: {e}")
            return 0.0
    
    async def verify_speakers(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Verify if two embeddings belong to the same speaker"""
        try:
            similarity = await self.calculate_similarity(embedding1, embedding2)
            
            is_same_speaker = similarity >= threshold
            confidence = similarity if is_same_speaker else 1.0 - similarity
            
            return {
                'is_same_speaker': is_same_speaker,
                'similarity_score': similarity,
                'confidence': confidence,
                'threshold_used': threshold
            }
            
        except Exception as e:
            logger.exception(f"Error in speaker verification: {e}")
            return {
                'is_same_speaker': False,
                'similarity_score': 0.0,
                'confidence': 0.0,
                'threshold_used': threshold
            }
    
    async def cluster_embeddings(
        self,
        embeddings: List[np.ndarray],
        n_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Cluster embeddings to identify speakers"""
        try:
            from sklearn.cluster import AgglomerativeClustering, KMeans
            import scipy.cluster.hierarchy as sch
            
            if len(embeddings) < 2:
                return {
                    'labels': [0] * len(embeddings),
                    'n_clusters': min(1, len(embeddings)),
                    'silhouette_score': 0.0
                }
            
            embeddings_array = np.array(embeddings)
            
            if n_speakers is None:
                distances = sch.distance.pdist(embeddings_array, metric='cosine')
                linkage_matrix = sch.linkage(distances, method='ward')
                n_speakers = min(len(embeddings), self._estimate_clusters(linkage_matrix))
            
            if n_speakers == 1:
                labels = [0] * len(embeddings)
                silhouette_score = 0.0
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=n_speakers,
                    linkage='ward',
                    metric='euclidean'
                )
                
                labels = clusterer.fit_predict(embeddings_array)
                
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette_score = silhouette_score(embeddings_array, labels)
                except:
                    silhouette_score = 0.0
            
            return {
                'labels': labels.tolist(),
                'n_clusters': n_speakers,
                'silhouette_score': float(silhouette_score)
            }
            
        except Exception as e:
            logger.exception(f"Error in embedding clustering: {e}")
            return {
                'labels': list(range(len(embeddings))),
                'n_clusters': len(embeddings),
                'silhouette_score': 0.0
            }
    
    def _estimate_clusters(self, linkage_matrix: np.ndarray, max_clusters: int = 10) -> int:
        """Estimate optimal number of clusters from linkage matrix"""
        try:
            distances = linkage_matrix[:, 2]
            
            if len(distances) < 2:
                return 1
            
            gaps = np.diff(sorted(distances, reverse=True))
            
            if len(gaps) == 0:
                return 1
            
            max_gap_idx = np.argmax(gaps)
            estimated_clusters = min(max_gap_idx + 2, max_clusters)
            
            return max(1, estimated_clusters)
            
        except Exception as e:
            logger.exception(f"Error estimating clusters: {e}")
            return 2
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get speaker model information"""
        if self.use_onnx:
            model_type = "ONNX WeSpeaker"
        elif isinstance(self.model, str):
            model_type = self.model
        elif hasattr(self.model, '__class__'):
            model_type = self.model.__class__.__name__
        else:
            model_type = "unknown"
        
        return {
            'model_type': model_type,
            'embedding_dimension': 256,
            'sample_rate': 16000,
            'min_duration': 0.5,
            'device': str(self.device),
            'similarity_threshold': getattr(model_config, 'speaker_similarity_threshold', 0.6)
        }


# Keep TritonSpeakerInference unchanged
class TritonSpeakerInference(TritonModelWrapper):
    """Triton-based speaker inference wrapper"""
    
    def __init__(self, triton_client: TritonClient):
        super().__init__(
            model_name=model_config.triton_speaker_model,
            triton_client=triton_client,
            input_names=['audio', 'sample_rate'],
            output_names=['embedding']
        )
    
    async def extract_embedding(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Extract speaker embedding using Triton server"""
        try:
            inputs = {
                'audio': audio_data.astype(np.float32),
                'sample_rate': np.array([sample_rate], dtype=np.int32)
            }
            
            results = await self.predict(inputs)
            embedding = results['embedding'].flatten()
            
            if len(embedding) != 256:
                if len(embedding) > 256:
                    embedding = embedding[:256]
                else:
                    embedding = np.pad(embedding, (0, 256 - len(embedding)))
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.exception(f"Error in Triton speaker inference: {e}")
            return np.zeros(256)