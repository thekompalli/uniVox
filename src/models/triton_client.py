"""
Triton Inference Server Client
GPU serving interface for ML models
"""
import logging
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Union
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

from src.config.app_config import app_config, model_config

logger = logging.getLogger(__name__)


class TritonClient:
    """Client for Triton Inference Server integration"""
    
    def __init__(self):
        self.http_client = None
        self.grpc_client = None
        self.use_grpc = app_config.triton_use_grpc
        self.server_url = app_config.triton_server_url
        self.model_configs = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize Triton clients"""
        try:
            logger.info(f"Initializing Triton client: {self.server_url}")
            
            if self.use_grpc:
                self.grpc_client = grpcclient.InferenceServerClient(
                    url=self.server_url,
                    verbose=False
                )
                
                # Check server health
                if not self.grpc_client.is_server_live():
                    raise ConnectionError("Triton gRPC server is not live")
                    
                if not self.grpc_client.is_server_ready():
                    raise ConnectionError("Triton gRPC server is not ready")
                
                logger.info("Connected to Triton gRPC server")
                
            else:
                self.http_client = httpclient.InferenceServerClient(
                    url=self.server_url,
                    verbose=False
                )
                
                # Check server health
                if not self.http_client.is_server_live():
                    raise ConnectionError("Triton HTTP server is not live")
                    
                if not self.http_client.is_server_ready():
                    raise ConnectionError("Triton HTTP server is not ready")
                
                logger.info("Connected to Triton HTTP server")
            
            # Load model configurations
            await self._load_model_configs()
            
            self._initialized = True
            logger.info("Triton client initialized successfully")
            
        except Exception as e:
            logger.exception(f"Failed to initialize Triton client: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup Triton clients"""
        try:
            if self.grpc_client:
                self.grpc_client.close()
            if self.http_client:
                self.http_client.close()
            
            logger.info("Triton client cleanup completed")
            
        except Exception as e:
            logger.exception(f"Error during Triton client cleanup: {e}")
    
    async def _load_model_configs(self):
        """Load model configurations from Triton server"""
        try:
            client = self.grpc_client if self.use_grpc else self.http_client
            
            # Get list of available models
            models_list = client.get_model_repository_index()
            
            for model_info in models_list:
                model_name = model_info['name']
                try:
                    config = client.get_model_config(model_name)
                    self.model_configs[model_name] = config
                    logger.debug(f"Loaded config for model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load config for model {model_name}: {e}")
            
            logger.info(f"Loaded configurations for {len(self.model_configs)} models")
            
        except Exception as e:
            logger.exception(f"Error loading model configurations: {e}")
    
    async def infer(
        self, 
        model_name: str, 
        inputs: Dict[str, np.ndarray],
        outputs: Optional[List[str]] = None,
        model_version: str = ""
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on Triton server
        
        Args:
            model_name: Name of the model
            inputs: Dictionary of input tensors
            outputs: List of output names to return
            model_version: Model version to use
            
        Returns:
            Dictionary of output tensors
        """
        try:
            if not self._initialized:
                raise RuntimeError("Triton client not initialized")
            
            client = self.grpc_client if self.use_grpc else self.http_client
            
            # Prepare input tensors
            triton_inputs = []
            for name, data in inputs.items():
                # Ensure data is numpy array
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                
                # Create Triton input
                triton_input = self._create_triton_input(name, data)
                triton_inputs.append(triton_input)
            
            # Prepare outputs
            triton_outputs = []
            if outputs:
                for output_name in outputs:
                    triton_output = self._create_triton_output(output_name)
                    triton_outputs.append(triton_output)
            
            # Run inference
            if self.use_grpc:
                response = client.infer(
                    model_name=model_name,
                    inputs=triton_inputs,
                    outputs=triton_outputs,
                    model_version=model_version
                )
            else:
                response = client.infer(
                    model_name=model_name,
                    inputs=triton_inputs,
                    outputs=triton_outputs,
                    model_version=model_version
                )
            
            # Extract results
            results = {}
            if outputs:
                for output_name in outputs:
                    results[output_name] = response.as_numpy(output_name)
            else:
                # Get all outputs
                for output in response._result.outputs:
                    results[output.name] = response.as_numpy(output.name)
            
            return results
            
        except InferenceServerException as e:
            logger.exception(f"Triton inference error for model {model_name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error running inference on {model_name}: {e}")
            raise
    
    def _create_triton_input(self, name: str, data: np.ndarray):
        """Create Triton input tensor"""
        if self.use_grpc:
            return grpcclient.InferInput(
                name, data.shape, np_to_triton_dtype(data.dtype)
            ).set_data_from_numpy(data)
        else:
            return httpclient.InferInput(
                name, data.shape, np_to_triton_dtype(data.dtype)
            ).set_data_from_numpy(data)
    
    def _create_triton_output(self, name: str):
        """Create Triton output tensor"""
        if self.use_grpc:
            return grpcclient.InferRequestedOutput(name)
        else:
            return httpclient.InferRequestedOutput(name)
    
    async def load_model(self, model_name: str) -> bool:
        """Load model on Triton server"""
        try:
            client = self.grpc_client if self.use_grpc else self.http_client
            client.load_model(model_name)
            
            logger.info(f"Loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading model {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload model from Triton server"""
        try:
            client = self.grpc_client if self.use_grpc else self.http_client
            client.unload_model(model_name)
            
            logger.info(f"Unloaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.exception(f"Error unloading model {model_name}: {e}")
            return False
    
    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get model status from Triton server"""
        try:
            client = self.grpc_client if self.use_grpc else self.http_client
            status = client.get_model_metadata(model_name)
            
            return {
                'name': status.name,
                'versions': status.versions,
                'platform': status.platform,
                'inputs': [
                    {
                        'name': inp.name,
                        'shape': list(inp.shape),
                        'datatype': inp.datatype
                    } for inp in status.inputs
                ],
                'outputs': [
                    {
                        'name': out.name,
                        'shape': list(out.shape),
                        'datatype': out.datatype
                    } for out in status.outputs
                ]
            }
            
        except Exception as e:
            logger.exception(f"Error getting model status for {model_name}: {e}")
            return {}
    
    async def get_server_stats(self) -> Dict[str, Any]:
        """Get Triton server statistics"""
        try:
            client = self.grpc_client if self.use_grpc else self.http_client
            stats = client.get_inference_statistics()
            
            return {
                'model_stats': [
                    {
                        'name': model.name,
                        'version': model.version,
                        'last_inference': model.last_inference,
                        'inference_count': model.inference_count,
                        'execution_count': model.execution_count
                    } for model in stats.model_stats
                ]
            }
            
        except Exception as e:
            logger.exception(f"Error getting server stats: {e}")
            return {}
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """Check if model is ready for inference"""
        try:
            client = self.grpc_client if self.use_grpc else self.http_client
            return client.is_model_ready(model_name, model_version)
            
        except Exception as e:
            logger.exception(f"Error checking model readiness: {e}")
            return False
    
    async def batch_infer(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, np.ndarray]],
        outputs: Optional[List[str]] = None,
        max_batch_size: int = 8
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run batch inference with dynamic batching
        
        Args:
            model_name: Name of the model
            batch_inputs: List of input dictionaries
            outputs: List of output names
            max_batch_size: Maximum batch size
            
        Returns:
            List of output dictionaries
        """
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(batch_inputs), max_batch_size):
                batch = batch_inputs[i:i + max_batch_size]
                
                # Stack inputs for batching
                batched_inputs = {}
                for key in batch[0].keys():
                    batched_data = np.stack([item[key] for item in batch], axis=0)
                    batched_inputs[key] = batched_data
                
                # Run batch inference
                batch_results = await self.infer(
                    model_name=model_name,
                    inputs=batched_inputs,
                    outputs=outputs
                )
                
                # Unstack results
                batch_size = len(batch)
                for j in range(batch_size):
                    result = {}
                    for key, value in batch_results.items():
                        result[key] = value[j]
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in batch inference: {e}")
            raise


class TritonModelWrapper:
    """Base wrapper for Triton-served models"""
    
    def __init__(
        self, 
        model_name: str, 
        triton_client: TritonClient,
        input_names: List[str],
        output_names: List[str]
    ):
        self.model_name = model_name
        self.triton_client = triton_client
        self.input_names = input_names
        self.output_names = output_names
        self.model_ready = False
    
    async def initialize(self):
        """Initialize the model wrapper"""
        try:
            # Check if model is ready
            self.model_ready = self.triton_client.is_model_ready(self.model_name)
            
            if not self.model_ready:
                # Try to load the model
                await self.triton_client.load_model(self.model_name)
                self.model_ready = self.triton_client.is_model_ready(self.model_name)
            
            if not self.model_ready:
                raise RuntimeError(f"Model {self.model_name} is not ready")
            
            logger.info(f"Initialized Triton model wrapper: {self.model_name}")
            
        except Exception as e:
            logger.exception(f"Error initializing model wrapper {self.model_name}: {e}")
            raise
    
    async def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run prediction using the model"""
        try:
            if not self.model_ready:
                raise RuntimeError(f"Model {self.model_name} is not ready")
            
            # Validate inputs
            for input_name in self.input_names:
                if input_name not in inputs:
                    raise ValueError(f"Missing required input: {input_name}")
            
            # Run inference
            results = await self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=self.output_names
            )
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in model prediction: {e}")
            raise