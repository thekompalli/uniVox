"""
PS-06 API Client
Python client for interacting with the PS-06 backend API
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
import logging
import io

logger = logging.getLogger(__name__)

class PS06APIClient:
    """Client for PS-06 Competition System API"""

    def __init__(self, base_url: str = "http://localhost:8000/api/v1", timeout: int = 30):
        """
        Initialize API client

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

        # Set common headers
        self.session.headers.update({
            'User-Agent': 'PS06-Streamlit-Frontend/1.0.0',
            'Accept': 'application/json'
        })

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to API

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response data or None if error
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )

            response.raise_for_status()

            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
                if data.get('success'):
                    return data.get('data')
                else:
                    logger.error(f"API returned error: {data.get('error')}")
                    return None
            else:
                return {"raw_response": response.text}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def get_health(self) -> Optional[Dict[str, Any]]:
        """Get basic health status"""
        return self._make_request('GET', '/health')

    def get_detailed_health(self) -> Optional[Dict[str, Any]]:
        """Get detailed health status"""
        return self._make_request('GET', '/health/detailed')

    def get_version(self) -> Optional[Dict[str, Any]]:
        """Get version information"""
        return self._make_request('GET', '/version')

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get system metrics"""
        return self._make_request('GET', '/metrics')

    def submit_job(self, audio_file, filename: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submit audio processing job

        Args:
            audio_file: Audio file data
            filename: Original filename
            config: Processing configuration

        Returns:
            Job data or None if error
        """
        try:
            # Prepare files
            files = {
                'audio_file': (filename, audio_file, 'audio/wav')
            }

            # Prepare form data
            data = {
                'languages': ','.join(config.get('languages', ['english'])),
                'quality_mode': config.get('quality_mode', 'balanced'),
                'enable_overlaps': str(config.get('enable_overlaps', True)).lower(),
                'min_segment_duration': str(config.get('min_segment_duration', 0.5))
            }

            # Make request without using _make_request due to file upload
            url = f"{self.base_url}/process"
            response = self.session.post(
                url,
                files=files,
                data=data,
                timeout=300  # Longer timeout for file upload
            )

            response.raise_for_status()
            result = response.json()

            if result.get('success'):
                return result.get('data')
            else:
                logger.error(f"Job submission failed: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job processing status

        Args:
            job_id: Job identifier

        Returns:
            Job status data or None if error
        """
        return self._make_request('GET', f'/status/{job_id}')

    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job processing results

        Args:
            job_id: Job identifier

        Returns:
            Job results or None if error
        """
        return self._make_request('GET', f'/result/{job_id}')

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel processing job

        Args:
            job_id: Job identifier

        Returns:
            True if successful, False otherwise
        """
        result = self._make_request('DELETE', f'/job/{job_id}')
        return result is not None

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 5,
        max_wait_time: int = 3600,
        callback = None
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for job completion with polling

        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            max_wait_time: Maximum time to wait in seconds
            callback: Optional callback function called with status updates

        Returns:
            Final job results or None if failed/timeout
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_data = self.get_job_status(job_id)

            if not status_data:
                logger.error("Failed to get job status")
                return None

            status = status_data.get('status')

            # Call callback if provided
            if callback:
                callback(status_data)

            # Check terminal states
            if status == 'COMPLETED':
                return self.get_job_result(job_id)
            elif status in ['FAILED', 'CANCELLED']:
                logger.error(f"Job {job_id} terminated with status: {status}")
                return None

            # Wait before next poll
            time.sleep(poll_interval)

        logger.error(f"Job {job_id} timed out after {max_wait_time} seconds")
        return None

    def submit_batch_job(self, files: List[str], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submit batch processing job

        Args:
            files: List of file paths
            config: Processing configuration

        Returns:
            Batch job data or None if error
        """
        payload = {
            'files': files,
            'common_settings': config,
            'priority': config.get('priority', 1)
        }

        return self._make_request('POST', '/batch', json=payload)

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch processing status

        Args:
            batch_id: Batch identifier

        Returns:
            Batch status data or None if error
        """
        return self._make_request('GET', f'/batch/{batch_id}')

    def test_connection(self) -> bool:
        """
        Test API connection

        Returns:
            True if connected, False otherwise
        """
        try:
            health = self.get_health()
            return health is not None and health.get('status') == 'healthy'
        except Exception:
            return False

    def get_supported_formats(self) -> List[str]:
        """
        Get supported audio formats

        Returns:
            List of supported file extensions
        """
        # Based on API documentation
        return ['wav', 'mp3', 'flac', 'ogg', 'm4a']

    def get_supported_languages(self) -> List[str]:
        """
        Get supported languages

        Returns:
            List of supported language codes
        """
        # Based on API documentation
        return ['english', 'hindi', 'punjabi', 'bengali', 'nepali', 'dogri']

    def get_quality_modes(self) -> List[str]:
        """
        Get available quality modes

        Returns:
            List of quality mode options
        """
        return ['fast', 'balanced', 'high']