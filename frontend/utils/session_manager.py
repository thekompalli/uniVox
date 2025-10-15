"""
Session Manager
Manage Streamlit session state for the PS-06 frontend
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SessionManager:
    """Manage session state for the frontend application"""

    def __init__(self):
        """Initialize session manager"""
        self.session_keys = {
            'jobs': 'ps06_jobs',
            'current_job': 'current_job',
            'api_config': 'api_config',
            'processing_config': 'processing_config',
            'user_preferences': 'user_preferences',
            'cache': 'ps06_cache'
        }

    def initialize_session(self):
        """Initialize session state with default values"""
        # Initialize jobs tracking
        if self.session_keys['jobs'] not in st.session_state:
            st.session_state[self.session_keys['jobs']] = {}

        # Initialize API configuration
        if self.session_keys['api_config'] not in st.session_state:
            st.session_state[self.session_keys['api_config']] = {
                'host': 'localhost',
                'port': 8000,
                'base_url': 'http://localhost:8000/api/v1'
            }

        # Initialize processing configuration
        if self.session_keys['processing_config'] not in st.session_state:
            st.session_state[self.session_keys['processing_config']] = {
                'quality_mode': 'balanced',
                'languages': ['english', 'hindi', 'punjabi'],
                'enable_overlaps': True,
                'min_segment_duration': 0.5
            }

        # Initialize user preferences
        if self.session_keys['user_preferences'] not in st.session_state:
            st.session_state[self.session_keys['user_preferences']] = {
                'auto_refresh': True,
                'refresh_interval': 5,
                'display_mode': 'detailed',
                'theme': 'light'
            }

        # Initialize cache
        if self.session_keys['cache'] not in st.session_state:
            st.session_state[self.session_keys['cache']] = {}

    def add_job(self, job_id: str, job_data: Dict[str, Any]):
        """
        Add a job to the session

        Args:
            job_id: Job identifier
            job_data: Job information
        """
        jobs = st.session_state[self.session_keys['jobs']]
        jobs[job_id] = {
            **job_data,
            'added_at': datetime.now(),
            'status_history': []
        }
        st.session_state[self.session_keys['jobs']] = jobs

    def update_job_status(self, job_id: str, status_data: Dict[str, Any]):
        """
        Update job status

        Args:
            job_id: Job identifier
            status_data: Updated status information
        """
        jobs = st.session_state[self.session_keys['jobs']]
        if job_id in jobs:
            # Add to status history
            jobs[job_id]['status_history'].append({
                'timestamp': datetime.now(),
                'status': status_data.get('status'),
                'progress': status_data.get('progress', 0)
            })

            # Update current status
            jobs[job_id]['current_status'] = status_data
            jobs[job_id]['last_updated'] = datetime.now()

            st.session_state[self.session_keys['jobs']] = jobs

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job information

        Args:
            job_id: Job identifier

        Returns:
            Job data or None if not found
        """
        jobs = st.session_state[self.session_keys['jobs']]
        return jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all jobs

        Returns:
            Dictionary of all jobs
        """
        return st.session_state[self.session_keys['jobs']].copy()

    def remove_job(self, job_id: str):
        """
        Remove a job from the session

        Args:
            job_id: Job identifier
        """
        jobs = st.session_state[self.session_keys['jobs']]
        if job_id in jobs:
            del jobs[job_id]
            st.session_state[self.session_keys['jobs']] = jobs

    def set_current_job(self, job_id: str):
        """
        Set the current active job

        Args:
            job_id: Job identifier
        """
        st.session_state[self.session_keys['current_job']] = job_id

    def get_current_job(self) -> Optional[str]:
        """
        Get current active job ID

        Returns:
            Current job ID or None
        """
        return st.session_state.get(self.session_keys['current_job'])

    def get_recent_jobs(self, limit: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Get recently added jobs

        Args:
            limit: Maximum number of jobs to return

        Returns:
            Dictionary of recent jobs
        """
        jobs = st.session_state[self.session_keys['jobs']]

        # Sort by added_at timestamp
        sorted_jobs = sorted(
            jobs.items(),
            key=lambda x: x[1].get('added_at', datetime.min),
            reverse=True
        )

        return dict(sorted_jobs[:limit])

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Remove old jobs from session

        Args:
            max_age_hours: Maximum age in hours
        """
        jobs = st.session_state[self.session_keys['jobs']]
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        jobs_to_remove = [
            job_id for job_id, job_data in jobs.items()
            if job_data.get('added_at', datetime.now()) < cutoff_time
        ]

        for job_id in jobs_to_remove:
            del jobs[job_id]

        st.session_state[self.session_keys['jobs']] = jobs

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return st.session_state[self.session_keys['api_config']].copy()

    def update_api_config(self, config: Dict[str, Any]):
        """
        Update API configuration

        Args:
            config: New API configuration
        """
        current_config = st.session_state[self.session_keys['api_config']]
        current_config.update(config)
        st.session_state[self.session_keys['api_config']] = current_config

    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return st.session_state[self.session_keys['processing_config']].copy()

    def update_processing_config(self, config: Dict[str, Any]):
        """
        Update processing configuration

        Args:
            config: New processing configuration
        """
        current_config = st.session_state[self.session_keys['processing_config']]
        current_config.update(config)
        st.session_state[self.session_keys['processing_config']] = current_config

    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        return st.session_state[self.session_keys['user_preferences']].copy()

    def update_user_preferences(self, preferences: Dict[str, Any]):
        """
        Update user preferences

        Args:
            preferences: New preferences
        """
        current_prefs = st.session_state[self.session_keys['user_preferences']]
        current_prefs.update(preferences)
        st.session_state[self.session_keys['user_preferences']] = current_prefs

    def cache_data(self, key: str, data: Any, ttl_seconds: int = 300):
        """
        Cache data with TTL

        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds
        """
        cache = st.session_state[self.session_keys['cache']]
        cache[key] = {
            'data': data,
            'cached_at': datetime.now(),
            'ttl': ttl_seconds
        }
        st.session_state[self.session_keys['cache']] = cache

    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Get cached data

        Args:
            key: Cache key

        Returns:
            Cached data or None if expired/not found
        """
        cache = st.session_state[self.session_keys['cache']]

        if key not in cache:
            return None

        cached_item = cache[key]
        cached_at = cached_item['cached_at']
        ttl = cached_item['ttl']

        # Check if expired
        if datetime.now() - cached_at > timedelta(seconds=ttl):
            del cache[key]
            st.session_state[self.session_keys['cache']] = cache
            return None

        return cached_item['data']

    def clear_cache(self):
        """Clear all cached data"""
        st.session_state[self.session_keys['cache']] = {}

    def export_session_data(self) -> Dict[str, Any]:
        """
        Export session data for backup/debugging

        Returns:
            Dictionary of session data
        """
        return {
            key: st.session_state.get(session_key, {})
            for key, session_key in self.session_keys.items()
        }

    def import_session_data(self, data: Dict[str, Any]):
        """
        Import session data

        Args:
            data: Session data to import
        """
        for key, session_key in self.session_keys.items():
            if key in data:
                st.session_state[session_key] = data[key]

    def reset_session(self):
        """Reset all session data"""
        for session_key in self.session_keys.values():
            if session_key in st.session_state:
                del st.session_state[session_key]

        # Reinitialize
        self.initialize_session()

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics

        Returns:
            Dictionary of session statistics
        """
        jobs = st.session_state[self.session_keys['jobs']]
        cache = st.session_state[self.session_keys['cache']]

        # Job statistics
        job_statuses = {}
        for job_data in jobs.values():
            current_status = job_data.get('current_status', {})
            status = current_status.get('status', 'Unknown')
            job_statuses[status] = job_statuses.get(status, 0) + 1

        # Cache statistics
        cache_stats = {
            'total_items': len(cache),
            'expired_items': 0
        }

        now = datetime.now()
        for cached_item in cache.values():
            cached_at = cached_item['cached_at']
            ttl = cached_item['ttl']
            if now - cached_at > timedelta(seconds=ttl):
                cache_stats['expired_items'] += 1

        return {
            'total_jobs': len(jobs),
            'job_statuses': job_statuses,
            'cache_stats': cache_stats,
            'session_keys': list(self.session_keys.keys())
        }