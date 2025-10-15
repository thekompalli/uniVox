"""
Progress Tracker Component
Real-time progress tracking for audio processing jobs
"""

import streamlit as st
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Real-time progress tracking component"""

    def __init__(self, api_client):
        """
        Initialize progress tracker

        Args:
            api_client: PS06APIClient instance
        """
        self.api_client = api_client
        self.processing_steps = [
            'QUEUED',
            'PREPROCESSING',
            'DIARIZATION',
            'LANGUAGE_ID',
            'TRANSCRIPTION',
            'TRANSLATION',
            'FORMATTING',
            'COMPLETED'
        ]
        self.step_descriptions = {
            'QUEUED': 'Job queued for processing',
            'PREPROCESSING': 'Preprocessing audio file',
            'DIARIZATION': 'Speaker diarization in progress',
            'LANGUAGE_ID': 'Language identification',
            'TRANSCRIPTION': 'Speech recognition (ASR)',
            'TRANSLATION': 'Neural machine translation',
            'FORMATTING': 'Generating output files',
            'COMPLETED': 'Processing completed successfully',
            'FAILED': 'Processing failed',
            'CANCELLED': 'Processing cancelled'
        }

    def render(self, job_id: str):
        """
        Render progress tracking interface

        Args:
            job_id: Job identifier to track
        """
        # Initialize session state for this job
        if f"job_{job_id}_status" not in st.session_state:
            st.session_state[f"job_{job_id}_status"] = None
            st.session_state[f"job_{job_id}_start_time"] = datetime.now()

        # Create containers for dynamic updates
        status_container = st.empty()
        progress_container = st.empty()
        details_container = st.empty()
        timeline_container = st.empty()

        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Refresh Status", key=f"refresh_{job_id}"):
                self._refresh_status(job_id, status_container, progress_container, details_container, timeline_container)

        with col2:
            if st.button("❌ Cancel Job", key=f"cancel_{job_id}", type="secondary"):
                self._cancel_job(job_id)

        with col3:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=True, key=f"auto_refresh_{job_id}")

        # Initial status check
        self._refresh_status(job_id, status_container, progress_container, details_container, timeline_container)

        # Auto-refresh loop
        if auto_refresh:
            current_status = st.session_state.get(f"job_{job_id}_status")
            if current_status and current_status.get('status') not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                time.sleep(3)  # Wait 3 seconds
                st.rerun()

    def _refresh_status(self, job_id: str, status_container, progress_container, details_container, timeline_container):
        """
        Refresh job status and update containers

        Args:
            job_id: Job identifier
            status_container: Streamlit container for status display
            progress_container: Streamlit container for progress display
            details_container: Streamlit container for details
            timeline_container: Streamlit container for timeline
        """
        try:
            # Get current status
            status_data = self.api_client.get_job_status(job_id)

            if status_data:
                # Store in session state
                st.session_state[f"job_{job_id}_status"] = status_data

                # Update status display
                self._update_status_display(status_data, status_container)

                # Update progress display
                self._update_progress_display(status_data, progress_container)

                # Update details
                self._update_details_display(status_data, details_container)

                # Update timeline
                self._update_timeline_display(job_id, status_data, timeline_container)

            else:
                status_container.error("❌ Failed to get job status")

        except Exception as e:
            logger.exception(f"Error refreshing status: {e}")
            status_container.error(f"❌ Error: {str(e)}")

    def _update_status_display(self, status_data: Dict[str, Any], container):
        """Update status display"""
        status = status_data.get('status', 'UNKNOWN')
        progress = status_data.get('progress', 0.0)
        current_step = status_data.get('current_step', status)

        with container:
            col1, col2, col3 = st.columns(3)

            with col1:
                # Status badge
                if status == 'COMPLETED':
                    st.success(f"✅ {status}")
                elif status == 'FAILED':
                    st.error(f"❌ {status}")
                elif status == 'CANCELLED':
                    st.warning(f"⚠️ {status}")
                else:
                    st.info(f"🔄 {status}")

            with col2:
                # Progress percentage
                st.metric("Progress", f"{progress:.1%}")

            with col3:
                # Current step
                st.metric("Current Step", current_step)

    def _update_progress_display(self, status_data: Dict[str, Any], container):
        """Update progress bar and steps"""
        status = status_data.get('status', 'UNKNOWN')
        progress = status_data.get('progress', 0.0)

        with container:
            # Overall progress bar
            st.progress(progress)

            # Step indicators
            current_step = status_data.get('current_step', status)
            self._render_step_indicators(current_step, status)

    def _render_step_indicators(self, current_step: str, status: str):
        """Render processing step indicators"""
        steps_html = "<div style='display: flex; justify-content: space-between; margin: 20px 0;'>"

        current_step_index = -1
        if current_step in self.processing_steps:
            current_step_index = self.processing_steps.index(current_step)

        for i, step in enumerate(self.processing_steps):
            # Determine step status
            if status == 'FAILED' and i >= current_step_index:
                step_status = 'failed'
            elif status == 'CANCELLED' and i >= current_step_index:
                step_status = 'cancelled'
            elif i < current_step_index:
                step_status = 'completed'
            elif i == current_step_index:
                step_status = 'active'
            else:
                step_status = 'pending'

            # Style based on status
            if step_status == 'completed':
                bg_color = '#28a745'
                text_color = 'white'
                icon = '✅'
            elif step_status == 'active':
                bg_color = '#007bff'
                text_color = 'white'
                icon = '🔄'
            elif step_status == 'failed':
                bg_color = '#dc3545'
                text_color = 'white'
                icon = '❌'
            elif step_status == 'cancelled':
                bg_color = '#ffc107'
                text_color = 'black'
                icon = '⚠️'
            else:
                bg_color = '#f8f9fa'
                text_color = '#6c757d'
                icon = '⏳'

            step_html = f"""
            <div style='
                background-color: {bg_color};
                color: {text_color};
                padding: 10px;
                border-radius: 5px;
                text-align: center;
                min-width: 100px;
                font-size: 12px;
                border: 1px solid #dee2e6;
            '>
                <div style='font-size: 16px; margin-bottom: 5px;'>{icon}</div>
                <div style='font-weight: bold;'>{step}</div>
            </div>
            """

            steps_html += step_html

            # Add arrow between steps (except last)
            if i < len(self.processing_steps) - 1:
                steps_html += "<div style='align-self: center; margin: 0 5px; color: #6c757d;'>→</div>"

        steps_html += "</div>"

        st.markdown(steps_html, unsafe_allow_html=True)

    def _update_details_display(self, status_data: Dict[str, Any], container):
        """Update detailed information display"""
        with container:
            with st.expander("📋 Job Details", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("⏰ Timing")
                    created_at = status_data.get('created_at')
                    updated_at = status_data.get('updated_at')
                    estimated_completion = status_data.get('estimated_completion')

                    if created_at:
                        st.text(f"Started: {created_at}")
                    if updated_at:
                        st.text(f"Last Update: {updated_at}")
                    if estimated_completion:
                        st.text(f"Est. Completion: {estimated_completion}")

                with col2:
                    st.subheader("🔧 Processing Info")
                    st.text(f"Job ID: {status_data.get('job_id', 'N/A')}")
                    st.text(f"Status: {status_data.get('status', 'N/A')}")
                    st.text(f"Progress: {status_data.get('progress', 0):.1%}")

                # Error information if available
                error_msg = status_data.get('error_msg')
                if error_msg:
                    st.subheader("❌ Error Information")
                    st.error(error_msg)

    def _update_timeline_display(self, job_id: str, status_data: Dict[str, Any], container):
        """Update processing timeline visualization"""
        with container:
            with st.expander("📈 Processing Timeline", expanded=False):
                self._create_timeline_chart(job_id, status_data)

    def _create_timeline_chart(self, job_id: str, status_data: Dict[str, Any]):
        """Create timeline chart showing processing steps"""
        try:
            # Get processing timeline from session state or create new
            timeline_key = f"job_{job_id}_timeline"
            if timeline_key not in st.session_state:
                st.session_state[timeline_key] = []

            # Update timeline with current status
            current_time = datetime.now()
            current_step = status_data.get('current_step', status_data.get('status'))

            # Add new timeline entry if step changed
            timeline = st.session_state[timeline_key]
            if not timeline or timeline[-1]['step'] != current_step:
                timeline.append({
                    'step': current_step,
                    'timestamp': current_time,
                    'status': status_data.get('status'),
                    'progress': status_data.get('progress', 0)
                })
                st.session_state[timeline_key] = timeline

            if len(timeline) > 1:
                # Create timeline chart
                steps = [entry['step'] for entry in timeline]
                timestamps = [entry['timestamp'] for entry in timeline]
                progress_values = [entry['progress'] for entry in timeline]

                fig = go.Figure()

                # Add progress line
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=progress_values,
                    mode='lines+markers',
                    name='Progress',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))

                # Update layout
                fig.update_layout(
                    title="Processing Progress Over Time",
                    xaxis_title="Time",
                    yaxis_title="Progress",
                    height=300,
                    yaxis=dict(range=[0, 1], tickformat='.0%'),
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show step timeline
                st.subheader("🕐 Step Timeline")
                for i, entry in enumerate(timeline):
                    duration = ""
                    if i > 0:
                        prev_time = timeline[i-1]['timestamp']
                        duration = f" ({(entry['timestamp'] - prev_time).total_seconds():.1f}s)"

                    st.text(f"{entry['timestamp'].strftime('%H:%M:%S')} - {entry['step']}{duration}")

        except Exception as e:
            logger.exception(f"Error creating timeline chart: {e}")
            st.error(f"Error creating timeline: {str(e)}")

    def _cancel_job(self, job_id: str):
        """Cancel processing job"""
        try:
            with st.spinner("Cancelling job..."):
                success = self.api_client.cancel_job(job_id)

            if success:
                st.success("✅ Job cancelled successfully")
                # Update session state
                if f"job_{job_id}_status" in st.session_state:
                    st.session_state[f"job_{job_id}_status"]['status'] = 'CANCELLED'
                st.rerun()
            else:
                st.error("❌ Failed to cancel job")

        except Exception as e:
            logger.exception(f"Error cancelling job: {e}")
            st.error(f"❌ Error cancelling job: {str(e)}")

    def get_processing_duration(self, job_id: str) -> Optional[timedelta]:
        """
        Get total processing duration

        Args:
            job_id: Job identifier

        Returns:
            Processing duration or None
        """
        try:
            start_time = st.session_state.get(f"job_{job_id}_start_time")
            status_data = st.session_state.get(f"job_{job_id}_status")

            if start_time and status_data:
                if status_data.get('status') in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    # Use updated_at if available
                    end_time_str = status_data.get('updated_at')
                    if end_time_str:
                        try:
                            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                            return end_time - start_time
                        except:
                            pass

                # Fallback to current time
                return datetime.now() - start_time

            return None

        except Exception:
            return None