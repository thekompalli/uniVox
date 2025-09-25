#!/usr/bin/env python3
"""
PS-06 Competition System Frontend
Streamlit-based web interface for audio processing
"""

import streamlit as st
import requests
import time
import json
import io
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from frontend.components.audio_uploader import AudioUploader
from frontend.components.progress_tracker import ProgressTracker
from frontend.components.results_display import ResultsDisplay
from frontend.utils.api_client import PS06APIClient
from frontend.utils.audio_utils import AudioUtils
from frontend.utils.session_manager import SessionManager

# Page configuration
st.set_page_config(
    page_title="PS-06 Competition System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/ps06-system',
        'Report a bug': 'https://github.com/your-repo/ps06-system/issues',
        'About': """
        # PS-06 Competition System

        Language Agnostic Speaker Identification & Diarization System

        **Features:**
        - Speaker Identification & Diarization
        - Language Identification
        - Automatic Speech Recognition (ASR)
        - Neural Machine Translation (NMT)
        - Multi-language support (English, Hindi, Punjabi, Bengali, Nepali, Dogri)

        **Version:** 1.0.0
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }

    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }

    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }

    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }

    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }

    .processing-step {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        background-color: #f8f9fa;
    }

    .processing-step.active {
        background-color: #007bff;
        color: white;
    }

    .processing-step.completed {
        background-color: #28a745;
        color: white;
    }

    .processing-step.failed {
        background-color: #dc3545;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""

    # Initialize session manager
    session_manager = SessionManager()
    session_manager.initialize_session()

    # Main header
    st.markdown('<h1 class="main-header">🎤 PS-06 Competition System</h1>', unsafe_allow_html=True)
    st.markdown("**Language Agnostic Speaker Identification & Diarization System**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Configuration
        st.subheader("Backend Connection")
        api_host = st.text_input("API Host", value="localhost", help="Backend server host")
        api_port = st.number_input("API Port", value=8000, min_value=1, max_value=65535, help="Backend server port")

        # Processing Configuration
        st.subheader("Processing Settings")
        quality_mode = st.selectbox(
            "Quality Mode",
            ["fast", "balanced", "high"],
            index=1,
            help="Processing quality vs speed trade-off"
        )

        supported_languages = ["english", "hindi", "punjabi", "bengali", "nepali", "dogri"]
        default_languages = ["english", "hindi", "punjabi"]

        selected_languages = st.multiselect(
            "Expected Languages",
            supported_languages,
            default=default_languages,
            help="Languages expected in the audio"
        )

        enable_overlaps = st.checkbox(
            "Enable Overlap Detection",
            value=True,
            help="Detect and handle overlapping speech"
        )

        min_segment_duration = st.slider(
            "Min Segment Duration (seconds)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Minimum duration for speech segments"
        )

        # Store configuration in session state
        st.session_state.api_config = {
            "host": api_host,
            "port": api_port,
            "base_url": f"http://{api_host}:{api_port}/api/v1"
        }

        st.session_state.processing_config = {
            "quality_mode": quality_mode,
            "languages": selected_languages,
            "enable_overlaps": enable_overlaps,
            "min_segment_duration": min_segment_duration
        }

    # Initialize API client
    api_client = PS06APIClient(st.session_state.api_config["base_url"])

    # Check backend connection
    with st.sidebar:
        st.subheader("🔌 Backend Status")
        if st.button("Check Connection", type="secondary"):
            try:
                health = api_client.get_health()
                if health and health.get("status") == "healthy":
                    st.success("✅ Connected")
                    st.json(health)
                else:
                    st.error("❌ Backend not responding")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "📈 Analytics", "🔧 System Info"])

    with tab1:
        st.header("Audio Processing")

        # Audio upload component
        audio_uploader = AudioUploader()
        uploaded_file, upload_info = audio_uploader.render()

        if uploaded_file is not None:
            # Display audio info
            st.subheader("📁 File Information")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("File Name", upload_info["name"])
            with col2:
                st.metric("Size", f"{upload_info['size']:.1f} MB")
            with col3:
                st.metric("Format", upload_info["type"])
            with col4:
                st.metric("Duration", f"{upload_info.get('duration', 'N/A')}")

            # Audio player
            st.audio(uploaded_file, format=upload_info["type"])

            # Process button
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):

                # Validate configuration
                if not selected_languages:
                    st.error("Please select at least one language")
                    st.stop()

                # Submit job
                with st.spinner("Submitting job to backend..."):
                    try:
                        job_data = api_client.submit_job(
                            uploaded_file,
                            upload_info["name"],
                            st.session_state.processing_config
                        )

                        if job_data:
                            st.session_state.current_job = job_data
                            st.success(f"✅ Job submitted successfully! Job ID: {job_data['job_id']}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to submit job")

                    except Exception as e:
                        st.error(f"❌ Error submitting job: {str(e)}")

        # Progress tracking
        if "current_job" in st.session_state:
            st.subheader("📊 Processing Progress")
            progress_tracker = ProgressTracker(api_client)
            progress_tracker.render(st.session_state.current_job["job_id"])

    with tab2:
        st.header("Processing Results")

        if "current_job" in st.session_state:
            results_display = ResultsDisplay(api_client)
            results_display.render(st.session_state.current_job["job_id"])
        else:
            st.info("No processing job in progress. Upload and process an audio file to see results.")

    with tab3:
        st.header("Analytics Dashboard")
        render_analytics_tab(api_client)

    with tab4:
        st.header("System Information")
        render_system_info_tab(api_client)

def render_analytics_tab(api_client):
    """Render analytics dashboard"""
    try:
        metrics = api_client.get_metrics()

        if metrics:
            # System metrics
            st.subheader("🖥️ System Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "System Load",
                    f"{metrics.get('system_load', 0):.2%}",
                    help="CPU usage percentage"
                )

            with col2:
                st.metric(
                    "Memory Usage",
                    f"{metrics.get('memory_usage', 0):.2%}",
                    help="Memory usage percentage"
                )

            with col3:
                st.metric(
                    "GPU Utilization",
                    f"{metrics.get('gpu_utilization', 0):.2%}",
                    help="GPU usage percentage"
                )

            with col4:
                st.metric(
                    "Active Jobs",
                    metrics.get('jobs_queued', 0),
                    help="Jobs currently being processed"
                )

            # Request statistics
            st.subheader("📈 Request Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Requests",
                    metrics.get('requests_total', 0)
                )

            with col2:
                st.metric(
                    "Success Rate",
                    f"{(metrics.get('requests_successful', 0) / max(metrics.get('requests_total', 1), 1) * 100):.1f}%",
                    delta=f"{metrics.get('requests_failed', 0)} failed"
                )

            with col3:
                st.metric(
                    "Avg Response Time",
                    f"{metrics.get('average_response_time', 0):.3f}s"
                )

            # Processing statistics
            st.subheader("⚙️ Processing Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Jobs Processed",
                    metrics.get('jobs_processed', 0)
                )

            with col2:
                st.metric(
                    "Jobs Queued",
                    metrics.get('jobs_queued', 0)
                )

            with col3:
                st.metric(
                    "Jobs Failed",
                    metrics.get('jobs_failed', 0)
                )

        else:
            st.warning("Unable to fetch system metrics")

    except Exception as e:
        st.error(f"Error fetching analytics: {str(e)}")

def render_system_info_tab(api_client):
    """Render system information tab"""
    try:
        # Version information
        st.subheader("📋 Version Information")
        version_info = api_client.get_version()

        if version_info:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("System Version", version_info.get('version', 'Unknown'))
                st.metric("Python Version", version_info.get('python_version', 'Unknown'))
                st.metric("Build Date", version_info.get('build_date', 'Unknown'))

            with col2:
                st.metric("Git Commit", version_info.get('git_commit', 'Unknown')[:8])

                # Dependencies
                deps = version_info.get('dependencies', {})
                if deps:
                    st.subheader("📦 Dependencies")
                    for name, version in deps.items():
                        st.text(f"{name}: {version}")

        # Detailed health check
        st.subheader("🏥 Detailed Health Check")
        if st.button("Run Health Check", type="secondary"):
            health_info = api_client.get_detailed_health()

            if health_info:
                # Component status
                components = health_info.get('components', {})

                st.subheader("🔧 Component Status")
                for component, status in components.items():
                    if status.get('healthy', False):
                        st.success(f"✅ {component.title()}: Healthy")
                    else:
                        st.error(f"❌ {component.title()}: Unhealthy")

                # System information
                system_info = health_info.get('system_info', {})
                if system_info:
                    st.subheader("💻 System Information")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Platform", system_info.get('platform', 'Unknown'))
                        st.metric("CPU Cores", system_info.get('cpu_count', 'Unknown'))
                        st.metric("Total Memory", f"{system_info.get('total_memory', 0):.1f} GB")

                    with col2:
                        st.metric("Available Memory", f"{system_info.get('available_memory', 0):.1f} GB")
                        st.metric("GPU Count", system_info.get('gpu_count', 0))
                        st.metric("Uptime", health_info.get('uptime', 'Unknown'))

    except Exception as e:
        st.error(f"Error fetching system information: {str(e)}")

if __name__ == "__main__":
    main()