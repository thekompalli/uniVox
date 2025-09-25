"""
Results Display Component
Comprehensive display of audio processing results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import io
import json

logger = logging.getLogger(__name__)

class ResultsDisplay:
    """Component for displaying processing results"""

    def __init__(self, api_client):
        """
        Initialize results display

        Args:
            api_client: PS06APIClient instance
        """
        self.api_client = api_client

    def render(self, job_id: str):
        """
        Render results display interface

        Args:
            job_id: Job identifier
        """
        # Check job status first
        status_data = self.api_client.get_job_status(job_id)

        if not status_data:
            st.error("❌ Unable to fetch job status")
            return

        status = status_data.get('status')

        if status == 'COMPLETED':
            self._render_completed_results(job_id, status_data)
        elif status in ['FAILED', 'CANCELLED']:
            self._render_failed_results(job_id, status_data)
        else:
            self._render_processing_status(job_id, status_data)

    def _render_completed_results(self, job_id: str, status_data: Dict[str, Any]):
        """Render completed job results"""
        try:
            # Get detailed results
            results = self.api_client.get_job_result(job_id)

            if not results:
                st.error("❌ Unable to fetch job results")
                return

            # Results overview
            self._render_results_overview(results)

            # Detailed results tabs
            self._render_results_tabs(results)

        except Exception as e:
            logger.exception(f"Error rendering completed results: {e}")
            st.error(f"❌ Error displaying results: {str(e)}")

    def _render_results_overview(self, results: Dict[str, Any]):
        """Render results overview section"""
        st.subheader("📊 Processing Results Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Processing Time",
                f"{results.get('processing_time', 0):.1f}s",
                help="Total time taken to process the audio"
            )

        with col2:
            st.metric(
                "Segments",
                len(results.get('segments', [])),
                help="Number of speech segments detected"
            )

        with col3:
            st.metric(
                "Speakers",
                results.get('speakers_detected', 0),
                help="Number of unique speakers identified"
            )

        with col4:
            languages = results.get('languages_detected', [])
            st.metric(
                "Languages",
                len(languages),
                delta=", ".join(languages[:2]) if languages else "None",
                help="Languages detected in the audio"
            )

        # Audio specifications
        audio_specs = results.get('audio_specs', {})
        if audio_specs:
            st.subheader("🎵 Audio Specifications")
            spec_col1, spec_col2, spec_col3, spec_col4 = st.columns(4)

            with spec_col1:
                st.metric("Format", audio_specs.get('format', 'Unknown'))

            with spec_col2:
                st.metric("Duration", f"{audio_specs.get('duration', 0):.1f}s")

            with spec_col3:
                st.metric("Sample Rate", f"{audio_specs.get('sample_rate', 0)} Hz")

            with spec_col4:
                st.metric("Channels", audio_specs.get('channels', 1))

    def _render_results_tabs(self, results: Dict[str, Any]):
        """Render detailed results in tabs"""
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📝 Transcription", "🗣️ Speakers", "🌐 Languages",
            "📈 Visualizations", "📊 Metrics", "📁 Downloads"
        ])

        with tab1:
            self._render_transcription_tab(results)

        with tab2:
            self._render_speakers_tab(results)

        with tab3:
            self._render_languages_tab(results)

        with tab4:
            self._render_visualizations_tab(results)

        with tab5:
            self._render_metrics_tab(results)

        with tab6:
            self._render_downloads_tab(results)

    def _render_transcription_tab(self, results: Dict[str, Any]):
        """Render transcription results tab"""
        st.subheader("📝 Transcription Results")

        segments = results.get('segments', [])

        if not segments:
            st.info("No transcription segments available")
            return

        # Search and filter options
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search_text = st.text_input("🔍 Search transcription", placeholder="Enter text to search...")

        with col2:
            selected_speaker = st.selectbox(
                "Filter by Speaker",
                ["All"] + list(set([s.get('speaker', 'Unknown') for s in segments]))
            )

        with col3:
            selected_language = st.selectbox(
                "Filter by Language",
                ["All"] + list(set([s.get('language', 'Unknown') for s in segments]))
            )

        # Apply filters
        filtered_segments = self._filter_segments(segments, search_text, selected_speaker, selected_language)

        # Display segments
        if filtered_segments:
            st.info(f"Showing {len(filtered_segments)} of {len(segments)} segments")

            # Segment display options
            display_mode = st.radio(
                "Display Mode",
                ["Detailed View", "Compact View", "Table View"],
                horizontal=True
            )

            if display_mode == "Detailed View":
                self._render_detailed_segments(filtered_segments)
            elif display_mode == "Compact View":
                self._render_compact_segments(filtered_segments)
            else:
                self._render_table_segments(filtered_segments)

        else:
            st.warning("No segments match the current filters")

    def _filter_segments(self, segments: List[Dict], search_text: str, speaker_filter: str, language_filter: str) -> List[Dict]:
        """Filter segments based on criteria"""
        filtered = segments

        # Text search
        if search_text:
            filtered = [
                s for s in filtered
                if search_text.lower() in s.get('text', '').lower() or
                   search_text.lower() in s.get('translated_text', '').lower()
            ]

        # Speaker filter
        if speaker_filter != "All":
            filtered = [s for s in filtered if s.get('speaker') == speaker_filter]

        # Language filter
        if language_filter != "All":
            filtered = [s for s in filtered if s.get('language') == language_filter]

        return filtered

    def _render_detailed_segments(self, segments: List[Dict]):
        """Render segments in detailed view"""
        for i, segment in enumerate(segments):
            with st.container():
                # Header with timing and speaker
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    st.markdown(f"**Segment {i+1}**")

                with col2:
                    st.text(f"🕒 {segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s")

                with col3:
                    st.text(f"🗣️ {segment.get('speaker', 'Unknown')}")

                with col4:
                    st.text(f"🌐 {segment.get('language', 'Unknown')}")

                # Original text
                st.markdown("**Original:**")
                st.text(segment.get('text', 'No text available'))

                # Translated text (if different)
                translated = segment.get('translated_text', '')
                original = segment.get('text', '')
                if translated and translated != original:
                    st.markdown("**Translation:**")
                    st.text(translated)

                # Confidence
                confidence = segment.get('confidence', 0)
                st.progress(confidence, text=f"Confidence: {confidence:.2%}")

                st.divider()

    def _render_compact_segments(self, segments: List[Dict]):
        """Render segments in compact view"""
        for i, segment in enumerate(segments):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', 'No text available')
            translated = segment.get('translated_text', '')

            # Use translated text if available and different
            display_text = translated if translated and translated != text else text

            st.markdown(f"""
            **[{start_time:.1f}s - {end_time:.1f}s] {speaker}:** {display_text}
            """)

    def _render_table_segments(self, segments: List[Dict]):
        """Render segments in table view"""
        # Prepare data for table
        table_data = []
        for segment in segments:
            row = {
                'Start': f"{segment.get('start', 0):.1f}s",
                'End': f"{segment.get('end', 0):.1f}s",
                'Duration': f"{(segment.get('end', 0) - segment.get('start', 0)):.1f}s",
                'Speaker': segment.get('speaker', 'Unknown'),
                'Language': segment.get('language', 'Unknown'),
                'Text': segment.get('text', 'No text available'),
                'Translation': segment.get('translated_text', ''),
                'Confidence': f"{segment.get('confidence', 0):.2%}"
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def _render_speakers_tab(self, results: Dict[str, Any]):
        """Render speaker analysis tab"""
        st.subheader("🗣️ Speaker Analysis")

        segments = results.get('segments', [])
        if not segments:
            st.info("No speaker data available")
            return

        # Speaker statistics
        speaker_stats = self._calculate_speaker_stats(segments)

        if speaker_stats:
            # Speaker overview
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👥 Speaker Distribution")
                speaker_df = pd.DataFrame([
                    {'Speaker': speaker, 'Segments': stats['segment_count'],
                     'Duration': f"{stats['total_duration']:.1f}s",
                     'Percentage': f"{stats['percentage']:.1f}%"}
                    for speaker, stats in speaker_stats.items()
                ])
                st.dataframe(speaker_df, use_container_width=True)

            with col2:
                st.subheader("📊 Speaking Time Distribution")
                # Pie chart
                fig = px.pie(
                    values=[stats['total_duration'] for stats in speaker_stats.values()],
                    names=list(speaker_stats.keys()),
                    title="Speaking Time by Speaker"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Timeline visualization
            self._render_speaker_timeline(segments)

    def _calculate_speaker_stats(self, segments: List[Dict]) -> Dict[str, Dict]:
        """Calculate speaker statistics"""
        stats = {}
        total_duration = sum(segment.get('end', 0) - segment.get('start', 0) for segment in segments)

        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            duration = segment.get('end', 0) - segment.get('start', 0)

            if speaker not in stats:
                stats[speaker] = {
                    'segment_count': 0,
                    'total_duration': 0,
                    'percentage': 0
                }

            stats[speaker]['segment_count'] += 1
            stats[speaker]['total_duration'] += duration

        # Calculate percentages
        for speaker in stats:
            if total_duration > 0:
                stats[speaker]['percentage'] = (stats[speaker]['total_duration'] / total_duration) * 100

        return stats

    def _render_speaker_timeline(self, segments: List[Dict]):
        """Render speaker timeline visualization"""
        st.subheader("⏱️ Speaker Timeline")

        # Create timeline chart
        fig = go.Figure()

        speakers = list(set([s.get('speaker', 'Unknown') for s in segments]))
        colors = px.colors.qualitative.Set3[:len(speakers)]
        speaker_colors = dict(zip(speakers, colors))

        for segment in segments:
            speaker = segment.get('speaker', 'Unknown')
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')[:50] + "..." if len(segment.get('text', '')) > 50 else segment.get('text', '')

            fig.add_trace(go.Scatter(
                x=[start, end, end, start, start],
                y=[speaker, speaker, speaker, speaker, speaker],
                mode='lines',
                fill='tonexty' if segment != segments[0] else None,
                fillcolor=speaker_colors[speaker],
                line=dict(color=speaker_colors[speaker], width=20),
                name=speaker,
                showlegend=speaker not in [trace.name for trace in fig.data],
                hovertemplate=f"<b>{speaker}</b><br>Time: {start:.1f}s - {end:.1f}s<br>Text: {text}<extra></extra>"
            ))

        fig.update_layout(
            title="Speaker Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Speaker",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_languages_tab(self, results: Dict[str, Any]):
        """Render language analysis tab"""
        st.subheader("🌐 Language Analysis")

        segments = results.get('segments', [])
        languages_detected = results.get('languages_detected', [])

        if not segments:
            st.info("No language data available")
            return

        # Language statistics
        language_stats = self._calculate_language_stats(segments)

        if language_stats:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🗣️ Language Distribution")
                lang_df = pd.DataFrame([
                    {'Language': lang, 'Segments': stats['segment_count'],
                     'Duration': f"{stats['total_duration']:.1f}s",
                     'Percentage': f"{stats['percentage']:.1f}%"}
                    for lang, stats in language_stats.items()
                ])
                st.dataframe(lang_df, use_container_width=True)

            with col2:
                st.subheader("📊 Language Time Distribution")
                fig = px.pie(
                    values=[stats['total_duration'] for stats in language_stats.values()],
                    names=list(language_stats.keys()),
                    title="Speaking Time by Language"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Language timeline
        self._render_language_timeline(segments)

    def _calculate_language_stats(self, segments: List[Dict]) -> Dict[str, Dict]:
        """Calculate language statistics"""
        stats = {}
        total_duration = sum(segment.get('end', 0) - segment.get('start', 0) for segment in segments)

        for segment in segments:
            language = segment.get('language', 'Unknown')
            duration = segment.get('end', 0) - segment.get('start', 0)

            if language not in stats:
                stats[language] = {
                    'segment_count': 0,
                    'total_duration': 0,
                    'percentage': 0
                }

            stats[language]['segment_count'] += 1
            stats[language]['total_duration'] += duration

        # Calculate percentages
        for language in stats:
            if total_duration > 0:
                stats[language]['percentage'] = (stats[language]['total_duration'] / total_duration) * 100

        return stats

    def _render_language_timeline(self, segments: List[Dict]):
        """Render language timeline visualization"""
        st.subheader("🌍 Language Timeline")

        # Create timeline chart
        fig = go.Figure()

        languages = list(set([s.get('language', 'Unknown') for s in segments]))
        colors = px.colors.qualitative.Pastel[:len(languages)]
        language_colors = dict(zip(languages, colors))

        y_pos = 0
        for segment in segments:
            language = segment.get('language', 'Unknown')
            start = segment.get('start', 0)
            end = segment.get('end', 0)

            fig.add_trace(go.Bar(
                x=[end - start],
                y=[y_pos],
                base=start,
                orientation='h',
                name=language,
                marker_color=language_colors[language],
                showlegend=language not in [trace.name for trace in fig.data],
                hovertemplate=f"<b>{language}</b><br>Time: {start:.1f}s - {end:.1f}s<extra></extra>"
            ))
            y_pos += 1

        fig.update_layout(
            title="Language Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Segments",
            height=max(400, len(segments) * 20),
            barmode='overlay'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_visualizations_tab(self, results: Dict[str, Any]):
        """Render visualization tab"""
        st.subheader("📈 Advanced Visualizations")

        segments = results.get('segments', [])
        if not segments:
            st.info("No data available for visualizations")
            return

        # Confidence distribution
        self._render_confidence_distribution(segments)

        # Segment duration analysis
        self._render_duration_analysis(segments)

        # Combined timeline
        self._render_combined_timeline(segments)

    def _render_confidence_distribution(self, segments: List[Dict]):
        """Render confidence score distribution"""
        st.subheader("📊 Confidence Score Distribution")

        confidences = [segment.get('confidence', 0) for segment in segments]

        fig = px.histogram(
            x=confidences,
            nbins=20,
            title="Distribution of Confidence Scores",
            labels={'x': 'Confidence Score', 'y': 'Number of Segments'}
        )

        fig.add_vline(x=np.mean(confidences), line_dash="dash",
                     annotation_text=f"Mean: {np.mean(confidences):.2%}")

        st.plotly_chart(fig, use_container_width=True)

    def _render_duration_analysis(self, segments: List[Dict]):
        """Render segment duration analysis"""
        st.subheader("⏱️ Segment Duration Analysis")

        durations = [segment.get('end', 0) - segment.get('start', 0) for segment in segments]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                x=durations,
                nbins=20,
                title="Segment Duration Distribution",
                labels={'x': 'Duration (seconds)', 'y': 'Number of Segments'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                y=durations,
                title="Segment Duration Statistics",
                labels={'y': 'Duration (seconds)'}
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_combined_timeline(self, segments: List[Dict]):
        """Render combined speaker and language timeline"""
        st.subheader("🔗 Combined Speaker & Language Timeline")

        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Speakers', 'Languages'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )

        # Speaker timeline
        speakers = list(set([s.get('speaker', 'Unknown') for s in segments]))
        speaker_colors = dict(zip(speakers, px.colors.qualitative.Set3[:len(speakers)]))

        for i, segment in enumerate(segments):
            speaker = segment.get('speaker', 'Unknown')
            start = segment.get('start', 0)
            end = segment.get('end', 0)

            fig.add_trace(go.Bar(
                x=[end - start],
                y=[speakers.index(speaker)],
                base=start,
                orientation='h',
                name=f"Speaker: {speaker}",
                marker_color=speaker_colors[speaker],
                showlegend=False
            ), row=1, col=1)

        # Language timeline
        languages = list(set([s.get('language', 'Unknown') for s in segments]))
        language_colors = dict(zip(languages, px.colors.qualitative.Pastel[:len(languages)]))

        for segment in segments:
            language = segment.get('language', 'Unknown')
            start = segment.get('start', 0)
            end = segment.get('end', 0)

            fig.add_trace(go.Bar(
                x=[end - start],
                y=[languages.index(language)],
                base=start,
                orientation='h',
                name=f"Language: {language}",
                marker_color=language_colors[language],
                showlegend=False
            ), row=2, col=1)

        fig.update_layout(height=600, title="Combined Timeline View")
        fig.update_xaxes(title_text="Time (seconds)")
        fig.update_yaxes(title_text="Speakers", row=1, col=1, ticktext=speakers, tickvals=list(range(len(speakers))))
        fig.update_yaxes(title_text="Languages", row=2, col=1, ticktext=languages, tickvals=list(range(len(languages))))

        st.plotly_chart(fig, use_container_width=True)

    def _render_metrics_tab(self, results: Dict[str, Any]):
        """Render quality metrics tab"""
        st.subheader("📊 Quality Metrics")

        metrics = results.get('metrics', {})

        if not metrics:
            st.info("No quality metrics available")
            return

        # Performance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Accuracy Metrics")
            if 'speaker_identification_accuracy' in metrics:
                st.metric("Speaker ID Accuracy", f"{metrics['speaker_identification_accuracy']:.2%}")
            if 'language_identification_accuracy' in metrics:
                st.metric("Language ID Accuracy", f"{metrics['language_identification_accuracy']:.2%}")

        with col2:
            st.subheader("📈 Error Rates")
            if 'diarization_error_rate' in metrics:
                st.metric("Diarization Error Rate", f"{metrics['diarization_error_rate']:.2%}")
            if 'word_error_rate' in metrics:
                st.metric("Word Error Rate", f"{metrics['word_error_rate']:.2%}")

        with col3:
            st.subheader("🚀 Performance")
            if 'real_time_factor' in metrics:
                st.metric("Real-time Factor", f"{metrics['real_time_factor']:.2f}x")
            if 'bleu_score' in metrics:
                st.metric("BLEU Score", f"{metrics['bleu_score']:.1f}")

        # Detailed metrics
        if metrics:
            with st.expander("📋 All Metrics", expanded=False):
                metrics_df = pd.DataFrame([
                    {'Metric': key.replace('_', ' ').title(), 'Value': value}
                    for key, value in metrics.items()
                ])
                st.dataframe(metrics_df, use_container_width=True)

    def _render_downloads_tab(self, results: Dict[str, Any]):
        """Render downloads tab"""
        st.subheader("📁 Download Results")

        # File downloads
        file_paths = {
            'Speaker ID (CSV)': results.get('sid_csv'),
            'Speaker Diarization (CSV)': results.get('sd_csv'),
            'Language ID (CSV)': results.get('lid_csv'),
            'ASR Transcription (TRN)': results.get('asr_trn'),
            'Translation (TXT)': results.get('nmt_txt')
        }

        available_files = {name: path for name, path in file_paths.items() if path}

        if available_files:
            st.info(f"📂 {len(available_files)} result files available for download")

            for file_name, file_path in available_files.items():
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.text(f"📄 {file_name}")
                    st.caption(f"Path: {file_path}")

                with col2:
                    if st.button(f"📥 Download", key=f"download_{file_name}", type="secondary"):
                        # Note: In a real implementation, you'd implement file download
                        st.info(f"Download functionality would be implemented for: {file_path}")

        # JSON export
        st.subheader("📊 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Export as JSON", type="secondary", use_container_width=True):
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=f"results_{results.get('job_id', 'unknown')}.json",
                    mime="application/json"
                )

        with col2:
            if st.button("📊 Export Segments CSV", type="secondary", use_container_width=True):
                segments = results.get('segments', [])
                if segments:
                    df = pd.DataFrame(segments)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_data,
                        file_name=f"segments_{results.get('job_id', 'unknown')}.csv",
                        mime="text/csv"
                    )

    def _render_failed_results(self, job_id: str, status_data: Dict[str, Any]):
        """Render failed job information"""
        status = status_data.get('status')
        error_msg = status_data.get('error_msg')

        if status == 'FAILED':
            st.error("❌ Processing Failed")
            st.write("The audio processing job has failed.")
        else:
            st.warning("⚠️ Processing Cancelled")
            st.write("The audio processing job was cancelled.")

        if error_msg:
            st.subheader("🔍 Error Details")
            st.code(error_msg)

        # Show job details
        with st.expander("📋 Job Information"):
            st.json(status_data)

    def _render_processing_status(self, job_id: str, status_data: Dict[str, Any]):
        """Render processing status for incomplete jobs"""
        status = status_data.get('status')
        progress = status_data.get('progress', 0)

        st.info(f"🔄 Processing in progress: {status}")
        st.progress(progress, text=f"Progress: {progress:.1%}")

        st.write("Results will be available once processing is complete.")

        # Auto-refresh button
        if st.button("🔄 Refresh Results", type="primary"):
            st.rerun()