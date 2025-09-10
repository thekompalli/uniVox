"""
Format Utilities
Competition output format generation utilities
"""
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import csv
import json
from datetime import datetime

from src.config.app_config import app_config

logger = logging.getLogger(__name__)


class FormatUtils:
    """Utility class for generating competition output formats"""
    
    def __init__(self):
        self.output_dir = app_config.data_dir / "results"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_competition_outputs(
        self,
        job_id: str,
        evaluation_id: str,
        processing_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate all competition output formats
        
        Args:
            job_id: Job identifier
            evaluation_id: Competition evaluation ID (e.g., "01", "02")
            processing_results: Complete processing results
            
        Returns:
            Dictionary of output file paths
        """
        try:
            logger.info(f"Generating competition outputs for job {job_id}, evaluation {evaluation_id}")
            
            output_files = {}
            
            # Generate each required format
            output_files['sid_csv'] = self._generate_sid_csv(
                job_id, evaluation_id, processing_results
            )
            
            output_files['sd_csv'] = self._generate_sd_csv(
                job_id, evaluation_id, processing_results
            )
            
            output_files['lid_csv'] = self._generate_lid_csv(
                job_id, evaluation_id, processing_results
            )
            
            output_files['asr_trn'] = self._generate_asr_trn(
                job_id, evaluation_id, processing_results
            )
            
            output_files['nmt_txt'] = self._generate_nmt_txt(
                job_id, evaluation_id, processing_results
            )
            
            logger.info(f"Generated all competition outputs for job {job_id}")
            return output_files
            
        except Exception as e:
            logger.exception(f"Error generating competition outputs: {e}")
            raise
    
    def _generate_sid_csv(
        self,
        job_id: str,
        evaluation_id: str,
        results: Dict[str, Any]
    ) -> str:
        """Generate SID_XX.csv file"""
        try:
            filename = f"SID_{evaluation_id}.csv"
            filepath = self.output_dir / job_id / filename
            filepath.parent.mkdir(exist_ok=True)
            
            # Extract speaker identification data
            speaker_segments = results.get('speaker_identification', {}).get('segments', [])
            audio_filename = results.get('audio_filename', f'ps6_{evaluation_id}_001.wav')
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header comment (optional)
                # writer.writerow(['# Audio File Name', 'speaker ID', 'confidence score (%)', 'start TS', 'end TS'])
                
                for segment in speaker_segments:
                    writer.writerow([
                        audio_filename,
                        segment.get('speaker_id', segment.get('speaker', 'unknown')),
                        int(segment.get('confidence', 0.8) * 100),  # Convert to percentage
                        f"{segment.get('start', 0.0):.3f}",
                        f"{segment.get('end', 0.0):.3f}"
                    ])
            
            logger.debug(f"Generated SID CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.exception(f"Error generating SID CSV: {e}")
            raise
    
    def _generate_sd_csv(
        self,
        job_id: str,
        evaluation_id: str,
        results: Dict[str, Any]
    ) -> str:
        """Generate SD_XX.csv file"""
        try:
            filename = f"SD_{evaluation_id}.csv"
            filepath = self.output_dir / job_id / filename
            filepath.parent.mkdir(exist_ok=True)
            
            # Extract speaker diarization data
            diarization_segments = results.get('diarization', {}).get('segments', [])
            audio_filename = results.get('audio_filename', f'ps6_{evaluation_id}_001.wav')
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                for segment in diarization_segments:
                    writer.writerow([
                        audio_filename,
                        segment.get('speaker', 'speaker1'),
                        int(segment.get('confidence', 0.8) * 100),  # Convert to percentage
                        f"{segment.get('start', 0.0):.3f}",
                        f"{segment.get('end', 0.0):.3f}"
                    ])
            
            logger.debug(f"Generated SD CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.exception(f"Error generating SD CSV: {e}")
            raise
    
    def _generate_lid_csv(
        self,
        job_id: str,
        evaluation_id: str,
        results: Dict[str, Any]
    ) -> str:
        """Generate LID_XX.csv file"""
        try:
            filename = f"LID_{evaluation_id}.csv"
            filepath = self.output_dir / job_id / filename
            filepath.parent.mkdir(exist_ok=True)
            
            # Extract language identification data
            language_segments = results.get('language_identification', {}).get('segments', [])
            audio_filename = results.get('audio_filename', f'ps6_{evaluation_id}_001.wav')
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                for segment in language_segments:
                    writer.writerow([
                        audio_filename,
                        segment.get('language', 'unknown'),
                        int(segment.get('confidence', 0.8) * 100),  # Convert to percentage
                        f"{segment.get('start', 0.0):.3f}",
                        f"{segment.get('end', 0.0):.3f}"
                    ])
            
            logger.debug(f"Generated LID CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.exception(f"Error generating LID CSV: {e}")
            raise
    
    def _generate_asr_trn(
        self,
        job_id: str,
        evaluation_id: str,
        results: Dict[str, Any]
    ) -> str:
        """Generate ASR_XX.trn file"""
        try:
            filename = f"ASR_{evaluation_id}.trn"
            filepath = self.output_dir / job_id / filename
            filepath.parent.mkdir(exist_ok=True)
            
            # Extract ASR transcription data
            transcription_segments = results.get('transcription', {}).get('segments', [])
            audio_filename = results.get('audio_filename', f'ps6_{evaluation_id}_001.wav')
            
            with open(filepath, 'w', encoding='utf-8') as trnfile:
                for segment in transcription_segments:
                    transcript = segment.get('text', '').strip()
                    if transcript:  # Only write non-empty transcripts
                        trnfile.write(
                            f"{audio_filename}, "
                            f"{segment.get('start', 0.0):.3f}, "
                            f"{segment.get('end', 0.0):.3f}, "
                            f"{transcript}\n"
                        )
            
            logger.debug(f"Generated ASR TRN: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.exception(f"Error generating ASR TRN: {e}")
            raise
    
    def _generate_nmt_txt(
        self,
        job_id: str,
        evaluation_id: str,
        results: Dict[str, Any]
    ) -> str:
        """Generate NMT_XX.txt file"""
        try:
            filename = f"NMT_{evaluation_id}.txt"
            filepath = self.output_dir / job_id / filename
            filepath.parent.mkdir(exist_ok=True)
            
            # Extract translation data
            translation_segments = results.get('translation', {}).get('segments', [])
            audio_filename = results.get('audio_filename', f'ps6_{evaluation_id}_001.wav')
            
            with open(filepath, 'w', encoding='utf-8') as txtfile:
                for segment in translation_segments:
                    translation = segment.get('translated_text', '').strip()
                    if translation:  # Only write non-empty translations
                        txtfile.write(
                            f"{audio_filename}, "
                            f"{segment.get('start', 0.0):.3f}, "
                            f"{segment.get('end', 0.0):.3f}, "
                            f"{translation}\n"
                        )
            
            logger.debug(f"Generated NMT TXT: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.exception(f"Error generating NMT TXT: {e}")
            raise
    
    def generate_solution_hash(self, job_id: str) -> str:
        """Generate MD5 hash for solution verification"""
        try:
            # Create a string representing the complete solution
            solution_data = {
                'job_id': job_id,
                'timestamp': datetime.utcnow().isoformat(),
                'system_version': app_config.version,
                'processing_config': {
                    'models': {
                        'whisper': 'large-v3',
                        'pyannote': '3.1',
                        'indictrans2': '1B'
                    }
                }
            }
            
            # Convert to string and hash
            solution_string = json.dumps(solution_data, sort_keys=True)
            hash_md5 = hashlib.md5(solution_string.encode('utf-8')).hexdigest()
            
            # Save hash file
            hash_filename = f"PS_06_{job_id}_solution.hash"
            hash_filepath = self.output_dir / job_id / hash_filename
            hash_filepath.parent.mkdir(exist_ok=True)
            
            with open(hash_filepath, 'w') as hash_file:
                hash_file.write(hash_md5)
            
            logger.info(f"Generated solution hash: {hash_md5}")
            return hash_md5
            
        except Exception as e:
            logger.exception(f"Error generating solution hash: {e}")
            return "error_generating_hash"
    
    def validate_output_format(self, file_path: str, format_type: str) -> bool:
        """Validate output file format compliance"""
        try:
            filepath = Path(file_path)
            
            if not filepath.exists():
                logger.error(f"Output file does not exist: {file_path}")
                return False
            
            if format_type in ['sid_csv', 'sd_csv', 'lid_csv']:
                return self._validate_csv_format(filepath, format_type)
            elif format_type == 'asr_trn':
                return self._validate_trn_format(filepath)
            elif format_type == 'nmt_txt':
                return self._validate_txt_format(filepath)
            else:
                logger.error(f"Unknown format type: {format_type}")
                return False
                
        except Exception as e:
            logger.exception(f"Error validating output format: {e}")
            return False
    
    def _validate_csv_format(self, filepath: Path, format_type: str) -> bool:
        """Validate CSV format compliance"""
        try:
            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                for i, row in enumerate(reader):
                    if len(row) != 5:
                        logger.error(f"Row {i+1} has {len(row)} columns, expected 5")
                        return False
                    
                    # Validate audio filename
                    if not row[0].strip():
                        logger.error(f"Row {i+1} has empty audio filename")
                        return False
                    
                    # Validate timestamps
                    try:
                        start_time = float(row[3])
                        end_time = float(row[4])
                        if end_time <= start_time:
                            logger.error(f"Row {i+1} has invalid timestamps: {start_time} >= {end_time}")
                            return False
                    except ValueError:
                        logger.error(f"Row {i+1} has invalid timestamp format")
                        return False
                    
                    # Validate confidence score
                    try:
                        confidence = int(row[2])
                        if not (0 <= confidence <= 100):
                            logger.error(f"Row {i+1} has invalid confidence: {confidence}")
                            return False
                    except ValueError:
                        logger.error(f"Row {i+1} has invalid confidence format")
                        return False
            
            return True
            
        except Exception as e:
            logger.exception(f"Error validating CSV format: {e}")
            return False
    
    def _validate_trn_format(self, filepath: Path) -> bool:
        """Validate TRN format compliance"""
        try:
            with open(filepath, 'r', encoding='utf-8') as trnfile:
                for i, line in enumerate(trnfile, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for required comma-separated fields
                    parts = line.split(', ', 3)
                    if len(parts) != 4:
                        logger.error(f"Line {i} has {len(parts)} parts, expected 4")
                        return False
                    
                    # Validate timestamps
                    try:
                        start_time = float(parts[1])
                        end_time = float(parts[2])
                        if end_time <= start_time:
                            logger.error(f"Line {i} has invalid timestamps")
                            return False
                    except ValueError:
                        logger.error(f"Line {i} has invalid timestamp format")
                        return False
                    
                    # Check transcript is not empty
                    if not parts[3].strip():
                        logger.error(f"Line {i} has empty transcript")
                        return False
            
            return True
            
        except Exception as e:
            logger.exception(f"Error validating TRN format: {e}")
            return False
    
    def _validate_txt_format(self, filepath: Path) -> bool:
        """Validate TXT format compliance (same as TRN)"""
        return self._validate_trn_format(filepath)
    
    def create_submission_package(
        self,
        job_id: str,
        evaluation_id: str,
        output_files: Dict[str, str],
        solution_description: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create complete submission package for competition"""
        try:
            import tarfile
            
            package_name = f"PS_06_{job_id}_{evaluation_id}.tar.gz"
            package_path = self.output_dir / package_name
            
            with tarfile.open(package_path, 'w:gz') as tar:
                # Add all output files
                for file_type, file_path in output_files.items():
                    if Path(file_path).exists():
                        tar.add(file_path, arcname=Path(file_path).name)
                
                # Add solution hash
                hash_file = self.output_dir / job_id / f"PS_06_{job_id}_solution.hash"
                if hash_file.exists():
                    tar.add(hash_file, arcname=hash_file.name)
                
                # Add solution description if provided
                if solution_description:
                    desc_file = self.output_dir / job_id / "solution_description.json"
                    with open(desc_file, 'w') as f:
                        json.dump(solution_description, f, indent=2)
                    tar.add(desc_file, arcname="solution_description.json")
            
            logger.info(f"Created submission package: {package_path}")
            return str(package_path)
            
        except Exception as e:
            logger.exception(f"Error creating submission package: {e}")
            raise
    
    def get_performance_summary(
        self, 
        processing_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract performance summary for competition metrics"""
        try:
            summary = {}
            
            # Speaker Identification Accuracy
            speaker_metrics = processing_results.get('speaker_identification', {}).get('quality_metrics', {})
            summary['speaker_identification_accuracy'] = speaker_metrics.get('average_confidence', 0.0)
            
            # Diarization Error Rate (estimated)
            diar_metrics = processing_results.get('diarization', {}).get('quality_metrics', {})
            # DER is complex to calculate without ground truth, use inverse of confidence as estimate
            summary['diarization_error_rate'] = 1.0 - diar_metrics.get('average_confidence', 0.8)
            
            # Language Identification Error Rate
            lang_metrics = processing_results.get('language_identification', {}).get('quality_metrics', {})
            summary['language_error_rate'] = 1.0 - lang_metrics.get('average_confidence', 0.8)
            
            # Word Error Rate (estimated)
            asr_metrics = processing_results.get('transcription', {}).get('quality_metrics', {})
            summary['word_error_rate'] = 1.0 - asr_metrics.get('average_confidence', 0.8)
            
            # BLEU Score (estimated)
            trans_metrics = processing_results.get('translation', {}).get('quality_metrics', {})
            summary['bleu_score'] = trans_metrics.get('average_confidence', 0.0) * 100
            
            return summary
            
        except Exception as e:
            logger.exception(f"Error getting performance summary: {e}")
            return {}
    
    def cleanup_output_files(self, job_id: str, keep_final: bool = True):
        """Clean up intermediate output files"""
        try:
            job_output_dir = self.output_dir / job_id
            
            if not job_output_dir.exists():
                return
            
            if keep_final:
                # Keep final competition format files
                final_files = ['SID_', 'SD_', 'LID_', 'ASR_', 'NMT_', '.hash']
                
                for file_path in job_output_dir.iterdir():
                    if not any(final_file in file_path.name for final_file in final_files):
                        file_path.unlink()
                        logger.debug(f"Cleaned up intermediate file: {file_path}")
            else:
                # Remove entire job directory
                import shutil
                shutil.rmtree(job_output_dir)
                logger.info(f"Cleaned up all output files for job {job_id}")
                
        except Exception as e:
            logger.exception(f"Error cleaning up output files: {e}")